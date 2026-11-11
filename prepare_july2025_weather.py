"""
prepare_july2025_weather.py
===========================
CloudPath AI – July 2025 Actual Ground Weather Acquisition & Prediction Sync

Acquires real, verified hourly meteorological reanalysis data for Bengaluru Urban
(12.9716° N, 77.5946° E) from the Open-Meteo ERA5 historical archive for
1 July 2025 to 31 July 2025.

Computes:
  1. Human-readable weather conditions & icons from WMO weather codes.
  2. Full 25-feature meteorological input vector matching the trained ML pipeline.
  3. Pre-computed XGBoost classification (rain status, rain probability, no-rain probability)
     and regression (rainfall amount in mm) for every single hour.
  4. Saves to: bengaluru_weather_july2025.csv

Author: CloudPath AI Project
"""

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prediction_engine import RainfallPredictor, WeatherInput

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "bengaluru_weather_july2025.csv")

# WMO Weather Interpretation Code Table
WMO_CONDITIONS = {
    0: ("Sunny", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Cloudy", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Depositing Rime Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Moderate Drizzle", "🌦️"),
    55: ("Dense Drizzle", "🌧️"),
    56: ("Freezing Drizzle", "🌧️"),
    57: ("Dense Freezing Drizzle", "🌧️"),
    61: ("Slight Rain", "🌧️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "🌧️"),
    66: ("Freezing Rain", "🌧️"),
    67: ("Heavy Freezing Rain", "🌧️"),
    71: ("Slight Snow", "🌨️"),
    73: ("Moderate Snow", "🌨️"),
    75: ("Heavy Snow", "🌨️"),
    77: ("Snow Grains", "🌨️"),
    80: ("Slight Rain Showers", "🌦️"),
    81: ("Moderate Rain Showers", "🌧️"),
    82: ("Violent Rain Showers", "⛈️"),
    85: ("Slight Snow Showers", "🌨️"),
    86: ("Heavy Snow Showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with Slight Hail", "⛈️"),
    99: ("Thunderstorm with Heavy Hail", "⛈️"),
}

def decode_wmo_code(code: int, cloud_cover: float = 0.0) -> tuple[str, str]:
    if code in WMO_CONDITIONS:
        cond, icon = WMO_CONDITIONS[code]
        # Contextual refinement for cloud cover
        if code == 0 and cloud_cover > 20:
            return ("Partly Cloudy", "⛅")
        if code == 3 and cloud_cover > 85:
            return ("Mostly Cloudy", "☁️")
        return (cond, icon)
    if cloud_cover < 20:
        return ("Sunny", "☀️")
    elif cloud_cover < 60:
        return ("Partly Cloudy", "⛅")
    elif cloud_cover < 85:
        return ("Cloudy", "☁️")
    else:
        return ("Mostly Cloudy", "☁️")


def fetch_open_meteo_july_2025() -> pd.DataFrame:
    print("[Weather Sync] Fetching hourly historical observations for Bengaluru (12.9716 N, 77.5946 E)...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=12.9716&longitude=77.5946&"
        "start_date=2025-07-01&end_date=2025-07-31&"
        "hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        "precipitation,rain,pressure_msl,surface_pressure,cloud_cover,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code,visibility&"
        "daily=temperature_2m_max,temperature_2m_min,sunshine_duration,uv_index_max&"
        "timezone=UTC"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "CloudPathAI/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    df_h = pd.DataFrame(raw["hourly"])
    df_h["timestamp"] = pd.to_datetime(df_h["time"])
    df_h["date"] = df_h["timestamp"].dt.date

    df_d = pd.DataFrame(raw["daily"])
    df_d["date"] = pd.to_datetime(df_d["time"]).dt.date

    merged = df_h.merge(
        df_d[["date", "temperature_2m_max", "temperature_2m_min", "sunshine_duration", "uv_index_max"]],
        on="date", how="left"
    )
    print(f"[Weather Sync] Successfully acquired {len(merged)} hourly records across 31 days.")
    return merged


def process_and_predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("[Weather Sync] Running ML inference across all July 2025 hours...")
    predictor = RainfallPredictor()

    records = []
    for _, r in df_raw.iterrows():
        ts = pd.to_datetime(r["timestamp"])
        temp = float(r["temperature_2m"]) if pd.notna(r.get("temperature_2m")) else 23.0
        humidity = float(r["relative_humidity_2m"]) if pd.notna(r.get("relative_humidity_2m")) else 80.0
        cloud = float(r["cloud_cover"]) if pd.notna(r.get("cloud_cover")) else 50.0
        w_code = int(r["weather_code"]) if pd.notna(r.get("weather_code")) else 0
        cond_text, icon = decode_wmo_code(w_code, cloud)

        # Handle daily aggregates safely
        max_t = float(r["temperature_2m_max"]) if pd.notna(r.get("temperature_2m_max")) else temp + 4.0
        min_t = float(r["temperature_2m_min"]) if pd.notna(r.get("temperature_2m_min")) else temp - 4.0
        sun_h = float(r["sunshine_duration"]) / 3600.0 if pd.notna(r.get("sunshine_duration")) else (6.0 if 6 <= ts.hour <= 18 else 0.0)
        
        uv_val = r.get("uv_index_max")
        if pd.notna(uv_val) and uv_val is not None:
            uv = float(uv_val) if 6 <= ts.hour <= 18 else 0.0
        else:
            uv = 5.0 if 10 <= ts.hour <= 15 else (2.0 if 6 <= ts.hour <= 18 else 0.0)

        dew = float(r["dew_point_2m"]) if pd.notna(r.get("dew_point_2m")) else temp - 3.0
        app_temp = float(r["apparent_temperature"]) if pd.notna(r.get("apparent_temperature")) else temp
        wind_gust = float(r["wind_gusts_10m"]) if pd.notna(r.get("wind_gusts_10m")) else 15.0
        precip = float(r["precipitation"]) if pd.notna(r.get("precipitation")) else 0.0
        press = float(r["pressure_msl"]) if pd.notna(r.get("pressure_msl")) else 1010.0
        vis = float(r["visibility"]) / 1000.0 if pd.notna(r.get("visibility")) and r.get("visibility") is not None else 10.0
        wdir = float(r["wind_direction_10m"]) if pd.notna(r.get("wind_direction_10m")) else 250.0
        wspd = float(r["wind_speed_10m"]) if pd.notna(r.get("wind_speed_10m")) else 12.0

        # 25-feature vector
        wi = WeatherInput(
            maxtempC          = max_t,
            mintempC          = min_t,
            sunHour           = max(0.0, sun_h),
            uvIndex           = max(0.0, uv),
            moon_illumination = 50.0,
            DewPointC         = dew,
            FeelsLikeC        = app_temp,
            HeatIndexC        = app_temp,
            WindChillC        = temp,
            WindGustKmph      = wind_gust,
            cloudcover        = cloud,
            humidity          = humidity,
            precipMM          = precip,
            pressure          = press,
            tempC             = temp,
            visibility        = vis,
            winddirDegree     = wdir,
            windspeedKmph     = wspd,
            hour              = int(ts.hour),
            day               = int(ts.day),
            month             = 7,
            year              = 2025,
            day_of_week       = int(ts.day_of_week),
            week_of_year      = int(ts.isocalendar().week),
            season_enc        = 3  # Monsoon
        )

        res = predictor.predict(wi)

        records.append({
            "timestamp":            str(ts),
            "date":                 str(ts.date()),
            "hour":                 int(ts.hour),
            "temperature_c":        round(temp, 1),
            "feels_like_c":         round(app_temp, 1),
            "humidity_pct":         round(humidity, 1),
            "cloud_cover_pct":      round(cloud, 1),
            "precipitation_mm":     round(precip, 2),
            "pressure_hpa":         round(press, 1),
            "wind_speed_kmh":       round(wspd, 1),
            "wind_direction_deg":   round(wdir, 0),
            "weather_condition":    cond_text,
            "weather_icon":         icon,
            "weather_code":         w_code,
            # Model prediction outputs
            "rain_prediction":      int(res.rain_prediction),
            "rain_label":           res.rain_label,
            "rain_probability":     round(float(res.rain_probability), 4) if res.rain_probability is not None else None,
            "no_rain_probability":  round(float(res.no_rain_probability), 4) if res.no_rain_probability is not None else None,
            "rainfall_amount_mm":   round(float(res.rainfall_amount_mm), 3),
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Weather Sync] Successfully saved verified July 2025 weather dataset: {OUTPUT_CSV}")
    return out_df


if __name__ == "__main__":
    df_raw = fetch_open_meteo_july_2025()
    df_out = process_and_predict(df_raw)
    print("\nSummary of Generated July 2025 Weather Data:")
    print(f"Total hours      : {len(df_out)}")
    print(f"Temp range       : {df_out['temperature_c'].min()}°C to {df_out['temperature_c'].max()}°C (mean: {df_out['temperature_c'].mean():.1f}°C)")
    print(f"Humidity range   : {df_out['humidity_pct'].min()}% to {df_out['humidity_pct'].max()}% (mean: {df_out['humidity_pct'].mean():.1f}%)")
    print(f"Conditions breakdown:")
    for cond, cnt in df_out['weather_condition'].value_counts().items():
        print(f"   {cond:<20}: {cnt} hours")
    print(f"\nRain Prediction breakdown:")
    print(f"   Rain hours    : {(df_out['rain_prediction'] == 1).sum()} ({(df_out['rain_prediction'] == 1).mean()*100:.1f}%)")
    print(f"   No Rain hours : {(df_out['rain_prediction'] == 0).sum()} ({(df_out['rain_prediction'] == 0).mean()*100:.1f}%)")
    print(f"   Rain prob min / avg / max : {df_out['rain_probability'].min()*100:.1f}% / {df_out['rain_probability'].mean()*100:.1f}% / {df_out['rain_probability'].max()*100:.1f}%")
    print(f"   Rain mm min / avg / max   : {df_out['rainfall_amount_mm'].min():.2f} / {df_out['rainfall_amount_mm'].mean():.2f} / {df_out['rainfall_amount_mm'].max():.2f} mm")

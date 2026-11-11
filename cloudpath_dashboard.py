"""
cloudpath_dashboard.py
======================
CloudPath AI – Phase 5: Integration Dashboard
Console-based dashboard showing satellite observations + rainfall predictions.

Runs in two modes:
  --live     : Enter weather values interactively for a real-time prediction.
  --batch    : Run predictions over a slice of historical weather data.
  --status   : Show current satellite data summary without weather input.

Author: CloudPath AI Project
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integration_pipeline import (
    load_weather_data, load_satellite_summary, load_movement_data,
    check_temporal_overlap, build_satellite_feature_table, build_merged_dataset,
)
from prediction_engine import (
    RainfallPredictor, WeatherInput, SatelliteContext,
    weather_input_from_row, satellite_context_from_row,
)

SEP = "=" * 65


def print_header():
    print("\n" + SEP)
    print("  CloudPath AI - Cloud Movement Tracking & Rainfall Prediction")
    print(SEP)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)


def print_satellite_panel(sat_row: pd.Series, latest: bool = True):
    tag = "Latest" if latest else "Observation"
    print(f"\n[SATELLITE] {tag}: {sat_row.get('timestamp', 'N/A')}")
    print(f"  Source              : INSAT-3DS / NRSC-ISRO (actual data)")
    print(f"  Cloud Coverage      : {sat_row.get('cloud_coverage_pct', 'N/A'):.2f}%")
    print(f"  Clear Sky           : {sat_row.get('clear_coverage_pct', 'N/A'):.2f}%")
    print(f"  Cloud Regions       : {int(sat_row.get('n_regions', 0))}")
    print(f"  Delta Coverage      : {sat_row.get('delta_cloud_pct', 0):+.2f}%  "
          f"({'increasing' if sat_row.get('delta_cloud_pct',0) > 0 else 'decreasing' if sat_row.get('delta_cloud_pct',0) < 0 else 'stable'})")

    speed = sat_row.get("mean_speed_kmh")
    if pd.notna(speed) and speed is not None:
        print(f"  Mean Cloud Speed    : {speed:.1f} km/h")
        print(f"  Movement Direction  : {sat_row.get('dominant_dir', 'N/A')}")
        conf = sat_row.get("mean_conf_score")
        if pd.notna(conf) and conf is not None:
            conf_label = "High" if conf > 0.7 else "Medium" if conf > 0.4 else "Low"
            print(f"  Tracking Confidence : {conf_label} ({conf:.2f})")
    else:
        print("  Mean Cloud Speed    : N/A (no tracked clouds in this slot)")
        print("  Movement Direction  : N/A")


def print_prediction_panel(result):
    rain_sym = "RAIN" if result.rain_prediction == 1 else "NO RAIN"
    print(f"\n[RAINFALL PREDICTION]")
    print(f"  Prediction          : {rain_sym}")
    if result.rain_probability is not None:
        pct = result.rain_probability * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  Rain Probability    : {pct:.1f}%  [{bar}]")
    print(f"  Predicted Amount    : {result.rainfall_amount_mm:.3f} mm")
    print(f"  Classifier          : {result.classifier_name}")
    print(f"  Regressor           : {result.regressor_name}")
    print(f"  Input Complete      : {'Yes' if result.input_complete else 'No'}")
    if result.warnings:
        for w in result.warnings:
            print(f"  [!] {w}")


def print_overlap_warning(overlap: dict):
    if not overlap["has_overlap"]:
        print("\n[DATA COMPATIBILITY WARNING]")
        print(f"  Weather data  : {overlap['weather_start'].date()} to {overlap['weather_end'].date()}")
        print(f"  Satellite data: {overlap['satellite_start'].date()} to {overlap['satellite_end'].date()}")
        print("  TEMPORAL OVERLAP: NONE")
        print("  -> Satellite features are displayed as context only.")
        print("  -> The ML models use weather features from user input.")
        print("  -> Satellite + weather COMBINED validation is not")
        print("     possible until a dataset with both types for the")
        print("     same time period is available.")
    else:
        print(f"\n[DATA COMPATIBILITY] Overlap from "
              f"{overlap['overlap_start'].date()} to "
              f"{overlap['overlap_end'].date()}")


def mode_status(sat_feat: pd.DataFrame, overlap: dict):
    """Show satellite status dashboard without weather prediction."""
    print_header()
    print_overlap_warning(overlap)

    latest = sat_feat.iloc[-1]
    print_satellite_panel(latest, latest=True)

    # 4-day trend
    print("\n[4-DAY SATELLITE TREND]")
    print(f"  {'Date/Time':<22}  {'Cloud%':>7}  {'Regions':>7}  "
          f"{'Speed(km/h)':>11}  {'Dir':>5}")
    print("  " + "-" * 58)
    step = max(1, len(sat_feat) // 12)
    for _, r in sat_feat.iloc[::step].iterrows():
        speed_str = f"{r['mean_speed_kmh']:.1f}" if pd.notna(r.get("mean_speed_kmh")) else "  N/A"
        dir_str   = r.get("dominant_dir", "N/A") or "N/A"
        print(f"  {str(r['timestamp']):<22}  {r['cloud_coverage_pct']:>7.1f}  "
              f"{int(r['n_regions']):>7}  {speed_str:>11}  {dir_str:>5}")

    print(f"\n  Observations : {len(sat_feat)}")
    print(f"  Date range   : {sat_feat['timestamp'].min()} to "
          f"{sat_feat['timestamp'].max()}")


def mode_live(predictor: RainfallPredictor,
              sat_feat: pd.DataFrame,
              overlap: dict):
    """Interactive mode: enter weather values and get prediction."""
    print_header()
    print_overlap_warning(overlap)

    # Show latest satellite observation
    latest_sat = sat_feat.iloc[-1]
    print_satellite_panel(latest_sat, latest=True)

    print("\n[WEATHER INPUT]  Enter current hourly weather values.")
    print("  (Press Enter to use example/default values for testing)\n")

    def ask(prompt, default, cast=float):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    wi = WeatherInput(
        maxtempC          = ask("Max temp (°C)",       30),
        mintempC          = ask("Min temp (°C)",       22),
        sunHour           = ask("Sun hours",           6),
        uvIndex           = ask("UV index",            5),
        moon_illumination = ask("Moon illumination",   50),
        DewPointC         = ask("Dew point (°C)",      20),
        FeelsLikeC        = ask("Feels like (°C)",     30),
        HeatIndexC        = ask("Heat index (°C)",     30),
        WindChillC        = ask("Wind chill (°C)",     30),
        WindGustKmph      = ask("Wind gust (km/h)",    20),
        cloudcover        = ask("Cloud cover (%)",     60),
        humidity          = ask("Humidity (%)",        75),
        precipMM          = ask("Current precip (mm)", 0),
        pressure          = ask("Pressure (hPa)",      1010),
        tempC             = ask("Temperature (°C)",    27),
        visibility        = ask("Visibility (km)",     8),
        winddirDegree     = ask("Wind direction (°)",  270),
        windspeedKmph     = ask("Wind speed (km/h)",   15),
        hour              = ask("Hour (0-23)",         12, int),
        day               = ask("Day",                 1, int),
        month             = ask("Month",               10, int),
        year              = ask("Year",                2025, int),
        day_of_week       = ask("Day of week (0=Mon)", 2, int),
        week_of_year      = ask("Week of year",        40, int),
        season_enc        = ask("Season (0=Win,1=Spr,2=Sum,3=Mon,4=Aut)", 3, int),
    )

    sat_ctx = satellite_context_from_row(latest_sat)
    result  = predictor.predict(wi, satellite=sat_ctx)
    print_prediction_panel(result)
    print(f"\n  [Note] {result.validation_note}")


def mode_batch(predictor: RainfallPredictor,
               weather_df: pd.DataFrame,
               sat_feat: pd.DataFrame,
               overlap: dict,
               n_samples: int = 20):
    """Run predictions on a sample of historical weather data."""
    print_header()
    print_overlap_warning(overlap)

    print(f"\n[BATCH PREDICTION]  Running on {n_samples} historical weather samples")
    print("  (weather data only — no temporally matching satellite data)\n")

    # Use test portion (last 20%)
    test_df = weather_df.tail(n_samples * 10).sample(n=n_samples, random_state=42)
    test_df = test_df.reset_index(drop=True)

    correct_clf = 0
    records = []
    for _, row in test_df.iterrows():
        wi     = weather_input_from_row(row)
        result = predictor.predict(wi)
        actual = int(row["rain_status"])
        actual_mm = float(row["next_hour_rainfall"])
        match  = (result.rain_prediction == actual)
        correct_clf += int(match)
        records.append({
            "date_time":    row["date_time"],
            "actual_label": actual,
            "pred_label":   result.rain_prediction,
            "correct":      match,
            "actual_mm":    actual_mm,
            "pred_mm":      result.rainfall_amount_mm,
            "rain_prob":    result.rain_probability,
        })

    df_res = pd.DataFrame(records)
    acc = correct_clf / n_samples

    print(f"  {'Date/Time':<22}  {'Actual':>7}  {'Predicted':>9}  "
          f"{'OK':>4}  {'ActualMM':>9}  {'PredMM':>9}  {'Prob':>6}")
    print("  " + "-" * 75)
    for _, r in df_res.iterrows():
        ok = "OK" if r["correct"] else "MISS"
        print(f"  {str(r['date_time']):<22}  "
              f"{'Rain' if r['actual_label'] else 'NoRain':>7}  "
              f"{'Rain' if r['pred_label'] else 'NoRain':>9}  "
              f"{ok:>4}  {r['actual_mm']:>9.3f}  "
              f"{r['pred_mm']:>9.3f}  "
              f"{r['rain_prob']:.3f}" if r['rain_prob'] is not None else "  N/A")

    print(f"\n  Accuracy on this sample: {acc*100:.1f}%  ({correct_clf}/{n_samples})")
    print(f"\n  [Note] These are real historical weather inputs from the original")
    print(f"  test set. Satellite context was not available for these dates.")

    # Save batch results
    out_path = os.path.join(PROJECT_ROOT, "integration_outputs",
                             "batch_predictions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"\n  Saved batch results: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CloudPath AI Integration Dashboard"
    )
    parser.add_argument("--mode", choices=["status", "live", "batch"],
                        default="status",
                        help="status: satellite view | live: interactive prediction "
                             "| batch: historical test")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of samples for batch mode")
    args = parser.parse_args()

    # Load all data
    print("Loading data ...", flush=True)
    weather_df   = load_weather_data()
    satellite_df = load_satellite_summary()
    movement_df  = load_movement_data()
    overlap      = check_temporal_overlap(weather_df, satellite_df)
    sat_feat     = build_satellite_feature_table(satellite_df, movement_df)
    predictor    = RainfallPredictor()

    if args.mode == "status":
        mode_status(sat_feat, overlap)
    elif args.mode == "live":
        mode_live(predictor, sat_feat, overlap)
    elif args.mode == "batch":
        mode_batch(predictor, weather_df, sat_feat, overlap, args.samples)


if __name__ == "__main__":
    main()


import os
import sys
import pandas as pd
from integration_pipeline import load_satellite_summary, load_movement_data, build_satellite_feature_table

def main():
    sat = build_satellite_feature_table(load_satellite_summary(), load_movement_data())
    sat["dt"] = pd.to_datetime(sat["timestamp"])
    wx = pd.read_csv("bengaluru_weather_july2025.csv")
    wx["dt"] = pd.to_datetime(wx["timestamp"])
    sat = sat.sort_values("dt")
    wx = wx.sort_values("dt")
    m = pd.merge_asof(sat, wx, on="dt", direction="nearest", suffixes=("", "_wx"))
    m["delta_min"] = ((m["dt"] - pd.to_datetime(m["timestamp_wx"])).abs().dt.total_seconds() / 60).astype(int)

    samples = [
        ("Early July Daytime", "2025-07-01 10:00:00"),
        ("Early July Nighttime", "2025-07-05 21:30:00"),
        ("Mid-July Daytime", "2025-07-15 08:30:00"),
        ("Late July Daytime", "2025-07-23 11:00:00"),
        ("Late July Nighttime", "2025-07-31 20:30:00"),
    ]

    print("=" * 70)
    print("VERIFICATION OF 5 DISTINCT JULY 2025 OBSERVATIONS")
    print("=" * 70)

    for label, ts_target in samples:
        sub = m[m["timestamp"] == ts_target]
        if len(sub) == 0:
            dt_target = pd.to_datetime(ts_target)
            idx = (m["dt"] - dt_target).abs().idxmin()
            sub = m.iloc[[idx]]
        row = sub.iloc[0]
        prob_pct = row["rain_probability"] * 100.0 if pd.notna(row["rain_probability"]) else 0.0
        print(f"[{label}]")
        print(f"  Observation Index   : {sub.index[0]}")
        print(f"  Selected Timestamp  : {row['timestamp']} UTC")
        print(f"  Matched Weather TS  : {row['timestamp_wx']} UTC (delta: {row['delta_min']} min)")
        print(f"  Weather Condition   : {row['weather_condition']} {row['weather_icon']}")
        print(f"  Temperature         : {row['temperature_c']:.1f} °C")
        print(f"  Humidity            : {row['humidity_pct']:.1f} %")
        print(f"  Feels Like          : {row['feels_like_c']:.1f} °C")
        print(f"  Cloud Coverage      : {row['cloud_coverage_pct']:.2f} %")
        print(f"  Cloud Regions       : {int(row['n_regions'])}")
        print(f"  Cloud Speed         : {row['mean_speed_kmh']} km/h ({row['dominant_dir']})")
        print(f"  Rain/No-Rain Pred   : {row['rain_label']}")
        print(f"  Rain Probability    : {prob_pct:.2f} %")
        print(f"  Next Hr Rainfall    : {row['rainfall_amount_mm']:.3f} mm")
        print("-" * 70)

if __name__ == "__main__":
    main()

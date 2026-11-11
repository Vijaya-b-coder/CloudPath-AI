"""
integration_pipeline.py
========================
CloudPath AI – Phase 5: System Integration
Module 1: Load and inspect all data sources, check temporal compatibility,
          and build the merged feature set when dates overlap.

KEY FINDING (from actual data inspection):
  Weather dataset  : 2009-01-01 to 2020-01-01  (hourly)
  Satellite data   : 2025-07-01 to 2025-07-31  (30-min)
  OVERLAP          : NONE — the two datasets cover different years.

This module:
  - Loads both data sources.
  - Reports the overlap status honestly.
  - Builds a per-timestamp satellite feature table for use with the
    prediction pipeline even without historical overlap.
  - Extracts matching rows when a future dataset does overlap.

Author: CloudPath AI Project
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── File paths ───────────────────────────────────────────────────────
WEATHER_CSV   = os.path.join(PROJECT_ROOT, "bengaluru_processed.csv")
SATELLITE_CSV = os.path.join(PROJECT_ROOT, "satellite_data_outputs",
                              "bengaluru_satellite_cloud_summary.csv")
MOVEMENT_CSV  = os.path.join(PROJECT_ROOT, "satellite_tracking_outputs",
                              "cloud_movement_summary.csv")
TRACKS_CSV    = os.path.join(PROJECT_ROOT, "satellite_tracking_outputs",
                              "cloud_tracking_results.csv")


def load_weather_data() -> pd.DataFrame:
    """Load the processed weather dataset (Phase 1 output)."""
    df = pd.read_csv(WEATHER_CSV, parse_dates=["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    return df


def load_satellite_summary() -> pd.DataFrame:
    """Load the per-observation satellite cloud summary (Phase 3 output)."""
    df = pd.read_csv(SATELLITE_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_movement_data() -> pd.DataFrame:
    """Load per-step cloud movement records (Phase 4 output)."""
    df = pd.read_csv(MOVEMENT_CSV, parse_dates=["prev_timestamp", "curr_timestamp"])
    df = df.sort_values("curr_timestamp").reset_index(drop=True)
    return df


def check_temporal_overlap(weather_df: pd.DataFrame,
                            satellite_df: pd.DataFrame) -> dict:
    """
    Compare date ranges of both datasets.
    Returns a dict with overlap status and an explanation.
    """
    w_start = weather_df["date_time"].min()
    w_end   = weather_df["date_time"].max()
    s_start = satellite_df["timestamp"].min()
    s_end   = satellite_df["timestamp"].max()

    overlap_start = max(w_start, s_start)
    overlap_end   = min(w_end,   s_end)
    has_overlap   = overlap_start <= overlap_end

    return {
        "weather_start":    w_start,
        "weather_end":      w_end,
        "satellite_start":  s_start,
        "satellite_end":    s_end,
        "has_overlap":      has_overlap,
        "overlap_start":    overlap_start if has_overlap else None,
        "overlap_end":      overlap_end   if has_overlap else None,
        "explanation": (
            f"Temporal overlap found from {overlap_start} to {overlap_end}."
            if has_overlap else
            f"NO TEMPORAL OVERLAP.\n"
            f"  Weather data : {w_start.date()} to {w_end.date()}\n"
            f"  Satellite    : {s_start.date()} to {s_end.date()}\n"
            f"  Combined validation using actual matched records is NOT possible\n"
            f"  with the current datasets. The rainfall prediction models can\n"
            f"  still run on current weather inputs, and satellite features can\n"
            f"  still be displayed alongside predictions."
        )
    }


def build_satellite_feature_table(satellite_df: pd.DataFrame,
                                   movement_df:  pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-timestamp satellite features suitable for display and
    optional future integration with the ML models.

    Per observation (30-min slot):
      - cloud_coverage_pct, clear_coverage_pct
      - n_regions
      - avg_region_area_px     (from tracking results if available)
      - mean_speed_kmh         (average across tracked clouds at this timestamp)
      - dominant_direction     (most common direction label)
      - mean_confidence_score  (HIGH=1, MEDIUM=0.5, LOW=0.25)
      - delta_cloud_pct        (change in cloud coverage vs previous observation)
    """
    feat = satellite_df[["timestamp", "cloud_coverage_pct", "clear_coverage_pct",
                          "n_regions", "flag_high_thick", "flag_low_thick",
                          "flag_cirrus", "flag_partial"]].copy()

    # Coverage change per step
    feat["delta_cloud_pct"] = feat["cloud_coverage_pct"].diff().fillna(0.0)

    # Aggregate movement features per curr_timestamp
    conf_score = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25, "NEW": 0.0}
    movement_df["conf_score"] = movement_df["confidence"].map(conf_score).fillna(0.0)

    speed_agg = (movement_df
                 .groupby("curr_timestamp")
                 .agg(
                     mean_speed_kmh  = ("speed_kmh",       "mean"),
                     max_speed_kmh   = ("speed_kmh",       "max"),
                     mean_conf_score = ("conf_score",      "mean"),
                     n_tracked       = ("track_id",        "nunique"),
                     dominant_dir    = ("direction_label", lambda x: x.mode().iloc[0]
                                        if len(x) > 0 else "N/A"),
                 )
                 .reset_index()
                 .rename(columns={"curr_timestamp": "timestamp"}))

    feat = feat.merge(speed_agg, on="timestamp", how="left")

    # Round numerics
    for col in ["mean_speed_kmh", "max_speed_kmh", "mean_conf_score"]:
        if col in feat.columns:
            feat[col] = feat[col].round(2)

    feat["dominant_dir"] = feat["dominant_dir"].fillna("N/A")
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    return feat


def build_merged_dataset(weather_df: pd.DataFrame,
                          sat_feat_df: pd.DataFrame,
                          overlap_info: dict) -> pd.DataFrame | None:
    """
    Attempt to merge weather and satellite data on matching timestamps.
    Returns None if there is no temporal overlap (honest reporting).
    """
    if not overlap_info["has_overlap"]:
        return None

    # Round weather timestamps to nearest 30 min to align with satellite slots
    w = weather_df.copy()
    w["ts_rounded"] = w["date_time"].dt.floor("30min")

    s = sat_feat_df.copy()
    s = s.rename(columns={"timestamp": "ts_rounded"})

    merged = w.merge(s, on="ts_rounded", how="inner")
    merged = merged.sort_values("date_time").reset_index(drop=True)
    return merged if len(merged) > 0 else None


# ─── standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    weather_df    = load_weather_data()
    satellite_df  = load_satellite_summary()
    movement_df   = load_movement_data()
    overlap       = check_temporal_overlap(weather_df, satellite_df)
    sat_feat      = build_satellite_feature_table(satellite_df, movement_df)
    merged        = build_merged_dataset(weather_df, sat_feat, overlap)

    print(overlap["explanation"])
    print(f"\nSatellite feature table shape: {sat_feat.shape}")
    print(sat_feat.head(3).to_string())
    if merged is not None:
        print(f"\nMerged dataset shape: {merged.shape}")
    else:
        print("\nMerged dataset: NOT available (no temporal overlap)")


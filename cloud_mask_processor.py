"""
cloud_mask_processor.py
=======================
CloudPath AI – Phase 3: Satellite Data Processing
Module 3: Aggregate observations across all four NetCDF files, sort by timestamp,
          identify gaps, and build the master observation table.

Author: CloudPath AI Project
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from regional_extractor import extract_all_observations_from_file


# Expected minimum interval between consecutive observations (minutes)
EXPECTED_INTERVAL_MIN = 30


def process_all_nc_files(nc_file_paths: list[str]) -> tuple[list[dict], pd.DataFrame]:
    """
    Process every NetCDF file sequentially (one at a time to keep memory low).

    Returns
    -------
    observations : list of raw observation dicts (contains numpy arrays)
    summary_df   : pandas DataFrame — one row per observation with scalar columns
    """
    all_obs = []

    for nc_path in sorted(nc_file_paths):
        print(f"  Processing: {os.path.basename(nc_path)} ...", end=" ", flush=True)
        obs_list = extract_all_observations_from_file(nc_path)
        print(f"{len(obs_list)} observations")
        all_obs.extend(obs_list)

    # Sort chronologically by timestamp
    all_obs.sort(key=lambda o: (o["timestamp"] is None, o["timestamp"]))

    # Build a flat summary DataFrame (no numpy arrays — only scalars)
    rows = []
    for obs in all_obs:
        ts = obs["timestamp"]
        fc = obs.get("flag_counts", {})
        rows.append({
            "timestamp":          ts,
            "nc_file":            obs.get("nc_file", ""),
            "cloud_coverage_pct": round(obs["cloud_pct"], 4) if not np.isnan(obs["cloud_pct"]) else np.nan,
            "clear_coverage_pct": round(obs["clear_pct"], 4) if not np.isnan(obs["clear_pct"]) else np.nan,
            "n_total_pixels":     obs["n_total"],
            "n_valid_pixels":     obs["n_valid"],
            "n_cloudy_pixels":    obs["n_cloudy"],
            "n_clear_pixels":     obs["n_clear"],
            "flag_high_thick":    fc.get(1, 0),
            "flag_low_thick":     fc.get(2, 0),
            "flag_cirrus":        fc.get(3, 0),
            "flag_partial":       fc.get(4, 0),
            "region_rows":        obs["region_shape"][0],
            "region_cols":        obs["region_shape"][1],
            "lat_min":            obs["lat_bounds"][0],
            "lat_max":            obs["lat_bounds"][1],
            "lon_min":            obs["lon_bounds"][0],
            "lon_max":            obs["lon_bounds"][1],
            "valid_observation":  True,
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df.sort_values("timestamp", inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    return all_obs, summary_df


def identify_gaps(summary_df: pd.DataFrame,
                  expected_interval_min: int = EXPECTED_INTERVAL_MIN) -> pd.DataFrame:
    """
    Compare consecutive timestamps and identify time gaps larger than expected.

    Returns a DataFrame of gaps with columns:
        gap_start, gap_end, gap_minutes, missing_slots_estimated
    """
    if summary_df.empty or len(summary_df) < 2:
        return pd.DataFrame()

    gaps = []
    ts_list = summary_df["timestamp"].dropna().tolist()

    for i in range(1, len(ts_list)):
        delta_min = (ts_list[i] - ts_list[i - 1]).total_seconds() / 60
        if delta_min > expected_interval_min + 5:   # allow 5 min tolerance
            missing = max(0, int(round(delta_min / expected_interval_min)) - 1)
            gaps.append({
                "gap_start":               ts_list[i - 1],
                "gap_end":                 ts_list[i],
                "gap_minutes":             int(delta_min),
                "missing_slots_estimated": missing,
            })

    return pd.DataFrame(gaps)


def get_processing_summary(nc_files: list[str],
                            all_obs: list[dict],
                            summary_df: pd.DataFrame,
                            gaps_df: pd.DataFrame) -> dict:
    """
    Build the final processing summary dictionary for reporting.
    """
    total_obs = len(all_obs)
    valid_obs  = int(summary_df["valid_observation"].sum()) if not summary_df.empty else 0
    first_ts   = summary_df["timestamp"].min() if not summary_df.empty else None
    last_ts    = summary_df["timestamp"].max() if not summary_df.empty else None

    # Expected theoretical observations: 48 slots/day × 4 days = 192
    days_covered = len(set(o["nc_file"] for o in all_obs))
    theoretical  = days_covered * 48

    # Compute average interval
    if len(summary_df) > 1:
        deltas = summary_df["timestamp"].diff().dropna().dt.total_seconds() / 60
        avg_interval = round(float(deltas.median()), 1)
    else:
        avg_interval = None

    # Cloud coverage statistics
    cc = summary_df["cloud_coverage_pct"].dropna()
    region_shape = (
        int(summary_df["region_rows"].iloc[0]),
        int(summary_df["region_cols"].iloc[0])
    ) if not summary_df.empty else (0, 0)

    return {
        "zip_files":              len(nc_files),
        "nc_files_processed":     len(nc_files),
        "total_observations":     total_obs,
        "valid_observations":     valid_obs,
        "first_timestamp":        first_ts,
        "last_timestamp":         last_ts,
        "theoretical_max_slots":  theoretical,
        "missing_slots":          theoretical - total_obs,
        "gap_events":             len(gaps_df),
        "avg_interval_min":       avg_interval,
        "region_shape":           region_shape,
        "cloud_coverage_mean":    round(float(cc.mean()), 2) if len(cc) > 0 else None,
        "cloud_coverage_min":     round(float(cc.min()),  2) if len(cc) > 0 else None,
        "cloud_coverage_max":     round(float(cc.max()),  2) if len(cc) > 0 else None,
        "cloud_mask_encoding": {
            "0":   "Clear Sky",
            "1":   "Cloud Covered",
            "NaN": "Fill / Space (no valid data)",
        },
        "flag_encoding": {
            "1": "High level thick clouds",
            "2": "Low level thick clouds",
            "3": "Semi-transparent cirrus clouds",
            "4": "Partial clouds",
        },
    }


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nc_dir = r"d:\bengaluru.csv\satellite_nc"
    import glob
    nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    print(f"Found {len(nc_files)} NetCDF files")

    all_obs, summary_df = process_all_nc_files(nc_files)
    gaps_df = identify_gaps(summary_df)
    report  = get_processing_summary(nc_files, all_obs, summary_df, gaps_df)

    print(f"\nTotal observations : {report['total_observations']}")
    print(f"First timestamp    : {report['first_timestamp']}")
    print(f"Last timestamp     : {report['last_timestamp']}")
    print(f"Missing slots      : {report['missing_slots']}")
    print(f"Gap events         : {report['gap_events']}")
    print(f"Mean cloud cover   : {report['cloud_coverage_mean']}%")
    print(f"\nGaps found:\n{gaps_df.to_string()}")


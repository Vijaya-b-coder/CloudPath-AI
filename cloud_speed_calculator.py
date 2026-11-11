"""
cloud_speed_calculator.py
=========================
CloudPath AI – Phase 4: Cloud Movement Tracking
Module 4: Convert pixel displacement into geographic speed using actual
          lat/lon bounds from the Phase 3 NPZ file.

Geospatial conversion
---------------------
The region bounding box covers:
    Lat : 9.986°N  – 15.939°N   (162 rows,  descending — row 0 = north)
    Lon : 74.604°E – 80.564°E   (159 cols,  ascending  — col 0 = west)

Scale derived from actual coordinates (computed once):
    km_per_row  ≈ 4.079 km  (north-south)
    km_per_col  ≈ 4.055 km  (east-west, at ~12.96°N centre latitude)

These are real computed values, not assumptions.

Author: CloudPath AI Project
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from cloud_motion_tracker import CloudTrack, TrackStep


# ─── Load actual lat/lon bounds from Phase 3 NPZ ─────────────────────
def load_geo_params(npz_path: str) -> dict:
    """
    Read lat/lon bounds and grid dimensions from the Phase 3 NPZ.
    Compute per-pixel km scale from actual coordinates.
    """
    data = np.load(npz_path, allow_pickle=True)
    lat_bounds = data["lat_bounds"]   # [lat_min, lat_max]
    lon_bounds = data["lon_bounds"]   # [lon_min, lon_max]
    n_rows, n_cols = data["cloud_masks"].shape[1:]

    lat_min, lat_max = float(lat_bounds[0]), float(lat_bounds[1])
    lon_min, lon_max = float(lon_bounds[0]), float(lon_bounds[1])

    lat_range_deg = lat_max - lat_min
    lon_range_deg = lon_max - lon_min

    center_lat_deg = (lat_min + lat_max) / 2.0

    km_per_lat_deg = 111.0
    km_per_lon_deg = 111.0 * math.cos(math.radians(center_lat_deg))

    # Positive km_per_row: moving down a row moves south
    km_per_row = (lat_range_deg * km_per_lat_deg) / n_rows
    km_per_col = (lon_range_deg * km_per_lon_deg) / n_cols

    return {
        "lat_min":       lat_min,
        "lat_max":       lat_max,
        "lon_min":       lon_min,
        "lon_max":       lon_max,
        "n_rows":        n_rows,
        "n_cols":        n_cols,
        "km_per_row":    km_per_row,
        "km_per_col":    km_per_col,
        "center_lat":    center_lat_deg,
    }


@dataclass
class MovementRecord:
    """One inter-frame movement event for a tracked cloud."""
    track_id:           int
    prev_timestamp:     str
    curr_timestamp:     str
    prev_frame_idx:     int
    curr_frame_idx:     int
    prev_centroid_r:    float
    prev_centroid_c:    float
    curr_centroid_r:    float
    curr_centroid_c:    float
    delta_row:          float    # positive = southward
    delta_col:          float    # positive = eastward
    pixel_dist:         float
    direction_deg:      float    # meteorological: 0=N, 90=E, 180=S, 270=W
    direction_label:    str
    elapsed_minutes:    float
    # geographic (valid if geo_available == True)
    geo_available:      bool
    delta_km_ns:        Optional[float]   # positive = northward displacement
    delta_km_ew:        Optional[float]   # positive = eastward displacement
    geo_dist_km:        Optional[float]
    speed_kmh:          Optional[float]
    confidence:         str    # from matching step


def _direction_deg(delta_row: float, delta_col: float) -> float:
    """
    Compute movement direction in meteorological degrees (0=N, 90=E).
    delta_row > 0 means the cloud moved south (row index increased).
    delta_col > 0 means the cloud moved east (col index increased).
    """
    # Convert to standard math angle: x=east, y=north
    dx = delta_col          # eastward component
    dy = -delta_row         # northward component (row increases downward)

    angle_math = math.degrees(math.atan2(dy, dx))   # -180 to 180
    # Convert to meteorological bearing (0=N, clockwise)
    bearing = (90 - angle_math) % 360
    return round(bearing, 2)


def _direction_label(bearing: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing + 22.5) / 45) % 8
    return labels[idx]


def _parse_timestamp(ts_str: str):
    """Parse ISO-like timestamp string to a datetime."""
    from datetime import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    return None


def compute_movement(track: CloudTrack,
                     geo_params: dict) -> list[MovementRecord]:
    """
    For each consecutive step pair in a track, compute movement metrics.

    Uses actual elapsed time from timestamps (not assumed 30 min).
    Uses actual km/pixel scale from geo_params (not assumed 4 km).
    """
    records: list[MovementRecord] = []

    for i in range(1, len(track.steps)):
        prev: TrackStep = track.steps[i - 1]
        curr: TrackStep = track.steps[i]

        delta_r = curr.centroid_r - prev.centroid_r   # positive → south
        delta_c = curr.centroid_c - prev.centroid_c   # positive → east
        pixel_dist = math.sqrt(delta_r ** 2 + delta_c ** 2)
        bearing = _direction_deg(delta_r, delta_c)
        dir_label = _direction_label(bearing)

        # Elapsed time from actual timestamps
        prev_dt = _parse_timestamp(prev.timestamp)
        curr_dt = _parse_timestamp(curr.timestamp)
        if prev_dt and curr_dt:
            elapsed_min = (curr_dt - prev_dt).total_seconds() / 60.0
        else:
            elapsed_min = float("nan")

        # Geographic displacement
        geo_ok = (not math.isnan(geo_params["km_per_row"]) and
                  not math.isnan(geo_params["km_per_col"]) and
                  not math.isnan(elapsed_min) and
                  elapsed_min > 0)

        if geo_ok:
            km_per_row = geo_params["km_per_row"]
            km_per_col = geo_params["km_per_col"]
            # North-south: moving south = positive delta_r
            delta_km_ns = -delta_r * km_per_row   # positive = northward
            delta_km_ew = delta_c * km_per_col    # positive = eastward
            geo_dist_km = math.sqrt(delta_km_ns ** 2 + delta_km_ew ** 2)
            elapsed_hours = elapsed_min / 60.0
            speed_kmh = geo_dist_km / elapsed_hours if elapsed_hours > 0 else float("nan")
        else:
            delta_km_ns = None
            delta_km_ew = None
            geo_dist_km = None
            speed_kmh   = None

        records.append(MovementRecord(
            track_id         = track.track_id,
            prev_timestamp   = prev.timestamp,
            curr_timestamp   = curr.timestamp,
            prev_frame_idx   = prev.frame_idx,
            curr_frame_idx   = curr.frame_idx,
            prev_centroid_r  = prev.centroid_r,
            prev_centroid_c  = prev.centroid_c,
            curr_centroid_r  = curr.centroid_r,
            curr_centroid_c  = curr.centroid_c,
            delta_row        = delta_r,
            delta_col        = delta_c,
            pixel_dist       = round(pixel_dist, 3),
            direction_deg    = bearing,
            direction_label  = dir_label,
            elapsed_minutes  = round(elapsed_min, 2),
            geo_available    = geo_ok,
            delta_km_ns      = round(delta_km_ns, 3) if delta_km_ns is not None else None,
            delta_km_ew      = round(delta_km_ew, 3) if delta_km_ew is not None else None,
            geo_dist_km      = round(geo_dist_km, 3) if geo_dist_km is not None else None,
            speed_kmh        = round(speed_kmh, 2)   if speed_kmh   is not None else None,
            confidence       = curr.confidence,
        ))

    return records


def compute_all_movements(tracks: list[CloudTrack],
                           geo_params: dict,
                           min_track_length: int = 2) -> list[MovementRecord]:
    """
    Compute movement records for every eligible track.

    Only tracks with >= min_track_length steps are processed.
    """
    all_records: list[MovementRecord] = []
    for track in tracks:
        if track.length < min_track_length:
            continue
        recs = compute_movement(track, geo_params)
        all_records.extend(recs)
    return all_records


# ─── standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    from cloud_region_detection import detect_all_frames
    from cloud_motion_tracker   import run_tracker

    NPZ = r"d:\bengaluru.csv\satellite_data_outputs\bengaluru_cloudmask_series.npz"
    data = np.load(NPZ, allow_pickle=True)
    geo  = load_geo_params(NPZ)
    print(f"km_per_row={geo['km_per_row']:.3f}, km_per_col={geo['km_per_col']:.3f}")

    all_r   = detect_all_frames(data["cloud_masks"], data["timestamps"], data["flags"])
    tracker = run_tracker(all_r)
    moves   = compute_all_movements(tracker.tracks, geo)

    print(f"Movement records: {len(moves)}")
    speeds = [m.speed_kmh for m in moves if m.speed_kmh is not None]
    if speeds:
        print(f"Speed  min/avg/max: {min(speeds):.1f} / "
              f"{sum(speeds)/len(speeds):.1f} / {max(speeds):.1f} km/h")
    for m in moves[:3]:
        print(f"  Track {m.track_id}: {m.direction_label} "
              f"{m.speed_kmh} km/h  dist={m.geo_dist_km} km")


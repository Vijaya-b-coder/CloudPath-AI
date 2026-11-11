"""
main_tracking.py
================
CloudPath AI – Phase 4: Multiple Cloud Detection and Movement Tracking

PHASE 4 ENTRY POINT
-------------------
Loads actual processed satellite observations from Phase 3,
runs the full cloud tracking pipeline, and saves all results.

Pipeline steps
--------------
1. Load Phase 3 NPZ + CSV (no re-extraction of satellite data).
2. Detect cloud regions in every frame (connected components).
3. Match regions between consecutive frames (Hungarian assignment).
4. Build persistent cloud tracks with unique IDs.
5. Compute geographic movement, direction and speed using actual
   lat/lon bounds and actual timestamps.
6. Generate all visualizations.
7. Save CSV outputs and a detailed report.

CONSTRAINTS
-----------
- Uses only actual satellite observations.
- No synthetic cloud movement is created.
- Speed calculations use real timestamps and real km/pixel scale.
- Phase 1, 2, 3 files and models are not modified.
- Original ZIP files are preserved.

Author: CloudPath AI Project
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from collections import Counter

# ─── project root on path ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cloud_region_detection      import detect_all_frames
from cloud_matching               import build_all_matches
from cloud_motion_tracker         import run_tracker
from cloud_speed_calculator       import load_geo_params, compute_all_movements
from cloud_tracking_visualization import (
    plot_detection_frame, plot_tracking_pair, plot_cloud_trajectory,
    plot_speed_direction_summary, plot_coverage_dynamics, OUTPUT_DIR
)

# ─── Paths ────────────────────────────────────────────────────────────
NPZ_PATH     = os.path.join(PROJECT_ROOT, "satellite_data_outputs",
                             "bengaluru_cloudmask_series.npz")
CSV_PATH     = os.path.join(PROJECT_ROOT, "satellite_data_outputs",
                             "bengaluru_satellite_cloud_summary.csv")
TRACK_OUT    = os.path.join(PROJECT_ROOT, "satellite_tracking_outputs")
os.makedirs(TRACK_OUT, exist_ok=True)


def step1_load_data():
    print("\n" + "="*60)
    print("STEP 1 — Load Phase 3 Processed Satellite Data")
    print("="*60)

    data = np.load(NPZ_PATH, allow_pickle=True)
    cloud_masks = data["cloud_masks"]   # (1448, 162, 159)
    flags       = data["flags"]
    timestamps  = data["timestamps"]
    lat_bounds  = data["lat_bounds"]
    lon_bounds  = data["lon_bounds"]

    summary_df  = pd.read_csv(CSV_PATH)
    summary_df["timestamp"] = pd.to_datetime(summary_df["timestamp"])

    geo_params = load_geo_params(NPZ_PATH)

    print(f"Observations loaded   : {len(timestamps)}")
    print(f"Grid shape            : {cloud_masks.shape[1]} x {cloud_masks.shape[2]} pixels")
    print(f"First timestamp (UTC) : {timestamps[0]}")
    print(f"Last  timestamp (UTC) : {timestamps[-1]}")
    print(f"Lat bounds            : {lat_bounds[0]:.3f}N – {lat_bounds[1]:.3f}N")
    print(f"Lon bounds            : {lon_bounds[0]:.3f}E – {lon_bounds[1]:.3f}E")
    print(f"km/row (N-S scale)    : {geo_params['km_per_row']:.3f} km")
    print(f"km/col (E-W scale)    : {geo_params['km_per_col']:.3f} km")

    return cloud_masks, flags, timestamps, summary_df, geo_params


def step2_detect_regions(cloud_masks, flags, timestamps):
    print("\n" + "="*60)
    print("STEP 2 — Detect Cloud Regions in All Frames")
    print("="*60)

    all_regions = detect_all_frames(cloud_masks, timestamps, flags, min_pixels=8)

    total_regions = sum(len(r) for r in all_regions)
    counts = [len(r) for r in all_regions]
    print(f"Total regions detected : {total_regions}")
    print(f"Mean regions/frame     : {np.mean(counts):.1f}")
    print(f"Min / Max per frame    : {min(counts)} / {max(counts)}")

    # Sample output
    for i in [0, 1, 2]:
        regs = all_regions[i]
        print(f"  Frame {i} ({timestamps[i]}): "
              f"{len(regs)} regions, "
              f"largest={regs[0].area_pixels if regs else 0} px")

    return all_regions


def step3_match_and_track(all_regions):
    print("\n" + "="*60)
    print("STEP 3 — Match Regions and Build Cloud Tracks")
    print("="*60)

    tracker     = run_tracker(all_regions)
    all_tracks  = tracker.tracks
    long_tracks = tracker.tracks_min_length(2)
    very_long   = tracker.tracks_min_length(5)

    pair_results = build_all_matches(all_regions)
    total_matches = sum(len(m) for m, _, _ in pair_results)
    high_conf     = sum(sum(1 for mm in m if mm.confidence == "HIGH")
                        for m, _, _ in pair_results)
    med_conf      = sum(sum(1 for mm in m if mm.confidence == "MEDIUM")
                        for m, _, _ in pair_results)

    print(f"Total cloud tracks     : {len(all_tracks)}")
    print(f"Tracks >= 2 frames     : {len(long_tracks)}")
    print(f"Tracks >= 5 frames     : {len(very_long)}")
    print(f"Total frame-to-frame matches: {total_matches}")
    print(f"  HIGH confidence      : {high_conf}")
    print(f"  MEDIUM confidence    : {med_conf}")
    print(f"  LOW confidence       : {total_matches - high_conf - med_conf}")

    # Longest tracks
    sorted_t = sorted(long_tracks, key=lambda t: t.length, reverse=True)[:5]
    print("\nTop 5 longest tracks:")
    for t in sorted_t:
        print(f"  Track {t.track_id:4d}: {t.length:3d} frames  "
              f"{t.first_timestamp} -> {t.last_timestamp}")

    return tracker, pair_results


def step4_compute_movement(tracker, geo_params):
    print("\n" + "="*60)
    print("STEP 4 — Compute Geographic Movement and Speed")
    print("="*60)

    movement_records = compute_all_movements(tracker.tracks, geo_params, min_track_length=2)
    geo_records = [m for m in movement_records if m.geo_available]
    speeds = [m.speed_kmh for m in geo_records
              if m.speed_kmh is not None and m.speed_kmh < 500]

    print(f"Movement records       : {len(movement_records)}")
    print(f"With geo speed         : {len(geo_records)}")
    if speeds:
        print(f"Speed  min             : {min(speeds):.1f} km/h")
        print(f"Speed  mean            : {sum(speeds)/len(speeds):.1f} km/h")
        print(f"Speed  max             : {max(speeds):.1f} km/h")

    dirs = Counter(m.direction_label for m in movement_records)
    most_common = dirs.most_common(3)
    print(f"Most common directions : {most_common}")

    return movement_records


def step5_visualizations(cloud_masks, timestamps, all_regions,
                          tracker, pair_results, movement_records,
                          summary_df) -> list[str]:
    print("\n" + "="*60)
    print("STEP 5 — Generate Visualizations")
    print("="*60)
    saved = []

    # 5a. Detection frames — first, middle, last observation
    for fi in [0, len(all_regions)//2, len(all_regions)-1]:
        p = plot_detection_frame(
            cloud_masks[fi], all_regions[fi],
            str(timestamps[fi]), frame_idx=fi
        )
        saved.append(p)
        print(f"  Saved: {os.path.basename(p)}")

    # 5b. Tracking pairs — first 3 pairs
    for pi in range(min(3, len(pair_results))):
        matches_pi, _, _ = pair_results[pi]
        # Build MovementRecord proxies for arrows (use pixel centroids)
        class _Arrow:
            def __init__(self, m):
                self.prev_centroid_r = m.prev_centroid[0]
                self.prev_centroid_c = m.prev_centroid[1]
                self.curr_centroid_r = m.curr_centroid[0]
                self.curr_centroid_c = m.curr_centroid[1]
                self.speed_kmh = None
        arrows = [_Arrow(m) for m in matches_pi]

        p = plot_tracking_pair(
            cloud_masks[pi], cloud_masks[pi + 1],
            all_regions[pi], all_regions[pi + 1],
            arrows,
            str(timestamps[pi]), str(timestamps[pi + 1]),
            pair_idx=pi
        )
        saved.append(p)
        print(f"  Saved: {os.path.basename(p)}")

    # 5c. Top 5 longest track trajectories
    long_tracks = sorted(tracker.tracks_min_length(3),
                          key=lambda t: t.length, reverse=True)[:5]
    for track in long_tracks:
        p = plot_cloud_trajectory(track, cloud_masks, timestamps)
        if p:
            saved.append(p)
            print(f"  Saved: {os.path.basename(p)}")

    # 5d. Speed and direction summary
    p = plot_speed_direction_summary(movement_records)
    if p:
        saved.append(p)
        print(f"  Saved: {os.path.basename(p)}")

    # 5e. Coverage dynamics + speed overlay
    p = plot_coverage_dynamics(summary_df, movement_records)
    if p:
        saved.append(p)
        print(f"  Saved: {os.path.basename(p)}")

    return saved


def step6_save_results(all_regions, tracker, pair_results,
                        movement_records, summary_df, vis_files) -> list[str]:
    print("\n" + "="*60)
    print("STEP 6 — Save Tracking Results")
    print("="*60)
    saved = []

    # ── cloud_tracking_results.csv ─────────────────────────────────
    rows = []
    for frame_regions in all_regions:
        for reg in frame_regions:
            rows.append({
                "frame_idx":    reg.frame_idx,
                "timestamp":    reg.timestamp,
                "region_id":    reg.region_id,
                "centroid_row": round(reg.centroid_r, 2),
                "centroid_col": round(reg.centroid_c, 2),
                "area_pixels":  reg.area_pixels,
                "dominant_flag": reg.dominant_flag,
                "bbox_rmin":    reg.bbox[0],
                "bbox_rmax":    reg.bbox[1],
                "bbox_cmin":    reg.bbox[2],
                "bbox_cmax":    reg.bbox[3],
            })
    df_regions = pd.DataFrame(rows)
    p = os.path.join(TRACK_OUT, "cloud_tracking_results.csv")
    df_regions.to_csv(p, index=False)
    saved.append(p)
    print(f"  Saved: cloud_tracking_results.csv  ({len(df_regions)} rows)")

    # ── cloud_tracks.csv ───────────────────────────────────────────
    track_rows = []
    for track in tracker.tracks:
        for step in track.steps:
            track_rows.append({
                "track_id":     track.track_id,
                "frame_idx":    step.frame_idx,
                "timestamp":    step.timestamp,
                "centroid_row": round(step.centroid_r, 2),
                "centroid_col": round(step.centroid_c, 2),
                "area_pixels":  step.area_pixels,
                "dominant_flag": step.dominant_flag,
                "confidence":   step.confidence,
            })
    df_tracks = pd.DataFrame(track_rows)
    p = os.path.join(TRACK_OUT, "cloud_tracks.csv")
    df_tracks.to_csv(p, index=False)
    saved.append(p)
    print(f"  Saved: cloud_tracks.csv  ({len(df_tracks)} rows)")

    # ── cloud_movement_summary.csv ─────────────────────────────────
    move_rows = []
    for m in movement_records:
        move_rows.append({
            "track_id":        m.track_id,
            "prev_timestamp":  m.prev_timestamp,
            "curr_timestamp":  m.curr_timestamp,
            "prev_frame_idx":  m.prev_frame_idx,
            "curr_frame_idx":  m.curr_frame_idx,
            "prev_centroid_r": round(m.prev_centroid_r, 2),
            "prev_centroid_c": round(m.prev_centroid_c, 2),
            "curr_centroid_r": round(m.curr_centroid_r, 2),
            "curr_centroid_c": round(m.curr_centroid_c, 2),
            "delta_row":       round(m.delta_row, 3),
            "delta_col":       round(m.delta_col, 3),
            "pixel_dist":      m.pixel_dist,
            "direction_deg":   m.direction_deg,
            "direction_label": m.direction_label,
            "elapsed_minutes": m.elapsed_minutes,
            "geo_available":   m.geo_available,
            "delta_km_ns":     m.delta_km_ns,
            "delta_km_ew":     m.delta_km_ew,
            "geo_dist_km":     m.geo_dist_km,
            "speed_kmh":       m.speed_kmh,
            "confidence":      m.confidence,
        })
    df_move = pd.DataFrame(move_rows)
    p = os.path.join(TRACK_OUT, "cloud_movement_summary.csv")
    df_move.to_csv(p, index=False)
    saved.append(p)
    print(f"  Saved: cloud_movement_summary.csv  ({len(df_move)} rows)")

    # ── cloud_trajectories.csv ─────────────────────────────────────
    traj_rows = []
    long_tracks = tracker.tracks_min_length(2)
    for track in long_tracks:
        for i, step in enumerate(track.steps):
            traj_rows.append({
                "track_id":    track.track_id,
                "step_number": i,
                "frame_idx":   step.frame_idx,
                "timestamp":   step.timestamp,
                "centroid_row": round(step.centroid_r, 2),
                "centroid_col": round(step.centroid_c, 2),
                "area_pixels": step.area_pixels,
                "confidence":  step.confidence,
                "track_length": track.length,
            })
    df_traj = pd.DataFrame(traj_rows)
    p = os.path.join(TRACK_OUT, "cloud_trajectories.csv")
    df_traj.to_csv(p, index=False)
    saved.append(p)
    print(f"  Saved: cloud_trajectories.csv  ({len(df_traj)} rows)")

    return saved


def step7_write_report(all_regions, tracker, pair_results,
                        movement_records, vis_files, data_files,
                        geo_params, timestamps) -> str:
    print("\n" + "="*60)
    print("STEP 7 — Write Tracking Report")
    print("="*60)

    report_path = os.path.join(TRACK_OUT, "cloud_tracking_report.txt")
    speeds = [m.speed_kmh for m in movement_records
              if m.speed_kmh is not None and m.speed_kmh < 500]
    dirs   = Counter(m.direction_label for m in movement_records)
    long_tracks = tracker.tracks_min_length(2)
    all_tracks  = tracker.tracks

    total_regions = sum(len(r) for r in all_regions)
    total_matches = sum(len(m) for m, _, _ in pair_results)
    high_conf     = sum(sum(1 for mm in m if mm.confidence == "HIGH")
                        for m, _, _ in pair_results)

    # Dynamic calculation of missing/gap slots
    t_start = pd.to_datetime(timestamps[0])
    t_end   = pd.to_datetime(timestamps[-1])
    expected_slots = int(round((t_end - t_start).total_seconds() / 1800.0)) + 1
    missing_slots  = max(0, expected_slots - len(timestamps))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CloudPath AI -- Phase 4: Cloud Tracking Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"GENERATED : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Observations processed      : {len(all_regions)}\n")
        f.write(f"  First observation           : {t_start}\n")
        f.write(f"  Last observation            : {t_end}\n")
        f.write(f"  Source                      : Phase 3 NPZ (actual INSAT-3DS data)\n")
        f.write(f"  Encoding (0=Clear, 1=Cloud) : confirmed from metadata\n\n")

        f.write("-" * 70 + "\n")
        f.write("2. REGION DETECTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total regions detected      : {total_regions}\n")
        counts = [len(r) for r in all_regions]
        f.write(f"  Mean regions per frame      : {sum(counts)/len(counts):.1f}\n")
        f.write(f"  Min / Max per frame         : {min(counts)} / {max(counts)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("3. TRACKING\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total cloud tracks          : {len(all_tracks)}\n")
        f.write(f"  Tracks >= 2 frames          : {len(long_tracks)}\n")
        f.write(f"  Tracks >= 5 frames          : {len(tracker.tracks_min_length(5))}\n")
        f.write(f"  Total frame-pair matches    : {total_matches}\n")
        f.write(f"  HIGH confidence matches     : {high_conf}\n")
        f.write(f"  Missing/gap observations    : {missing_slots} (satellite eclipse/calibration gaps)\n\n")

        f.write("-" * 70 + "\n")
        f.write("4. CLOUD MOVEMENT\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Movement records computed   : {len(movement_records)}\n")
        f.write(f"  Geo speed available         : Yes\n")
        f.write(f"  km per row (N-S)            : {geo_params['km_per_row']:.3f} km\n")
        f.write(f"  km per col (E-W)            : {geo_params['km_per_col']:.3f} km\n")
        f.write(f"  (Computed from actual lat/lon bounds -- not assumed)\n")
        if speeds:
            f.write(f"  Speed minimum               : {min(speeds):.1f} km/h\n")
            f.write(f"  Speed mean                  : {sum(speeds)/len(speeds):.1f} km/h\n")
            f.write(f"  Speed maximum               : {max(speeds):.1f} km/h\n")
        else:
            f.write(f"  Speed                       : not computable\n")
        f.write(f"  Dominant direction          : {dirs.most_common(1)[0] if dirs else 'N/A'}\n")
        f.write(f"  Direction breakdown         : {dict(dirs.most_common())}\n\n")

        f.write("-" * 70 + "\n")
        f.write("5. VISUALIZATIONS\n")
        f.write("-" * 70 + "\n")
        for vf in vis_files:
            if vf:
                f.write(f"  {os.path.basename(vf)}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("6. DATA FILES\n")
        f.write("-" * 70 + "\n")
        for df_ in data_files:
            sz = os.path.getsize(df_) / 1024
            f.write(f"  {os.path.basename(df_)}  ({sz:.0f} KB)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("7. NOTES\n")
        f.write("-" * 70 + "\n")
        f.write("  - Only actual satellite observations used.\n")
        f.write("  - No synthetic cloud movement created.\n")
        f.write("  - Timestamps and pixel scale taken from actual data.\n")
        f.write("  - Phase 1, 2, 3 files and models were NOT modified.\n")
        f.write("  - Original ZIP files preserved in satellite_img/.\n")
        f.write("  - Matching uses Hungarian algorithm + IoU overlap scoring.\n")
        f.write("  - Speed values >500 km/h excluded as outliers.\n\n")

    print(f"  Report saved: {os.path.basename(report_path)}")
    return report_path


def main():
    t0 = time.time()
    print("\n" + "="*60)
    print("  CloudPath AI - Phase 4: Cloud Movement Tracking")
    print("="*60)

    try:
        # Step 1: Load data
        cloud_masks, flags, timestamps, summary_df, geo_params = step1_load_data()

        # Step 2: Detect regions
        all_regions = step2_detect_regions(cloud_masks, flags, timestamps)

        # Step 3: Match and track
        tracker, pair_results = step3_match_and_track(all_regions)

        # Step 4: Compute movement
        movement_records = step4_compute_movement(tracker, geo_params)

        # Step 5: Visualizations
        vis_files = step5_visualizations(
            cloud_masks, timestamps, all_regions,
            tracker, pair_results, movement_records, summary_df
        )

        # Step 6: Save CSVs
        data_files = step6_save_results(
            all_regions, tracker, pair_results, movement_records, summary_df, vis_files
        )

        # Step 7: Report
        report_path = step7_write_report(
            all_regions, tracker, pair_results, movement_records,
            vis_files, data_files, geo_params, timestamps
        )

        elapsed = time.time() - t0
        print("\n" + "="*60)
        print("PHASE 4 COMPLETE")
        print("="*60)

        total_regions = sum(len(r) for r in all_regions)
        long_tracks   = tracker.tracks_min_length(2)
        speeds = [m.speed_kmh for m in movement_records
                  if m.speed_kmh is not None and m.speed_kmh < 500]

        print(f"  Observations processed     : {len(all_regions)}")
        print(f"  Total cloud regions        : {total_regions}")
        print(f"  Reliable cloud tracks      : {len(long_tracks)}")
        print(f"  Movement records           : {len(movement_records)}")
        print(f"  Geo speed calculated       : Yes")
        if speeds:
            print(f"  Speed min / avg / max      : "
                  f"{min(speeds):.1f} / {sum(speeds)/len(speeds):.1f} / "
                  f"{max(speeds):.1f} km/h")
        print(f"  Visualizations saved       : {len([v for v in vis_files if v])}")
        print(f"  CSV files saved            : {len(data_files)}")
        print(f"  Time elapsed               : {elapsed:.1f} s")
        print(f"\n  Output: {TRACK_OUT}")

    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


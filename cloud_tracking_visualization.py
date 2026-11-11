"""
cloud_tracking_visualization.py
================================
CloudPath AI – Phase 4: Cloud Movement Tracking
Module 5: All Phase 4 visualizations.

Saves to: satellite_tracking_outputs/

Author: CloudPath AI Project
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from typing import Optional

from cloud_region_detection  import CloudRegion
from cloud_speed_calculator  import MovementRecord
from cloud_motion_tracker    import CloudTrack

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "satellite_tracking_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette for cloud type flags
FLAG_COLORS = {1: "#1f78b4", 2: "#33a02c", 3: "#ff7f00", 4: "#e31a1c"}
FLAG_NAMES  = {1: "High-thick", 2: "Low-thick", 3: "Cirrus", 4: "Partial"}


def _rgb_from_mask(cloud_mask: np.ndarray) -> np.ndarray:
    h, w = cloud_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[cloud_mask == 0]      = [0.53, 0.81, 0.98]
    rgb[cloud_mask == 1]      = [0.93, 0.95, 1.0]
    rgb[np.isnan(cloud_mask)] = [0.18, 0.18, 0.18]
    return rgb


def _save(fig, filename: str, out_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── 1. Detection overlay (regions + centroids + IDs) ────────────────
def plot_detection_frame(cloud_mask: np.ndarray,
                          regions: list[CloudRegion],
                          timestamp: str,
                          frame_idx: int,
                          max_regions: int = 30) -> str:
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(_rgb_from_mask(cloud_mask), origin="upper", aspect="auto")

    cmap = plt.get_cmap("tab20")
    for i, reg in enumerate(regions[:max_regions]):
        color = cmap(i % 20)
        ax.plot(reg.centroid_c, reg.centroid_r, "o", color=color,
                markersize=5, zorder=4)
        ax.text(reg.centroid_c + 1, reg.centroid_r - 1,
                f"R{reg.region_id}", fontsize=6, color=color,
                fontweight="bold", zorder=5)

    ax.set_title(f"Cloud Region Detection | Frame {frame_idx}\n{timestamp}",
                 fontsize=10)
    ax.set_xlabel("Column index (W→E)")
    ax.set_ylabel("Row index (N→S)")
    legend_elems = [
        mpatches.Patch(color=[0.53, 0.81, 0.98], label="Clear Sky"),
        mpatches.Patch(color=[0.93, 0.95, 1.0],  label="Cloud", ec="grey"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
    return _save(fig, f"detection_frame_{frame_idx:03d}.png")


# ─── 2. Tracking arrows between two consecutive frames ───────────────
def plot_tracking_pair(mask_prev: np.ndarray,
                        mask_curr: np.ndarray,
                        regions_prev: list[CloudRegion],
                        regions_curr: list[CloudRegion],
                        movement_records: list[MovementRecord],
                        ts_prev: str,
                        ts_curr: str,
                        pair_idx: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, mask, ts, regs, label in [
        (axes[0], mask_prev, ts_prev, regions_prev, "Previous"),
        (axes[1], mask_curr, ts_curr, regions_curr, "Current"),
    ]:
        ax.imshow(_rgb_from_mask(mask), origin="upper", aspect="auto")
        for reg in regs[:25]:
            ax.plot(reg.centroid_c, reg.centroid_r, "r+", markersize=7,
                    markeredgewidth=1.5, zorder=4)
            ax.text(reg.centroid_c + 1, reg.centroid_r - 1,
                    f"R{reg.region_id}", fontsize=6, color="yellow",
                    fontweight="bold", zorder=5)
        ax.set_title(f"{label}\n{ts}", fontsize=9)
        ax.set_xlabel("Col (W→E)")
        ax.set_ylabel("Row (N→S)")

    # Draw movement arrows on the current-frame panel
    ax_curr = axes[1]
    for mrec in movement_records:
        ax_curr.annotate(
            "", 
            xy=(mrec.curr_centroid_c, mrec.curr_centroid_r),
            xytext=(mrec.prev_centroid_c, mrec.prev_centroid_r),
            arrowprops=dict(arrowstyle="->", color="lime",
                            lw=1.4, mutation_scale=10),
            zorder=6,
        )

    plt.suptitle(f"Cloud Tracking — Frame Pair {pair_idx}", fontsize=11)
    plt.tight_layout()
    return _save(fig, f"tracking_pair_{pair_idx:03d}.png")


# ─── 3. Individual cloud trajectory ──────────────────────────────────
def plot_cloud_trajectory(track: CloudTrack,
                           cloud_masks: np.ndarray,
                           timestamps: np.ndarray) -> str:
    """Plot the path of one cloud across all its frames."""
    if track.length < 2:
        return ""

    rows = [s.centroid_r for s in track.steps]
    cols = [s.centroid_c for s in track.steps]

    # Show background of first and last frame
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, step_idx, label in [(axes[0], 0, "Start"), (axes[1], -1, "End")]:
        fi = track.steps[step_idx].frame_idx
        ax.imshow(_rgb_from_mask(cloud_masks[fi]), origin="upper", aspect="auto")
        ax.plot(cols, rows, "o-", color="red", markersize=4, linewidth=1.5,
                zorder=5, label="Trajectory")
        ax.plot(cols[step_idx], rows[step_idx], "*", color="yellow",
                markersize=12, zorder=6)
        ax.set_title(f"{label}: {track.steps[step_idx].timestamp}", fontsize=9)
        ax.set_xlabel("Col (W→E)")
        ax.set_ylabel("Row (N→S)")

    plt.suptitle(f"Cloud Track {track.track_id} — {track.length} frames  "
                 f"({track.first_timestamp[:10]})", fontsize=11)
    plt.tight_layout()
    return _save(fig, f"trajectory_track_{track.track_id:04d}.png")


# ─── 4. Speed and direction summary chart ────────────────────────────
def plot_speed_direction_summary(movement_records: list[MovementRecord]) -> str:
    valid = [m for m in movement_records
             if m.speed_kmh is not None and not np.isnan(m.speed_kmh)
             and m.speed_kmh < 300]   # filter implausible outliers

    if len(valid) < 2:
        return ""

    speeds = [m.speed_kmh for m in valid]
    bearings = [m.direction_deg for m in valid]
    dirs = [m.direction_label for m in valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Speed distribution
    axes[0].hist(speeds, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(speeds), color="red", linestyle="--",
                    label=f"Mean: {np.mean(speeds):.1f} km/h")
    axes[0].set_xlabel("Speed (km/h)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cloud Speed Distribution")
    axes[0].legend(fontsize=9)

    # Panel 2: Direction rose (polar histogram)
    ax_pol = fig.add_subplot(1, 3, 2, projection="polar")
    # 16 bins for compass directions
    bin_edges = np.linspace(0, 2 * np.pi, 17)
    bearing_rad = np.radians(bearings)
    counts, _ = np.histogram(bearing_rad, bins=bin_edges)
    width = bin_edges[1] - bin_edges[0]
    ax_pol.bar(bin_edges[:-1], counts, width=width, align="edge",
               color="coral", edgecolor="white", alpha=0.8)
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_title("Movement Direction Rose", pad=15, fontsize=9)

    # Panel 3: Direction label counts
    from collections import Counter
    dir_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    dir_counts = Counter(dirs)
    bar_vals = [dir_counts.get(d, 0) for d in dir_order]
    axes[2].bar(dir_order, bar_vals, color="coral", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("Direction")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Movement Direction Counts")

    plt.suptitle("Cloud Movement — Speed and Direction Summary", fontsize=12, y=1.01)
    plt.tight_layout()
    return _save(fig, "speed_direction_summary.png")


# ─── 5. Cloud coverage dynamics over time ────────────────────────────
def plot_coverage_dynamics(summary_df, movement_records: list[MovementRecord]) -> str:
    """Re-plot coverage + overlay average speed per timestep."""
    import pandas as pd

    df = summary_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Average speed per frame (current frame)
    if movement_records:
        from collections import defaultdict
        speed_by_ts = defaultdict(list)
        for m in movement_records:
            if m.speed_kmh is not None:
                speed_by_ts[m.curr_timestamp].append(m.speed_kmh)
        df["mean_speed"] = df["timestamp"].astype(str).map(
            lambda t: np.mean(speed_by_ts[t]) if speed_by_ts[t] else np.nan
        )

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].fill_between(df["timestamp"], df["cloud_coverage_pct"],
                         alpha=0.7, color="steelblue", label="Cloud %")
    axes[0].set_ylabel("Cloud Coverage (%)")
    month_str = df["timestamp"].iloc[0].strftime("%B %Y") if not df.empty else "July 2025"
    axes[0].set_title(f"Bengaluru Region — Cloud & Movement Analysis (INSAT-3DS, {month_str})")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.35)

    axes[1].bar(df["timestamp"], df["n_regions"],
                width=0.015, color="coral", alpha=0.8, label="Detected Regions")
    axes[1].set_ylabel("Cloud Regions")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.35)

    if "mean_speed" in df.columns:
        axes[2].plot(df["timestamp"], df["mean_speed"],
                     "o-", color="green", markersize=2, linewidth=1.0,
                     label="Mean Cloud Speed (km/h)")
        axes[2].set_ylabel("Speed (km/h)")
        axes[2].legend(fontsize=9)
        axes[2].grid(axis="y", alpha=0.35)
    else:
        axes[2].set_visible(False)

    axes[-1].set_xlabel("Timestamp (UTC)")
    step = max(1, len(df) // 16)
    ticks = range(0, len(df), step)
    axes[-1].set_xticks([df["timestamp"].iloc[t] for t in ticks])
    axes[-1].set_xticklabels([df["timestamp"].iloc[t].strftime("%d %b") for t in ticks],
                             rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    return _save(fig, "cloud_coverage_dynamics.png")


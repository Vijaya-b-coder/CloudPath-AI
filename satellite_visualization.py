"""
satellite_visualization.py
===========================
CloudPath AI – Phase 3: Satellite Data Processing
Module 5: Generate all required visualizations from processed satellite data.

Outputs saved to satellite_data_outputs/ folder.

Author: CloudPath AI Project
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
from scipy import ndimage

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "satellite_data_outputs")


def _ensure_output_dir(out_dir: str = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ──────────────────────────────────────────────────────────────────────
# 1. Raw cloud-mask image for a single observation
# ──────────────────────────────────────────────────────────────────────
def plot_cloud_mask(cloud_mask: np.ndarray,
                    timestamp,
                    title_prefix: str = "Cloud Mask",
                    filename: str = None,
                    out_dir: str = OUTPUT_DIR) -> str:
    """
    Plot a single cloud-mask array as a colour image.

    Colour scheme (from actual metadata):
        0 (Clear Sky)  → light blue
        1 (Cloudy)     → white
        NaN (Fill)     → dark grey
    """
    _ensure_output_dir(out_dir)

    # Build RGB image
    h, w = cloud_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    nan_mask   = np.isnan(cloud_mask)
    clear_mask = (cloud_mask == 0)
    cloud_mask_bool = (cloud_mask == 1)

    rgb[clear_mask]      = [0.53, 0.81, 0.98]   # light blue  = clear
    rgb[cloud_mask_bool] = [1.0,  1.0,  1.0]    # white       = cloud
    rgb[nan_mask]        = [0.15, 0.15, 0.15]   # dark grey   = fill/space

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rgb, origin="upper", aspect="auto")
    ax.set_title(f"{title_prefix}\n{timestamp}", fontsize=11, pad=8)
    ax.set_xlabel("Longitude index (west → east)")
    ax.set_ylabel("Latitude index (north → south)")

    legend_patches = [
        mpatches.Patch(color=[0.53, 0.81, 0.98], label="Clear Sky (0)"),
        mpatches.Patch(color=[1.0,  1.0,  1.0],  label="Cloud Covered (1)",
                       ec="lightgrey"),
        mpatches.Patch(color=[0.15, 0.15, 0.15], label="Fill / Space (NaN)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.85)

    if filename is None:
        ts_str = str(timestamp).replace(":", "").replace(" ", "_").replace("-", "")
        filename = f"cloud_mask_{ts_str}.png"

    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 2. Cloud region detection overlay
# ──────────────────────────────────────────────────────────────────────
def plot_cloud_regions(cloud_mask: np.ndarray,
                       labelled_array: np.ndarray,
                       n_regions: int,
                       region_centroids: dict,
                       timestamp,
                       filename: str = None,
                       out_dir: str = OUTPUT_DIR) -> str:
    """
    Overlay detected cloud region boundaries and centroids on the cloud mask.
    """
    _ensure_output_dir(out_dir)

    h, w = cloud_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    nan_mask        = np.isnan(cloud_mask)
    clear_mask      = (cloud_mask == 0)
    cloud_mask_bool = (cloud_mask == 1)

    rgb[clear_mask]       = [0.53, 0.81, 0.98]
    rgb[cloud_mask_bool]  = [0.85, 0.93, 1.0]
    rgb[nan_mask]         = [0.15, 0.15, 0.15]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(rgb, origin="upper", aspect="auto")

    # Draw region boundaries using contours
    if n_regions > 0 and labelled_array is not None:
        binary = (labelled_array > 0).astype(np.float32)
        from scipy.ndimage import find_objects
        for lbl, (r_centroid, c_centroid) in region_centroids.items():
            ax.plot(c_centroid, r_centroid, "r+", markersize=6, markeredgewidth=1.2)

        ax.contour(binary, levels=[0.5], colors=["red"], linewidths=0.6, alpha=0.8)

    ax.set_title(
        f"Bengaluru Region – Cloud Regions Detected: {n_regions}\n{timestamp}",
        fontsize=11, pad=8
    )
    ax.set_xlabel("Longitude index")
    ax.set_ylabel("Latitude index")

    legend_patches = [
        mpatches.Patch(color=[0.53, 0.81, 0.98], label="Clear Sky"),
        mpatches.Patch(color=[0.85, 0.93, 1.0],  label="Cloud Covered",
                       ec="lightgrey"),
        mpatches.Patch(color="red", label=f"Region Boundaries ({n_regions})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.85)

    if filename is None:
        ts_str = str(timestamp).replace(":", "").replace(" ", "_").replace("-", "")
        filename = f"cloud_regions_{ts_str}.png"

    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 3. Cloud coverage time-series
# ──────────────────────────────────────────────────────────────────────
def plot_coverage_timeseries(summary_df: pd.DataFrame,
                              filename: str = "coverage_timeseries.png",
                              out_dir: str = OUTPUT_DIR) -> str:
    """
    Plot cloud coverage percentage over all observations across all 4 days.
    """
    _ensure_output_dir(out_dir)

    df = summary_df.dropna(subset=["timestamp", "cloud_coverage_pct"]).copy()
    df = df.sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Cloud vs Clear percentage
    axes[0].fill_between(df["timestamp"], df["cloud_coverage_pct"],
                         alpha=0.6, color="steelblue", label="Cloud %")
    axes[0].fill_between(df["timestamp"], df["clear_coverage_pct"],
                         alpha=0.4, color="skyblue", label="Clear Sky %")
    axes[0].set_ylabel("Coverage (%)")
    month_str = df["timestamp"].iloc[0].strftime("%B %Y") if not df.empty else "July 2025"
    axes[0].set_title(f"Bengaluru Region – Cloud Coverage Over Time (INSAT-3DS, {month_str})",
                      fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.4)

    # Panel 2: Number of cloud regions
    if "n_regions" in df.columns:
        axes[1].bar(df["timestamp"], df["n_regions"],
                    width=0.015, color="coral", alpha=0.8, label="Cloud Regions")
        axes[1].set_ylabel("Number of Cloud Regions")
        axes[1].set_xlabel("Timestamp (UTC)")
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.4)
    else:
        axes[1].set_visible(False)

    step = max(1, len(df) // 16)
    ticks = range(0, len(df), step)
    axes[1].set_xticks([df["timestamp"].iloc[t] for t in ticks])
    axes[1].set_xticklabels([df["timestamp"].iloc[t].strftime("%d %b") for t in ticks],
                             rotation=30, ha="right", fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 4. Daily cloud mask progression (4 panels — one per day)
# ──────────────────────────────────────────────────────────────────────
def plot_daily_progression(observations: list[dict],
                            filename: str = "daily_cloud_progression.png",
                            out_dir: str = OUTPUT_DIR) -> str:
    """
    Show one representative cloud-mask per day as a 4-panel figure.
    Selects the midday observation (closest to 1200 UTC) for each day.
    """
    _ensure_output_dir(out_dir)

    from datetime import time as dt_time
    from collections import defaultdict

    days = defaultdict(list)
    for obs in observations:
        if obs.get("timestamp"):
            days[obs["timestamp"].date()].append(obs)

    import math
    sorted_days = sorted(days.keys())
    n_days = len(sorted_days)
    if n_days == 0:
        return None

    ncols = min(n_days, 7)
    nrows = int(math.ceil(n_days / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 3.0 * nrows), squeeze=False)
    axes = axes_grid.flatten()

    for i, day in enumerate(sorted_days):
        ax = axes[i]
        # Find observation closest to 1200 UTC
        day_obs = days[day]
        def noon_dist(o):
            ts = o["timestamp"]
            return abs(ts.hour * 60 + ts.minute - 720)
        rep = min(day_obs, key=noon_dist)

        cm = rep["cloud_mask"]
        h, w = cm.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[cm == 0]          = [0.53, 0.81, 0.98]
        rgb[cm == 1]          = [1.0,  1.0,  1.0]
        rgb[np.isnan(cm)]     = [0.15, 0.15, 0.15]

        ax.imshow(rgb, origin="upper", aspect="auto")
        ax.set_title(f"{day.strftime('%d %b')}\n{rep['timestamp'].strftime('%H:%M')} UTC", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cc = rep.get("cloud_pct", float("nan"))
        ax.text(0.03, 0.04, f"{cc:.0f}%", transform=ax.transAxes,
                fontsize=8, color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))

    # Hide unused axes
    for j in range(len(sorted_days), len(axes)):
        axes[j].set_visible(False)

    month_str = sorted_days[0].strftime("%B %Y") if sorted_days else "July 2025"
    fig.suptitle(f"Bengaluru Region – Daily Midday Cloud Mask (INSAT-3DS, {month_str})",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 5. Flag/cloud-type breakdown bar chart
# ──────────────────────────────────────────────────────────────────────
def plot_cloud_type_breakdown(summary_df: pd.DataFrame,
                               filename: str = "cloud_type_breakdown.png",
                               out_dir: str = OUTPUT_DIR) -> str:
    """
    Stacked bar chart showing flag breakdown (cloud types) per observation.
    """
    _ensure_output_dir(out_dir)

    flag_cols = ["flag_high_thick", "flag_low_thick", "flag_cirrus", "flag_partial"]
    available = [c for c in flag_cols if c in summary_df.columns]
    if not available:
        return None

    df = summary_df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp")

    colors = ["#2166ac", "#92c5de", "#d1e5f0", "#f4a582"]
    labels = ["High-level thick", "Low-level thick", "Cirrus", "Partial clouds"]

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(df))
    for col, label, color in zip(available, labels, colors):
        vals = df[col].fillna(0).values
        ax.bar(range(len(df)), vals, bottom=bottom,
               label=label, color=color, alpha=0.85, width=1.0)
        bottom += vals

    step = max(1, len(df) // 16)
    ticks = range(0, len(df), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [df["timestamp"].iloc[i].strftime("%d %b") for i in ticks],
        fontsize=8, rotation=30, ha="right"
    )
    ax.set_xlabel("Observation (UTC)")
    ax.set_ylabel("Pixels in Bengaluru region")
    month_str = df["timestamp"].iloc[0].strftime("%B %Y") if not df.empty else "July 2025"
    ax.set_title(f"Cloud Type Breakdown per Observation (INSAT-3DS, {month_str})", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# 6. Single-frame Bengaluru highlight (zoomed region with lat/lon axes)
# ──────────────────────────────────────────────────────────────────────
def plot_bengaluru_highlight(cloud_mask: np.ndarray,
                              lat_bounds: tuple, lon_bounds: tuple,
                              timestamp,
                              cloud_pct: float,
                              filename: str = None,
                              out_dir: str = OUTPUT_DIR) -> str:
    """
    Plot the Bengaluru region with real lat/lon tick labels.
    """
    _ensure_output_dir(out_dir)

    h, w = cloud_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[cloud_mask == 0]       = [0.53, 0.81, 0.98]
    rgb[cloud_mask == 1]       = [1.0,  1.0,  1.0]
    rgb[np.isnan(cloud_mask)]  = [0.15, 0.15, 0.15]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin="upper", aspect="auto",
              extent=[lon_bounds[0], lon_bounds[1],
                      lat_bounds[0], lat_bounds[1]])

    # Mark Bengaluru city centre
    ax.plot(77.59, 12.97, "r*", markersize=12, zorder=5)
    ax.annotate("Bengaluru", (77.59, 12.97), xytext=(77.85, 12.70),
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"Bengaluru Region Cloud Mask — {timestamp}\n"
        f"Cloud Coverage: {cloud_pct:.1f}%",
        fontsize=11
    )
    legend_patches = [
        mpatches.Patch(color=[0.53, 0.81, 0.98], label="Clear Sky"),
        mpatches.Patch(color=[1.0, 1.0, 1.0],  label="Cloud Covered", ec="lightgrey"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    if filename is None:
        ts_str = str(timestamp).replace(":", "").replace(" ", "_").replace("-", "")
        filename = f"bengaluru_highlight_{ts_str}.png"

    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


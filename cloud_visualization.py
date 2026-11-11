"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: cloud_visualization.py
===================================================================
Author : B.Tech Final Year Project
Role   : Generates diagnostic, tracking, motion vector, future extrapolation,
         and cloud coverage trend visualizations saved to output directories.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines


def save_detection_visualization(
    raw_image: np.ndarray,
    preprocessed_img: np.ndarray,
    binary_mask: np.ndarray,
    density_mask: np.ndarray,
    output_path: str,
    title: str = "Cloud Detection & Density Classification"
):
    """
    Renders a 4-panel comparison:
    1. Raw Satellite Image
    2. Preprocessed & Enhanced (CLAHE + Bilateral)
    3. Binary Cloud Mask (Otsu)
    4. Multi-level Cloud Density Classification
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Raw
    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("1. Raw Satellite Observation")
    axes[0].axis("off")

    # 2. Enhanced
    axes[1].imshow(preprocessed_img, cmap="magma")
    axes[1].set_title("2. Preprocessed (CLAHE + Bilateral)")
    axes[1].axis("off")

    # 3. Binary Mask
    axes[2].imshow(binary_mask, cmap="Blues_r")
    axes[2].set_title("3. Binary Cloud Mask")
    axes[2].axis("off")

    # 4. Density Map
    # Custom color map: 0=Navy, 1=LightCyan (Thin), 2=SkyBlue (Medium), 3=White/Crimson (Deep Convective)
    cmap_dens = matplotlib.colors.ListedColormap(["#0d1b2a", "#8ecae6", "#219ebc", "#ffb703"])
    bounds = [0, 1, 2, 3, 4]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap_dens.N)
    im4 = axes[3].imshow(density_mask, cmap=cmap_dens, norm=norm)
    axes[3].set_title("4. Cloud Density (Thin/Med/Convective)")
    axes[3].axis("off")

    cbar = fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(["Clear", "Thin", "Medium", "Convective"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_tracking_and_motion_visualization(
    base_image: np.ndarray,
    tracked_clouds: List[Any],
    motion_vectors: List[Any],
    output_path: str,
    timestamp_str: str = "",
    target_coord: Optional[Tuple[float, float]] = None,
    title: str = "Multiple Cloud Tracking & Motion Vectors"
):
    """
    Renders satellite imagery with overlaid cloud contours, centroids,
    unique tracking IDs, and velocity arrows.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_image, cmap="gray")

    # Color palette for distinct cloud IDs
    colors = ["#00f0ff", "#39ff14", "#ff007f", "#ffea00", "#ff5400", "#b5179e", "#4cc9f0"]

    motion_map = {mv.cloud_id: mv for mv in motion_vectors}

    for cloud in tracked_clouds:
        cid = cloud.cloud_id
        color = colors[(cid - 1) % len(colors)]
        cx, cy = cloud.centroid
        bx, by, bw, bh = cloud.bbox

        # 1. Bounding box
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=1.8, edgecolor=color, facecolor="none", linestyle="--")
        ax.add_patch(rect)

        # 2. Centroid marker
        ax.plot(cx, cy, marker="o", markersize=6, color=color, markeredgecolor="black")

        # 3. Cloud ID badge & status
        status_tag = f" [{cloud.status}]" if cloud.status != "ACTIVE" else ""
        label_text = f"ID:{cid}{status_tag}\nArea:{int(cloud.area)}px"
        ax.text(bx, max(15, by - 5), label_text, color=color, fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor=color))

        # 4. Movement arrow from motion vector
        if cid in motion_map:
            mv = motion_map[cid]
            px, py = mv.prev_centroid
            dx, dy = mv.dx_pixels, mv.dy_pixels

            # Draw trajectory path
            if len(cloud.history_centroids) > 1:
                hist_x = [pt[0] for pt in cloud.history_centroids]
                hist_y = [pt[1] for pt in cloud.history_centroids]
                ax.plot(hist_x, hist_y, color=color, linewidth=1.5, linestyle=":", alpha=0.8)

            # Draw prominent motion arrow scaled for visibility
            if abs(dx) > 1 or abs(dy) > 1:
                arrow_scale = 1.8
                ax.annotate(
                    "", xy=(cx + dx * arrow_scale, cy + dy * arrow_scale), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                                    color="yellow", lw=2.2, mutation_scale=15)
                )
                speed_str = f"{mv.speed_kmh:.1f} km/h" if mv.speed_kmh is not None else f"{mv.pixel_speed_per_hour:.1f} px/h"
                ax.text(cx + dx * arrow_scale + 5, cy + dy * arrow_scale + 5,
                        f"{mv.direction_compass} ({speed_str})", color="yellow",
                        fontsize=7.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1a1a", alpha=0.7, edgecolor="yellow"))

    # Highlight Target City/Station (e.g. Bengaluru) if provided
    if target_coord is not None:
        tx, ty = target_coord
        ax.plot(tx, ty, marker="*", markersize=14, color="red", markeredgecolor="white")
        circ = patches.Circle((tx, ty), radius=60, edgecolor="red", facecolor="none", linestyle=":", linewidth=1.5)
        ax.add_patch(circ)
        ax.text(tx + 8, ty - 8, "Bengaluru (Target Station)", color="red", fontsize=9,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_title(f"{title}\n{timestamp_str}", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_prediction_extrapolation_visualization(
    base_image: np.ndarray,
    extrapolations_by_cloud: Dict[int, List[Any]],
    current_clouds: List[Any],
    output_path: str,
    target_coord: Optional[Tuple[float, float]] = None,
    title: str = "Cloud Position Extrapolation (+30m, +1h, +2h)"
):
    """
    Renders extrapolated future positions of tracked cloud clusters with
    expanding uncertainty contours and projection trajectories.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_image, cmap="gray")

    lead_time_colors = {
        30: "#00f0ff",   # +30 min
        60: "#ffe600",   # +1 hr
        120: "#ff0055"   # +2 hr
    }

    # Draw current cloud centroids
    for c in current_clouds:
        ax.plot(c.centroid[0], c.centroid[1], marker="s", markersize=7, color="#00ff66", markeredgecolor="black")
        ax.text(c.centroid[0] + 5, c.centroid[1] - 5, f"ID:{c.cloud_id} (Now)", color="#00ff66",
                fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.6))

    # Draw extrapolated future centroids and uncertainty envelopes
    for cid, preds in extrapolations_by_cloud.items():
        curr_cloud = next((c for c in current_clouds if c.cloud_id == cid), None)
        if not curr_cloud:
            continue

        traj_x = [curr_cloud.centroid[0]]
        traj_y = [curr_cloud.centroid[1]]

        for p in preds:
            color = lead_time_colors.get(p.lead_time_minutes, "#ffffff")
            px, py = p.predicted_centroid
            traj_x.append(px)
            traj_y.append(py)

            # Draw predicted centroid
            ax.plot(px, py, marker="o", markersize=6, color=color, markeredgecolor="black")

            # Draw uncertainty dispersion radius
            u_circ = patches.Circle((px, py), radius=p.uncertainty_radius_px,
                                    edgecolor=color, facecolor=color, alpha=0.18, linestyle="--", linewidth=1.2)
            ax.add_patch(u_circ)

            ax.text(px + 4, py + 4, f"+{p.lead_time_minutes}m", color=color, fontsize=7.5,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="#111", alpha=0.7))

            if p.will_intersect_target:
                ax.text(px, py - p.uncertainty_radius_px - 8, "ALERT: INTERSECTING", color="red",
                        fontsize=8, fontweight="bold", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.9))

        # Draw trajectory line
        ax.plot(traj_x, traj_y, color="#ffffff", linestyle="-.", linewidth=1.3, alpha=0.7)

    # Highlight Target Station
    if target_coord is not None:
        tx, ty = target_coord
        ax.plot(tx, ty, marker="*", markersize=14, color="red", markeredgecolor="white")
        ax.text(tx + 8, ty - 8, "Bengaluru (Target)", color="red", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color="#00ff66", marker="s", linestyle="None", markersize=8, label="Current Position"),
        mlines.Line2D([], [], color="#00f0ff", marker="o", linestyle="None", markersize=7, label="+30 min Extrapolation"),
        mlines.Line2D([], [], color="#ffe600", marker="o", linestyle="None", markersize=7, label="+1 hour Extrapolation"),
        mlines.Line2D([], [], color="#ff0055", marker="o", linestyle="None", markersize=7, label="+2 hour Extrapolation"),
        mlines.Line2D([], [], color="red", marker="*", linestyle="None", markersize=10, label="Bengaluru Station")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5, framealpha=0.85)

    disclaimer = "[DISCLAIMER: Extrapolated positions based on linear motion physics. Not guaranteed forecasts.]"
    ax.set_title(f"{title}\n{disclaimer}", fontsize=11, fontweight="bold", color="darkred")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_coverage_trend_visualization(
    timestamps: List[str],
    total_coverage_pcts: List[float],
    thin_pcts: List[float],
    med_pcts: List[float],
    dense_pcts: List[float],
    output_path: str,
    title: str = "Cloud Coverage & Density Dynamics Over Time"
):
    """
    Renders cloud coverage evolution graph and stacked density changes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1. Total Cloud Coverage %
    axes[0].plot(timestamps, total_coverage_pcts, marker="o", color="#1976D2", linewidth=2.2, label="Total Cloud Coverage (%)")
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_ylim(0, min(100.0, max(total_coverage_pcts or [10.0]) + 15.0))
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Trend calculation
    if len(total_coverage_pcts) >= 2:
        diff = total_coverage_pcts[-1] - total_coverage_pcts[0]
        trend_str = f"Trend: {'INCREASING (+%.1f%%)' % diff if diff > 0 else 'DECREASING (%.1f%%)' % diff}"
        axes[0].annotate(trend_str, xy=(0.75, 0.85), xycoords="axes fraction",
                         fontsize=9.5, fontweight="bold",
                         bbox=dict(boxstyle="round", facecolor="#e3f2fd", edgecolor="#1976D2"))
    axes[0].legend(loc="upper left")

    # 2. Stacked Cloud Density Breakdown
    axes[1].stackplot(
        timestamps,
        thin_pcts, med_pcts, dense_pcts,
        labels=["Thin / Low Clouds", "Medium Clouds", "Deep Convective Clouds"],
        colors=["#8ecae6", "#219ebc", "#ffb703"],
        alpha=0.85
    )
    axes[1].set_ylabel("Density Share (%)")
    axes[1].set_xlabel("Observation Timestamp")
    axes[1].set_ylim(0, max(total_coverage_pcts or [10.0]) + 10.0)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=25, ha="right", fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


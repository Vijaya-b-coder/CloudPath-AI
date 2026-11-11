# -*- coding: utf-8 -*-
"""
Generate 4 Clean, High-Resolution Technical Diagrams for CloudPath AI Project Synopsis:
1. Overall System Architecture
2. Proposed Methodology Workflow
3. Satellite Cloud Movement Tracking Workflow
4. Rainfall Prediction Workflow
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = r"D:\bengaluru.csv\CloudPath_AI_Diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# Color Palette
NAVY = "#0F172A"       # Title & Frame
DARK_BLUE = "#1E293B"  # Primary Node
CYAN_ACC = "#0284C7"   # Secondary Node / Accents
TEAL_ACC = "#0D9488"   # Tracking Node
BG_COLOR = "#FFFFFF"   # Crisp White Canvas
BOX_BG = "#F8FAFC"     # Light background for containers
TEXT_COLOR = "#0F172A"
BOX_BORDER = "#64748B"

def draw_rounded_box(ax, xy, width, height, title="", subtitle="", box_color="#FFFFFF", border_color="#334155", title_color="#0F172A", sub_color="#475569", title_size=10, sub_size=8, pad=0.02):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=box_color, edgecolor=border_color, linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    if subtitle:
        cy1 = xy[1] + height * 0.62
        cy2 = xy[1] + height * 0.28
        ax.text(cx, cy1, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=4)
        ax.text(cx, cy2, subtitle, ha='center', va='center', fontsize=sub_size, color=sub_color, zorder=4)
    else:
        cy = xy[1] + height / 2.0
        ax.text(cx, cy, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=4)

def draw_arrow(ax, start, end, label="", color="#334155", lw=1.8, label_pos=0.5):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, shrinkA=3, shrinkB=3),
        zorder=5
    )
    if label:
        lx = start[0] + (end[0] - start[0]) * label_pos
        ly = start[1] + (end[1] - start[1]) * label_pos + 0.025
        ax.text(lx, ly, label, ha='center', va='bottom', fontsize=8, fontweight='bold', color=color, zorder=6)

# ------------------------------------------------------------------------------
# DIAGRAM 1: OVERALL SYSTEM ARCHITECTURE
# ------------------------------------------------------------------------------
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Title
    ax.text(5.0, 5.2, "CloudPath AI - Overall System Architecture", ha='center', va='center', fontsize=13, fontweight='bold', color=NAVY)

    # Branch 1: Satellite Tracking Pipeline (Top)
    draw_rounded_box(ax, (0.5, 3.8), 2.2, 0.7, "Satellite Data", "INSAT-3DS NetCDF4 (4km)", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (3.2, 3.8), 1.9, 0.7, "Preprocessing", "Crop 162x159 & Otsu Mask", "#F0FDF4", "#22C55E", "#14532D", "#15803D")
    draw_rounded_box(ax, (5.5, 3.8), 1.8, 0.7, "Cloud Detection", "8-Conn CCA Centroids", "#F0FDFA", "#14B8A6", "#134E4A", "#0F766E")
    draw_rounded_box(ax, (7.7, 3.8), 1.8, 0.7, "Cloud Tracking", "Hungarian Match & Kinematics", "#F0FDFA", "#14B8A6", "#134E4A", "#0F766E")

    draw_arrow(ax, (2.7, 4.15), (3.2, 4.15))
    draw_arrow(ax, (5.1, 4.15), (5.5, 4.15))
    draw_arrow(ax, (7.3, 4.15), (7.7, 4.15))

    # Branch 2: Weather & Rainfall Prediction Pipeline (Bottom)
    draw_rounded_box(ax, (0.5, 2.3), 2.2, 0.7, "Historical Weather Data", "96,431 Records (2009-2020)", "#FFFBEB", "#F59E0B", "#78350F", "#B45309")
    draw_rounded_box(ax, (3.2, 2.3), 1.9, 0.7, "Data Preprocessing", "Cleaning & 80/20 Split", "#FFFBEB", "#F59E0B", "#78350F", "#B45309")
    draw_rounded_box(ax, (5.5, 2.3), 1.8, 0.7, "Feature Extraction", "25 Atmospheric Vars", "#FEF3C7", "#D97706", "#78350F", "#92400E")
    draw_rounded_box(ax, (7.7, 2.3), 1.8, 0.7, "XGBoost Models", "Classification + Regression", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")

    draw_arrow(ax, (2.7, 2.65), (3.2, 2.65))
    draw_arrow(ax, (5.1, 2.65), (5.5, 2.65))
    draw_arrow(ax, (7.3, 2.65), (7.7, 2.65))

    # Unified Convergence: Dashboard & User
    draw_rounded_box(ax, (3.0, 0.8), 4.0, 0.75, "CloudPath AI Dashboard", "Interactive Web Interface (Midnight Navy / Cyan)", "#F8FAFC", "#0F172A", "#0F172A", "#334155")
    draw_rounded_box(ax, (7.8, 0.8), 1.7, 0.75, "End User", "Meteorologist / Planner", "#F1F5F9", "#475569", "#0F172A", "#475569")

    # Connect Pipelines to Dashboard
    draw_arrow(ax, (8.6, 3.8), (6.5, 1.55), "Tracking Metrics")
    draw_arrow(ax, (8.6, 2.3), (5.5, 1.55), "Precipitation Forecast")
    draw_arrow(ax, (7.0, 1.175), (7.8, 1.175), "Presents")

    out_path = os.path.join(OUT_DIR, "synopsis_fig1_architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("Generated:", out_path)

# ------------------------------------------------------------------------------
# DIAGRAM 2: PROPOSED METHODOLOGY WORKFLOW
# ------------------------------------------------------------------------------
def generate_methodology_workflow():
    fig, ax = plt.subplots(figsize=(10, 6.0), dpi=300)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis('off')

    ax.text(5.0, 5.7, "CloudPath AI - Proposed Methodology Workflow", ha='center', va='center', fontsize=13, fontweight='bold', color=NAVY)

    # 9-Step Flowchart arranged in 3 rows
    # Row 1: Steps 1, 2, 3
    draw_rounded_box(ax, (0.5, 4.4), 2.6, 0.8, "1. Ingest Satellite Data", "INSAT-3DS July 2025 Archive", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (3.7, 4.4), 2.6, 0.8, "2. Spatial Preprocessing", "Crop Bengaluru Grid (162x159)", "#F0FDF4", "#22C55E", "#14532D", "#15803D")
    draw_rounded_box(ax, (6.9, 4.4), 2.6, 0.8, "3. Cloud Segmentation", "Otsu Mask & 8-Connectivity CCA", "#F0FDFA", "#14B8A6", "#134E4A", "#0F766E")

    draw_arrow(ax, (3.1, 4.8), (3.7, 4.8))
    draw_arrow(ax, (6.3, 4.8), (6.9, 4.8))
    draw_arrow(ax, (8.2, 4.4), (8.2, 3.8))

    # Row 2: Steps 4, 5, 6 (Right to Left)
    draw_rounded_box(ax, (6.9, 2.9), 2.6, 0.8, "4. Track Cloud Formations", "Hungarian Association (t-1 to t)", "#F0FDFA", "#14B8A6", "#134E4A", "#0F766E")
    draw_rounded_box(ax, (3.7, 2.9), 2.6, 0.8, "5. Estimate Kinematics", "Centroid Speed & Heading Angle", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (0.5, 2.9), 2.6, 0.8, "6. Preprocess Weather Data", "96,431 Records (2009-2020)", "#FFFBEB", "#F59E0B", "#78350F", "#B45309")

    draw_arrow(ax, (6.9, 3.3), (6.3, 3.3))
    draw_arrow(ax, (3.7, 3.3), (3.1, 3.3))
    draw_arrow(ax, (1.8, 2.9), (1.8, 2.3))

    # Row 3: Steps 7, 8, 9 (Left to Right)
    draw_rounded_box(ax, (0.5, 1.4), 2.6, 0.8, "7. Train XGBoost Models", "25 Features (Classifier & Regressor)", "#FFFBEB", "#F59E0B", "#78350F", "#B45309")
    draw_rounded_box(ax, (3.7, 1.4), 2.6, 0.8, "8. Generate Forecast", "P(Rain) & Rainfall Volume (mm)", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (6.9, 1.4), 2.6, 0.8, "9. Render Dashboard", "Unified UI (Tracks & Weather)", "#F8FAFC", "#0F172A", "#0F172A", "#334155")

    draw_arrow(ax, (3.1, 1.8), (3.7, 1.8))
    draw_arrow(ax, (6.3, 1.8), (6.9, 1.8))

    out_path = os.path.join(OUT_DIR, "synopsis_fig2_methodology.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("Generated:", out_path)

# ------------------------------------------------------------------------------
# DIAGRAM 3: SATELLITE CLOUD MOVEMENT TRACKING WORKFLOW
# ------------------------------------------------------------------------------
def generate_satellite_tracking_workflow():
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    ax.text(5.0, 4.6, "Module A: Satellite Cloud Movement Tracking Workflow", ha='center', va='center', fontsize=13, fontweight='bold', color=NAVY)

    # 6 Linear Steps with detailed descriptors
    draw_rounded_box(ax, (0.4, 3.0), 2.7, 0.85, "1. Satellite Ingestion", "Extract INSAT-3DS NetCDF4\nCrop 162x159 px Bengaluru ROI", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (3.6, 3.0), 2.7, 0.85, "2. Cloud-Mask Extraction", "Filter Invalid Pixels & Apply\nOtsu Adaptive Thresholding", "#F0FDF4", "#22C55E", "#14532D", "#15803D")
    draw_rounded_box(ax, (6.8, 3.0), 2.8, 0.85, "3. Formation Detection", "8-Connectivity CCA Labeling\nCentroids (x_bar, y_bar) & Area", "#F0FDFA", "#14B8A6", "#134E4A", "#0F766E")

    draw_arrow(ax, (3.1, 3.425), (3.6, 3.425))
    draw_arrow(ax, (6.3, 3.425), (6.8, 3.425))
    draw_arrow(ax, (8.2, 3.0), (8.2, 2.3))

    draw_rounded_box(ax, (6.8, 1.4), 2.8, 0.85, "4. Cloud Tracking (t-1 to t)", "Gated Distance Cost Matrix\nHungarian Assignment Algorithm", "#F0FDFA", "#14B8A6", "#134E4A", "#0F766E")
    draw_rounded_box(ax, (3.6, 1.4), 2.7, 0.85, "5. Kinematic Estimation", "Compute Velocity (vx, vy)\nSpeed (km/h) & Heading Azimuth", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (0.4, 1.4), 2.7, 0.85, "6. Spatial Visualization", "GeoJSON Vector Overlays &\nTimeline Cloud Scrubber", "#F8FAFC", "#0F172A", "#0F172A", "#334155")

    draw_arrow(ax, (6.8, 1.825), (6.3, 1.825))
    draw_arrow(ax, (3.6, 1.825), (3.1, 1.825))

    out_path = os.path.join(OUT_DIR, "synopsis_fig3_tracking_workflow.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("Generated:", out_path)

# ------------------------------------------------------------------------------
# DIAGRAM 4: RAINFALL PREDICTION WORKFLOW
# ------------------------------------------------------------------------------
def generate_rainfall_prediction_workflow():
    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    ax.text(5.0, 4.6, "Module B: Machine Learning Rainfall Prediction Workflow", ha='center', va='center', fontsize=13, fontweight='bold', color=NAVY)

    # 5 Sequential Stages
    draw_rounded_box(ax, (0.4, 2.2), 2.0, 1.0, "1. Weather Ingestion", "96,431 Hourly Records\n(2009-2020 Reanalysis)\nClean & Temporal Sort", "#FFFBEB", "#F59E0B", "#78350F", "#B45309")
    draw_rounded_box(ax, (2.8, 2.2), 2.0, 1.0, "2. Feature Extraction", "25 Atmospheric Variables\nTemp, Dew, Humidity, Wind,\nRadiation & Pressure", "#FEF3C7", "#D97706", "#78350F", "#92400E")
    draw_rounded_box(ax, (5.2, 2.2), 2.1, 1.0, "3. Stage 1: Classifier", "Tuned XGBoost Binary Class\nP(Rain) Threshold >= 0.50\n(94.13% Accuracy)", "#EFF6FF", "#3B82F6", "#1E3A8A", "#1D4ED8")
    draw_rounded_box(ax, (7.6, 2.8), 2.0, 0.9, "4A. If No Rain", "P(Rain) < 0.50\nVolume = 0.00 mm", "#F0FDF4", "#22C55E", "#14532D", "#15803D")
    draw_rounded_box(ax, (7.6, 1.5), 2.0, 0.9, "4B. Stage 2: Regressor", "Tuned XGBoost Regressor\nPredicts Amount in mm\n(MAE = 0.0597 mm)", "#FEF2F2", "#EF4444", "#7F1D1D", "#B91C1C")

    draw_arrow(ax, (2.4, 2.7), (2.8, 2.7))
    draw_arrow(ax, (4.8, 2.7), (5.2, 2.7))
    draw_arrow(ax, (7.3, 2.9), (7.6, 3.15), "Dry")
    draw_arrow(ax, (7.3, 2.5), (7.6, 2.05), "Wet")

    # Output node
    draw_rounded_box(ax, (3.2, 0.4), 3.6, 0.7, "5. Unified Prediction Visualization", "Dynamic KPI Cards & Weather Conditions Display", "#F8FAFC", "#0F172A", "#0F172A", "#334155")
    draw_arrow(ax, (7.6, 2.8), (6.8, 0.75))
    draw_arrow(ax, (7.6, 1.95), (6.8, 0.75))

    out_path = os.path.join(OUT_DIR, "synopsis_fig4_rainfall_workflow.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print("Generated:", out_path)

if __name__ == "__main__":
    generate_system_architecture()
    generate_methodology_workflow()
    generate_satellite_tracking_workflow()
    generate_rainfall_prediction_workflow()
    print("All 4 Synopsis Diagrams Generated Successfully!")


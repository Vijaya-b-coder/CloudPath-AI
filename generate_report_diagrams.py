"""
generate_report_diagrams.py
===========================
Generates 10 publication-quality technical diagrams for the CloudPath AI
B.Tech Final-Year Project Report.
Matching the Midnight Navy, Electric Cyan, and Teal palette on clean white canvas.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, BoxStyle

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CloudPath_AI_Diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

# Common styling constants
FONT_FAMILY = "sans-serif"
COLOR_NAVY = "#172554"       # Deep Slate Blue
COLOR_DARK_BLUE = "#1E293B"  # Secondary Dark Blue
COLOR_CYAN = "#0284C7"       # Electric Cyan / Deep
COLOR_TEAL = "#0D9488"       # Teal
COLOR_SKY = "#3B82F6"        # Sky Blue
COLOR_CORAL = "#DC2626"      # Coral Red / Alert
COLOR_EMERALD = "#059669"    # Emerald Green / Pass
COLOR_BG_BOX = "#F8FAFC"     # Off-white / light slate fill
COLOR_BORDER = "#334155"     # Dark Slate Border


def create_box(ax, x, y, w, h, title, subtitle="", box_color="#EFF6FF", border_color="#1E293B", text_color="#0F172A", title_size=10, sub_size=8, corner=0.04):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=BoxStyle("Round", pad=0.01, rounding_size=corner),
                         facecolor=box_color, edgecolor=border_color, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_color, zorder=3, family=FONT_FAMILY)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=sub_size, color="#475569", zorder=3, family=FONT_FAMILY)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=text_color, zorder=3, family=FONT_FAMILY)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#1E293B", lw=1.5, label_pos=0.5, label_offset=(0, 0.02), font_size=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, shrinkA=0, shrinkB=0), zorder=4)
    if label:
        lx = x1 + (x2 - x1) * label_pos + label_offset[0]
        ly = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=font_size, color="#334155",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9), zorder=5, family=FONT_FAMILY)


# ==============================================================================
# Diagram 1: Overall System Architecture
# ==============================================================================
def draw_figure_1():
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title Banner
    ax.text(50, 97, "CloudPath AI – Multi-Tier System Architecture", ha="center", va="center", fontsize=15, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)
    ax.text(50, 94, "Parallel Situational Integration of Satellite Cloud Tracking & Meteorological ML Nowcasting", ha="center", va="center", fontsize=9, color="#64748B", family=FONT_FAMILY)

    # 4 Layers (Background Groupings)
    layers = [
        ("Layer 4: Presentation & Operational Dashboard Layer", 73, 18, "#F0FDF4", "#059669"),
        ("Layer 3: Analytical & Machine Learning Inference Engine", 50, 19, "#EFF6FF", "#1D4ED8"),
        ("Layer 2: Preprocessing, Geospatial Cropping & Segmentation Layer", 27, 19, "#F0FDFA", "#0D9488"),
        ("Layer 1: Raw Ingestion & Multi-Source Data Layer", 4, 19, "#F8FAFC", "#475569")
    ]
    for title, y, h, bg, border in layers:
        bg_box = FancyBboxPatch((3, y), 94, h, boxstyle=BoxStyle("Round", pad=0.01, rounding_size=0.02),
                                facecolor=bg, edgecolor=border, linewidth=1.2, linestyle="--", alpha=0.8, zorder=1)
        ax.add_patch(bg_box)
        ax.text(5, y + h - 2.2, title, ha="left", va="center", fontsize=9.5, fontweight="bold", color=border, zorder=3, family=FONT_FAMILY)

    # Layer 1 Components (Data Layer)
    create_box(ax, 6, 6, 26, 12, "Historical Weather Dataset", "96,431 records (2009–2020)\nHourly Ground Station Observations", box_color="#FFFFFF", border_color="#3B82F6")
    create_box(ax, 37, 6, 26, 12, "INSAT-3DS Satellite Archive", "31 NetCDF-4 Files (July 2025)\n4km Cloud Cover Product Ver02", box_color="#FFFFFF", border_color="#0D9488")
    create_box(ax, 68, 6, 26, 12, "Bengaluru Reanalysis (2025)", "744 Hourly ERA5 Observations\nGround Sensor Synchronization", box_color="#FFFFFF", border_color="#6366F1")

    # Layer 2 Components (Preprocessing)
    create_box(ax, 6, 29, 26, 13, "Weather Preprocessing Module", "Missing value imputation, Min-Max\nCyclical sin/cos temporal encoding", box_color="#FFFFFF", border_color="#3B82F6")
    create_box(ax, 37, 29, 26, 13, "Regional Crop & Mask Extraction", "162x159 pixel Bengaluru grid\nLat: 12.35–13.67°N, Lon: 77.05–78.38°E", box_color="#FFFFFF", border_color="#0D9488")
    create_box(ax, 68, 29, 26, 13, "Connected Component Segmentation", "8-way connectivity (cv2.connectedComp)\n16,321 cloud formations segmented", box_color="#FFFFFF", border_color="#0D9488")

    # Layer 3 Components (Inference & Tracking)
    create_box(ax, 6, 52, 26, 14, "XGBoost Rainfall Models", "Classifier: Acc 94.13%, ROC 0.975\nRegressor: MAE 0.0597 mm (R² 0.760)", box_color="#EFF6FF", border_color="#1D4ED8")
    create_box(ax, 37, 52, 26, 14, "Hungarian Assignment Tracker", "Pairwise centroid matching\nCost threshold d_max = 120 km", box_color="#F0FDFA", border_color="#0D9488")
    create_box(ax, 68, 52, 26, 14, "Geospatial Velocity Calculator", "4.079 km/row, 4.055 km/col\nSpeed (km/h) & 8-compass direction", box_color="#F0FDFA", border_color="#0D9488")

    # Layer 4 Components (Presentation)
    create_box(ax, 6, 75, 26, 13, "Operational Live Overview", "Synchronized 1,448 Observation Slider\nTop KPIs (Coverage, Regions, Speed, Status)", box_color="#FFFFFF", border_color="#059669")
    create_box(ax, 37, 75, 26, 13, "Dynamic Cloud Tracker UI", "Cloud-mask frames with green motion vectors\nTrajectory analysis & displacement roses", box_color="#FFFFFF", border_color="#059669")
    create_box(ax, 68, 75, 26, 13, "Weather Conditions & Console", "Real-time Ground Weather Sync\nDynamic ML Rain Probability & Next Hr mm", box_color="#FFFFFF", border_color="#059669")

    # Vertical Inter-Layer Arrows
    draw_arrow(ax, 19, 18, 19, 29, "25 raw features")
    draw_arrow(ax, 50, 18, 50, 29, "NetCDF-4 V2")
    draw_arrow(ax, 50, 42, 50, 52, "Centroids & Areas")
    draw_arrow(ax, 19, 42, 19, 52, "Feature Matrix X")
    draw_arrow(ax, 63, 35, 68, 35, "Binary Mask")
    draw_arrow(ax, 63, 59, 68, 59, "Matched Pairs")
    draw_arrow(ax, 19, 66, 19, 75, "Rain Pred & mm")
    draw_arrow(ax, 50, 66, 50, 75, "Vectors & Speeds")
    draw_arrow(ax, 81, 66, 81, 75, "Aligned Context")

    # Horizontal Bridge (Parallel Awareness)
    draw_arrow(ax, 32, 59, 37, 59, "Audit Bridge", color="#64748B", lw=1.2)
    draw_arrow(ax, 37, 81, 32, 81, "Context", color="#64748B", lw=1.2)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig1_system_architecture.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 1: {fig_path}")


# ==============================================================================
# Diagram 2: Level 0 DFD (Context Diagram)
# ==============================================================================
def draw_figure_2():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 95, "CloudPath AI – Level 0 Data Flow Diagram (Context Diagram)", ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)

    # External Entities (Rectangles)
    create_box(ax, 5, 60, 22, 18, "Meteorological Data\nArchive (Ground)", "IMD / Open-Meteo\nHourly Weather Records", box_color="#EFF6FF", border_color="#1D4ED8")
    create_box(ax, 5, 20, 22, 18, "INSAT-3DS Satellite\nRepository (MOSDAC)", "ISRO / NRSC Portal\nDaily Cloud Mask NetCDF", box_color="#F0FDFA", border_color="#0D9488")
    create_box(ax, 73, 40, 22, 20, "User / Operational\nMeteorologist", "Web Browser Client\nDecision Makers / Public", box_color="#FEF2F2", border_color="#DC2626")

    # Central System Process (Circle / Rounded Box)
    sys_circle = FancyBboxPatch((38, 30), 24, 40, boxstyle=BoxStyle("Round", pad=0.02, rounding_size=0.15),
                                facecolor="#F8FAFC", edgecolor=COLOR_NAVY, linewidth=2.5, zorder=2)
    ax.add_patch(sys_circle)
    ax.text(50, 54, "0.0", ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_NAVY, zorder=3, family=FONT_FAMILY)
    ax.text(50, 48, "CloudPath AI\nCore System", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_NAVY, zorder=3, family=FONT_FAMILY)
    ax.text(50, 37, "Integrated Tracking &\nRainfall Nowcasting", ha="center", va="center", fontsize=8, color="#64748B", zorder=3, family=FONT_FAMILY)

    # Data Store (Bottom)
    store_box = FancyBboxPatch((36, 4), 28, 12, boxstyle=BoxStyle("Square", pad=0),
                               facecolor="#FFFFFF", edgecolor="#334155", linewidth=1.5, zorder=2)
    ax.add_patch(store_box)
    ax.plot([36, 64], [14, 14], color="#334155", lw=1.5, zorder=3)
    ax.text(38, 10, "D1", ha="left", va="center", fontsize=9, fontweight="bold", color="#334155", zorder=3, family=FONT_FAMILY)
    ax.text(51, 8.5, "Unified Project Store", ha="center", va="center", fontsize=9, fontweight="bold", color=COLOR_NAVY, zorder=3, family=FONT_FAMILY)
    ax.text(51, 5.5, "(Models, NPZ Arrays, CSV Summaries)", ha="center", va="center", fontsize=7.5, color="#64748B", zorder=3, family=FONT_FAMILY)

    # Arrows
    draw_arrow(ax, 27, 69, 38, 58, "Hourly Weather Data (temp, hum, precip)", font_size=7.5, label_offset=(0, 2))
    draw_arrow(ax, 27, 29, 38, 42, "INSAT-3DS NetCDF-4 V2 Granules", font_size=7.5, label_offset=(0, -2))
    draw_arrow(ax, 62, 55, 73, 55, "Forecast Visualizations & KPIs", font_size=7.5, label_offset=(0, 2))
    draw_arrow(ax, 73, 45, 62, 45, "Interactive Timeline Queries", font_size=7.5, label_offset=(0, -2))
    draw_arrow(ax, 47, 30, 47, 16, "Persist outputs", font_size=7.5, label_offset=(-5, 0))
    draw_arrow(ax, 53, 16, 53, 30, "Fetch artifacts", font_size=7.5, label_offset=(5, 0))

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig2_dfd_level0.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 2: {fig_path}")


# ==============================================================================
# Diagram 3: Level 1 DFD
# ==============================================================================
def draw_figure_3():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "CloudPath AI – Level 1 Data Flow Diagram", ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)

    # External Entities
    create_box(ax, 4, 76, 16, 12, "Ground Weather\nSource", box_color="#EFF6FF", border_color="#3B82F6", title_size=8.5)
    create_box(ax, 4, 34, 16, 12, "INSAT-3DS\nArchive", box_color="#F0FDFA", border_color="#0D9488", title_size=8.5)
    create_box(ax, 82, 54, 15, 14, "Forecaster /\nClient Browser", box_color="#FEF2F2", border_color="#DC2626", title_size=8.5)

    # Processes (Numbered circles/rounded boxes)
    p_style = dict(box_color="#FFFFFF", border_color=COLOR_NAVY, title_size=9, sub_size=7.5)
    create_box(ax, 27, 74, 21, 16, "1.0 Ingestion &\nFeature Preprocessing", "Missing value & cyclical encode", **p_style)
    create_box(ax, 55, 74, 21, 16, "2.0 XGBoost ML\nRainfall Predictor", "Classification & Regression", **p_style)
    
    create_box(ax, 27, 32, 21, 16, "3.0 NetCDF-4 Regional\nSubgrid Extraction", "162x159 Bengaluru Domain", **p_style)
    create_box(ax, 55, 32, 21, 16, "4.0 Cloud Segmentation &\nCentroid Extraction", "Connected Components (8-way)", **p_style)
    
    create_box(ax, 27, 4, 21, 16, "5.0 Hungarian Cloud\nMotion Tracking", "Bipartite assignment d<=120km", **p_style)
    create_box(ax, 55, 4, 21, 16, "6.0 Multi-Modal Parallel\nIntegration Service", "Situational context aggregation", **p_style)

    # Data Stores
    def draw_store(x, y, w, h, id_str, name):
        box = FancyBboxPatch((x, y), w, h, boxstyle=BoxStyle("Square", pad=0), facecolor="#F8FAFC", edgecolor="#475569", linewidth=1.2, zorder=2)
        ax.add_patch(box)
        ax.plot([x, x + w], [y + h * 0.7, y + h * 0.7], color="#475569", lw=1.2, zorder=3)
        ax.text(x + 2, y + h * 0.82, id_str, ha="left", va="center", fontsize=7.5, fontweight="bold", color="#475569", zorder=3)
        ax.text(x + w / 2, y + h * 0.35, name, ha="center", va="center", fontsize=8, fontweight="bold", color=COLOR_NAVY, zorder=3)

    draw_store(27, 56, 21, 10, "D1", "Cleaned Weather DB")
    draw_store(55, 56, 21, 10, "D2", "Trained Models")
    draw_store(82, 16, 15, 14, "D3", "Satellite NPZ &\nTracking Store")

    # Connecting Flow Arrows
    draw_arrow(ax, 20, 82, 27, 82, "Raw CSV")
    draw_arrow(ax, 37, 74, 37, 66, "Cleaned")
    draw_arrow(ax, 48, 61, 55, 61, "Features")
    draw_arrow(ax, 65, 66, 65, 74, "Weights")
    draw_arrow(ax, 66, 82, 82, 63, "Rain prob & mm")
    
    draw_arrow(ax, 20, 40, 27, 40, "Daily .nc")
    draw_arrow(ax, 48, 40, 55, 40, "162x159 mask")
    draw_arrow(ax, 65, 32, 65, 20, "Regions & Centroids")
    draw_arrow(ax, 55, 12, 48, 12, "Displacements")
    draw_arrow(ax, 37, 32, 37, 20, "Prev/Curr Frames")
    draw_arrow(ax, 66, 12, 82, 23, "Tracks")
    
    draw_arrow(ax, 76, 12, 85, 54, "Motion Vectors")
    draw_arrow(ax, 82, 58, 76, 78, "Slider Input")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig3_dfd_level1.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 3: {fig_path}")


# ==============================================================================
# Diagram 4: Satellite Data Processing Flowchart
# ==============================================================================
def draw_figure_4():
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97, "Satellite Data Processing Flowchart", ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)
    ax.text(50, 95, "INSAT-3DS Level-3 Cloud Mask Extraction & Regional Ingestion", ha="center", va="center", fontsize=8.5, color="#64748B", family=FONT_FAMILY)

    nodes = [
        (50, 90, "Start Processing Pipeline", "#FEF3C7", "#D97706", "Terminal"),
        (50, 80, "Scan `satellite_img/` Directory\nDetect 31 Daily ZIP Files (July 2025)", "#FFFFFF", "#334155", "Process"),
        (50, 69, "Extract & Inspect NetCDF-4 Granules\nINS3S_L3_CF_4km_Ver02 (930–995 MB/day)", "#FFFFFF", "#334155", "Process"),
        (50, 58, "Parse Global Coordinate Grids\nExtract Latitude, Longitude & Time Vectors", "#FFFFFF", "#334155", "Process"),
        (50, 47, "Slice Bengaluru Regional Bounding Box\n12.35°–13.67° N, 77.05°–78.38° E (162 × 159 px)", "#EFF6FF", "#1D4ED8", "Process"),
        (50, 36, "Decode Byte Cloud Mask (CM_FLAG)\n0 = Clear Sky, 1 = Cloud Covered", "#FFFFFF", "#0D9488", "Process"),
        (50, 25, "Perform Connected Component Analysis\n8-Way Neighborhood Segmentation (cv2)", "#F0FDFA", "#0D9488", "Process"),
        (50, 14, "Calculate Metrics per Observation\nCloud Coverage %, Clear %, Centroids & Area", "#FFFFFF", "#059669", "Process"),
        (50, 4, "Export to `bengaluru_cloudmask_series.npz`\n1,448 Observations Saved for Tracking", "#DCFCE7", "#059669", "Terminal")
    ]

    for x, y, title, bg, border, ntype in nodes:
        w, h = 64, 7.5
        rounding = 0.25 if ntype == "Terminal" else 0.04
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=BoxStyle("Round", pad=0.01, rounding_size=rounding),
                             facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=3, family=FONT_FAMILY)

    # Vertical Connecting Arrows
    for i in range(len(nodes) - 1):
        y_top = nodes[i][1] - 3.75
        y_bot = nodes[i + 1][1] + 3.75
        draw_arrow(ax, 50, y_top, 50, y_bot, lw=1.5)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig4_satellite_flowchart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 4: {fig_path}")


# ==============================================================================
# Diagram 5: Rainfall Prediction Workflow
# ==============================================================================
def draw_figure_5():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "Rainfall Prediction Workflow (Two-Stage XGBoost Pipeline)", ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)

    # Step 1: Input Vector
    create_box(ax, 5, 42, 20, 40, "25 Ground Weather\nFeatures Ingested",
               "Temperature (tempC, max, min)\nHumidity, Cloudcover, Pressure\nDewPoint, FeelsLike, HeatIndex\nWind speed, Gust, Direction\nSolar hour, UV index, Visibility\nCyclical temporal (hour, day,\nmonth, day_of_week, season)",
               box_color="#EFF6FF", border_color="#3B82F6", title_size=9.5, sub_size=7)

    # Step 2: Preprocessing
    create_box(ax, 30, 52, 18, 20, "Feature Scaling &\nValidation Checks", "StandardScaler transform\nVerify finite float vector\nImpute missing features", box_color="#FFFFFF", border_color="#1E293B", title_size=9, sub_size=7.5)

    # Step 3: Classification
    create_box(ax, 53, 64, 20, 22, "Stage 1: XGBoost\nRainfall Classifier", "model.predict_proba(X)\nDynamic class 1 lookup\nRain Probability P(Rain)\nP(No Rain) = 1 - P(Rain)", box_color="#FEF2F2", border_color="#DC2626", title_size=9.5, sub_size=7.5)

    # Decision Diamond
    diamond = patches.Polygon([[80, 75], [87, 83], [94, 75], [87, 67]], closed=True, facecolor="#FEF3C7", edgecolor="#D97706", lw=1.5, zorder=2)
    ax.add_patch(diamond)
    ax.text(87, 75, "P(Rain)\n>= 0.50 ?", ha="center", va="center", fontsize=8, fontweight="bold", color="#78350F", zorder=3, family=FONT_FAMILY)

    # Step 4: Regression
    create_box(ax, 53, 20, 20, 22, "Stage 2: XGBoost\nRainfall Regressor", "Continuous precipitation\nPrediction: y_hat (mm)\nNon-negative bounding:\ny_pred = max(0.0, y_hat)", box_color="#F0FDF4", border_color="#059669", title_size=9.5, sub_size=7.5)

    # Outputs
    create_box(ax, 80, 45, 17, 14, "Output: RAIN\nAlert Active", "Display P(Rain) %\nDisplay y_pred (mm)\nNext Hour Estimate", box_color="#FEE2E2", border_color="#DC2626", title_size=9, sub_size=7)
    create_box(ax, 80, 20, 17, 14, "Output: NO RAIN\nClear Skies", "Display P(No Rain) %\nPrecipitation: 0.00 mm\nStable Condition", box_color="#DCFCE7", border_color="#059669", title_size=9, sub_size=7)

    # Arrows
    draw_arrow(ax, 25, 62, 30, 62, "25-d vector")
    draw_arrow(ax, 48, 67, 53, 73, "Standardized X")
    draw_arrow(ax, 48, 57, 53, 33, "Pass to Regressor")
    draw_arrow(ax, 73, 75, 80, 75, "Probabilities")
    draw_arrow(ax, 87, 67, 87, 59, "Yes (Rain)")
    draw_arrow(ax, 87, 83, 98, 83)
    draw_arrow(ax, 98, 83, 98, 27)
    draw_arrow(ax, 98, 27, 97, 27, "No (Dry)")
    draw_arrow(ax, 73, 31, 80, 48, "Predicted mm")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig5_rainfall_workflow.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 5: {fig_path}")


# ==============================================================================
# Diagram 6: Cloud Detection and Tracking Workflow
# ==============================================================================
def draw_figure_6():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "Cloud Detection and Movement Tracking Workflow", ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)

    # Frame t & Frame t+1 inputs
    create_box(ax, 4, 60, 20, 24, "Frame t\nCloud Mask M_t", "INSAT-3DS Obs (t)\nConnected Components\nCentroids C_t = {(r_i, c_i)}\nAreas A_t = {a_i}", box_color="#EFF6FF", border_color="#3B82F6", title_size=9.5, sub_size=7.5)
    create_box(ax, 4, 18, 20, 24, "Frame t+1\nCloud Mask M_{t+1}", "INSAT-3DS Obs (t + 30 min)\nConnected Components\nCentroids C_{t+1} = {(r_j, c_j)}\nAreas A_{t+1} = {a_j}", box_color="#F0FDFA", border_color="#0D9488", title_size=9.5, sub_size=7.5)

    # Cost Matrix Formulator
    create_box(ax, 28, 38, 22, 28, "Bipartite Cost Matrix\nFormulation", "Pairwise Euclidean distance:\nD_ij = sqrt((r_i-r_j)^2 + (c_i-c_j)^2)\nDistance Gating Threshold:\nCost_ij = D_ij if D_ij <= d_max\nCost_ij = infinity if D_ij > d_max\n(d_max = 120 km / ~30 pixels)", box_color="#FFFFFF", border_color="#1E293B", title_size=9.5, sub_size=7)

    # Hungarian Optimizer
    create_box(ax, 54, 42, 20, 20, "Hungarian Optimization\n(Munkres Algorithm)", "Global minimum cost matching\nscipy.optimize.linear_sum_assignment\nComplexity O(n^3)\nDeterministic 1-to-1 Association", box_color="#FDF4FF", border_color="#A855F7", title_size=9.5, sub_size=7)

    # Output Branches
    create_box(ax, 78, 68, 19, 20, "Matched Trajectories\n(Persistent Tracks)", "Delta row = r_j - r_i\nDelta col = c_j - c_i\nVelocity V_kmh & Compass Dir\nConfidence Score (Area overlap)", box_color="#DCFCE7", border_color="#059669", title_size=9, sub_size=7)
    create_box(ax, 78, 41, 19, 18, "Unmatched C_{t+1}\n(New Formations)", "Initialize new Track IDs\nRecord creation timestamp\nAwaiting subsequent frames", box_color="#EFF6FF", border_color="#3B82F6", title_size=9, sub_size=7)
    create_box(ax, 78, 14, 19, 18, "Unmatched C_t\n(Dissipated Clouds)", "Terminate active Track IDs\nCompute total track lifespan\nArchive trajectory history", box_color="#FEF2F2", border_color="#DC2626", title_size=9, sub_size=7)

    # Connecting Arrows
    draw_arrow(ax, 24, 72, 28, 55, "C_t centroids")
    draw_arrow(ax, 24, 30, 28, 47, "C_{t+1} centroids")
    draw_arrow(ax, 50, 52, 54, 52, "Cost Matrix")
    draw_arrow(ax, 74, 56, 78, 75, "Matched (i -> j)")
    draw_arrow(ax, 74, 52, 78, 50, "Unmatched in t+1")
    draw_arrow(ax, 74, 46, 78, 25, "Unmatched in t")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig6_cloud_tracking_workflow.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 6: {fig_path}")


# ==============================================================================
# Diagram 7: Hungarian Assignment Algorithm Workflow
# ==============================================================================
def draw_figure_7():
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97, "Hungarian Assignment Algorithm Execution Flow", ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)
    ax.text(50, 95, "Optimal Bipartite Cloud Centroid Matching (Kuhn-Munkres)", ha="center", va="center", fontsize=8.5, color="#64748B", family=FONT_FAMILY)

    steps = [
        (50, 90, "Start: Ingest n x m Centroid Cost Matrix C", "#FEF3C7", "#D97706", "Terminal"),
        (50, 80, "Step 1: Row Reduction\nSubtract minimum element of each row from all elements in that row", "#FFFFFF", "#334155", "Process"),
        (50, 69, "Step 2: Column Reduction\nSubtract minimum element of each column from all elements in that column", "#FFFFFF", "#334155", "Process"),
        (50, 58, "Step 3: Zero Covering Test\nDraw minimum number of horizontal and vertical lines (k) to cover all zeros", "#EFF6FF", "#1D4ED8", "Process"),
        (50, 45, "Optimal Matching Check\nIs total covering lines k == Matrix Dimension N ?", "#FDF4FF", "#A855F7", "Decision"),
        (50, 31, "Step 4: Shift Matrix Elements\nFind smallest uncovered element delta;\nSubtract delta from uncovered, add delta to doubly covered elements", "#FEF2F2", "#DC2626", "Process"),
        (50, 18, "Step 5: Extract Zero Assignments\nSelect independent zeros such that each row & column has exactly one match", "#F0FDFA", "#0D9488", "Process"),
        (50, 7, "Output: Optimal 1-to-1 Cloud Associations\nCompute displacement vectors, speeds (km/h) & compass bearings", "#DCFCE7", "#059669", "Terminal")
    ]

    for x, y, title, bg, border, ntype in steps:
        w, h = 66, 8
        rounding = 0.25 if ntype == "Terminal" else (0.15 if ntype == "Decision" else 0.04)
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=BoxStyle("Round", pad=0.01, rounding_size=rounding),
                             facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A", zorder=3, family=FONT_FAMILY)

    # Arrows
    draw_arrow(ax, 50, 86, 50, 84)
    draw_arrow(ax, 50, 76, 50, 73)
    draw_arrow(ax, 50, 65, 50, 62)
    draw_arrow(ax, 50, 54, 50, 49)
    
    # Branching from Step 3 Check
    draw_arrow(ax, 50, 41, 50, 35, "No (k < N)")
    # Loop back from Step 4 to Step 3
    ax.annotate("", xy=(83, 58), xytext=(83, 31),
                arrowprops=dict(arrowstyle="-", color=COLOR_CORAL, lw=1.5, shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate("", xy=(50 + 33, 58), xytext=(83, 58),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_CORAL, lw=1.5, mutation_scale=14, shrinkA=0, shrinkB=0), zorder=4)
    ax.text(85, 45, "Repeat with\nnew zeros", ha="left", va="center", fontsize=7.5, color=COLOR_CORAL, family=FONT_FAMILY)

    # Yes branch directly to Step 5
    # Curve around left
    ax.annotate("", xy=(17, 45), xytext=(17, 18),
                arrowprops=dict(arrowstyle="-", color=COLOR_EMERALD, lw=1.5, shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate("", xy=(17, 45), xytext=(50 - 33, 45),
                arrowprops=dict(arrowstyle="-", color=COLOR_EMERALD, lw=1.5, shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate("", xy=(50 - 33, 18), xytext=(17, 18),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_EMERALD, lw=1.5, mutation_scale=14, shrinkA=0, shrinkB=0), zorder=4)
    ax.text(15, 31, "Yes (k == N)\nOptimal", ha="right", va="center", fontsize=7.5, color=COLOR_EMERALD, family=FONT_FAMILY)

    draw_arrow(ax, 50, 14, 50, 11)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig7_hungarian_algorithm.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 7: {fig_path}")


# ==============================================================================
# Diagram 8: UML Use Case Diagram
# ==============================================================================
def draw_figure_8():
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "CloudPath AI – UML Use Case Diagram", ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)

    # System Boundary Box
    sys_box = FancyBboxPatch((28, 8), 58, 82, boxstyle=BoxStyle("Round", pad=0.01, rounding_size=0.02),
                             facecolor="#F8FAFC", edgecolor=COLOR_NAVY, linewidth=2, zorder=1)
    ax.add_patch(sys_box)
    ax.text(31, 87, "CloudPath AI Platform Boundary", ha="left", va="center", fontsize=11, fontweight="bold", color=COLOR_NAVY, zorder=3, family=FONT_FAMILY)

    # Actor Function
    def draw_actor(x, y, name):
        # Head
        circle = patches.Circle((x, y), 2.2, facecolor="#FFFFFF", edgecolor=COLOR_NAVY, lw=1.5, zorder=4)
        ax.add_patch(circle)
        # Body
        ax.plot([x, x], [y - 2.2, y - 7], color=COLOR_NAVY, lw=1.5, zorder=4)
        # Arms
        ax.plot([x - 3.5, x + 3.5], [y - 4, y - 4], color=COLOR_NAVY, lw=1.5, zorder=4)
        # Legs
        ax.plot([x, x - 3], [y - 7, y - 11], color=COLOR_NAVY, lw=1.5, zorder=4)
        ax.plot([x, x + 3], [y - 7, y - 11], color=COLOR_NAVY, lw=1.5, zorder=4)
        ax.text(x, y - 13.5, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=COLOR_NAVY, zorder=4, family=FONT_FAMILY)

    draw_actor(14, 75, "Meteorologist /\nOperational Forecaster")
    draw_actor(14, 30, "Automated Ingestion\nDaemon Service")
    draw_actor(93, 50, "System Admin /\nResearcher")

    # Use Cases (Ellipses)
    use_cases = [
        (57, 82, "UC1: Inspect INSAT-3DS Satellite\nObservation Timeline (1,448 frames)"),
        (57, 72, "UC2: Visualize Dynamic Cloud Formations\n& Vector Displacement Overlays"),
        (57, 62, "UC3: Execute 25-Feature XGBoost\nRainfall Classification & Regression"),
        (57, 52, "UC4: Monitor Real-Time Ground Weather\nConditions & Synchronized Status"),
        (57, 42, "UC5: Extract Regional Bounding Subgrid\nfrom INSAT-3DS NetCDF-4 Files"),
        (57, 32, "UC6: Run Hungarian Assignment Tracking\nAcross Consecutive Observations"),
        (57, 22, "UC7: Benchmark Model Metrics\n(Accuracy, ROC-AUC, RMSE, R² Score)"),
        (57, 12, "UC8: Audit Multi-Source Temporal\nAlignment & System Health")
    ]

    for x, y, text in use_cases:
        ellipse = FancyBboxPatch((x - 22, y - 3.5), 44, 7, boxstyle=BoxStyle("Round", pad=0.01, rounding_size=0.35),
                                 facecolor="#FFFFFF", edgecolor=COLOR_CYAN, linewidth=1.4, zorder=2)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="#0F172A", zorder=3, family=FONT_FAMILY)

    # Actor Connections (Lines)
    # Forecaster -> UC1, UC2, UC3, UC4, UC7
    for u_idx in [0, 1, 2, 3, 6]:
        uy = use_cases[u_idx][1]
        ax.plot([18, 35], [75, uy], color="#334155", lw=1.2, linestyle="-", zorder=2)

    # Daemon Service -> UC5, UC6, UC8
    for u_idx in [4, 5, 7]:
        uy = use_cases[u_idx][1]
        ax.plot([18, 35], [30, uy], color="#0D9488", lw=1.2, linestyle="-", zorder=2)

    # Admin -> UC3, UC7, UC8
    for u_idx in [2, 6, 7]:
        uy = use_cases[u_idx][1]
        ax.plot([89, 79], [50, uy], color="#DC2626", lw=1.2, linestyle="-", zorder=2)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig8_uml_usecase.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 8: {fig_path}")


# ==============================================================================
# Diagram 9: UML Activity Diagram
# ==============================================================================
def draw_figure_9():
    fig, ax = plt.subplots(figsize=(9, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97, "CloudPath AI – UML Activity Diagram", ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)
    ax.text(50, 95, "End-to-End Operational Lifecycle: Ingestion, Tracking & Rainfall Inference", ha="center", va="center", fontsize=8.5, color="#64748B", family=FONT_FAMILY)

    # Initial Node (Filled Circle)
    init_circle = patches.Circle((50, 90), 2.2, facecolor=COLOR_NAVY, edgecolor=COLOR_NAVY, zorder=3)
    ax.add_patch(init_circle)
    ax.text(50, 86.5, "Start", ha="center", va="center", fontsize=8, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)

    # Action 1: Ingest
    create_box(ax, 25, 78, 50, 7, "Ingest INSAT-3DS NetCDF-4 Granule (July 2025)", box_color="#FFFFFF", border_color=COLOR_NAVY, title_size=8.5)
    draw_arrow(ax, 50, 87.8, 50, 85)

    # Fork Bar (Parallel processing start)
    ax.plot([15, 85], [74, 74], color=COLOR_NAVY, lw=4, zorder=4)
    draw_arrow(ax, 50, 78, 50, 74)

    # Branch A: Satellite Path (Left)
    ax.text(25, 72.5, "Satellite Processing Branch", ha="center", va="center", fontsize=8, fontweight="bold", color=COLOR_TEAL, family=FONT_FAMILY)
    create_box(ax, 8, 62, 36, 7, "Extract 162x159 px Bengaluru Subgrid", box_color="#F0FDFA", border_color="#0D9488", title_size=8)
    create_box(ax, 8, 50, 36, 7, "Segment Clouds (cv2 Connected Comp)", box_color="#F0FDFA", border_color="#0D9488", title_size=8)
    create_box(ax, 8, 38, 36, 7, "Execute Hungarian Assignment (d<=120km)", box_color="#F0FDFA", border_color="#0D9488", title_size=8)
    create_box(ax, 8, 26, 36, 7, "Compute Velocity Vectors, Speed & Direction", box_color="#F0FDFA", border_color="#0D9488", title_size=8)

    draw_arrow(ax, 26, 74, 26, 69)
    draw_arrow(ax, 26, 62, 26, 57)
    draw_arrow(ax, 26, 50, 26, 45)
    draw_arrow(ax, 26, 38, 26, 33)

    # Branch B: Ground Weather ML Path (Right)
    ax.text(75, 72.5, "Meteorological ML Branch", ha="center", va="center", fontsize=8, fontweight="bold", color=COLOR_SKY, family=FONT_FAMILY)
    create_box(ax, 56, 62, 36, 7, "Retrieve Synchronized Weather Observation", box_color="#EFF6FF", border_color="#3B82F6", title_size=8)
    create_box(ax, 56, 50, 36, 7, "Assemble 25-Feature Meteorological Vector", box_color="#EFF6FF", border_color="#3B82F6", title_size=8)
    create_box(ax, 56, 38, 36, 7, "Execute XGBoost Classifier (P(Rain))", box_color="#EFF6FF", border_color="#3B82F6", title_size=8)
    create_box(ax, 56, 26, 36, 7, "Execute XGBoost Regressor (Rainfall mm)", box_color="#EFF6FF", border_color="#3B82F6", title_size=8)

    draw_arrow(ax, 74, 74, 74, 69)
    draw_arrow(ax, 74, 62, 74, 57)
    draw_arrow(ax, 74, 50, 74, 45)
    draw_arrow(ax, 74, 38, 74, 33)

    # Join Bar (Synchronization)
    ax.plot([15, 85], [21, 21], color=COLOR_NAVY, lw=4, zorder=4)
    draw_arrow(ax, 26, 26, 26, 21)
    draw_arrow(ax, 74, 26, 74, 21)

    # Unified Display
    create_box(ax, 25, 10, 50, 7, "Render Unified Interactive Web Dashboard", box_color="#DCFCE7", border_color="#059669", title_size=8.5)
    draw_arrow(ax, 50, 21, 50, 17)

    # Final Node (Bullseye Circle)
    fin_outer = patches.Circle((50, 4), 2.5, facecolor="#FFFFFF", edgecolor=COLOR_NAVY, lw=1.5, zorder=3)
    fin_inner = patches.Circle((50, 4), 1.6, facecolor=COLOR_NAVY, edgecolor=COLOR_NAVY, zorder=4)
    ax.add_patch(fin_outer)
    ax.add_patch(fin_inner)
    draw_arrow(ax, 50, 10, 50, 6.5)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig9_uml_activity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 9: {fig_path}")


# ==============================================================================
# Diagram 10: UML Sequence Diagram
# ==============================================================================
def draw_figure_10():
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "CloudPath AI – UML Sequence Diagram", ha="center", va="center", fontsize=14, fontweight="bold", color=COLOR_NAVY, family=FONT_FAMILY)
    ax.text(50, 93, "Timeline Navigation, Satellite Processing & ML Inference Cycle", ha="center", va="center", fontsize=8.5, color="#64748B", family=FONT_FAMILY)

    # Lifeline Lifeline Objects (Boxes at top)
    lifelines = [
        (10, "User Browser\n(Client Web UI)"),
        (30, "HTTP Server\n(dashboard_app.py)"),
        (50, "Satellite Feature\nEngine (Phase 3/4)"),
        (70, "ML Prediction\nEngine (XGBoost)"),
        (90, "Model & NPZ\nData Store")
    ]

    for lx, title in lifelines:
        create_box(ax, lx - 8, 81, 16, 9, title, box_color="#F8FAFC", border_color=COLOR_NAVY, title_size=8, sub_size=7)
        # Lifeline vertical dashed line
        ax.plot([lx, lx], [81, 8], color="#94A3B8", linestyle="--", lw=1.2, zorder=1)

    # Sequence Messages
    msgs = [
        (10, 30, 75, "1. onSlideObservation(frame_idx = 1046)", "#1E293B", True),
        (30, 50, 68, "2. getObservationContext(idx)", "#0284C7", True),
        (50, 90, 61, "3. loadCloudmaskAndVectors(idx)", "#0D9488", True),
        (90, 50, 54, "4. return NPZ array & motion track rows", "#0D9488", False),
        (50, 30, 47, "5. return coverage %, speed, dominant dir", "#0284C7", False),
        (30, 70, 40, "6. executePrediction(weather_input)", "#1D4ED8", True),
        (70, 90, 34, "7. loadClassifier & Regressor artifacts", "#475569", True),
        (90, 70, 28, "8. return trained pipeline objects", "#475569", False),
        (70, 30, 22, "9. return P(Rain)=100%, amount=1.18 mm", "#DC2626", False),
        (30, 10, 15, "10. HTTP 200 JSON Response (All Metrics)", "#059669", False)
    ]

    for x1, x2, y, label, col, is_sync in msgs:
        # Activation Box on target
        act_box = patches.Rectangle((x2 - 1, y - 2), 2, 4, facecolor="#E2E8F0", edgecolor=col, lw=1, zorder=2)
        ax.add_patch(act_box)
        
        # Message Arrow
        style = "-|>" if is_sync else "--|>"
        ls = "-" if is_sync else "--"
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, linestyle=ls, mutation_scale=12, shrinkA=0, shrinkB=0), zorder=3)
        ax.text((x1 + x2) / 2, y + 1.6, label, ha="center", va="center", fontsize=7.5, fontweight="bold" if is_sync else "normal", color=col,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor="none", alpha=0.9), zorder=4, family=FONT_FAMILY)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fig10_uml_sequence.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Generated Fig 10: {fig_path}")


def main():
    print("=" * 70)
    print("GENERATING CLOUDPATH AI TECHNICAL DIAGRAMS")
    print("=" * 70)
    draw_figure_1()
    draw_figure_2()
    draw_figure_3()
    draw_figure_4()
    draw_figure_5()
    draw_figure_6()
    draw_figure_7()
    draw_figure_8()
    draw_figure_9()
    draw_figure_10()
    print("=" * 70)
    print("ALL 10 TECHNICAL DIAGRAMS GENERATED SUCCESSFULLY IN:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()


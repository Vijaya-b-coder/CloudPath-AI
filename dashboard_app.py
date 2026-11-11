"""
dashboard_app.py
================
CloudPath AI – Phase 6: Interactive Web Dashboard
A modern, zero-dependency web application powered by Python's built-in http.server.

Features:
  - Responsive Single-Page Application (SPA) with Dark Mode UI
  - Real-time Satellite Observation Explorer (1,448 observations, July 1–31, 2025)
  - Interactive Cloud Tracking Visualizer (frame-to-frame movement arrows & trajectories)
  - Interactive Rainfall Prediction Console (live XGBoost inference with preset scenarios)
  - Model Performance Benchmark Matrices (Classification & Regression)
  - Academic Integrity Assurance (clear separation of historical ground weather and satellite data)

Usage:
  python dashboard_app.py [--port 8050] [--host 0.0.0.0]

Author: CloudPath AI Project
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prediction_engine import (
    RainfallPredictor, WeatherInput, SatelliteContext,
    REQUIRED_FEATURES, weather_input_from_row, satellite_context_from_row
)
from integration_pipeline import (
    load_weather_data, load_satellite_summary, load_movement_data,
    check_temporal_overlap, build_satellite_feature_table
)

# ── Load Core Data into Memory at Startup ────────────────────────────
print("[Dashboard] Initializing CloudPath AI Engine...", flush=True)
predictor = RainfallPredictor()
weather_df = load_weather_data()
satellite_df = load_satellite_summary()
movement_df = load_movement_data()
overlap_audit = check_temporal_overlap(weather_df, satellite_df)
satellite_features = build_satellite_feature_table(satellite_df, movement_df)
eval_df = pd.read_csv(os.path.join(PROJECT_ROOT, "model_evaluation.csv"))

# ── Load July 2025 Satellite NPZ Series & Motion Index ───────────────
NPZ_PATH = os.path.join(PROJECT_ROOT, "satellite_data_outputs", "bengaluru_cloudmask_series.npz")
_npz_data = np.load(NPZ_PATH, allow_pickle=True)
CLOUD_MASKS = _npz_data["cloud_masks"]  # (1448, 162, 159)
TIMESTAMPS = _npz_data["timestamps"]

_movement_by_frame = {}
for _, m_row in movement_df.iterrows():
    f_idx = int(m_row["curr_frame_idx"])
    if f_idx not in _movement_by_frame:
        _movement_by_frame[f_idx] = []
    _movement_by_frame[f_idx].append(m_row)

_render_cache = {}

def render_cloudmask_png(frame_idx: int) -> bytes:
    if frame_idx in _render_cache:
        return _render_cache[frame_idx]

    frame_idx = max(0, min(frame_idx, len(CLOUD_MASKS) - 1))
    mask = CLOUD_MASKS[frame_idx]
    h, w = mask.shape
    scale = 3

    # BGR array for OpenCV
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 0] = [235, 206, 135]      # Sky blue (BGR)
    rgb[mask == 1] = [255, 255, 255]      # Cloud white
    rgb[np.isnan(mask)] = [32, 17, 11]    # Midnight Navy #0B1120 (BGR)

    img = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Draw movement vectors and centroids using theme colors:
    # Teal (#2DD4BF -> BGR: 191, 212, 45) for tracking vectors/arrows
    # Electric Cyan (#38BDF8 -> BGR: 248, 189, 56) for detected centroids
    arrows = _movement_by_frame.get(frame_idx, [])
    for m in arrows[:45]:
        pr, pc = int(m["prev_centroid_r"] * scale), int(m["prev_centroid_c"] * scale)
        cr, cc = int(m["curr_centroid_r"] * scale), int(m["curr_centroid_c"] * scale)
        cv2.circle(img, (cc, cr), 4, (248, 189, 56), -1)  # Electric Cyan centroid
        if (pr, pc) != (cr, cc):
            cv2.arrowedLine(img, (pc, pr), (cc, cr), (191, 212, 45), 2, tipLength=0.28)  # Teal movement arrow

    # Info banner at bottom in Midnight Navy (#0B1120)
    ts_str = str(TIMESTAMPS[frame_idx]) + " UTC"
    obs_info = OBSERVATIONS_JSON[frame_idx]
    cov_str = f"Obs {frame_idx+1}/{len(TIMESTAMPS)} | Cloud: {obs_info['cloud_coverage_pct']:.1f}%"
    cv2.rectangle(img, (0, h * scale - 26), (w * scale, h * scale), (32, 17, 11), -1)
    cv2.putText(img, ts_str, (8, h * scale - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (252, 250, 248), 1, cv2.LINE_AA)  # White #F8FAFC
    cv2.putText(img, cov_str, (w * scale - 215, h * scale - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (248, 189, 56), 1, cv2.LINE_AA)  # Electric Cyan #38BDF8

    _, buf = cv2.imencode(".png", img)
    png_bytes = buf.tobytes()
    if len(_render_cache) < 300:
        _render_cache[frame_idx] = png_bytes
    return png_bytes

print("[Dashboard] Data and ML Models Loaded Successfully.", flush=True)

# Pre-compute satellite observations list merged with July 2025 weather & predictions
weather_july_path = os.path.join(PROJECT_ROOT, "bengaluru_weather_july2025.csv")
if os.path.exists(weather_july_path):
    print(f"[Dashboard] Loading verified July 2025 ground weather dataset: {weather_july_path}", flush=True)
    df_july_wx = pd.read_csv(weather_july_path)
    df_july_wx["dt"] = pd.to_datetime(df_july_wx["timestamp"])
    df_july_wx = df_july_wx.sort_values("dt").reset_index(drop=True)
else:
    print("[Dashboard] WARNING: bengaluru_weather_july2025.csv not found!", flush=True)
    df_july_wx = None

sat_temp = satellite_features.copy()
sat_temp["dt"] = pd.to_datetime(sat_temp["timestamp"])
sat_temp = sat_temp.sort_values("dt")

if df_july_wx is not None:
    merged_sat_wx = pd.merge_asof(sat_temp, df_july_wx, on="dt", direction="nearest", suffixes=("", "_wx"))
    merged_sat_wx["delta_min"] = ((merged_sat_wx["dt"] - pd.to_datetime(merged_sat_wx["timestamp_wx"])).abs().dt.total_seconds() / 60).astype(int)
else:
    merged_sat_wx = sat_temp
    merged_sat_wx["delta_min"] = 0

OBSERVATIONS_JSON = []
for idx, row in merged_sat_wx.iterrows():
    obs_dict = {
        "index": int(idx),
        "timestamp": str(row["timestamp"]),
        "cloud_coverage_pct": round(float(row["cloud_coverage_pct"]), 2),
        "clear_coverage_pct": round(float(row["clear_coverage_pct"]), 2),
        "n_regions": int(row["n_regions"]),
        "delta_cloud_pct": round(float(row.get("delta_cloud_pct", 0.0)), 2),
        "mean_speed_kmh": round(float(row["mean_speed_kmh"]), 1) if pd.notna(row.get("mean_speed_kmh")) else None,
        "dominant_dir": str(row.get("dominant_dir", "N/A")),
        "mean_conf_score": round(float(row["mean_conf_score"]), 2) if pd.notna(row.get("mean_conf_score")) else None,
        # Synchronized Ground Weather Parameters
        "temperature_c": round(float(row.get("temperature_c", 24.0)), 1),
        "feels_like_c": round(float(row.get("feels_like_c", 24.0)), 1),
        "humidity_pct": round(float(row.get("humidity_pct", 75.0)), 1),
        "cloud_cover_pct": round(float(row.get("cloud_cover_pct", 80.0)), 1),
        "precipitation_mm": round(float(row.get("precipitation_mm", 0.0)), 2),
        "pressure_hpa": round(float(row.get("pressure_hpa", 1010.0)), 1),
        "wind_speed_kmh": round(float(row.get("wind_speed_kmh", 15.0)), 1),
        "wind_direction_deg": round(float(row.get("wind_direction_deg", 250.0)), 0),
        "weather_condition": str(row.get("weather_condition", "Mostly Cloudy")),
        "weather_icon": str(row.get("weather_icon", "☁️")),
        "weather_code": int(row.get("weather_code", 3)),
        # Dynamic ML Predictions
        "rain_prediction": int(row.get("rain_prediction", 0)),
        "rain_label": str(row.get("rain_label", "No Rain")),
        "rain_probability": round(float(row.get("rain_probability", 0.0)), 4) if pd.notna(row.get("rain_probability")) else None,
        "no_rain_probability": round(float(row.get("no_rain_probability", 1.0)), 4) if pd.notna(row.get("no_rain_probability")) else None,
        "rainfall_amount_mm": round(float(row.get("rainfall_amount_mm", 0.0)), 3),
        "weather_timestamp": str(row.get("timestamp_wx", row["timestamp"])),
        "time_delta_min": int(row.get("delta_min", 0)),
    }
    OBSERVATIONS_JSON.append(obs_dict)
print(f"[Dashboard] Pre-computed {len(OBSERVATIONS_JSON)} synchronized satellite + weather + ML frames.", flush=True)


# ── Embedded HTML / CSS / JS Single Page App ─────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CloudPath AI – Cloud Tracking & Rainfall Prediction</title>
  <style>
    :root {
      /* Base Theme Palette - Midnight Navy + Electric Cyan + Teal */
      --bg-midnight: #0B1120;       /* Midnight Navy */
      --card-main: #172554;         /* Deep Slate Blue */
      --card-sec: #1E293B;          /* Dark Blue */
      --border-card: rgba(56, 189, 248, 0.16);
      --border-subtle: #24355A;
      
      /* Typography */
      --text-heading: #F8FAFC;      /* White */
      --text-body: #E2E8F0;
      --text-muted: #94A3B8;        /* Cool Grey */
      
      /* Feature Accents */
      --accent-cyan: #38BDF8;       /* Electric Cyan - Cloud Coverage */
      --accent-teal: #2DD4BF;       /* Teal - Cloud Tracking & Motion */
      --accent-sky: #60A5FA;        /* Sky Blue - Rain Prediction */
      --accent-coral: #F87171;      /* Coral Red - Rain Alert */
      --accent-emerald: #34D399;    /* Emerald Green - No-rain status */
      --accent-indigo: #818CF8;
      --accent-amber: #FBBF24;
      
      /* Shadows & Glassmorphism */
      --shadow-card: 0 4px 20px -2px rgba(3, 7, 18, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.05);
      --shadow-hover: 0 8px 28px -4px rgba(56, 189, 248, 0.22);
      --glow-cyan: 0 0 16px rgba(56, 189, 248, 0.35);
      --glow-teal: 0 0 16px rgba(45, 212, 191, 0.35);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
    
    body {
      background: radial-gradient(ellipse at 50% 0%, #172554 0%, #0B1120 70%);
      background-color: var(--bg-midnight);
      background-attachment: fixed;
      color: var(--text-body);
      min-height: 100vh;
      line-height: 1.5;
    }
    
    /* Header */
    header {
      background: rgba(11, 17, 32, 0.95);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-card);
      padding: 1.15rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 2px 16px rgba(3, 7, 18, 0.6);
    }
    .logo-area { display: flex; align-items: center; gap: 0.85rem; }
    .logo-icon {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
      border-radius: 10px;
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 1.3rem;
      color: var(--bg-midnight);
      box-shadow: var(--glow-cyan);
    }
    .header-title h1 {
      font-size: 1.4rem;
      font-weight: 800;
      color: var(--text-heading);
      letter-spacing: -0.5px;
    }
    .header-title p {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 0.1rem;
    }
    .system-badges { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
    .badge {
      font-size: 0.72rem;
      padding: 0.28rem 0.75rem;
      border-radius: 999px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      letter-spacing: 0.3px;
    }
    .badge-live {
      background: rgba(52, 211, 153, 0.12);
      color: var(--accent-emerald);
      border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-sat {
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent-cyan);
      border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .badge-ml {
      background: rgba(96, 165, 250, 0.12);
      color: var(--accent-sky);
      border: 1px solid rgba(96, 165, 250, 0.35);
    }
    .badge-teal {
      background: rgba(45, 212, 191, 0.12);
      color: var(--accent-teal);
      border: 1px solid rgba(45, 212, 191, 0.35);
    }

    /* Navigation Tabs */
    nav {
      background: #0F172A;
      padding: 0.5rem 2rem;
      border-bottom: 1px solid var(--border-card);
      display: flex;
      gap: 0.7rem;
      overflow-x: auto;
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.65rem 1.15rem;
      border-radius: 8px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.22s ease-in-out;
      white-space: nowrap;
    }
    .tab-btn:hover {
      color: var(--text-heading);
      background: rgba(56, 189, 248, 0.08);
    }
    .tab-btn.active {
      color: var(--accent-cyan);
      background: var(--card-main);
      border-bottom: 2px solid var(--accent-cyan);
      box-shadow: 0 -2px 10px rgba(56, 189, 248, 0.15);
    }

    /* Main Container */
    main { padding: 1.8rem 2rem; max-width: 1400px; margin: 0 auto; }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.25s ease-in-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* Academic Integrity Notice Banner */
    .integrity-banner {
      background: linear-gradient(135deg, #1E293B 0%, #152238 100%);
      border: 1px solid var(--border-card);
      border-left: 4px solid var(--accent-teal);
      padding: 0.95rem 1.25rem;
      border-radius: 10px;
      margin-bottom: 1.6rem;
      display: flex;
      gap: 0.9rem;
      font-size: 0.82rem;
      box-shadow: var(--shadow-card);
    }
    .integrity-banner strong { color: var(--accent-teal); }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.1rem;
      margin-bottom: 1.8rem;
    }
    .kpi-card {
      background: linear-gradient(145deg, #172554 0%, #101c3d 100%);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 1.25rem;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-card);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-hover);
    }
    .kpi-card::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--accent-cyan);
    }
    .kpi-card.regions-card::before { background: var(--accent-sky); }
    .kpi-card.speed-card::before { background: var(--accent-teal); }
    .kpi-card.rain-card::before { background: var(--kpi-rain-bar, var(--accent-coral)); }
    .kpi-title {
      font-size: 0.72rem;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 700;
      letter-spacing: 0.6px;
    }
    .kpi-val {
      font-size: 1.95rem;
      font-weight: 800;
      margin: 0.35rem 0;
      color: var(--text-heading);
      letter-spacing: -0.5px;
    }
    .kpi-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      gap: 0.4rem;
      align-items: center;
    }

    /* Layout Grids */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
    
    .card {
      background: linear-gradient(145deg, #172554 0%, #101c3d 100%);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 1.4rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--shadow-card);
      transition: border-color 0.2s ease;
    }
    .card-title {
      font-size: 1.08rem;
      font-weight: 700;
      margin-bottom: 1rem;
      color: var(--text-heading);
      display: flex;
      justify-content: space-between;
      align-items: center;
      letter-spacing: -0.2px;
    }

    /* Visual Media & Scrubber */
    .media-container {
      text-align: center;
      background: var(--card-sec);
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border-card);
      padding: 0.6rem;
    }
    .media-container img {
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      display: inline-block;
      border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .slider-box {
      margin-top: 1rem;
      background: var(--card-sec);
      padding: 1.1rem;
      border-radius: 10px;
      border: 1px solid var(--border-card);
    }
    .slider-box label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }
    .slider-box input[type=range] {
      width: 100%;
      margin: 0.6rem 0;
      cursor: pointer;
      accent-color: var(--accent-cyan);
    }

    /* Weather Conditions Card */
    .weather-hero {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1.15rem 1.3rem;
      background: linear-gradient(135deg, #1E293B 0%, #131E35 100%);
      border-radius: 10px;
      border: 1px solid rgba(45, 212, 191, 0.25);
      margin-bottom: 0.95rem;
    }
    .weather-hero-main { display: flex; align-items: center; gap: 1.1rem; }
    .weather-hero-icon {
      font-size: 2.9rem;
      line-height: 1;
      filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.35));
    }
    .weather-hero-title {
      font-size: 1.4rem;
      font-weight: 800;
      color: var(--text-heading);
      line-height: 1.2;
    }
    .weather-hero-desc {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }
    .weather-metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.75rem;
      margin-bottom: 0.8rem;
    }
    .weather-metric-box {
      background: var(--card-sec);
      border: 1px solid var(--border-card);
      border-radius: 10px;
      padding: 0.75rem 0.95rem;
      transition: border-color 0.2s ease;
    }
    .weather-metric-box:hover {
      border-color: rgba(56, 189, 248, 0.35);
    }
    .weather-metric-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      letter-spacing: 0.5px;
    }
    .weather-metric-val {
      font-size: 1.3rem;
      font-weight: 800;
      color: var(--text-heading);
      margin-top: 0.25rem;
    }

    /* Prediction Form & Interactive Controls */
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 0.9rem;
      margin-bottom: 1.2rem;
    }
    .form-group label {
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      display: block;
      margin-bottom: 0.35rem;
      letter-spacing: 0.5px;
    }
    .form-group input, .form-group select {
      width: 100%;
      background: var(--card-sec);
      border: 1px solid #2B3D6B;
      color: var(--text-heading);
      padding: 0.6rem 0.8rem;
      border-radius: 8px;
      font-size: 0.85rem;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .form-group input:focus, .form-group select:focus {
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
      outline: none;
    }
    .preset-row {
      display: flex;
      gap: 0.55rem;
      flex-wrap: wrap;
      margin-bottom: 1.2rem;
      align-items: center;
    }
    .preset-btn {
      background: var(--card-sec);
      border: 1px solid #2B3D6B;
      color: var(--text-body);
      font-size: 0.75rem;
      padding: 0.4rem 0.8rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s ease;
    }
    .preset-btn:hover {
      background: #172554;
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }
    .predict-btn {
      background: linear-gradient(135deg, #0284C7 0%, #0D9488 100%);
      border: none;
      color: var(--text-heading);
      padding: 0.85rem 1.6rem;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s ease;
      display: block;
      width: 100%;
      box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35);
    }
    .predict-btn:hover {
      opacity: 0.94;
      transform: translateY(-1px);
      box-shadow: 0 6px 22px rgba(45, 212, 191, 0.4);
    }

    /* Prediction Result Card */
    .result-card {
      background: var(--card-sec);
      border: 1px solid var(--border-card);
      border-radius: 10px;
      padding: 1.25rem;
      margin-top: 1rem;
    }
    .pred-pill {
      display: inline-block;
      font-size: 1.15rem;
      font-weight: 800;
      padding: 0.45rem 1.2rem;
      border-radius: 8px;
      margin-bottom: 0.85rem;
      letter-spacing: 0.5px;
    }
    .pred-rain {
      background: rgba(248, 113, 113, 0.16);
      color: var(--accent-coral);
      border: 1.5px solid var(--accent-coral);
      box-shadow: 0 0 16px rgba(248, 113, 113, 0.25);
    }
    .pred-norain {
      background: rgba(52, 211, 153, 0.16);
      color: var(--accent-emerald);
      border: 1.5px solid var(--accent-emerald);
      box-shadow: 0 0 16px rgba(52, 211, 153, 0.25);
    }
    .prob-bar-bg {
      width: 100%;
      height: 12px;
      background: var(--bg-midnight);
      border: 1px solid #2B3D6B;
      border-radius: 6px;
      overflow: hidden;
      margin: 0.5rem 0 0.85rem 0;
    }
    .prob-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-sky) 0%, var(--accent-cyan) 60%, var(--accent-coral) 100%);
      transition: width 0.4s ease;
    }

    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.5rem; }
    th, td { padding: 0.7rem 0.9rem; text-align: left; border-bottom: 1px solid var(--card-sec); }
    th {
      background: var(--card-sec);
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.72rem;
      letter-spacing: 0.5px;
    }
    tr:hover { background: rgba(56, 189, 248, 0.05); }

    /* Responsive SVG Timeline */
    .chart-container {
      background: var(--card-sec);
      border-radius: 10px;
      border: 1px solid var(--border-card);
      padding: 1rem;
      overflow-x: auto;
    }
    svg { display: block; width: 100%; height: auto; }
  </style>
</head>
<body>

  <header>
    <div class="logo-area">
      <div class="logo-icon">&#9729;</div>
      <div class="header-title">
        <h1>CloudPath AI</h1>
        <p>AI-Based Cloud Movement Tracking and Rainfall Prediction System</p>
      </div>
    </div>
    <div class="system-badges">
      <span class="badge badge-live">&#9679; Engine Active</span>
      <span class="badge badge-sat">INSAT-3DS Ver02 (4km)</span>
      <span class="badge badge-ml">XGBoost ML Pipeline</span>
    </div>
  </header>

  <nav>
    <button class="tab-btn active" onclick="switchTab('dashboard')">Operational Dashboard</button>
    <button class="tab-btn" onclick="switchTab('prediction')">Interactive Prediction Console</button>
    <button class="tab-btn" onclick="switchTab('satellite')">Satellite Cloud Dynamics</button>
    <button class="tab-btn" onclick="switchTab('models')">Model Benchmark & Evaluation</button>
    <button class="tab-btn" onclick="switchTab('architecture')">Architecture & Documentation</button>
  </nav>

  <main>

    <!-- Integrity Notice Banner -->
    <div class="integrity-banner">
      <div style="font-size: 1.3rem;">&#9888;</div>
      <div>
        <strong>Academic & Meteorological Integrity Notice:</strong>
        Ground-station weather models were trained and validated on historical hourly records from <strong>2009–2020</strong>. Geostationary satellite cloud movement tracking was analyzed from <strong>INSAT-3DS (July 1–31, 2025)</strong>. Because these datasets do not temporally overlap, satellite features operate in parallel as live situational context rather than fabricated multimodal inputs.
      </div>
    </div>

    <!-- TAB 1: OPERATIONAL DASHBOARD -->
    <section id="tab-dashboard" class="tab-content active">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-title">Satellite Cloud Coverage</div>
          <div class="kpi-val" id="kpi-coverage" style="color:var(--accent-cyan);">--%</div>
          <div class="kpi-sub" id="kpi-coverage-sub">Regional Domain: 162 x 159 px</div>
        </div>
        <div class="kpi-card regions-card">
          <div class="kpi-title">Active Cloud Formations</div>
          <div class="kpi-val" id="kpi-regions" style="color:var(--accent-sky);">--</div>
          <div class="kpi-sub">Connected Components (8-way)</div>
        </div>
        <div class="kpi-card speed-card">
          <div class="kpi-title">Cloud Movement Speed</div>
          <div class="kpi-val" id="kpi-speed" style="color:var(--accent-teal);">-- km/h</div>
          <div class="kpi-sub" id="kpi-speed-dir">Direction: --</div>
        </div>
        <div class="kpi-card rain-card">
          <div class="kpi-title">Rainfall Forecast Status</div>
          <div class="kpi-val" id="kpi-rain-status">--</div>
          <div class="kpi-sub" id="kpi-rain-amount">Predicted: -- mm in next hr</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>INSAT-3DS Satellite Cloud Tracking</span>
            <span style="font-size:0.75rem; color:var(--accent-cyan); font-weight:600;" id="obs-ts-label">Observation: --</span>
          </div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.6rem;">
            Bengaluru Regional Domain (162 &times; 159 px, ~650 &times; 650 km) | Active Month: <strong style="color:var(--accent-cyan);">July 2025</strong>
          </div>
          <div class="media-container" style="padding:0.6rem; text-align:center;">
            <img id="dashboard-sat-img" src="/api/cloudmask-render/0" alt="INSAT-3DS Cloud Motion Vectors" style="max-height: 420px; width: auto; max-width: 100%; border-radius: 6px;">
          </div>
          <div class="slider-box" style="margin-top:1rem;">
            <label style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
              <span id="slider-range-label" style="font-weight:700; color:var(--text-heading);">Observation Timeline (1 to 1,448) &mdash; July 1&ndash;31, 2025</span>
              <span id="slider-val-disp" style="color:var(--accent-cyan); font-weight:700;">Index: 0</span>
            </label>
            <input type="range" id="obs-slider" min="0" max="1447" value="0" oninput="onSlideObservation(this.value)">
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
              <span>01 Jul 2025 00:30 UTC</span>
              <span style="color:var(--accent-teal);">16 Jul 2025 12:00 UTC</span>
              <span>31 Jul 2025 23:30 UTC</span>
            </div>
            <div style="display:flex; gap:0.5rem; margin-top:0.8rem; justify-content:center;">
              <button class="preset-btn" onclick="stepObservation(-observations.length)">&#9198; First (01 Jul)</button>
              <button class="preset-btn" onclick="stepObservation(-2)">&#9664; -1 Hour</button>
              <button class="preset-btn" onclick="stepObservation(2)">&#9654; +1 Hour</button>
              <button class="preset-btn" onclick="stepObservation(observations.length)">&#9197; Latest (31 Jul)</button>
            </div>
          </div>
        </div>

        <div>
          <!-- Card 1: WEATHER CONDITIONS – BENGALURU -->
          <div class="card" id="card-weather-conditions">
            <div class="card-title">
              <span>WEATHER CONDITIONS &ndash; BENGALURU</span>
              <span class="badge badge-teal" id="wx-status-badge">&#9679; Ground Sensor Sync</span>
            </div>
            
            <div class="weather-hero">
              <div class="weather-hero-main">
                <div class="weather-hero-icon" id="wx-icon">&#9729;</div>
                <div>
                  <div class="weather-hero-title" id="wx-condition">Mostly Cloudy</div>
                  <div class="weather-hero-desc" id="wx-timestamp">Observation: -- UTC</div>
                </div>
              </div>
              <div style="text-align:right;">
                <span class="badge badge-sat" id="wx-delta-badge">&plusmn; 0 min sync</span>
              </div>
            </div>

            <div class="weather-metrics-grid">
              <div class="weather-metric-box">
                <div class="weather-metric-label"><span style="color:var(--accent-cyan); font-size:0.95rem;">&#127777;</span> Temperature</div>
                <div class="weather-metric-val" id="wx-temp">-- &deg;C</div>
              </div>
              <div class="weather-metric-box">
                <div class="weather-metric-label"><span style="color:var(--accent-sky); font-size:0.95rem;">&#128167;</span> Humidity</div>
                <div class="weather-metric-val" id="wx-humidity">-- %</div>
              </div>
              <div class="weather-metric-box">
                <div class="weather-metric-label"><span style="color:var(--accent-teal); font-size:0.95rem;">&#127777;</span> Feels Like</div>
                <div class="weather-metric-val" id="wx-feelslike">-- &deg;C</div>
              </div>
              <div class="weather-metric-box">
                <div class="weather-metric-label"><span style="color:var(--text-muted); font-size:0.95rem;">&#128168;</span> Wind &amp; Pressure</div>
                <div class="weather-metric-val" style="font-size:1.05rem; padding-top:0.2rem;" id="wx-wind-press">-- km/h &bull; -- hPa</div>
              </div>
            </div>
          </div>

          <!-- Card 2: Rainfall Prediction Engine Overview -->
          <div class="card">
            <div class="card-title">
              <span>Rainfall Prediction Engine Overview</span>
              <span class="badge badge-ml">XGBoost Tuned</span>
            </div>
            <p style="font-size:0.83rem; color:var(--text-muted); margin-bottom:0.6rem;">
              Dynamic inference for the selected July 2025 timeline observation. Test scenario presets:
            </p>

            <div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.9rem;">
              <button class="preset-btn" style="font-size:0.72rem; padding:0.28rem 0.6rem;" onclick="loadPreset('monsoon')">&#127783; Monsoon Downpour</button>
              <button class="preset-btn" style="font-size:0.72rem; padding:0.28rem 0.6rem;" onclick="loadPreset('dry')">&#9728; Dry Winter</button>
              <button class="preset-btn" style="font-size:0.72rem; padding:0.28rem 0.6rem;" onclick="loadPreset('summer')">&#9925; Summer Convection</button>
              <button class="preset-btn" style="font-size:0.72rem; padding:0.28rem 0.6rem;" onclick="loadPreset('cyclonic')">&#9928; Cyclonic Squall</button>
            </div>

            <div class="result-card">
              <div style="font-size:0.75rem; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.5px;">Current State Forecast</div>
              <div id="quick-pred-badge" class="pred-pill pred-norain">NO RAIN</div>
              <div>
                <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                  <span style="color:var(--text-muted);">Rain Probability:</span>
                  <strong id="quick-prob-text" style="color:var(--accent-sky);">--%</strong>
                </div>
                <div class="prob-bar-bg">
                  <div id="quick-prob-bar" class="prob-bar-fill" style="width: 0%;"></div>
                </div>
              </div>
              <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-top:0.6rem;">
                <span style="color:var(--text-muted);">Next Hour Rainfall Amount:</span>
                <strong id="quick-amount-text" style="color:var(--accent-sky);">0.00 mm</strong>
              </div>
            </div>

            <div style="margin-top:1.1rem;">
              <div style="font-size:0.75rem; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.4rem; letter-spacing:0.5px;">Complementary Satellite Context:</div>
              <div style="background:var(--card-sec); padding:0.85rem; border-radius:8px; font-size:0.8rem; border:1px solid var(--border-card);">
                <div>&bull; <strong>Cloud Coverage:</strong> <span id="quick-sat-cov" style="color:var(--accent-cyan); font-weight:600;">--</span></div>
                <div>&bull; <strong>Tracking Vector:</strong> <span id="quick-sat-vec" style="color:var(--accent-teal); font-weight:600;">--</span></div>
                <div>&bull; <strong>Observed Timestamp:</strong> <span id="quick-sat-ts" style="color:var(--text-heading); font-weight:600;">--</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: INTERACTIVE PREDICTION CONSOLE -->
    <section id="tab-prediction" class="tab-content">
      <div class="card">
        <div class="card-title">
          <span>Interactive Rainfall Prediction Console</span>
          <span style="font-size:0.8rem; color:var(--text-muted);">Model: 25-Feature XGBoost Pipeline</span>
        </div>

        <div class="preset-row">
          <span style="font-size:0.78rem; font-weight:700; color:var(--text-muted);">Load Scenarios:</span>
          <button class="preset-btn" onclick="loadPreset('monsoon')">Monsoon Downpour</button>
          <button class="preset-btn" onclick="loadPreset('dry')">Dry Winter Noon</button>
          <button class="preset-btn" onclick="loadPreset('summer')">Summer Convection</button>
          <button class="preset-btn" onclick="loadPreset('cyclonic')">Cyclonic Squall</button>
        </div>

        <form id="prediction-form" onsubmit="handlePredictSubmit(event)">
          <div class="form-grid">
            <div class="form-group"><label>Temp (°C)</label><input type="number" id="inp-tempC" step="0.1" required></div>
            <div class="form-group"><label>Max Temp (°C)</label><input type="number" id="inp-maxtempC" step="0.1" required></div>
            <div class="form-group"><label>Min Temp (°C)</label><input type="number" id="inp-mintempC" step="0.1" required></div>
            <div class="form-group"><label>Humidity (%)</label><input type="number" id="inp-humidity" min="0" max="100" required></div>
            <div class="form-group"><label>Cloud Cover (%)</label><input type="number" id="inp-cloudcover" min="0" max="100" required></div>
            <div class="form-group"><label>Pressure (hPa)</label><input type="number" id="inp-pressure" step="0.1" required></div>
            <div class="form-group"><label>Current Precip (mm)</label><input type="number" id="inp-precipMM" step="0.1" required></div>
            <div class="form-group"><label>Dew Point (°C)</label><input type="number" id="inp-DewPointC" step="0.1" required></div>
            <div class="form-group"><label>Feels Like (°C)</label><input type="number" id="inp-FeelsLikeC" step="0.1" required></div>
            <div class="form-group"><label>Wind Speed (km/h)</label><input type="number" id="inp-windspeedKmph" step="0.1" required></div>
            <div class="form-group"><label>Wind Gust (km/h)</label><input type="number" id="inp-WindGustKmph" step="0.1" required></div>
            <div class="form-group"><label>Wind Dir (°)</label><input type="number" id="inp-winddirDegree" min="0" max="360" required></div>
            <div class="form-group"><label>Sun Hours</label><input type="number" id="inp-sunHour" step="0.1" required></div>
            <div class="form-group"><label>UV Index</label><input type="number" id="inp-uvIndex" step="1" required></div>
            <div class="form-group"><label>Visibility (km)</label><input type="number" id="inp-visibility" step="0.5" required></div>
            <div class="form-group">
              <label>Season</label>
              <select id="inp-season_enc">
                <option value="3">Monsoon</option>
                <option value="0">Winter</option>
                <option value="1">Spring</option>
                <option value="2">Summer</option>
                <option value="4">Autumn</option>
              </select>
            </div>
            <div class="form-group"><label>Hour (0-23)</label><input type="number" id="inp-hour" min="0" max="23" required></div>
            <div class="form-group"><label>Month (1-12)</label><input type="number" id="inp-month" min="1" max="12" required></div>
          </div>
          <button type="submit" class="predict-btn">Execute Rainfall Prediction</button>
        </form>

        <div id="prediction-output-box" class="result-card" style="display:none;">
          <div class="card-title" style="margin-bottom:0.5rem;">Prediction Results</div>
          <div id="res-badge" class="pred-pill pred-norain">--</div>
          <div style="margin-bottom:0.6rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
              <span>Rainfall Probability:</span>
              <strong id="res-prob-val">0.0%</strong>
            </div>
            <div class="prob-bar-bg">
              <div id="res-prob-bar" class="prob-bar-fill" style="width:0%;"></div>
            </div>
          </div>
          <div style="font-size:0.9rem; margin-bottom:0.6rem;">
            <span>Expected Precipitation in Next Hour:</span>
            <strong id="res-amount-val" style="color:var(--accent-blue);">0.00 mm</strong>
          </div>
          <div id="res-warnings" style="font-size:0.75rem; color:var(--accent-amber); margin-top:0.4rem;"></div>
        </div>
      </div>
    </section>

    <!-- TAB 3: SATELLITE CLOUD DYNAMICS -->
    <section id="tab-satellite" class="tab-content">
      <div class="card">
        <div class="card-title">INSAT-3DS Multi-Day Cloud Coverage Progression</div>
        <p style="font-size:0.82rem; color:var(--text-muted); margin-bottom:1rem;">
          Continuous 30-minute cloud-mask observations for the Bengaluru region (July 1–31, 2025). Gaps represent scheduled satellite eclipse and calibration pauses.
        </p>
        <div class="media-container">
          <img src="/api/images/cloud_coverage_dynamics.png" alt="Cloud Coverage Dynamics">
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">Inter-Frame Cloud Motion Vectors (Arrows)</div>
          <div class="media-container">
            <img src="/api/images/tracking_pair_001.png" alt="Tracking Pairs with Arrows">
          </div>
          <p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.5rem;">
            Green vectors indicate detected centroid displacement between consecutive timestamped frames.
          </p>
        </div>
        <div class="card">
          <div class="card-title">Long-Duration Cloud Trajectories</div>
          <div class="media-container">
            <img src="/api/images/trajectory_track_2759.png" alt="Track 2759 Trajectory">
          </div>
          <p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.5rem;">
            Track 2759: Tracked across 508 consecutive frames spanning 10 days (July 17–27) across the Bengaluru domain.
          </p>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Cloud Speed & Direction Statistics</div>
        <div class="media-container">
          <img src="/api/images/speed_direction_summary.png" alt="Speed & Direction Rose">
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>July 2025 Chronological Observation Table</span>
          <span style="font-size:0.75rem; color:var(--accent-blue);" id="table-summary-tag">1,448 Observations Loaded</span>
        </div>
        <p style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.8rem;">
          Verified 30-minute INSAT-3DS observations covering the full month of July 2025. Click &quot;Jump to Frame&quot; on any row to inspect it on the timeline visualizer.
        </p>
        <div style="max-height: 380px; overflow-y: auto; border: 1px solid var(--card-border); border-radius: 8px;">
          <table>
            <thead>
              <tr style="position: sticky; top: 0; background: #0f172a; z-index: 5;">
                <th>Index</th>
                <th>Timestamp (UTC)</th>
                <th>Cloud %</th>
                <th>Weather Condition</th>
                <th>Temp</th>
                <th>Rain Forecast</th>
                <th>Speed</th>
                <th>Direction</th>
                <th>Navigate</th>
              </tr>
            </thead>
            <tbody id="sat-table-body">
              <tr><td colspan="8" style="text-align:center; padding:1.5rem; color:var(--text-muted);">Loading July 2025 observation records...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 4: MODEL BENCHMARK & EVALUATION -->
    <section id="tab-models" class="tab-content">
      <div class="card">
        <div class="card-title">Classification Benchmark (Rain Occurrence)</div>
        <p style="font-size:0.82rem; color:var(--text-muted); margin-bottom:0.8rem;">
          Evaluated strictly on held-out chronological test data (19,287 records, 2017–2020). No data leakage.
        </p>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>
            </thead>
            <tbody id="clf-table-body">
              <tr><td>Logistic Regression</td><td>84.36%</td><td>53.00%</td><td>92.49%</td><td>0.6739</td><td>0.9567</td></tr>
              <tr><td>Decision Tree</td><td>92.89%</td><td>75.40%</td><td>88.04%</td><td>0.8123</td><td>0.9608</td></tr>
              <tr><td>Random Forest</td><td>92.72%</td><td>72.41%</td><td>94.21%</td><td>0.8188</td><td>0.9801</td></tr>
              <tr style="font-weight:bold; background:rgba(56,189,248,0.08);">
                <td>XGBoost (Selected & Tuned)</td><td>94.13%</td><td>82.08%</td><td>84.96%</td><td>0.8349</td><td>0.9753</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Regression Benchmark (Rainfall Amount in mm)</div>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr><th>Model</th><th>MAE (mm)</th><th>MSE</th><th>RMSE (mm)</th><th>R² Score</th></tr>
            </thead>
            <tbody id="reg-table-body">
              <tr><td>Linear Regression</td><td>0.0969</td><td>0.1040</td><td>0.3224</td><td>0.6302</td></tr>
              <tr><td>Decision Tree Regressor</td><td>0.0780</td><td>0.1027</td><td>0.3204</td><td>0.6348</td></tr>
              <tr><td>Random Forest Regressor</td><td>0.0700</td><td>0.0858</td><td>0.2929</td><td>0.6949</td></tr>
              <tr style="font-weight:bold; background:rgba(56,189,248,0.08);">
                <td>XGBoost Regressor (Selected & Tuned)</td><td>0.0597</td><td>0.0675</td><td>0.2599</td><td>0.7598</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">Confusion Matrix (Best Classifier)</div>
          <div class="media-container"><img src="/api/images/03_confusion_matrix.png" alt="Confusion Matrix"></div>
        </div>
        <div class="card">
          <div class="card-title">ROC Curves Comparison</div>
          <div class="media-container"><img src="/api/images/04_roc_curves.png" alt="ROC Curves"></div>
        </div>
      </div>
    </section>

    <!-- TAB 5: ARCHITECTURE & DOCUMENTATION -->
    <section id="tab-architecture" class="tab-content">
      <div class="card">
        <div class="card-title">Project Roadmap & Methodology (Phases 1–6)</div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:1.1rem; margin-top:1rem;">
          <div style="background:var(--card-sec); padding:1.15rem; border-radius:10px; border:1px solid var(--border-card); box-shadow:var(--shadow-card);">
            <strong style="color:var(--accent-cyan); font-weight:700;">Phase 1: Preprocessing &amp; EDA</strong>
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">
              96,431 hourly records cleaned; cyclical temporal features engineered; 11 comprehensive EDA charts generated.
            </p>
          </div>
          <div style="background:var(--card-sec); padding:1.15rem; border-radius:10px; border:1px solid var(--border-card); box-shadow:var(--shadow-card);">
            <strong style="color:var(--accent-sky); font-weight:700;">Phase 2: ML Model Training</strong>
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">
              8 models trained (4 classification, 4 regression). XGBoost selected as champion for both targets.
            </p>
          </div>
          <div style="background:var(--card-sec); padding:1.15rem; border-radius:10px; border:1px solid var(--border-card); box-shadow:var(--shadow-card);">
            <strong style="color:var(--accent-teal); font-weight:700;">Phase 3: Satellite NetCDF Extraction</strong>
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">
              Processed 31 INSAT-3DS NetCDF files (~30 GB total); extracted 162x159 Bengaluru domain across 1,448 time slots (July 1–31, 2025).
            </p>
          </div>
          <div style="background:var(--card-sec); padding:1.15rem; border-radius:10px; border:1px solid var(--border-card); box-shadow:var(--shadow-card);">
            <strong style="color:var(--accent-cyan); font-weight:700;">Phase 4: Cloud Motion Tracking</strong>
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">
              Connected components + Hungarian assignment. Detected 16,321 regions, built 4,624 tracks (2,831 multi-frame), computed km/h speeds (avg 70.1 km/h).
            </p>
          </div>
          <div style="background:var(--card-sec); padding:1.15rem; border-radius:10px; border:1px solid var(--border-card); box-shadow:var(--shadow-card);">
            <strong style="color:var(--accent-coral); font-weight:700;">Phase 5: System Integration</strong>
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">
              Connected models into unified prediction engine with temporal auditing and satellite context integration.
            </p>
          </div>
          <div style="background:var(--card-sec); padding:1.15rem; border-radius:10px; border:1px solid var(--border-card); box-shadow:var(--shadow-card);">
            <strong style="color:var(--accent-emerald); font-weight:700;">Phase 6: Verification &amp; Dashboard</strong>
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">
              Automated test suite; interactive web dashboard styled in modern Midnight Navy, Electric Cyan &amp; Teal.
            </p>
          </div>
        </div>
      </div>
    </section>

  </main>

  <script>
    let observations = [];

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
    }

    // Presets for testing
    const PRESETS = {
      monsoon: { tempC: 23.5, maxtempC: 26, mintempC: 21, humidity: 94, cloudcover: 98, pressure: 1006, precipMM: 5.2, DewPointC: 22, FeelsLikeC: 26, windspeedKmph: 22, WindGustKmph: 35, winddirDegree: 240, sunHour: 1.5, uvIndex: 3, visibility: 4, season_enc: 3, hour: 17, month: 8 },
      dry: { tempC: 26.0, maxtempC: 30, mintempC: 15, humidity: 32, cloudcover: 2, pressure: 1016, precipMM: 0.0, DewPointC: 9, FeelsLikeC: 25, windspeedKmph: 7, WindGustKmph: 10, winddirDegree: 90, sunHour: 11.2, uvIndex: 8, visibility: 10, season_enc: 0, hour: 13, month: 1 },
      summer: { tempC: 33.0, maxtempC: 36, mintempC: 22, humidity: 55, cloudcover: 65, pressure: 1009, precipMM: 0.2, DewPointC: 21, FeelsLikeC: 36, windspeedKmph: 14, WindGustKmph: 20, winddirDegree: 180, sunHour: 9.0, uvIndex: 9, visibility: 8, season_enc: 2, hour: 15, month: 5 },
      cyclonic: { tempC: 22.0, maxtempC: 24, mintempC: 20, humidity: 98, cloudcover: 100, pressure: 998, precipMM: 12.5, DewPointC: 21, FeelsLikeC: 23, windspeedKmph: 45, WindGustKmph: 68, winddirDegree: 210, sunHour: 0.0, uvIndex: 1, visibility: 2, season_enc: 3, hour: 11, month: 9 }
    };

    function getFormPayload() {
      const getVal = (id, def) => {
        const el = document.getElementById(id);
        return (el && el.value !== '' && !isNaN(parseFloat(el.value))) ? parseFloat(el.value) : def;
      };
      const getInt = (id, def) => {
        const el = document.getElementById(id);
        return (el && el.value !== '' && !isNaN(parseInt(el.value))) ? parseInt(el.value) : def;
      };

      return {
        tempC:             getVal('inp-tempC', 23.5),
        maxtempC:          getVal('inp-maxtempC', 26.0),
        mintempC:          getVal('inp-mintempC', 21.0),
        humidity:          getVal('inp-humidity', 94.0),
        cloudcover:        getVal('inp-cloudcover', 98.0),
        pressure:          getVal('inp-pressure', 1006.0),
        precipMM:          getVal('inp-precipMM', 5.2),
        DewPointC:         getVal('inp-DewPointC', 22.0),
        FeelsLikeC:        getVal('inp-FeelsLikeC', 26.0),
        windspeedKmph:     getVal('inp-windspeedKmph', 22.0),
        WindGustKmph:      getVal('inp-WindGustKmph', 35.0),
        winddirDegree:     getVal('inp-winddirDegree', 240.0),
        sunHour:           getVal('inp-sunHour', 1.5),
        uvIndex:           getVal('inp-uvIndex', 3.0),
        visibility:        getVal('inp-visibility', 4.0),
        season_enc:        getInt('inp-season_enc', 3),
        hour:              getInt('inp-hour', 17),
        month:             getInt('inp-month', 8),
        day:               15,
        year:              2020,
        day_of_week:       2,
        week_of_year:      25,
        HeatIndexC:        getVal('inp-FeelsLikeC', 26.0),
        WindChillC:        getVal('inp-tempC', 23.5),
        moon_illumination: 50.0
      };
    }

    async function executePredictionWithPayload(payload) {
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          throw new Error('HTTP ' + res.status + ': ' + (await res.text()));
        }
        const data = await res.json();
        renderPrediction(data);
      } catch (err) {
        console.error('Prediction API error:', err);
        renderPrediction({ prediction_valid: false, warnings: [err.message] });
      }
    }

    function loadPreset(key) {
      const p = PRESETS[key];
      if (!p) return;
      for (const k in p) {
        const el = document.getElementById('inp-' + k);
        if (el) el.value = p[k];
      }
      const payload = getFormPayload();
      executePredictionWithPayload(payload);
    }

    async function handlePredictSubmit(e) {
      if (e && e.preventDefault) e.preventDefault();
      const payload = getFormPayload();
      executePredictionWithPayload(payload);
    }

    function renderPrediction(data) {
      if (!data || !data.prediction_valid) {
        const kpiStatusEl = document.getElementById('kpi-rain-status');
        if (kpiStatusEl) { kpiStatusEl.textContent = 'UNAVAILABLE'; kpiStatusEl.style.color = 'var(--text-muted)'; }
        const kpiAmountEl = document.getElementById('kpi-rain-amount');
        if (kpiAmountEl) kpiAmountEl.textContent = 'Prediction model offline';
        const qb = document.getElementById('quick-pred-badge');
        if (qb) { qb.textContent = 'UNAVAILABLE'; qb.className = 'pred-pill'; }
        const qp = document.getElementById('quick-prob-text');
        if (qp) qp.textContent = 'N/A';
        const qa = document.getElementById('quick-amount-text');
        if (qa) qa.textContent = 'N/A';
        return;
      }

      const isRain = (data.rain_prediction === 1 || data.rain_label === 'Rain' || data.rain_status === 'Rain');
      const badgeText = isRain ? 'RAIN PREDICTED' : 'NO RAIN';
      const pillClass = isRain ? 'pred-pill pred-rain' : 'pred-pill pred-norain';
      const probPct = (data.rain_probability !== null && !isNaN(data.rain_probability))
        ? (data.rain_probability * 100).toFixed(1)
        : 'N/A';
      const noRainPct = (data.no_rain_probability !== null && !isNaN(data.no_rain_probability))
        ? (data.no_rain_probability * 100).toFixed(1)
        : (probPct !== 'N/A' ? (100.0 - parseFloat(probPct)).toFixed(1) : 'N/A');
      const amountMm = (data.rainfall_amount_mm !== null && !isNaN(data.rainfall_amount_mm))
        ? data.rainfall_amount_mm.toFixed(2)
        : '0.00';
      const amountStr = amountMm + ' mm';

      // ── 1. Top Summary Card (Tab 1) ──
      const kpiStatusEl = document.getElementById('kpi-rain-status');
      if (kpiStatusEl) {
        kpiStatusEl.textContent = isRain ? 'RAIN' : 'NO RAIN';
        kpiStatusEl.style.color = isRain ? 'var(--accent-coral)' : 'var(--accent-emerald)';
      }
      const rainCardEl = document.querySelector('.kpi-card.rain-card');
      if (rainCardEl) {
        rainCardEl.style.setProperty('--kpi-rain-bar', isRain ? 'var(--accent-coral)' : 'var(--accent-emerald)');
      }
      const kpiAmountEl = document.getElementById('kpi-rain-amount');
      if (kpiAmountEl) {
        kpiAmountEl.innerHTML = '<span style="color:var(--accent-sky); font-weight:600;">' + probPct + '% prob</span> | Next hr: <span style="color:' + (isRain ? 'var(--accent-coral)' : 'var(--accent-sky)') + '; font-weight:700;">' + amountStr + '</span>';
      }

      // ── 2. Detailed Prediction Panel (Tab 1) ──
      const quickBadge = document.getElementById('quick-pred-badge');
      if (quickBadge) {
        quickBadge.textContent = badgeText;
        quickBadge.className = pillClass;
      }
      const quickProb = document.getElementById('quick-prob-text');
      if (quickProb) {
        quickProb.textContent = probPct + '%';
      }
      const quickBar = document.getElementById('quick-prob-bar');
      if (quickBar) {
        quickBar.style.width = (data.rain_probability !== null ? (data.rain_probability * 100) : 0) + '%';
      }
      const quickAmount = document.getElementById('quick-amount-text');
      if (quickAmount) {
        quickAmount.textContent = amountStr;
        quickAmount.style.color = isRain ? 'var(--accent-coral)' : 'var(--accent-sky)';
      }

      // ── 3. Interactive Console Output Box (Tab 2) ──
      const box = document.getElementById('prediction-output-box');
      if (box) box.style.display = 'block';

      const resBadge = document.getElementById('res-badge');
      if (resBadge) {
        resBadge.textContent = badgeText;
        resBadge.className = pillClass;
      }
      const resProbVal = document.getElementById('res-prob-val');
      if (resProbVal) {
        resProbVal.textContent = probPct + '% (No-Rain Prob: ' + noRainPct + '%)';
      }
      const resProbBar = document.getElementById('res-prob-bar');
      if (resProbBar) {
        resProbBar.style.width = (data.rain_probability !== null ? (data.rain_probability * 100) : 0) + '%';
      }
      const resAmountVal = document.getElementById('res-amount-val');
      if (resAmountVal) {
        resAmountVal.textContent = amountStr;
        resAmountVal.style.color = isRain ? 'var(--accent-coral)' : 'var(--accent-sky)';
      }

      const wBox = document.getElementById('res-warnings');
      if (wBox) {
        if (data.warnings && data.warnings.length) {
          wBox.innerHTML = '&#9888; ' + data.warnings.join(' | ');
        } else {
          wBox.innerHTML = '';
        }
      }
    }

    function onSlideObservation(idx) {
      idx = parseInt(idx);
      document.getElementById('slider-val-disp').textContent = 'Index: ' + idx + ' / ' + (observations.length - 1);
      if (observations[idx]) {
        updateDashboardCards(observations[idx]);
      }
    }

    function stepObservation(delta) {
      const slider = document.getElementById('obs-slider');
      let val = parseInt(slider.value) + delta;
      val = Math.max(0, Math.min(val, observations.length - 1));
      slider.value = val;
      onSlideObservation(val);
    }

    function jumpToObservation(idx) {
      switchTab('dashboard');
      const slider = document.getElementById('obs-slider');
      slider.value = idx;
      onSlideObservation(idx);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function updateDashboardCards(obs) {
      if (!obs) return;

      // ── 1. Satellite Cloud Tracking KPIs ──
      const covEl = document.getElementById('kpi-coverage');
      if (covEl) covEl.textContent = obs.cloud_coverage_pct + '%';
      const covSubEl = document.getElementById('kpi-coverage-sub');
      if (covSubEl) covSubEl.textContent = 'Clear: ' + obs.clear_coverage_pct + '%';
      const regEl = document.getElementById('kpi-regions');
      if (regEl) regEl.textContent = obs.n_regions;
      const spdEl = document.getElementById('kpi-speed');
      if (spdEl) spdEl.textContent = (obs.mean_speed_kmh ? obs.mean_speed_kmh : 'N/A') + ' km/h';
      const spdDirEl = document.getElementById('kpi-speed-dir');
      if (spdDirEl) spdDirEl.textContent = 'Direction: ' + obs.dominant_dir;
      const tsLabelEl = document.getElementById('obs-ts-label');
      if (tsLabelEl) tsLabelEl.textContent = 'Observation ' + (obs.index + 1) + '/' + observations.length + ': ' + obs.timestamp + ' UTC';

      // ── 2. Dynamic July Cloud-Mask Frame Render ──
      const satImg = document.getElementById('dashboard-sat-img');
      if (satImg) satImg.src = '/api/cloudmask-render/' + obs.index;

      // ── 3. WEATHER CONDITIONS – BENGALURU ──
      const wxIcon = document.getElementById('wx-icon');
      if (wxIcon) wxIcon.textContent = obs.weather_icon || '☁️';
      const wxCond = document.getElementById('wx-condition');
      if (wxCond) wxCond.textContent = obs.weather_condition || 'Mostly Cloudy';
      const wxTs = document.getElementById('wx-timestamp');
      if (wxTs) wxTs.textContent = 'Observation: ' + obs.timestamp + ' UTC';
      const wxDelta = document.getElementById('wx-delta-badge');
      if (wxDelta) wxDelta.textContent = 'Δ ' + (obs.time_delta_min !== undefined ? obs.time_delta_min : 0) + ' min sync';
      const wxTemp = document.getElementById('wx-temp');
      if (wxTemp) wxTemp.textContent = (obs.temperature_c !== undefined ? obs.temperature_c.toFixed(1) : '--') + ' °C';
      const wxHum = document.getElementById('wx-humidity');
      if (wxHum) wxHum.textContent = (obs.humidity_pct !== undefined ? obs.humidity_pct.toFixed(1) : '--') + ' %';
      const wxFeels = document.getElementById('wx-feelslike');
      if (wxFeels) wxFeels.textContent = (obs.feels_like_c !== undefined ? obs.feels_like_c.toFixed(1) : '--') + ' °C';
      const wxWindPress = document.getElementById('wx-wind-press');
      if (wxWindPress) {
        const wspd = obs.wind_speed_kmh !== undefined ? obs.wind_speed_kmh.toFixed(1) : '--';
        const press = obs.pressure_hpa !== undefined ? obs.pressure_hpa.toFixed(0) : '--';
        wxWindPress.innerHTML = wspd + ' km/h &bull; ' + press + ' hPa';
      }

      // ── 4. Dynamic Rainfall Forecast (Top KPI Card + Detailed Prediction Panel) ──
      const isRain = obs.rain_prediction === 1 || (obs.rain_label && obs.rain_label.toLowerCase() === 'rain');
      const badgeText = isRain ? 'RAIN' : 'NO RAIN';
      const pillClass = isRain ? 'pred-pill pred-rain' : 'pred-pill pred-norain';
      const probVal = obs.rain_probability !== null && obs.rain_probability !== undefined ? (obs.rain_probability * 100) : 0;
      const probPct = probVal.toFixed(1);
      const noRainPct = obs.no_rain_probability !== null && obs.no_rain_probability !== undefined ? (obs.no_rain_probability * 100).toFixed(1) : (100 - probVal).toFixed(1);
      const amountVal = obs.rainfall_amount_mm !== undefined && obs.rainfall_amount_mm !== null ? obs.rainfall_amount_mm : 0;
      const amountStr = amountVal.toFixed(2) + ' mm';

      // Top KPI Card
      const kpiRainStatus = document.getElementById('kpi-rain-status');
      if (kpiRainStatus) {
        kpiRainStatus.textContent = badgeText;
        kpiRainStatus.style.color = isRain ? 'var(--accent-coral)' : 'var(--accent-emerald)';
      }
      const kpiRainCard = document.querySelector('.kpi-card.rain-card');
      if (kpiRainCard) {
        kpiRainCard.style.setProperty('--kpi-rain-bar', isRain ? 'var(--accent-coral)' : 'var(--accent-emerald)');
      }
      const kpiRainAmount = document.getElementById('kpi-rain-amount');
      if (kpiRainAmount) {
        kpiRainAmount.innerHTML = '<span style="color:var(--accent-sky); font-weight:600;">' + probPct + '% prob</span> | Next hr: <span style="color:' + (isRain ? 'var(--accent-coral)' : 'var(--accent-sky)') + '; font-weight:700;">' + amountStr + '</span>';
      }

      // Detailed Prediction Panel
      const quickBadge = document.getElementById('quick-pred-badge');
      if (quickBadge) {
        quickBadge.textContent = badgeText;
        quickBadge.className = pillClass;
      }
      const quickProb = document.getElementById('quick-prob-text');
      if (quickProb) {
        quickProb.textContent = probPct + '% (No-Rain: ' + noRainPct + '%)';
      }
      const quickBar = document.getElementById('quick-prob-bar');
      if (quickBar) {
        quickBar.style.width = probPct + '%';
      }
      const quickAmount = document.getElementById('quick-amount-text');
      if (quickAmount) {
        quickAmount.textContent = amountStr;
        quickAmount.style.color = isRain ? 'var(--accent-coral)' : 'var(--accent-sky)';
      }

      // ── 5. Complementary Satellite Context ──
      const qSatCov = document.getElementById('quick-sat-cov');
      if (qSatCov) qSatCov.textContent = obs.cloud_coverage_pct + '% (Clear: ' + obs.clear_coverage_pct + '%)';
      const qSatVec = document.getElementById('quick-sat-vec');
      if (qSatVec) qSatVec.textContent = (obs.mean_speed_kmh || '--') + ' km/h towards ' + obs.dominant_dir;
      const qSatTs = document.getElementById('quick-sat-ts');
      if (qSatTs) qSatTs.textContent = obs.timestamp + ' UTC';
    }

    function populateObservationTable() {
      const tbody = document.getElementById('sat-table-body');
      if (!tbody || !observations.length) return;
      
      let html = '';
      // Render sample observations across the 1448 frames (step = 14 to show ~100 distributed timestamps)
      const step = Math.max(1, Math.floor(observations.length / 100));
      for (let i = 0; i < observations.length; i += step) {
        const o = observations[i];
        const spd = o.mean_speed_kmh !== null ? o.mean_speed_kmh + ' km/h' : '<span style="color:var(--text-muted);">N/A</span>';
        const isR = o.rain_prediction === 1;
        const rainProb = o.rain_probability !== null && o.rain_probability !== undefined ? (o.rain_probability * 100).toFixed(0) : '0';
        const rainBadge = isR ? '<span style="color:var(--accent-coral); font-weight:700;">Rain (' + rainProb + '%)</span>' : '<span style="color:var(--accent-emerald); font-weight:600;">No Rain (' + rainProb + '%)</span>';
        html += '<tr>' +
          '<td><strong>#' + (o.index + 1) + '</strong></td>' +
          '<td>' + o.timestamp + '</td>' +
          '<td><span style="color:var(--accent-cyan); font-weight:600;">' + o.cloud_coverage_pct + '%</span></td>' +
          '<td>' + o.weather_condition + ' ' + (o.weather_icon || '') + '</td>' +
          '<td>' + (o.temperature_c !== undefined ? o.temperature_c.toFixed(1) + ' °C' : '--') + '</td>' +
          '<td>' + rainBadge + '</td>' +
          '<td><span style="color:var(--accent-teal);">' + spd + '</span></td>' +
          '<td>' + o.dominant_dir + '</td>' +
          '<td><button class="preset-btn" style="padding:0.25rem 0.6rem; font-size:0.72rem; color:var(--accent-teal); border-color:rgba(45,212,191,0.3);" onclick="jumpToObservation(' + o.index + ')">Jump &rarr;</button></td>' +
          '</tr>';
      }
      tbody.innerHTML = html;
      const tag = document.getElementById('table-summary-tag');
      if (tag) tag.textContent = observations.length + ' Observations (1–31 July 2025)';
    }

    // Startup Init
    async function init() {
      try {
        const res = await fetch('/api/satellite-observations');
        observations = await res.json();
        if (observations.length) {
          // Initialize with Observation 0 (July 1, 2025 00:30 UTC)
          const defaultObs = observations[0];
          const slider = document.getElementById('obs-slider');
          slider.max = observations.length - 1;
          slider.value = 0;
          document.getElementById('slider-val-disp').textContent = 'Index: 0 / ' + (observations.length - 1);
          document.getElementById('slider-range-label').textContent = 'Observation Timeline (1 to ' + observations.length + ') \u2014 July 1\u201331, 2025';
          updateDashboardCards(defaultObs);
          populateObservationTable();
        }
      } catch (e) {
        console.error('Init error:', e);
      }
    }

    window.onload = init;
  </script>
</body>
</html>
"""


# ── Custom HTTP Request Handler ──────────────────────────────────────
class CloudPathHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean console logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif path == "/api/status":
            latest = OBSERVATIONS_JSON[-1] if OBSERVATIONS_JSON else {}
            status_data = {
                "system": "CloudPath AI Operational Engine",
                "status": "ONLINE",
                "observations_count": len(OBSERVATIONS_JSON),
                "latest_satellite": latest,
                "temporal_overlap": overlap_audit["has_overlap"],
                "classifier": predictor._clf_name,
                "regressor": predictor._reg_name,
            }
            self._send_json(status_data)

        elif path == "/api/satellite-observations":
            self._send_json(OBSERVATIONS_JSON)

        elif path == "/api/model-benchmarks":
            records = eval_df.to_dict(orient="records")
            self._send_json(records)

        elif path.startswith("/api/images/"):
            img_name = urllib.parse.unquote(path.replace("/api/images/", ""))
            # Look in the three output folders
            candidates = [
                os.path.join(PROJECT_ROOT, "satellite_tracking_outputs", img_name),
                os.path.join(PROJECT_ROOT, "satellite_data_outputs", img_name),
                os.path.join(PROJECT_ROOT, "model_comparison", img_name),
            ]
            found_path = None
            for cp in candidates:
                if os.path.exists(cp):
                    found_path = cp
                    break

            if found_path:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                with open(found_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"Image {img_name} not found")

        elif path.startswith("/api/cloudmask-render/"):
            try:
                frame_idx = int(path.replace("/api/cloudmask-render/", ""))
                png_bytes = render_cloudmask_png(frame_idx)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(png_bytes)
            except Exception as e:
                self.send_error(400, f"Render error: {e}")

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                
                # Build WeatherInput dataclass
                wi = WeatherInput(
                    maxtempC          = float(data.get("maxtempC", 28.0)),
                    mintempC          = float(data.get("mintempC", 20.0)),
                    sunHour           = float(data.get("sunHour", 6.0)),
                    uvIndex           = float(data.get("uvIndex", 5.0)),
                    moon_illumination = float(data.get("moon_illumination", 50.0)),
                    DewPointC         = float(data.get("DewPointC", 20.0)),
                    FeelsLikeC        = float(data.get("FeelsLikeC", 28.0)),
                    HeatIndexC        = float(data.get("HeatIndexC", 28.0)),
                    WindChillC        = float(data.get("WindChillC", 28.0)),
                    WindGustKmph      = float(data.get("WindGustKmph", 20.0)),
                    cloudcover        = float(data.get("cloudcover", 70.0)),
                    humidity          = float(data.get("humidity", 85.0)),
                    precipMM          = float(data.get("precipMM", 1.0)),
                    pressure          = float(data.get("pressure", 1010.0)),
                    tempC             = float(data.get("tempC", 26.0)),
                    visibility        = float(data.get("visibility", 8.0)),
                    winddirDegree     = float(data.get("winddirDegree", 220.0)),
                    windspeedKmph     = float(data.get("windspeedKmph", 15.0)),
                    hour              = int(data.get("hour", 14)),
                    day               = int(data.get("day", 15)),
                    month             = int(data.get("month", 7)),
                    year              = int(data.get("year", 2020)),
                    day_of_week       = int(data.get("day_of_week", 2)),
                    week_of_year      = int(data.get("week_of_year", 28)),
                    season_enc        = int(data.get("season_enc", 3)),
                )

                # Attach latest satellite context
                latest_sat_row = satellite_features.iloc[-1]
                sat_ctx = satellite_context_from_row(latest_sat_row)

                # Run inference
                res = predictor.predict(wi, satellite=sat_ctx)

                resp = {
                    "rain_prediction": int(res.rain_prediction),
                    "rain_status": res.rain_label,
                    "rain_label": res.rain_label,
                    "rain_probability": res.rain_probability,
                    "no_rain_probability": res.no_rain_probability,
                    "rainfall_amount_mm": res.rainfall_amount_mm,
                    "input_complete": res.input_complete,
                    "warnings": res.warnings,
                    "validation_note": res.validation_note,
                    "prediction_timestamp": res.prediction_timestamp,
                    "prediction_valid": res.prediction_valid,
                }
                self._send_json(resp)
            except Exception as e:
                self.send_error(400, f"Inference error: {str(e)}")
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


def run_server(port=8050, host="0.0.0.0"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, CloudPathHandler)
    print("\n" + "=" * 65, flush=True)
    print("  CloudPath AI – Interactive Web Dashboard Server", flush=True)
    print("=" * 65, flush=True)
    print(f"  Server URL: http://localhost:{port}/", flush=True)
    print(f"  Status API: http://localhost:{port}/api/status", flush=True)
    print("  Press Ctrl+C to terminate.", flush=True)
    print("=" * 65 + "\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] Shutting down server gracefully...", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CloudPath AI Web Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


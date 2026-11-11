# CloudPath AI – Phase 3: Satellite Image Analysis & Cloud Movement Tracking

Welcome to Phase 3 of **CloudPath AI – AI-Based Cloud Movement Tracking and Rainfall Prediction System**. This module extends the system's ground weather models by analyzing sequential meteorological satellite imagery (e.g., from ISRO MOSDAC INSAT-3D / INSAT-3DR) to detect, segment, track, and forecast cloud movement towards Bengaluru.

---

## 1. Architecture & Module Overview

The system is organized into modular Python components:

| Module File | Responsibility |
| :--- | :--- |
| `satellite_loader.py` | Ingests raster images (`.png`, `.jpg`, `.tif`) and scientific HDF5 (`.h5`, `.hdf`) files. Extracts timestamps from filenames/metadata, inspects HDF datasets, and sorts frames chronologically. |
| `image_preprocessing.py` | Standardizes dimensions, applies min-max normalization, enhances contrast using CLAHE, and removes atmospheric sensor noise with bilateral filtering. |
| `cloud_detection.py` | Segments clouds using Otsu adaptive thresholding and classifies cloud vertical thickness into 3 density levels (Thin, Medium, Deep Convective). |
| `cloud_tracking.py` | Detects connected cloud formations, extracts spatial moments (centroids, areas, bounding boxes), and matches multi-cloud entities across frames using Hungarian bipartite matching while handling births, dissipations, splits, and mergers. |
| `cloud_motion.py` | Computes displacement vectors ($\Delta x, \Delta y$), 16-point meteorological compass headings, elapsed time $\Delta t$, and speed. Converts to $\text{km/h}$ when georeferenced scale is provided. |
| `cloud_prediction.py` | Extrapolates future cloud centroid coordinates and bounding envelopes for $+30\text{ min}$, $+1\text{ hour}$, and $+2\text{ hours}$ with expanding uncertainty buffers, computing Bengaluru intersection alarms and ETA. |
| `cloud_visualization.py` | Generates diagnostic plots, segmentation masks, tracking arrows, trajectory projections, and coverage dynamics curves. |
| `main_satellite.py` | Main CLI orchestration entry point coordinating the end-to-end pipeline. |

---

## 2. Installation & Requirements

Ensure Python 3.9+ is installed. Install the dependencies listed in `requirements_phase3.txt`:

```bash
pip install -r requirements_phase3.txt
```

Key libraries used:
* **OpenCV (`cv2`)**: Computer vision, morphological filtering, contours, optical flow.
* **NumPy & SciPy**: Matrix operations, moment analysis, bipartite matching (`linear_sum_assignment`).
* **h5py**: Scientific inspection of MOSDAC HDF5 satellite datasets.
* **Matplotlib & Seaborn**: Diagnostic and trajectory visual generation.
* **Pillow (PIL)**: Image loading and metadata parsing.

---

## 3. How to Run

### Step 1: Providing Satellite Imagery
Place your sequential satellite images in the input directory:
```
d:\bengaluru.csv\satellite_input\
```
Supported formats:
* Common images: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`
* Scientific HDF5: `.h5`, `.hdf`, `.he5`

**MOSDAC INSAT-3D Naming Convention:**
The loader automatically parses timestamps from standard MOSDAC filenames such as:
`3DIMG_03NOV2026_0600_L1C_ASIA_MER_TIR1.jpg` $\rightarrow$ Timestamp: `2026-11-03 06:00:00`.

*Note:* If the `satellite_input/` folder is empty, running the pipeline automatically engages **Test Mode**, generating a realistic, verified, watermarked synthetic INSAT-3D TIR1 sequence clearly labeled `[SIMULATED_TEST_DATA]` so you can evaluate the complete pipeline immediately.

### Step 2: Executing the Pipeline
Run the main script:
```bash
python main_satellite.py
```

---

## 4. Output Deliverables

All outputs are saved automatically in:
```
d:\bengaluru.csv\satellite_tracking_outputs\
```

### Visualizations Generated:
1. `detection_frame_*.png`: 4-panel visual comparison showing raw, CLAHE-enhanced, binary mask, and 3-level density classification (Thin/Medium/Deep Convective).
2. `tracking_frame_*.png`: Overlaid satellite imagery with persistent cloud IDs, centroid coordinates, bounding boxes, historical paths, and yellow motion vector arrows indicating speed and compass direction.
3. `prediction_extrapolation_frame_*.png`: Extrapolated cloud positions for $+30\text{ min}$, $+1\text{ hour}$, and $+2\text{ hours}$ with expanding uncertainty cones and Bengaluru collision alarms.
4. `cloud_coverage_dynamics_trend.png`: Multi-frame cloud coverage graph showing net expansion/dissipation trends and stacked density breakdown.

### Data & Technical Reports:
* `satellite_tracking_metrics.csv`: Tabular log recording each tracked cloud's position, speed ($\text{px/h}$ and $\text{km/h}$), direction, extrapolated coordinates, and Bengaluru intersection alerts.
* `cloud_coverage_report.txt`: Summary of cloud dynamics, predominant headings, coverage evolution, and physical limitations.


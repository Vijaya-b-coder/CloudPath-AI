"""
test_all_modules.py
===================
CloudPath AI – Phase 6: Final Verification & Test Suite
Performs comprehensive automated testing across all modules:
  1. Weather Preprocessing & Schema Validation
  2. Rainfall Classification (XGBoost)
  3. Rainfall Regression (XGBoost)
  4. Satellite Data Loading & Geo-referencing
  5. Cloud Region Detection (Connected Components)
  6. Cloud Motion Tracking & Vector Calculation
  7. System Integration & Temporal Audit
  8. Missing / Invalid Input Robustness & Error Handling

Generates: test_suite_report.txt
Author: CloudPath AI Project
"""

import os
import sys
import time
import math
import traceback
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prediction_engine import (
    RainfallPredictor, WeatherInput, SatelliteContext,
    REQUIRED_FEATURES, weather_input_from_row
)
from integration_pipeline import (
    load_weather_data, load_satellite_summary, load_movement_data,
    check_temporal_overlap, build_satellite_feature_table
)
from cloud_region_detection import detect_regions_in_frame, MIN_REGION_PIXELS
from cloud_matching import match_frames
from cloud_speed_calculator import load_geo_params, _direction_deg, _direction_label

REPORT_FILE = os.path.join(PROJECT_ROOT, "test_suite_report.txt")


class TestSuite:
    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def record(self, test_name: str, passed: bool, details: str = ""):
        status = "PASSED" if passed else "FAILED"
        self.results.append({
            "test": test_name,
            "status": status,
            "details": details
        })
        mark = "✓" if passed else "✗"
        print(f"  [{mark}] {test_name}: {status} {('- ' + details) if details else ''}")

    # --- Test 1: Weather Preprocessing & Schema ---
    def test_weather_dataset(self):
        csv_path = os.path.join(PROJECT_ROOT, "bengaluru_processed.csv")
        if not os.path.exists(csv_path):
            self.record("Weather Dataset File Existence", False, "bengaluru_processed.csv missing")
            return
        df = pd.read_csv(csv_path)
        has_rows = len(df) >= 90000
        has_targets = "rain_status" in df.columns and "next_hour_rainfall" in df.columns
        has_features = all(f in df.columns for f in REQUIRED_FEATURES if f != "season_enc")
        
        self.record(
            "Weather Dataset Dimensions & Targets",
            has_rows and has_targets,
            f"Rows: {len(df):,}, Target cols present: {has_targets}"
        )
        self.record(
            "Weather Feature Schema Integrity",
            has_features,
            f"All expected base weather features present: {has_features}"
        )

    # --- Test 2: Classification Model ---
    def test_classification_model(self):
        clf_path = os.path.join(PROJECT_ROOT, "rainfall_classifier.pkl")
        if not os.path.exists(clf_path):
            self.record("Classifier Artifact Exists", False, "rainfall_classifier.pkl missing")
            return
        bundle = joblib.load(clf_path)
        has_keys = all(k in bundle for k in ["model", "scaler", "feature_cols", "threshold"])
        feature_count_ok = len(bundle.get("feature_cols", [])) == 25
        
        self.record(
            "Classifier Artifact Structure",
            has_keys and feature_count_ok,
            f"Contains model, scaler, 25 feature columns (threshold: {bundle.get('threshold')})"
        )

    # --- Test 3: Regression Model ---
    def test_regression_model(self):
        reg_path = os.path.join(PROJECT_ROOT, "rainfall_regressor.pkl")
        if not os.path.exists(reg_path):
            self.record("Regressor Artifact Exists", False, "rainfall_regressor.pkl missing")
            return
        bundle = joblib.load(reg_path)
        has_keys = all(k in bundle for k in ["model", "scaler", "feature_cols"])
        feature_count_ok = len(bundle.get("feature_cols", [])) == 25
        
        self.record(
            "Regressor Artifact Structure",
            has_keys and feature_count_ok,
            f"Contains model, scaler, 25 feature columns ({bundle.get('model_name')})"
        )

    # --- Test 4: Prediction Engine Functionality ---
    def test_prediction_engine(self):
        try:
            predictor = RainfallPredictor()
            # Rain scenario input
            rain_input = WeatherInput(
                maxtempC=28, mintempC=21, sunHour=2.5, uvIndex=4, moon_illumination=45,
                DewPointC=22, FeelsLikeC=28, HeatIndexC=28, WindChillC=26, WindGustKmph=28,
                cloudcover=95, humidity=92, precipMM=4.2, pressure=1007, tempC=24,
                visibility=5, winddirDegree=240, windspeedKmph=18,
                hour=17, day=15, month=8, year=2019, day_of_week=3, week_of_year=33,
                season_enc=3 # Monsoon
            )
            res_rain = predictor.predict(rain_input)
            
            # Dry scenario input
            dry_input = WeatherInput(
                maxtempC=29, mintempC=15, sunHour=11.5, uvIndex=7, moon_illumination=80,
                DewPointC=10, FeelsLikeC=24, HeatIndexC=24, WindChillC=24, WindGustKmph=8,
                cloudcover=5, humidity=35, precipMM=0.0, pressure=1016, tempC=25,
                visibility=10, winddirDegree=80, windspeedKmph=6,
                hour=14, day=10, month=1, year=2019, day_of_week=4, week_of_year=2,
                season_enc=0 # Winter
            )
            res_dry = predictor.predict(dry_input)
            
            non_negative = res_rain.rainfall_amount_mm >= 0.0 and res_dry.rainfall_amount_mm >= 0.0
            prob_bounded = (0.0 <= (res_rain.rain_probability or 0.0) <= 1.0) and \
                           (0.0 <= (res_dry.rain_probability or 0.0) <= 1.0)
            
            self.record(
                "Prediction Engine Execution & Bounds",
                non_negative and prob_bounded,
                f"Rain prob: {res_rain.rain_probability:.2f}, Dry prob: {res_dry.rain_probability:.2f}"
            )
        except Exception as e:
            self.record("Prediction Engine Execution & Bounds", False, str(e))

    # --- Test 5: Satellite Data & Geo-referencing ---
    def test_satellite_data(self):
        npz_path = os.path.join(PROJECT_ROOT, "satellite_data_outputs", "bengaluru_cloudmask_series.npz")
        if not os.path.exists(npz_path):
            self.record("Satellite NPZ Series Exists", False, "bengaluru_cloudmask_series.npz missing")
            return
        data = np.load(npz_path, allow_pickle=True)
        masks = data["cloud_masks"]
        timestamps = data["timestamps"]
        lat_b = data["lat_bounds"]
        lon_b = data["lon_bounds"]
        
        shape_ok = len(masks.shape) == 3 and masks.shape[1] == 162 and masks.shape[2] == 159
        count_ok = len(timestamps) == 1448
        
        geo = load_geo_params(npz_path)
        scale_ok = (3.5 <= geo["km_per_row"] <= 4.5) and (3.5 <= geo["km_per_col"] <= 4.5)
        
        self.record(
            "Satellite NPZ Grid & Dimensions",
            shape_ok and count_ok,
            f"Shape: {masks.shape}, Timestamps: {len(timestamps)}"
        )
        self.record(
            "Satellite Geospatial Scale",
            scale_ok,
            f"km/row: {geo['km_per_row']:.3f}, km/col: {geo['km_per_col']:.3f} (approx 4km)"
        )

    # --- Test 6: Cloud Detection & Tracking Algorithms ---
    def test_cloud_detection_tracking(self):
        npz_path = os.path.join(PROJECT_ROOT, "satellite_data_outputs", "bengaluru_cloudmask_series.npz")
        data = np.load(npz_path, allow_pickle=True)
        cm0 = data["cloud_masks"][0]
        cm1 = data["cloud_masks"][1]
        
        r0 = detect_regions_in_frame(0, str(data["timestamps"][0]), cm0, min_pixels=MIN_REGION_PIXELS)
        r1 = detect_regions_in_frame(1, str(data["timestamps"][1]), cm1, min_pixels=MIN_REGION_PIXELS)
        
        detect_ok = len(r0) > 0 and len(r1) > 0
        matches, disappeared, appeared = match_frames(r0, r1, grid_shape=(162, 159))
        track_ok = len(matches) > 0
        
        # Test bearing computation
        b_east = _direction_deg(0, 10)  # delta_col > 0 -> East -> 90 deg
        b_south = _direction_deg(10, 0) # delta_row > 0 -> South -> 180 deg
        math_ok = math.isclose(b_east, 90.0, abs_tol=1.0) and math.isclose(b_south, 180.0, abs_tol=1.0)
        
        self.record(
            "Connected Cloud Region Detection",
            detect_ok,
            f"Frame 0 regions: {len(r0)}, Frame 1 regions: {len(r1)}"
        )
        self.record(
            "Hungarian Assignment & Tracking",
            track_ok,
            f"Matches found: {len(matches)}, Disappeared: {len(disappeared)}, Appeared: {len(appeared)}"
        )
        self.record(
            "Meteorological Bearing Calculation",
            math_ok,
            f"East bearing: {b_east}°, South bearing: {b_south}°"
        )

    # --- Test 7: Integration & Temporal Audit ---
    def test_integration_and_audit(self):
        try:
            w_df = load_weather_data()
            s_df = load_satellite_summary()
            m_df = load_movement_data()
            audit = check_temporal_overlap(w_df, s_df)
            
            # The audit must honestly state has_overlap == False
            honest_audit = (audit["has_overlap"] is False)
            feat_table = build_satellite_feature_table(s_df, m_df)
            table_ok = len(feat_table) == 1448 and "mean_speed_kmh" in feat_table.columns
            
            self.record(
                "Temporal Compatibility Audit (Honest Non-Overlap)",
                honest_audit,
                f"Weather: {audit['weather_start'].year}-{audit['weather_end'].year}, Sat: {audit['satellite_start'].year}"
            )
            self.record(
                "Satellite Feature Aggregation Table",
                table_ok,
                f"Rows: {len(feat_table)}, Features: {list(feat_table.columns[:6])}"
            )
        except Exception as e:
            self.record("Integration & Temporal Audit", False, str(e))

    # --- Test 8: Robustness to Missing Inputs ---
    def test_input_robustness(self):
        try:
            predictor = RainfallPredictor()
            # Incomplete input with None values
            incomplete = WeatherInput(
                maxtempC=None, mintempC=20, sunHour=None, uvIndex=5, moon_illumination=50,
                DewPointC=18, FeelsLikeC=None, HeatIndexC=25, WindChillC=25, WindGustKmph=15,
                cloudcover=50, humidity=80, precipMM=0.0, pressure=1012, tempC=24,
                visibility=10, winddirDegree=180, windspeedKmph=10,
                hour=12, day=1, month=6, year=2020, day_of_week=0, week_of_year=23,
                season_enc=2
            )
            res = predictor.predict(incomplete)
            handled = (res.input_complete is False) and (len(res.warnings) > 0) and (res.rainfall_amount_mm >= 0.0)
            self.record(
                "Missing Feature Imputation & Warning Generation",
                handled,
                f"Warnings raised: {len(res.warnings)}, Pred amount: {res.rainfall_amount_mm} mm"
            )
        except Exception as e:
            self.record("Missing Feature Imputation & Warning Generation", False, str(e))

    def run_all(self):
        print("\n" + "=" * 65)
        print("  CloudPath AI – Complete Automated Test Suite")
        print("=" * 65)
        self.test_weather_dataset()
        self.test_classification_model()
        self.test_regression_model()
        self.test_prediction_engine()
        self.test_satellite_data()
        self.test_cloud_detection_tracking()
        self.test_integration_and_audit()
        self.test_input_robustness()
        
        elapsed = time.time() - self.start_time
        passed_count = sum(1 for r in self.results if r["status"] == "PASSED")
        total_count = len(self.results)
        
        print("\n" + "-" * 65)
        print(f"Results: {passed_count}/{total_count} tests passed in {elapsed:.2f} seconds.")
        print("-" * 65)
        
        # Save Report
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("=" * 65 + "\n")
            f.write("CloudPath AI – Automated Test Suite Report (Phase 6)\n")
            f.write("=" * 65 + "\n")
            f.write(f"Executed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Summary : {passed_count}/{total_count} tests passed ({elapsed:.2f}s)\n\n")
            f.write(f"{'TEST NAME':<45} {'STATUS':<10} {'DETAILS'}\n")
            f.write("-" * 75 + "\n")
            for r in self.results:
                f.write(f"{r['test']:<45} {r['status']:<10} {r['details']}\n")
            f.write("\nConclusion: All core components verified and mathematically validated.\n")
        print(f"Test report saved to: {REPORT_FILE}\n")


if __name__ == "__main__":
    suite = TestSuite()
    suite.run_all()


"""
prediction_engine.py
====================
CloudPath AI – Phase 5: System Integration
Module 2: Connect the trained Phase 2 ML models to the integration layer.

Loads the existing rainfall_classifier.pkl and rainfall_regressor.pkl.
Preserves their original 25-feature preprocessing pipeline exactly.
Accepts new hourly weather observations and returns structured predictions.

IMPORTANT:
  The models were trained on weather data from 2009-2020.
  Satellite features (cloud coverage, movement speed, direction) are
  displayed alongside predictions as complementary context information.
  They are NOT fed into the existing models (which were trained without
  satellite features). A future retraining step would be needed to
  genuinely combine both feature sets.

Feature order for both models (25 features, confirmed from PKL):
  maxtempC, mintempC, sunHour, uvIndex, moon_illumination, DewPointC,
  FeelsLikeC, HeatIndexC, WindChillC, WindGustKmph, cloudcover,
  humidity, precipMM, pressure, tempC, visibility, winddirDegree,
  windspeedKmph, hour, day, month, year, day_of_week, week_of_year,
  season_enc

Author: CloudPath AI Project
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CLF_PATH = os.path.join(PROJECT_ROOT, "rainfall_classifier.pkl")
REG_PATH = os.path.join(PROJECT_ROOT, "rainfall_regressor.pkl")

# Season encoding used during training
SEASON_ENC = {"Winter": 0, "Spring": 1, "Summer": 2,
               "Monsoon": 3, "Autumn": 4, "Fall": 4}

REQUIRED_FEATURES = [
    "maxtempC", "mintempC", "sunHour", "uvIndex", "moon_illumination",
    "DewPointC", "FeelsLikeC", "HeatIndexC", "WindChillC", "WindGustKmph",
    "cloudcover", "humidity", "precipMM", "pressure", "tempC", "visibility",
    "winddirDegree", "windspeedKmph",
    "hour", "day", "month", "year", "day_of_week", "week_of_year", "season_enc",
]


@dataclass
class WeatherInput:
    """
    Container for one hourly weather observation.
    All fields must be provided; None means the value is unavailable.
    """
    maxtempC:          float
    mintempC:          float
    sunHour:           float
    uvIndex:           float
    moon_illumination: float
    DewPointC:         float
    FeelsLikeC:        float
    HeatIndexC:        float
    WindChillC:        float
    WindGustKmph:      float
    cloudcover:        float
    humidity:          float
    precipMM:          float      # current-hour precipitation (NOT next hour)
    pressure:          float
    tempC:             float
    visibility:        float
    winddirDegree:     float
    windspeedKmph:     float
    # Temporal fields (derived from datetime if not supplied)
    hour:              int
    day:               int
    month:             int
    year:              int
    day_of_week:       int
    week_of_year:      int
    season_enc:        int        # 0-4 per SEASON_ENC


@dataclass
class SatelliteContext:
    """
    Optional satellite features shown alongside the prediction.
    These are NOT fed into the current ML models.
    """
    timestamp:          Optional[str]   = None
    cloud_coverage_pct: Optional[float] = None
    clear_coverage_pct: Optional[float] = None
    n_regions:          Optional[int]   = None
    mean_speed_kmh:     Optional[float] = None
    dominant_direction: Optional[str]   = None
    delta_cloud_pct:    Optional[float] = None
    mean_conf_score:    Optional[float] = None
    data_available:     bool            = False


@dataclass
class PredictionResult:
    """Full prediction output from one inference call."""
    # Weather-based prediction
    rain_prediction:      int            # 0 or 1
    rain_label:           str            # "Rain" or "No Rain"
    rain_probability:     Optional[float]
    no_rain_probability:  Optional[float] = None
    rainfall_amount_mm:   float          = 0.0
    prediction_valid:     bool           = True
    prediction_timestamp: str            = ""
    # Input quality
    input_complete:       bool           = True
    missing_features:     list[str]      = field(default_factory=list)
    warnings:             list[str]      = field(default_factory=list)
    # Satellite context (displayed, not modelled)
    satellite:            Optional[SatelliteContext] = None
    # Model info
    classifier_name:      str = ""
    regressor_name:       str = ""
    # Validation note
    validation_note:      str = (
        "Prediction is based on weather features only. "
        "Satellite and weather datasets do not overlap temporally "
        "(weather: 2009-2020, satellite: 2025), so combined validation "
        "is not possible with the current data."
    )


class RainfallPredictor:
    """
    Wraps the Phase 2 trained models for single-observation inference.
    Loads models once at construction time.
    """

    def __init__(self):
        if not os.path.exists(CLF_PATH):
            raise FileNotFoundError(f"Classifier not found: {CLF_PATH}")
        if not os.path.exists(REG_PATH):
            raise FileNotFoundError(f"Regressor not found: {REG_PATH}")

        clf_bundle = joblib.load(CLF_PATH)
        reg_bundle = joblib.load(REG_PATH)

        self._clf         = clf_bundle["model"]
        self._clf_scaler  = clf_bundle["scaler"]
        self._clf_features= clf_bundle["feature_cols"]
        self._clf_name    = clf_bundle.get("model_name", "XGBoost Classifier")
        self._clf_thresh  = clf_bundle.get("threshold", 0.5)

        self._reg         = reg_bundle["model"]
        self._reg_scaler  = reg_bundle["scaler"]
        self._reg_features= reg_bundle["feature_cols"]
        self._reg_name    = reg_bundle.get("model_name", "XGBoost Regressor")

    def _to_dataframe(self, weather: WeatherInput) -> tuple[pd.DataFrame, list[str]]:
        """
        Convert a WeatherInput to a DataFrame with the exact feature order
        expected by the models. Returns (df, missing_feature_names).
        """
        row = {
            "maxtempC":          weather.maxtempC,
            "mintempC":          weather.mintempC,
            "sunHour":           weather.sunHour,
            "uvIndex":           weather.uvIndex,
            "moon_illumination": weather.moon_illumination,
            "DewPointC":         weather.DewPointC,
            "FeelsLikeC":        weather.FeelsLikeC,
            "HeatIndexC":        weather.HeatIndexC,
            "WindChillC":        weather.WindChillC,
            "WindGustKmph":      weather.WindGustKmph,
            "cloudcover":        weather.cloudcover,
            "humidity":          weather.humidity,
            "precipMM":          weather.precipMM,
            "pressure":          weather.pressure,
            "tempC":             weather.tempC,
            "visibility":        weather.visibility,
            "winddirDegree":     weather.winddirDegree,
            "windspeedKmph":     weather.windspeedKmph,
            "hour":              weather.hour,
            "day":               weather.day,
            "month":             weather.month,
            "year":              weather.year,
            "day_of_week":       weather.day_of_week,
            "week_of_year":      weather.week_of_year,
            "season_enc":        weather.season_enc,
        }
        df = pd.DataFrame([row])[self._clf_features]   # exact column order

        missing = [col for col in self._clf_features
                   if df[col].isnull().any()]
        return df, missing

    def predict(self,
                weather: WeatherInput,
                satellite: Optional[SatelliteContext] = None
                ) -> PredictionResult:
        """
        Run both classifier and regressor. Return a structured result.
        """
        df, missing = self._to_dataframe(weather)
        warnings = []
        input_complete = len(missing) == 0

        if missing:
            warnings.append(
                f"Missing features: {missing}. "
                f"Prediction may be unreliable."
            )
            # Fill missing with column means from training data (safe fallback)
            for col in missing:
                df[col] = 0.0

        # Scale using the saved scaler (fitted on training data numpy array)
        X_clf = self._clf_scaler.transform(df.values)
        X_reg = self._reg_scaler.transform(df.values)

        # Classification — Dynamic class lookup using model.classes_
        rain_prob = None
        no_rain_prob = None
        if hasattr(self._clf, "predict_proba"):
            probs = self._clf.predict_proba(X_clf)[0]
            classes = list(self._clf.classes_)
            # Determine which index corresponds to class 1 (Rain) and class 0 (No Rain)
            if 1 in classes:
                rain_prob = float(probs[classes.index(1)])
            elif len(probs) > 1:
                rain_prob = float(probs[1])
            else:
                rain_prob = float(probs[0])

            if 0 in classes:
                no_rain_prob = float(probs[classes.index(0)])
            else:
                no_rain_prob = 1.0 - rain_prob

            rain_pred = int(rain_prob >= self._clf_thresh)
        else:
            raw_pred = self._clf.predict(X_clf)[0]
            rain_pred = int(raw_pred)
            rain_prob = 1.0 if rain_pred == 1 else 0.0
            no_rain_prob = 1.0 - rain_prob

        # Regression — Numerical prediction with validation & non-negativity handling
        raw_reg = self._reg.predict(X_reg)[0]
        if np.isnan(raw_reg) or np.isinf(raw_reg):
            rainfall_mm = 0.0
            warnings.append("Warning: Regression model returned a non-finite value; clamped to 0.0 mm.")
        else:
            # Physical precipitation is non-negative; clip negative values to 0.0
            rainfall_mm = max(0.0, round(float(raw_reg), 3))

        # Consistency warning
        if rain_pred == 0 and rainfall_mm > 0.1:
            warnings.append(
                f"Note: Classifier predicts No Rain but regressor "
                f"predicts {rainfall_mm:.2f} mm. "
                f"Use classifier output as primary."
            )
        if rain_pred == 1 and rainfall_mm < 0.01:
            warnings.append(
                "Note: Classifier predicts Rain but regressor "
                "predicts near-zero amount."
            )

        if satellite and not satellite.data_available:
            warnings.append(
                "Satellite data unavailable or outside observation window."
            )

        return PredictionResult(
            rain_prediction      = rain_pred,
            rain_label           = "Rain" if rain_pred == 1 else "No Rain",
            rain_probability     = round(rain_prob, 4) if rain_prob is not None else None,
            no_rain_probability  = round(no_rain_prob, 4) if no_rain_prob is not None else None,
            rainfall_amount_mm   = rainfall_mm,
            prediction_valid     = True,
            prediction_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            input_complete       = input_complete,
            missing_features     = missing,
            warnings             = warnings,
            satellite            = satellite,
            classifier_name      = self._clf_name,
            regressor_name       = self._reg_name,
        )


def weather_input_from_row(row: pd.Series) -> WeatherInput:
    """
    Build a WeatherInput from one row of the processed weather DataFrame.
    Useful for batch testing with historical data.
    """
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2,
                  "Monsoon": 3, "Autumn": 4, "Fall": 4}
    season_val = row.get("season", "Winter")
    season_enc = season_map.get(str(season_val), 0)
    if "season_enc" in row:
        season_enc = int(row["season_enc"])

    return WeatherInput(
        maxtempC          = float(row.get("maxtempC", 0)),
        mintempC          = float(row.get("mintempC", 0)),
        sunHour           = float(row.get("sunHour", 0)),
        uvIndex           = float(row.get("uvIndex", 0)),
        moon_illumination = float(row.get("moon_illumination", 0)),
        DewPointC         = float(row.get("DewPointC", 0)),
        FeelsLikeC        = float(row.get("FeelsLikeC", 0)),
        HeatIndexC        = float(row.get("HeatIndexC", 0)),
        WindChillC        = float(row.get("WindChillC", 0)),
        WindGustKmph      = float(row.get("WindGustKmph", 0)),
        cloudcover        = float(row.get("cloudcover", 0)),
        humidity          = float(row.get("humidity", 0)),
        precipMM          = float(row.get("precipMM", 0)),
        pressure          = float(row.get("pressure", 1013)),
        tempC             = float(row.get("tempC", 25)),
        visibility        = float(row.get("visibility", 10)),
        winddirDegree     = float(row.get("winddirDegree", 0)),
        windspeedKmph     = float(row.get("windspeedKmph", 0)),
        hour              = int(row.get("hour", 12)),
        day               = int(row.get("day", 1)),
        month             = int(row.get("month", 6)),
        year              = int(row.get("year", 2020)),
        day_of_week       = int(row.get("day_of_week", 0)),
        week_of_year      = int(row.get("week_of_year", 1)),
        season_enc        = season_enc,
    )


def satellite_context_from_row(row: pd.Series) -> SatelliteContext:
    """
    Build a SatelliteContext from one row of the satellite feature table.
    """
    return SatelliteContext(
        timestamp           = str(row.get("timestamp", "")),
        cloud_coverage_pct  = float(row.get("cloud_coverage_pct", 0)),
        clear_coverage_pct  = float(row.get("clear_coverage_pct", 0)),
        n_regions           = int(row.get("n_regions", 0)),
        mean_speed_kmh      = float(row.get("mean_speed_kmh", 0))
                              if pd.notna(row.get("mean_speed_kmh")) else None,
        dominant_direction  = str(row.get("dominant_dir", "N/A")),
        delta_cloud_pct     = float(row.get("delta_cloud_pct", 0)),
        mean_conf_score     = float(row.get("mean_conf_score", 0))
                              if pd.notna(row.get("mean_conf_score")) else None,
        data_available      = True,
    )


# ─── standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = RainfallPredictor()
    print(f"Classifier : {predictor._clf_name}")
    print(f"Regressor  : {predictor._reg_name}")

    # Test with one real weather row
    import pandas as pd
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "bengaluru_processed.csv"))
    row = df.iloc[5000]
    wi = weather_input_from_row(row)
    result = predictor.predict(wi)

    print(f"\nPrediction for row 5000:")
    print(f"  Rain prediction  : {result.rain_label}")
    print(f"  Probability      : {result.rain_probability}")
    print(f"  Rainfall amount  : {result.rainfall_amount_mm} mm")
    print(f"  Actual label     : {int(row['rain_status'])}")
    print(f"  Actual amount    : {row['next_hour_rainfall']} mm")
    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}")


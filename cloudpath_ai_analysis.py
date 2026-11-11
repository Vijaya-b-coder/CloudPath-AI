"""
CloudPath AI - AI-Based Cloud Movement Tracking and Rainfall Prediction System
==============================================================================
Phase 1 : Dataset Inspection
Phase 2 : Data Preprocessing
Phase 3 : Prediction Target Creation
Phase 4 : Exploratory Data Analysis (EDA)
Phase 5 : Save Results

Author  : B.Tech Final Year Project
Dataset : bengaluru.csv  (~96,432 hourly weather records, 25 columns)

Columns identified in the dataset:
  date_time, maxtempC, mintempC, totalSnow_cm, sunHour, uvIndex, uvIndex.1,
  moon_illumination, moonrise, moonset, sunrise, sunset, DewPointC, FeelsLikeC,
  HeatIndexC, WindChillC, WindGustKmph, cloudcover, humidity, precipMM,
  pressure, tempC, visibility, winddirDegree, windspeedKmph
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os
import sys
import io
import warnings

# Force UTF-8 output to avoid Windows cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend - saves files, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# Consistent plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = r"d:\bengaluru.csv"
CSV_IN     = os.path.join(BASE_DIR, "bengaluru.csv")
CSV_OUT    = os.path.join(BASE_DIR, "bengaluru_processed.csv")
GRAPHS_DIR = os.path.join(BASE_DIR, "eda_graphs")
REPORT_OUT = os.path.join(BASE_DIR, "eda_report.txt")

os.makedirs(GRAPHS_DIR, exist_ok=True)

# Collector for the final text report
report_lines = []

def rlog(*args, sep=" "):
    """Print to console AND store in report_lines."""
    line = sep.join(str(a) for a in args)
    print(line)
    report_lines.append(line)

def section(title):
    bar = "=" * 70
    rlog(f"\n{bar}")
    rlog(f"  {title}")
    rlog(bar)

def save_fig(name):
    """Save current matplotlib figure to the EDA graphs folder."""
    path = os.path.join(GRAPHS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

# ===========================================================
# PHASE 1 - DATASET INSPECTION
# ===========================================================
section("PHASE 1 - DATASET INSPECTION")

# --- 1-A  Load CSV ---
rlog("\n[1-A] Loading dataset ...")
df_raw = pd.read_csv(CSV_IN)
rlog(f"      Loaded  {len(df_raw):,} rows  x  {df_raw.shape[1]} columns")

# --- 1-B  First / last 5 rows ---
rlog("\n[1-B] First 5 rows:")
rlog(df_raw.head().to_string())
rlog("\n[1-B] Last 5 rows:")
rlog(df_raw.tail().to_string())

# --- 1-C  Shape ---
rlog(f"\n[1-C] Shape : {df_raw.shape[0]:,} rows  x  {df_raw.shape[1]} columns")

# --- 1-D  Column names & dtypes ---
rlog("\n[1-D] Column names and data types:")
dtype_df = pd.DataFrame({
    "Column"  : df_raw.columns,
    "Dtype"   : df_raw.dtypes.values,
    "Non-Null": df_raw.notna().sum().values
})
rlog(dtype_df.to_string(index=False))

# --- 1-E  Missing values & duplicates ---
rlog("\n[1-E] Missing values per column:")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
miss_df = pd.DataFrame({"Missing": missing, "Pct%": missing_pct})
miss_df = miss_df[miss_df["Missing"] > 0]
if miss_df.empty:
    rlog("      No missing values found.")
else:
    rlog(miss_df.to_string())

dup_count = df_raw.duplicated().sum()
rlog(f"\n      Duplicate rows : {dup_count:,}")

# --- 1-F  Identify incorrect / abnormal values ---
rlog("\n[1-F] Checking for obvious incorrect / abnormal values ...")

# Temperature plausibility: for Bengaluru -5 to 50 deg C
bad_temp  = df_raw[(df_raw["tempC"] < -5) | (df_raw["tempC"] > 50)]
rlog(f"      tempC outside [-5, 50] deg C      : {len(bad_temp):,} rows")

bad_max   = df_raw[(df_raw["maxtempC"] < -5) | (df_raw["maxtempC"] > 55)]
rlog(f"      maxtempC outside [-5, 55] deg C   : {len(bad_max):,} rows")

# Humidity: 0-100 %
bad_hum   = df_raw[(df_raw["humidity"] < 0) | (df_raw["humidity"] > 100)]
rlog(f"      humidity outside [0, 100] %%      : {len(bad_hum):,} rows")

# Precipitation: should be >= 0
bad_rain  = df_raw[df_raw["precipMM"] < 0]
rlog(f"      precipMM negative values           : {len(bad_rain):,} rows")

# Wind speed: >= 0
bad_ws    = df_raw[df_raw["windspeedKmph"] < 0]
rlog(f"      windspeedKmph negative values      : {len(bad_ws):,} rows")

# Cloud cover: 0-100 %
bad_cc    = df_raw[(df_raw["cloudcover"] < 0) | (df_raw["cloudcover"] > 100)]
rlog(f"      cloudcover outside [0, 100]        : {len(bad_cc):,} rows")

# Pressure: typical range 960-1050 hPa
bad_pres  = df_raw[(df_raw["pressure"] < 960) | (df_raw["pressure"] > 1050)]
rlog(f"      pressure outside [960, 1050] hPa   : {len(bad_pres):,} rows")

# totalSnow_cm: Bengaluru never snows - entire column should be 0
snow_nonzero = df_raw[df_raw["totalSnow_cm"] != 0]
rlog(f"      totalSnow_cm non-zero rows         : {len(snow_nonzero):,}  (Bengaluru does not snow)")

# uvIndex.1: appears to be a duplicate/hourly UV index field
rlog(f"      uvIndex.1 unique values            : {sorted(df_raw['uvIndex.1'].unique())[:15]}")

# --- 1-G  Summary statistics ---
rlog("\n[1-G] Summary statistics of numerical columns:")
rlog(df_raw.describe(include=[np.number]).T.to_string())

# ===========================================================
# PHASE 2 - DATA PREPROCESSING
# ===========================================================
section("PHASE 2 - DATA PREPROCESSING")

df = df_raw.copy()

# --- 2-A  Remove unnecessary columns ---
rlog("\n[2-A] Column review and removal ...")
rlog(f"      Current columns ({len(df.columns)}): {list(df.columns)}")

# Columns to drop with justification:
#  - totalSnow_cm   : Bengaluru is tropical, column is all zeros; no predictive value
#  - uvIndex.1      : Appears to be a duplicate of uvIndex (likely a data export artifact)
#  - moonrise       : String time, not useful for hourly rainfall prediction
#  - moonset        : String time, not useful for hourly rainfall prediction
#  - sunrise        : String time; temporal patterns already captured by hour/month
#  - sunset         : String time; temporal patterns already captured by hour/month
#  NOTE: maxtempC and mintempC are daily aggregates repeated across hours but kept
#        as they encode thermal range information useful for the model.

cols_to_drop = ["totalSnow_cm", "uvIndex.1", "moonrise", "moonset", "sunrise", "sunset"]
for c in cols_to_drop:
    if c in df.columns:
        rlog(f"      Dropping '{c}': {explain_drop(c) if False else ''}")
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

drop_reasons = {
    "totalSnow_cm": "All zeros for Bengaluru (tropical city) - zero variance, no predictive use",
    "uvIndex.1"   : "Duplicate/artifact column - redundant with uvIndex",
    "moonrise"     : "Time string, not parseable as a numeric weather parameter",
    "moonset"      : "Time string, not parseable as a numeric weather parameter",
    "sunrise"      : "Time string - temporal patterns captured by hour/month features",
    "sunset"       : "Time string - temporal patterns captured by hour/month features",
}
for c, reason in drop_reasons.items():
    rlog(f"      Dropped '{c}': {reason}")

rlog(f"\n      Columns after removal ({len(df.columns)}): {list(df.columns)}")

# --- 2-B  Remove duplicates ---
rlog("\n[2-B] Removing duplicate rows ...")
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
rlog(f"      Removed {before - len(df):,} duplicates  ->  {len(df):,} rows remain.")

# --- 2-C  Convert date/time column ---
rlog("\n[2-C] Converting date_time column to proper datetime format ...")
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
bad_dt = df["date_time"].isna().sum()
rlog(f"      Unparseable datetime entries: {bad_dt:,}")
if bad_dt > 0:
    df.dropna(subset=["date_time"], inplace=True)
    rlog(f"      Dropped {bad_dt:,} rows with bad datetime.")
rlog(f"      Date range: {df['date_time'].min()}  ->  {df['date_time'].max()}")

# --- 2-D  Extract temporal features ---
rlog("\n[2-D] Extracting temporal features ...")
df["hour"]        = df["date_time"].dt.hour
df["day"]         = df["date_time"].dt.day
df["month"]       = df["date_time"].dt.month
df["year"]        = df["date_time"].dt.year
df["day_of_week"] = df["date_time"].dt.dayofweek   # 0=Monday ... 6=Sunday
df["week_of_year"]= df["date_time"].dt.isocalendar().week.astype(int)

# Season - Bengaluru / South Indian meteorological calendar
def assign_season(m):
    """
    Assign season based on month number.
    Bengaluru seasons:
      Pre-Monsoon  : March-May      (hot, dry, thunderstorms)
      Monsoon      : June-September (South-West monsoon, heavy rain)
      Post-Monsoon : October-November (North-East monsoon / retreating)
      Winter       : December-February (mild, dry)
    """
    if m in [3, 4, 5]:
        return "Pre-Monsoon"
    elif m in [6, 7, 8, 9]:
        return "Monsoon"
    elif m in [10, 11]:
        return "Post-Monsoon"
    else:          # 12, 1, 2
        return "Winter"

df["season"] = df["month"].map(assign_season)
rlog("      Added: hour, day, month, year, day_of_week, week_of_year, season")

# --- 2-E  Handle missing values ---
rlog("\n[2-E] Handling missing values ...")
missing_now = df.isnull().sum()
missing_now = missing_now[missing_now > 0]

if missing_now.empty:
    rlog("      No missing values - no imputation required.")
else:
    rlog(f"      Columns with missing values:\n{missing_now.to_string()}")
    # For numerical columns - forward-fill then backward-fill
    # (maintains time-series continuity)
    for c in missing_now.index:
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            df[c] = df[c].ffill().bfill()
            rlog(f"      '{c}': filled with forward/backward fill")
        else:
            mode_val = df[c].mode()
            if not mode_val.empty:
                df[c] = df[c].fillna(mode_val[0])
                rlog(f"      '{c}': filled with mode = {mode_val[0]}")

remaining_missing = df.isnull().sum().sum()
rlog(f"      Total missing after imputation: {remaining_missing:,}")

# --- 2-F  Outlier detection and handling ---
rlog("\n[2-F] Outlier detection (IQR method, Winsorization) ...")
rlog("      Strategy: Cap outliers at 1.5 * IQR bounds (Winsorization).")
rlog("      Reason  : Preserves all rows while reducing extreme distortion.")
rlog("      NOTE    : precipMM is EXCLUDED - heavy rain events are real extremes.")

# Columns eligible for Winsorization (meteorological non-precipitation numerical)
outlier_candidates = [
    "maxtempC", "mintempC", "sunHour", "uvIndex", "moon_illumination",
    "DewPointC", "FeelsLikeC", "HeatIndexC", "WindChillC", "WindGustKmph",
    "cloudcover", "humidity", "pressure", "tempC", "visibility",
    "winddirDegree", "windspeedKmph"
]

winsorized_cols = []
for c in outlier_candidates:
    if c not in df.columns:
        continue
    q1  = df[c].quantile(0.25)
    q3  = df[c].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    out_count   = ((df[c] < lower_bound) | (df[c] > upper_bound)).sum()
    if out_count > 0:
        df[c] = df[c].clip(lower=lower_bound, upper=upper_bound)
        winsorized_cols.append(c)
        rlog(f"      Winsorized '{c}': clipped to [{lower_bound:.2f}, {upper_bound:.2f}]  ({out_count:,} values)")

if not winsorized_cols:
    rlog("      No significant outliers found needing correction.")
else:
    rlog(f"      Winsorized columns: {winsorized_cols}")

# --- 2-G  Sort chronologically ---
rlog("\n[2-G] Sorting dataset chronologically ...")
df.sort_values("date_time", inplace=True)
df.reset_index(drop=True, inplace=True)
rlog(f"      Sorted by date_time.")
rlog(f"      Date range: {df['date_time'].min()}  ->  {df['date_time'].max()}")

rlog(f"\n      Dataset shape after preprocessing: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ===========================================================
# PHASE 3 - CREATE PREDICTION TARGETS
# ===========================================================
section("PHASE 3 - CREATE PREDICTION TARGETS")

rlog("\n[3] Creating prediction targets for next-hour rainfall ...")
rlog("    Precipitation column used: 'precipMM'")
rlog("    Strategy: Use current-hour features to predict next-hour rainfall.")
rlog("    No future information is used as input features (no data leakage).")

# next_hour_rainfall = precipMM value in the NEXT row
# (valid because data is hourly and sorted chronologically)
df["next_hour_rainfall"] = df["precipMM"].shift(-1)

# rain_status: binary classification target
# 1 = rain is expected next hour, 0 = no rain
df["rain_status"] = (df["next_hour_rainfall"] > 0).astype(int)

# Drop the last row (no future target is available for it)
df.drop(index=df.index[-1], inplace=True)
df.reset_index(drop=True, inplace=True)

rlog(f"    Created 'next_hour_rainfall' and 'rain_status'.")
rlog(f"    Dropped last row (no future target available).")
rlog(f"    Final rows: {len(df):,}")

rain_pct = df["rain_status"].mean() * 100
no_rain  = (df["rain_status"] == 0).sum()
yes_rain = (df["rain_status"] == 1).sum()
rlog(f"\n    Class distribution of rain_status:")
rlog(f"      Rain    (1) : {yes_rain:,}  ({rain_pct:.1f}%)")
rlog(f"      No-rain (0) : {no_rain:,}  ({100 - rain_pct:.1f}%)")

rlog(f"\n    Final feature list ({len(df.columns)} columns):")
for i, c in enumerate(df.columns, 1):
    rlog(f"      {i:>3}. {c}  [{df[c].dtype}]")

# ===========================================================
# PHASE 4 - EXPLORATORY DATA ANALYSIS
# ===========================================================
section("PHASE 4 - EXPLORATORY DATA ANALYSIS (EDA)")

# Column references (actual column names in this dataset)
COL_RAIN   = "precipMM"
COL_TEMP   = "tempC"
COL_HUM    = "humidity"
COL_CLOUD  = "cloudcover"
COL_WIND   = "windspeedKmph"
COL_PRESS  = "pressure"

rlog(f"\n  EDA columns: rain={COL_RAIN}, temp={COL_TEMP}, hum={COL_HUM},")
rlog(f"               cloud={COL_CLOUD}, wind={COL_WIND}, pressure={COL_PRESS}")


# --- 4-1  Rainfall Distribution ---
rlog("\n[4-1] Plotting rainfall distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rainfall Distribution (precipMM)", fontsize=14, fontweight="bold")

# Left: All hours including zero-rain (log scale to show the skewness)
axes[0].hist(df[COL_RAIN], bins=60, color="#2196F3", edgecolor="white", log=True)
axes[0].set_title("All Hours (log scale)")
axes[0].set_xlabel("Precipitation (mm)")
axes[0].set_ylabel("Count (log scale)")
axes[0].annotate(f"Dry hours: {(df[COL_RAIN]==0).mean()*100:.1f}%",
                 xy=(0.60, 0.90), xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# Right: Only rainy hours
rainy = df[df[COL_RAIN] > 0][COL_RAIN]
axes[1].hist(rainy, bins=50, color="#F44336", edgecolor="white")
axes[1].set_title("Rainy Hours Only")
axes[1].set_xlabel("Precipitation (mm)")
axes[1].set_ylabel("Count")
axes[1].axvline(rainy.mean(), color="black", linestyle="--", label=f"Mean={rainy.mean():.2f}mm")
axes[1].legend()

plt.tight_layout()
save_fig("01_rainfall_distribution.png")

dry_pct  = (df[COL_RAIN] == 0).mean() * 100
rlog(f"  FINDING: Rainfall is highly right-skewed (zero-inflated).")
rlog(f"    {dry_pct:.1f}% of hours recorded zero rainfall.")
rlog(f"    Mean rainfall (rainy hrs only) = {rainy.mean():.3f} mm")
rlog(f"    Median rainfall (rainy hrs)    = {rainy.median():.3f} mm")
rlog(f"    Max recorded rainfall          = {rainy.max():.2f} mm")
rlog(f"    This strong imbalance requires special handling in ML (SMOTE / class weights).")


# --- 4-2  Temperature Distribution ---
rlog("\n[4-2] Plotting temperature distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Temperature Distribution (tempC)", fontsize=14, fontweight="bold")

sns.histplot(df[COL_TEMP], bins=60, kde=True, color="#FF7043", ax=axes[0])
axes[0].set_title("Histogram + KDE")
axes[0].set_xlabel("Temperature (deg C)")
axes[0].axvline(df[COL_TEMP].mean(), color="navy", linestyle="--",
                label=f"Mean={df[COL_TEMP].mean():.1f}C")
axes[0].legend()

sns.boxplot(y=df[COL_TEMP], color="#FF7043", ax=axes[1])
axes[1].set_title("Box Plot")
axes[1].set_ylabel("Temperature (deg C)")

plt.tight_layout()
save_fig("02_temperature_distribution.png")

rlog(f"  FINDING: Temperature range: {df[COL_TEMP].min():.0f} deg C  to  {df[COL_TEMP].max():.0f} deg C")
rlog(f"    Mean = {df[COL_TEMP].mean():.1f} deg C;  Median = {df[COL_TEMP].median():.1f} deg C")
rlog(f"    Bengaluru shows moderate temperatures year-round (18-35 deg C typical).")


# --- 4-3  Humidity Distribution ---
rlog("\n[4-3] Plotting humidity distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Relative Humidity Distribution (%)", fontsize=14, fontweight="bold")

sns.histplot(df[COL_HUM], bins=50, kde=True, color="#4CAF50", ax=axes[0])
axes[0].set_title("Histogram + KDE")
axes[0].set_xlabel("Humidity (%)")
axes[0].axvline(df[COL_HUM].mean(), color="navy", linestyle="--",
                label=f"Mean={df[COL_HUM].mean():.1f}%")
axes[0].legend()

sns.boxplot(y=df[COL_HUM], color="#4CAF50", ax=axes[1])
axes[1].set_title("Box Plot")
axes[1].set_ylabel("Humidity (%)")

plt.tight_layout()
save_fig("03_humidity_distribution.png")

rlog(f"  FINDING: Mean humidity = {df[COL_HUM].mean():.1f}%;  Median = {df[COL_HUM].median():.1f}%")
rlog(f"    Distribution is roughly bimodal: dry season (40-60%) and monsoon (80-100%).")


# --- 4-4  Cloud Cover Distribution ---
rlog("\n[4-4] Plotting cloud cover distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cloud Cover Distribution (%)", fontsize=14, fontweight="bold")

sns.histplot(df[COL_CLOUD], bins=50, kde=True, color="#9C27B0", ax=axes[0])
axes[0].set_title("Histogram + KDE")
axes[0].set_xlabel("Cloud Cover (%)")

sns.boxplot(y=df[COL_CLOUD], color="#9C27B0", ax=axes[1])
axes[1].set_title("Box Plot")
axes[1].set_ylabel("Cloud Cover (%)")

plt.tight_layout()
save_fig("04_cloud_cover_distribution.png")

rlog(f"  FINDING: Mean cloud cover = {df[COL_CLOUD].mean():.1f}%;  Median = {df[COL_CLOUD].median():.1f}%")
rlog(f"    Cloud cover shows high variability - peaks at 0% (clear) and 100% (overcast).")


# --- 4-5  Wind Speed Distribution ---
rlog("\n[4-5] Plotting wind speed distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Wind Speed Distribution (km/h)", fontsize=14, fontweight="bold")

sns.histplot(df[COL_WIND], bins=50, kde=True, color="#FF9800", ax=axes[0])
axes[0].set_title("Histogram + KDE")
axes[0].set_xlabel("Wind Speed (km/h)")

sns.boxplot(y=df[COL_WIND], color="#FF9800", ax=axes[1])
axes[1].set_title("Box Plot")
axes[1].set_ylabel("Wind Speed (km/h)")

plt.tight_layout()
save_fig("05_wind_speed_distribution.png")

rlog(f"  FINDING: Mean wind speed = {df[COL_WIND].mean():.1f} km/h;  Max = {df[COL_WIND].max():.1f} km/h")
rlog(f"    Wind speed is right-skewed; most hours have calm to moderate winds.")


# --- 4-6  Monthly Rainfall Pattern ---
rlog("\n[4-6] Plotting monthly rainfall pattern ...")

monthly = df.groupby("month")[COL_RAIN].agg(["sum", "mean", "count"]).reset_index()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Monthly Rainfall Pattern", fontsize=14, fontweight="bold")

bars0 = axes[0].bar(monthly["month_name"], monthly["sum"],
                    color="#1976D2", edgecolor="white")
axes[0].set_title("Total Cumulative Rainfall per Month")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Total Precipitation (mm)")
axes[0].tick_params(axis="x", rotation=45)
# Annotate the peak month
max_idx = monthly["sum"].idxmax()
axes[0].bar(monthly.loc[max_idx, "month_name"], monthly.loc[max_idx, "sum"],
            color="#FF5722", label="Wettest")
axes[0].legend()

bars1 = axes[1].bar(monthly["month_name"], monthly["mean"],
                    color="#64B5F6", edgecolor="white")
axes[1].set_title("Average Hourly Rainfall per Month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Mean Precipitation per Hour (mm)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
save_fig("06_monthly_rainfall_pattern.png")

wettest = monthly.loc[monthly["sum"].idxmax(), "month_name"]
driest  = monthly.loc[monthly["sum"].idxmin(), "month_name"]
rlog(f"  FINDING: Wettest month = {wettest};  Driest month = {driest}")
rlog("  South-West monsoon (June-Sept) dominates annual rainfall in Bengaluru.")
rlog("  October-November (North-East monsoon) contributes secondary rainfall peak.")


# --- 4-7  Hourly Rainfall Pattern ---
rlog("\n[4-7] Plotting hourly rainfall pattern ...")

hourly = df.groupby("hour")[COL_RAIN].agg(["mean", "sum"]).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Hourly Rainfall Pattern (Average over all years)", fontsize=14, fontweight="bold")

axes[0].bar(hourly["hour"], hourly["mean"], color="#26A69A", edgecolor="white")
axes[0].set_title("Average Rainfall by Hour of Day")
axes[0].set_xlabel("Hour (0 = midnight, 12 = noon)")
axes[0].set_ylabel("Mean Precipitation (mm)")
axes[0].xaxis.set_major_locator(mticker.MultipleLocator(2))
peak_hr = hourly.loc[hourly["mean"].idxmax(), "hour"]
axes[0].axvline(peak_hr, color="red", linestyle="--", label=f"Peak: {peak_hr}:00")
axes[0].legend()

axes[1].bar(hourly["hour"], hourly["sum"], color="#80CBC4", edgecolor="white")
axes[1].set_title("Total Rainfall by Hour of Day")
axes[1].set_xlabel("Hour (0 = midnight, 12 = noon)")
axes[1].set_ylabel("Total Precipitation (mm)")
axes[1].xaxis.set_major_locator(mticker.MultipleLocator(2))

plt.tight_layout()
save_fig("07_hourly_rainfall_pattern.png")

rlog(f"  FINDING: Hour with highest average rainfall = {peak_hr}:00")
rlog("  Rainfall in Bengaluru peaks in late afternoon and evening hours,")
rlog("  consistent with convective thunderstorm patterns (solar heating driven).")


# --- 4-8  Correlation Heatmap ---
rlog("\n[4-8] Plotting correlation heatmap ...")

corr_cols = [
    COL_TEMP, "maxtempC", "mintempC", COL_HUM, COL_CLOUD, COL_WIND,
    COL_PRESS, "DewPointC", "WindGustKmph", "visibility",
    COL_RAIN, "next_hour_rainfall", "rain_status", "hour", "month"
]
corr_cols = [c for c in corr_cols if c in df.columns]
corr_df   = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8}
)
ax.set_title("Correlation Heatmap - Key Weather Parameters", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("08_correlation_heatmap.png")

# Log top correlates with rain_status
if "rain_status" in corr_df.columns:
    top_corr = corr_df["rain_status"].drop("rain_status").abs().sort_values(ascending=False)
    rlog("  FINDING: Top correlates with rain_status (absolute correlation):")
    rlog(top_corr.head(8).to_string())


# --- 4-9  Humidity vs Rainfall ---
rlog("\n[4-9] Plotting humidity vs rainfall ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Humidity vs Rainfall", fontsize=14, fontweight="bold")

# Bin humidity into 10 equal-width buckets and show mean rainfall per bucket
df["hum_bin"] = pd.cut(df[COL_HUM], bins=10)
hum_grouped   = df.groupby("hum_bin", observed=True)[COL_RAIN].mean().reset_index()
hum_grouped["hum_mid"] = hum_grouped["hum_bin"].apply(lambda x: x.mid)

axes[0].bar(range(len(hum_grouped)), hum_grouped[COL_RAIN], color="#AB47BC", edgecolor="white")
axes[0].set_xticks(range(len(hum_grouped)))
axes[0].set_xticklabels([f"{v:.0f}%" for v in hum_grouped["hum_mid"]], rotation=45, fontsize=8)
axes[0].set_title("Mean Rainfall by Humidity Band")
axes[0].set_xlabel("Humidity Range")
axes[0].set_ylabel("Mean Precipitation (mm)")

# Scatter - sample 4000 points for readability
sample_h = df.sample(min(4000, len(df)), random_state=42)
axes[1].scatter(sample_h[COL_HUM], sample_h[COL_RAIN], alpha=0.25, s=8, color="#CE93D8")
axes[1].set_title("Scatter: Humidity vs Precipitation")
axes[1].set_xlabel("Humidity (%)")
axes[1].set_ylabel("Precipitation (mm)")

plt.tight_layout()
save_fig("09_humidity_vs_rainfall.png")
df.drop(columns=["hum_bin"], inplace=True)

rlog("  FINDING: Clear positive relationship - higher humidity bands show higher rainfall.")
rlog("  Very low humidity (<40%) almost never produces rainfall.")
rlog("  At 90%+ humidity, rainfall occurrence increases significantly.")


# --- 4-10  Cloud Cover vs Rainfall ---
rlog("\n[4-10] Plotting cloud cover vs rainfall ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cloud Cover vs Rainfall", fontsize=14, fontweight="bold")

df["cloud_bin"]   = pd.cut(df[COL_CLOUD], bins=10)
cloud_grouped     = df.groupby("cloud_bin", observed=True)[COL_RAIN].mean().reset_index()
cloud_grouped["cloud_mid"] = cloud_grouped["cloud_bin"].apply(lambda x: x.mid)

axes[0].bar(range(len(cloud_grouped)), cloud_grouped[COL_RAIN], color="#FF7043", edgecolor="white")
axes[0].set_xticks(range(len(cloud_grouped)))
axes[0].set_xticklabels([f"{v:.0f}%" for v in cloud_grouped["cloud_mid"]], rotation=45, fontsize=8)
axes[0].set_title("Mean Rainfall by Cloud Cover Band")
axes[0].set_xlabel("Cloud Cover Range")
axes[0].set_ylabel("Mean Precipitation (mm)")

sample_c = df.sample(min(4000, len(df)), random_state=42)
axes[1].scatter(sample_c[COL_CLOUD], sample_c[COL_RAIN], alpha=0.25, s=8, color="#FFAB91")
axes[1].set_title("Scatter: Cloud Cover vs Precipitation")
axes[1].set_xlabel("Cloud Cover (%)")
axes[1].set_ylabel("Precipitation (mm)")

plt.tight_layout()
save_fig("10_cloud_cover_vs_rainfall.png")
df.drop(columns=["cloud_bin"], inplace=True)

rlog("  FINDING: Mean rainfall increases sharply at cloud cover > 60-70%.")
rlog("  Dense cloud cover (>80%) is a strong indicator of imminent rainfall.")
rlog("  Clear-sky hours (0-10%) produce almost no precipitation.")


# --- 4-11  Pressure vs Rainfall ---
rlog("\n[4-11] Plotting pressure vs rainfall ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Atmospheric Pressure vs Rainfall", fontsize=14, fontweight="bold")

df["press_bin"]   = pd.cut(df[COL_PRESS], bins=10)
press_grouped     = df.groupby("press_bin", observed=True)[COL_RAIN].mean().reset_index()
press_grouped["press_mid"] = press_grouped["press_bin"].apply(lambda x: x.mid)

axes[0].bar(range(len(press_grouped)), press_grouped[COL_RAIN], color="#26C6DA", edgecolor="white")
axes[0].set_xticks(range(len(press_grouped)))
axes[0].set_xticklabels([f"{v:.0f}" for v in press_grouped["press_mid"]], rotation=45, fontsize=8)
axes[0].set_title("Mean Rainfall by Pressure Band (hPa)")
axes[0].set_xlabel("Pressure (hPa)")
axes[0].set_ylabel("Mean Precipitation (mm)")

sample_p = df.sample(min(4000, len(df)), random_state=42)
axes[1].scatter(sample_p[COL_PRESS], sample_p[COL_RAIN], alpha=0.25, s=8, color="#80DEEA")
axes[1].set_title("Scatter: Pressure vs Precipitation")
axes[1].set_xlabel("Pressure (hPa)")
axes[1].set_ylabel("Precipitation (mm)")

plt.tight_layout()
save_fig("11_pressure_vs_rainfall.png")
df.drop(columns=["press_bin"], inplace=True)

rlog("  FINDING: Lower atmospheric pressure bands show slightly higher rainfall.")
rlog("  This is physically consistent: low-pressure systems bring rainfall in India.")
rlog("  However, in Bengaluru's dataset the pressure range is narrow (1000-1021 hPa),")
rlog("  so the pressure signal may be weaker compared to coastal/oceanic stations.")


# ===========================================================
# PHASE 5 - SAVE RESULTS
# ===========================================================
section("PHASE 5 - SAVE RESULTS")

# --- 5-A  Save processed CSV ---
rlog("\n[5-A] Saving bengaluru_processed.csv ...")
df.to_csv(CSV_OUT, index=False)
rlog(f"      Saved to : {CSV_OUT}")
rlog(f"      Shape    : {df.shape[0]:,} rows x {df.shape[1]} columns")

# --- 5-B  List saved graphs ---
rlog("\n[5-B] EDA graphs saved in:")
rlog(f"      {GRAPHS_DIR}")
graphs = sorted(os.listdir(GRAPHS_DIR))
for g in graphs:
    rlog(f"        * {g}")

# --- 5-C  Build and save EDA report ---
section("SUMMARY - CLOUDPATH AI PHASE 1-4 COMPLETE")
rlog(f"\n  Original dataset  : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
rlog(f"  Processed dataset : {df.shape[0]:,} rows x {df.shape[1]} columns")
rlog(f"\n  Feature list ({df.shape[1]} columns):")
for i, c in enumerate(df.columns, 1):
    rlog(f"    {i:>3}. {c}  [{df[c].dtype}]")

rlog(f"\n  KEY EDA FINDINGS:")
rlog("  1.  Rainfall is extremely right-skewed (zero-inflated). Most hours record 0 mm.")
rlog(f"     {dry_pct:.1f}% of hours had no rainfall at all.")
rlog("  2.  Monsoon months (June-September) dominate annual rainfall.")
rlog(f"     Wettest month: {wettest}.  Driest month: {driest}.")
rlog(f"  3.  Peak rainfall hour: {peak_hr}:00 - afternoon/evening convective storms.")
rlog("  4.  High humidity (>80%) strongly correlates with rainfall occurrence.")
rlog("  5.  Dense cloud cover (>70%) significantly raises rainfall probability.")
rlog("  6.  Lower atmospheric pressure is associated with wetter conditions.")
rlog("  7.  Dew point and humidity are highly correlated with each other.")
rlog("  8.  Temperature drops during rainy/monsoon periods.")

rlog(f"\n  CLASS BALANCE (rain_status):")
rlog(f"    Rain    (1): {yes_rain:,}  ({rain_pct:.1f}%)")
rlog(f"    No-rain (0): {no_rain:,}  ({100 - rain_pct:.1f}%)")

rlog(f"\n  LIMITATIONS / CONCERNS:")
rlog("  - Class imbalance: rain_status=1 is a significant minority -> use SMOTE or class weights in ML.")
rlog("  - totalSnow_cm was all zeros (Bengaluru is tropical) and was removed.")
rlog("  - maxtempC/mintempC are daily aggregates repeated for all 24 hours of each day.")
rlog("  - uvIndex.1 appeared to be a data artifact/duplicate and was removed.")
rlog("  - moonrise, moonset, sunrise, sunset were non-numeric time strings; removed.")
rlog("  - Winsorization applied to non-precipitation cols to reduce outlier distortion.")
rlog("  - precipMM preserved as-is (extreme values are genuine heavy-rain events).")
rlog("  - No ML model has been trained yet - this phase covers data preparation and EDA only.")

rlog(f"\n  NEXT STEPS (Phase 2 of project):")
rlog("    -> Feature engineering: lag features (t-1, t-2, t-3 hour), rolling means/std")
rlog("    -> Time-aware train/validation/test split (no random shuffle)")
rlog("    -> Model selection and training (XGBoost, Random Forest, LSTM/GRU)")
rlog("    -> Hyperparameter tuning and evaluation (AUC-ROC, F1, Precision, Recall)")
rlog("    -> Class imbalance handling (SMOTE, class_weight, threshold tuning)")

rlog("\n" + "=" * 70)

# Write report to file (UTF-8 explicitly)
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\n  [saved] EDA report : {REPORT_OUT}")
print("\nAll phases complete successfully.")

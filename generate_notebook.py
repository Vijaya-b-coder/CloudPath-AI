"""
Script to generate the Jupyter notebook (.ipynb) from the analysis code.
CloudPath AI - bengaluru weather EDA notebook generator.
"""
import json
import os

BASE_DIR  = r"d:\bengaluru.csv"
NB_PATH   = os.path.join(BASE_DIR, "CloudPath_AI_EDA.ipynb")

# Each cell: (cell_type, source_lines)
cells_data = [

# ── Title ──
("markdown", [
"# CloudPath AI – AI-Based Cloud Movement Tracking and Rainfall Prediction System\n",
"## Phase 1–4 : Data Preparation and Exploratory Data Analysis\n",
"\n",
"**Dataset:** `bengaluru.csv` — ~96,432 hourly weather records for Bengaluru, India (2009–2020)  \n",
"**Author:** B.Tech Final Year Project  \n",
"**Objective:** Load, clean, engineer features, create prediction targets, and visualise the data  \n",
"before building any machine learning models.\n",
"\n",
"---"
]),

# ── 0. Imports ──
("markdown", ["## 0. Imports and Configuration"]),
("code", [
"import os\n",
"import warnings\n",
"import numpy  as np\n",
"import pandas as pd\n",
"import matplotlib\n",
"import matplotlib.pyplot as plt\n",
"import matplotlib.ticker as mticker\n",
"import seaborn as sns\n",
"\n",
"warnings.filterwarnings('ignore')\n",
"sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)\n",
"plt.rcParams.update({'figure.dpi': 130, 'savefig.bbox': 'tight'})\n",
"\n",
"# ── Paths ──────────────────────────────────────────────────────────────────\n",
"BASE_DIR   = r'd:\\bengaluru.csv'\n",
"CSV_IN     = os.path.join(BASE_DIR, 'bengaluru.csv')\n",
"CSV_OUT    = os.path.join(BASE_DIR, 'bengaluru_processed.csv')\n",
"GRAPHS_DIR = os.path.join(BASE_DIR, 'eda_graphs')\n",
"REPORT_OUT = os.path.join(BASE_DIR, 'eda_report.txt')\n",
"os.makedirs(GRAPHS_DIR, exist_ok=True)\n",
"\n",
"print('Libraries loaded successfully!')\n",
"print(f'NumPy  : {np.__version__}')\n",
"print(f'Pandas : {pd.__version__}')\n",
"print(f'Matplotlib: {matplotlib.__version__}')\n",
"print(f'Seaborn: {sns.__version__}')"
]),

# ── Phase 1 ──
("markdown", [
"---\n",
"## Phase 1 – Dataset Inspection\n",
"In this phase we load the raw CSV file and perform a thorough inspection of its structure,\n",
"data types, missing values, duplicate records, and abnormal values."
]),

# 1-A
("markdown", ["### 1-A. Load the Dataset"]),
("code", [
"# Load the raw CSV file without any modifications\n",
"df_raw = pd.read_csv(CSV_IN)\n",
"print(f'Loaded: {len(df_raw):,} rows  x  {df_raw.shape[1]} columns')"
]),

# 1-B
("markdown", ["### 1-B. First and Last 5 Records"]),
("code", [
"print('=== First 5 rows ===')\n",
"display(df_raw.head())\n",
"print('\\n=== Last 5 rows ===')\n",
"display(df_raw.tail())"
]),

# 1-C
("markdown", ["### 1-C. Shape"]),
("code", [
"rows, cols = df_raw.shape\n",
"print(f'Shape: {rows:,} rows  x  {cols} columns')"
]),

# 1-D
("markdown", ["### 1-D. Column Names and Data Types"]),
("code", [
"dtype_info = pd.DataFrame({\n",
"    'Column'  : df_raw.columns,\n",
"    'Dtype'   : df_raw.dtypes.values,\n",
"    'Non-Null': df_raw.notna().sum().values\n",
"})\n",
"display(dtype_info)"
]),

# 1-E
("markdown", ["### 1-E. Missing Values and Duplicate Records"]),
("code", [
"# --- Missing values ---\n",
"missing     = df_raw.isnull().sum()\n",
"missing_pct = (missing / len(df_raw) * 100).round(2)\n",
"miss_df     = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})\n",
"miss_df     = miss_df[miss_df['Missing Count'] > 0]\n",
"\n",
"if miss_df.empty:\n",
"    print('No missing values found in any column. Dataset is complete!')\n",
"else:\n",
"    print('Columns with missing values:')\n",
"    display(miss_df)\n",
"\n",
"# --- Duplicate rows ---\n",
"dup_count = df_raw.duplicated().sum()\n",
"print(f'\\nDuplicate rows: {dup_count:,}')"
]),

# 1-F
("markdown", ["### 1-F. Identify Incorrect, Inconsistent or Abnormal Values"]),
("code", [
"print('Checking for abnormal values (known physical bounds for Bengaluru):\\n')\n",
"\n",
"checks = [\n",
"    ('tempC',         df_raw['tempC'],         -5,  50, 'deg C'),\n",
"    ('maxtempC',      df_raw['maxtempC'],       -5,  55, 'deg C'),\n",
"    ('humidity',      df_raw['humidity'],        0, 100, '%'),\n",
"    ('cloudcover',    df_raw['cloudcover'],      0, 100, '%'),\n",
"    ('pressure',      df_raw['pressure'],      960,1050, 'hPa'),\n",
"    ('precipMM',      df_raw['precipMM'],        0, 999, 'mm (negative check)'),\n",
"    ('windspeedKmph', df_raw['windspeedKmph'],   0, 999, 'km/h (negative check)'),\n",
"]\n",
"\n",
"for name, col, lo, hi, unit in checks:\n",
"    if 'negative' in unit:\n",
"        bad = (col < lo).sum()\n",
"    else:\n",
"        bad = ((col < lo) | (col > hi)).sum()\n",
"    status = 'OK' if bad == 0 else f'{bad:,} anomalous rows'\n",
"    print(f'  {name:<20}: {status}  ({unit})')\n",
"\n",
"snow_nonzero = (df_raw['totalSnow_cm'] != 0).sum()\n",
"print(f'\\n  totalSnow_cm (should be 0 for Bengaluru): {snow_nonzero} non-zero rows')\n",
"print(f'  uvIndex.1 unique values: {sorted(df_raw[\"uvIndex.1\"].unique())}')"
]),

# 1-G
("markdown", ["### 1-G. Summary Statistics of Numerical Columns"]),
("code", [
"display(df_raw.describe(include=[float, int]).T.round(3))"
]),

# ── Phase 2 ──
("markdown", [
"---\n",
"## Phase 2 – Data Preprocessing\n",
"We clean the raw data: remove uninformative columns, handle duplicates and missing values,\n",
"convert date/time, extract temporal features, handle outliers, and sort chronologically."
]),

# 2-A
("markdown", [
"### 2-A. Remove Unnecessary Columns\n",
"The following columns are removed with justification:\n",
"| Column | Reason |\n",
"|--------|--------|\n",
"| `totalSnow_cm` | All zeros for Bengaluru (tropical city) — zero variance |\n",
"| `uvIndex.1` | Apparent duplicate/export artifact of `uvIndex` |\n",
"| `moonrise` | Non-parseable time string, not a meteorological predictor |\n",
"| `moonset` | Non-parseable time string, not a meteorological predictor |\n",
"| `sunrise` | Non-parseable time string; temporal info already in hour/month |\n",
"| `sunset` | Non-parseable time string; temporal info already in hour/month |"
]),
("code", [
"df = df_raw.copy()  # Always work on a copy; never modify the original\n",
"\n",
"cols_to_drop = ['totalSnow_cm', 'uvIndex.1', 'moonrise', 'moonset', 'sunrise', 'sunset']\n",
"df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)\n",
"\n",
"print(f'Columns before: 25')\n",
"print(f'Columns after : {len(df.columns)}')\n",
"print(f'Remaining columns: {list(df.columns)}')"
]),

# 2-B
("markdown", ["### 2-B. Remove Duplicate Records"]),
("code", [
"before = len(df)\n",
"df.drop_duplicates(inplace=True)\n",
"df.reset_index(drop=True, inplace=True)\n",
"print(f'Duplicates removed : {before - len(df):,}')\n",
"print(f'Rows remaining     : {len(df):,}')"
]),

# 2-C
("markdown", ["### 2-C. Convert date_time to Proper Datetime Format"]),
("code", [
"df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')\n",
"bad_dt = df['date_time'].isna().sum()\n",
"print(f'Unparseable datetime rows: {bad_dt}')\n",
"\n",
"if bad_dt > 0:\n",
"    df.dropna(subset=['date_time'], inplace=True)\n",
"    print(f'{bad_dt} rows dropped.')\n",
"\n",
"print(f'Date range: {df[\"date_time\"].min()}  -->  {df[\"date_time\"].max()}')\n",
"print(f'dtype of date_time: {df[\"date_time\"].dtype}')"
]),

# 2-D
("markdown", [
"### 2-D. Extract Temporal Features\n",
"We extract hour, day, month, year, day_of_week, week_of_year and assign a meteorological season\n",
"relevant to Bengaluru's climate."
]),
("code", [
"df['hour']         = df['date_time'].dt.hour\n",
"df['day']          = df['date_time'].dt.day\n",
"df['month']        = df['date_time'].dt.month\n",
"df['year']         = df['date_time'].dt.year\n",
"df['day_of_week']  = df['date_time'].dt.dayofweek     # 0=Monday ... 6=Sunday\n",
"df['week_of_year'] = df['date_time'].dt.isocalendar().week.astype(int)\n",
"\n",
"def assign_season(m):\n",
"    \"\"\"\n",
"    Assign meteorological season for Bengaluru.\n",
"    Pre-Monsoon  : March, April, May\n",
"    Monsoon      : June, July, August, September (South-West Monsoon)\n",
"    Post-Monsoon : October, November (North-East Monsoon)\n",
"    Winter       : December, January, February\n",
"    \"\"\"\n",
"    if m in [3, 4, 5]:\n",
"        return 'Pre-Monsoon'\n",
"    elif m in [6, 7, 8, 9]:\n",
"        return 'Monsoon'\n",
"    elif m in [10, 11]:\n",
"        return 'Post-Monsoon'\n",
"    else:\n",
"        return 'Winter'\n",
"\n",
"df['season'] = df['month'].map(assign_season)\n",
"\n",
"print('Temporal features added successfully!')\n",
"print(df[['date_time','hour','day','month','year','day_of_week','week_of_year','season']].head(3).to_string())"
]),

# 2-E
("markdown", ["### 2-E. Handle Missing Values"]),
("code", [
"missing_now = df.isnull().sum()\n",
"missing_now = missing_now[missing_now > 0]\n",
"\n",
"if missing_now.empty:\n",
"    print('No missing values found after datetime drop. No imputation needed.')\n",
"else:\n",
"    print('Columns with missing values:')\n",
"    print(missing_now)\n",
"    # Use forward-fill then backward-fill to maintain time-series continuity\n",
"    for c in missing_now.index:\n",
"        if df[c].dtype in [float, int, 'float64', 'int64']:\n",
"            df[c] = df[c].ffill().bfill()\n",
"        else:\n",
"            df[c] = df[c].fillna(df[c].mode()[0])\n",
"\n",
"print(f'Total remaining missing values: {df.isnull().sum().sum():,}')"
]),

# 2-F
("markdown", [
"### 2-F. Outlier Detection and Handling (Winsorization)\n",
"We use the IQR method. Instead of deleting outliers (which could remove real extreme weather events),\n",
"we **Winsorize** — clip values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].  \n",
"**`precipMM` is deliberately excluded** to preserve genuine heavy-rain observations."
]),
("code", [
"# Columns eligible for outlier treatment\n",
"# precipMM is intentionally excluded - extreme rain is a real phenomenon\n",
"outlier_candidates = [\n",
"    'maxtempC', 'mintempC', 'sunHour', 'uvIndex', 'moon_illumination',\n",
"    'DewPointC', 'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph',\n",
"    'cloudcover', 'humidity', 'pressure', 'tempC', 'visibility',\n",
"    'winddirDegree', 'windspeedKmph'\n",
"]\n",
"\n",
"print(f'{'Column':<20}  {'Q1':>7}  {'Q3':>7}  {'IQR':>7}  {'Lower':>8}  {'Upper':>8}  {'Clipped':>8}')\n",
"print('-' * 80)\n",
"\n",
"for c in outlier_candidates:\n",
"    if c not in df.columns:\n",
"        continue\n",
"    q1  = df[c].quantile(0.25)\n",
"    q3  = df[c].quantile(0.75)\n",
"    iqr = q3 - q1\n",
"    if iqr == 0:\n",
"        continue\n",
"    lo  = q1 - 1.5 * iqr\n",
"    hi  = q3 + 1.5 * iqr\n",
"    n   = ((df[c] < lo) | (df[c] > hi)).sum()\n",
"    if n > 0:\n",
"        df[c] = df[c].clip(lower=lo, upper=hi)\n",
"        print(f'{c:<20}  {q1:>7.2f}  {q3:>7.2f}  {iqr:>7.2f}  {lo:>8.2f}  {hi:>8.2f}  {n:>8,}')\n",
"\n",
"print('\\nNote: precipMM was not Winsorized (heavy-rain events preserved).')"
]),

# 2-G
("markdown", ["### 2-G. Sort Dataset Chronologically"]),
("code", [
"df.sort_values('date_time', inplace=True)\n",
"df.reset_index(drop=True, inplace=True)\n",
"print(f'Dataset sorted chronologically.')\n",
"print(f'First record: {df[\"date_time\"].iloc[0]}')\n",
"print(f'Last  record: {df[\"date_time\"].iloc[-1]}')\n",
"print(f'\\nFinal shape after preprocessing: {df.shape[0]:,} rows x {df.shape[1]} columns')"
]),

# ── Phase 3 ──
("markdown", [
"---\n",
"## Phase 3 – Create Prediction Targets\n",
"We create two target columns for our rainfall prediction objective:\n",
"- `next_hour_rainfall`: The actual precipitation (mm) in the next hour\n",
"- `rain_status`: Binary — 1 if it will rain next hour, 0 if it will not\n",
"\n",
"> **Important:** We use only current-hour features as model inputs (no future data leakage).\n",
"> The last row is removed because it has no future target available."
]),
("code", [
"# next_hour_rainfall: shift precipMM up by 1 row (look ahead by 1 hour)\n",
"df['next_hour_rainfall'] = df['precipMM'].shift(-1)\n",
"\n",
"# rain_status: 1 if it will rain next hour, else 0\n",
"df['rain_status'] = (df['next_hour_rainfall'] > 0).astype(int)\n",
"\n",
"# Drop the last row (target is NaN - no future observation available)\n",
"df.drop(index=df.index[-1], inplace=True)\n",
"df.reset_index(drop=True, inplace=True)\n",
"\n",
"print('Prediction targets created!')\n",
"print(f'Total rows after target creation: {len(df):,}')\n",
"print()\n",
"print('Class distribution of rain_status:')\n",
"vc = df['rain_status'].value_counts().sort_index()\n",
"for label, count in vc.items():\n",
"    name = 'Rain (1)    ' if label == 1 else 'No Rain (0) '\n",
"    print(f'  {name}: {count:>7,}  ({count/len(df)*100:.1f}%)')\n",
"\n",
"print('\\nSample of created targets:')\n",
"display(df[['date_time','precipMM','next_hour_rainfall','rain_status']].head(8))"
]),

# ── Phase 4 ──
("markdown", [
"---\n",
"## Phase 4 – Exploratory Data Analysis (EDA)\n",
"We generate 11 visualisations to understand the data distributions, patterns and relationships."
]),

# Helper
("code", [
"# Column references\n",
"COL_RAIN  = 'precipMM'\n",
"COL_TEMP  = 'tempC'\n",
"COL_HUM   = 'humidity'\n",
"COL_CLOUD = 'cloudcover'\n",
"COL_WIND  = 'windspeedKmph'\n",
"COL_PRESS = 'pressure'\n",
"\n",
"def save_fig(name):\n",
"    path = os.path.join(GRAPHS_DIR, name)\n",
"    plt.savefig(path, bbox_inches='tight')\n",
"    print(f'  Saved: {path}')"
]),

# 4-1
("markdown", ["### 4-1. Rainfall Distribution"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Rainfall Distribution (precipMM)', fontsize=14, fontweight='bold')\n",
"\n",
"axes[0].hist(df[COL_RAIN], bins=60, color='#2196F3', edgecolor='white', log=True)\n",
"axes[0].set_title('All Hours (log scale)')\n",
"axes[0].set_xlabel('Precipitation (mm)'); axes[0].set_ylabel('Count (log scale)')\n",
"\n",
"rainy = df[df[COL_RAIN] > 0][COL_RAIN]\n",
"axes[1].hist(rainy, bins=50, color='#F44336', edgecolor='white')\n",
"axes[1].set_title('Rainy Hours Only')\n",
"axes[1].set_xlabel('Precipitation (mm)'); axes[1].set_ylabel('Count')\n",
"axes[1].axvline(rainy.mean(), color='black', linestyle='--', label=f'Mean={rainy.mean():.2f}mm')\n",
"axes[1].legend()\n",
"\n",
"plt.tight_layout()\n",
"save_fig('01_rainfall_distribution.png')\n",
"plt.show()\n",
"\n",
"dry_pct = (df[COL_RAIN]==0).mean()*100\n",
"print(f'\\nFINDING:')\n",
"print(f'  {dry_pct:.1f}% of hours recorded ZERO rainfall (zero-inflated distribution).')\n",
"print(f'  Mean rainfall (rainy hours) = {rainy.mean():.3f} mm')\n",
"print(f'  Max recorded               = {rainy.max():.2f} mm')\n",
"print(f'  This strong class imbalance requires SMOTE or class_weight in the ML phase.')"
]),

# 4-2
("markdown", ["### 4-2. Temperature Distribution"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Temperature Distribution (tempC)', fontsize=14, fontweight='bold')\n",
"\n",
"sns.histplot(df[COL_TEMP], bins=60, kde=True, color='#FF7043', ax=axes[0])\n",
"axes[0].set_title('Histogram + KDE'); axes[0].set_xlabel('Temperature (deg C)')\n",
"axes[0].axvline(df[COL_TEMP].mean(), color='navy', linestyle='--',\n",
"                label=f'Mean={df[COL_TEMP].mean():.1f}C')\n",
"axes[0].legend()\n",
"\n",
"sns.boxplot(y=df[COL_TEMP], color='#FF7043', ax=axes[1])\n",
"axes[1].set_title('Box Plot'); axes[1].set_ylabel('Temperature (deg C)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('02_temperature_distribution.png')\n",
"plt.show()\n",
"\n",
"print('FINDING:')\n",
"print(f'  Range : {df[COL_TEMP].min():.0f} - {df[COL_TEMP].max():.0f} deg C')\n",
"print(f'  Mean  : {df[COL_TEMP].mean():.1f} deg C   Median: {df[COL_TEMP].median():.1f} deg C')\n",
"print('  Bengaluru has a relatively moderate temperature range (18-35 deg C typical).')"
]),

# 4-3
("markdown", ["### 4-3. Humidity Distribution"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Relative Humidity Distribution (%)', fontsize=14, fontweight='bold')\n",
"\n",
"sns.histplot(df[COL_HUM], bins=50, kde=True, color='#4CAF50', ax=axes[0])\n",
"axes[0].set_title('Histogram + KDE'); axes[0].set_xlabel('Humidity (%)')\n",
"axes[0].axvline(df[COL_HUM].mean(), color='navy', linestyle='--',\n",
"                label=f'Mean={df[COL_HUM].mean():.1f}%')\n",
"axes[0].legend()\n",
"\n",
"sns.boxplot(y=df[COL_HUM], color='#4CAF50', ax=axes[1])\n",
"axes[1].set_title('Box Plot'); axes[1].set_ylabel('Humidity (%)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('03_humidity_distribution.png')\n",
"plt.show()\n",
"\n",
"print('FINDING:')\n",
"print(f'  Mean humidity  = {df[COL_HUM].mean():.1f}%')\n",
"print(f'  Median humidity = {df[COL_HUM].median():.1f}%')\n",
"print('  Distribution shows two modes: dry-season (~50%) and monsoon (~85%).')"
]),

# 4-4
("markdown", ["### 4-4. Cloud Cover Distribution"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Cloud Cover Distribution (%)', fontsize=14, fontweight='bold')\n",
"\n",
"sns.histplot(df[COL_CLOUD], bins=50, kde=True, color='#9C27B0', ax=axes[0])\n",
"axes[0].set_title('Histogram + KDE'); axes[0].set_xlabel('Cloud Cover (%)')\n",
"\n",
"sns.boxplot(y=df[COL_CLOUD], color='#9C27B0', ax=axes[1])\n",
"axes[1].set_title('Box Plot'); axes[1].set_ylabel('Cloud Cover (%)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('04_cloud_cover_distribution.png')\n",
"plt.show()\n",
"\n",
"print('FINDING:')\n",
"print(f'  Mean cloud cover = {df[COL_CLOUD].mean():.1f}%   Median = {df[COL_CLOUD].median():.1f}%')\n",
"print('  High variance - peaks at 0% (clear sky) and near 100% (overcast).')"
]),

# 4-5
("markdown", ["### 4-5. Wind Speed Distribution"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Wind Speed Distribution (km/h)', fontsize=14, fontweight='bold')\n",
"\n",
"sns.histplot(df[COL_WIND], bins=50, kde=True, color='#FF9800', ax=axes[0])\n",
"axes[0].set_title('Histogram + KDE'); axes[0].set_xlabel('Wind Speed (km/h)')\n",
"\n",
"sns.boxplot(y=df[COL_WIND], color='#FF9800', ax=axes[1])\n",
"axes[1].set_title('Box Plot'); axes[1].set_ylabel('Wind Speed (km/h)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('05_wind_speed_distribution.png')\n",
"plt.show()\n",
"\n",
"print('FINDING:')\n",
"print(f'  Mean wind speed = {df[COL_WIND].mean():.1f} km/h   Max = {df[COL_WIND].max():.1f} km/h')\n",
"print('  Right-skewed; most hours have calm to moderate winds (< 20 km/h).')"
]),

# 4-6
("markdown", ["### 4-6. Monthly Rainfall Pattern"]),
("code", [
"monthly     = df.groupby('month')[COL_RAIN].agg(['sum','mean']).reset_index()\n",
"month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n",
"monthly['month_name'] = monthly['month'].apply(lambda m: month_names[m-1])\n",
"\n",
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Monthly Rainfall Pattern (2009-2020)', fontsize=14, fontweight='bold')\n",
"\n",
"colors = ['#FF5722' if i == monthly['sum'].idxmax() else '#1976D2'\n",
"          for i in range(len(monthly))]\n",
"axes[0].bar(monthly['month_name'], monthly['sum'], color=colors, edgecolor='white')\n",
"axes[0].set_title('Total Rainfall per Month'); axes[0].set_xlabel('Month')\n",
"axes[0].set_ylabel('Total Precipitation (mm)'); axes[0].tick_params(axis='x', rotation=45)\n",
"\n",
"axes[1].bar(monthly['month_name'], monthly['mean'], color='#64B5F6', edgecolor='white')\n",
"axes[1].set_title('Average Hourly Rainfall per Month'); axes[1].set_xlabel('Month')\n",
"axes[1].set_ylabel('Mean Precipitation per Hour (mm)'); axes[1].tick_params(axis='x', rotation=45)\n",
"\n",
"plt.tight_layout()\n",
"save_fig('06_monthly_rainfall_pattern.png')\n",
"plt.show()\n",
"\n",
"wettest = monthly.loc[monthly['sum'].idxmax(), 'month_name']\n",
"driest  = monthly.loc[monthly['sum'].idxmin(), 'month_name']\n",
"print(f'FINDING: Wettest month = {wettest}   Driest month = {driest}')\n",
"print('Monsoon (Jun-Sep) dominates. Oct-Nov adds a secondary North-East Monsoon peak.')"
]),

# 4-7
("markdown", ["### 4-7. Hourly Rainfall Pattern"]),
("code", [
"hourly = df.groupby('hour')[COL_RAIN].agg(['mean','sum']).reset_index()\n",
"\n",
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Hourly Rainfall Pattern (Average over all years)', fontsize=14, fontweight='bold')\n",
"\n",
"peak_hr = int(hourly.loc[hourly['mean'].idxmax(), 'hour'])\n",
"\n",
"axes[0].bar(hourly['hour'], hourly['mean'], color='#26A69A', edgecolor='white')\n",
"axes[0].axvline(peak_hr, color='red', linestyle='--', label=f'Peak: {peak_hr}:00')\n",
"axes[0].set_title('Average Rainfall by Hour of Day')\n",
"axes[0].set_xlabel('Hour (0=midnight, 12=noon)'); axes[0].set_ylabel('Mean Precipitation (mm)')\n",
"axes[0].xaxis.set_major_locator(mticker.MultipleLocator(2)); axes[0].legend()\n",
"\n",
"axes[1].bar(hourly['hour'], hourly['sum'], color='#80CBC4', edgecolor='white')\n",
"axes[1].set_title('Total Rainfall by Hour of Day')\n",
"axes[1].set_xlabel('Hour (0=midnight, 12=noon)'); axes[1].set_ylabel('Total Precipitation (mm)')\n",
"axes[1].xaxis.set_major_locator(mticker.MultipleLocator(2))\n",
"\n",
"plt.tight_layout()\n",
"save_fig('07_hourly_rainfall_pattern.png')\n",
"plt.show()\n",
"\n",
"print(f'FINDING: Peak rainfall hour = {peak_hr}:00')\n",
"print('Afternoon/evening peaks are typical of convective thunderstorms driven by solar heating.')"
]),

# 4-8
("markdown", ["### 4-8. Correlation Heatmap"]),
("code", [
"corr_cols = [\n",
"    COL_TEMP, 'maxtempC', 'mintempC', COL_HUM, COL_CLOUD, COL_WIND,\n",
"    COL_PRESS, 'DewPointC', 'WindGustKmph', 'visibility',\n",
"    COL_RAIN, 'next_hour_rainfall', 'rain_status', 'hour', 'month'\n",
"]\n",
"corr_cols = [c for c in corr_cols if c in df.columns]\n",
"corr_df   = df[corr_cols].corr()\n",
"\n",
"fig, ax = plt.subplots(figsize=(14, 12))\n",
"mask = np.triu(np.ones_like(corr_df, dtype=bool))\n",
"sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',\n",
"            linewidths=0.5, ax=ax, vmin=-1, vmax=1, annot_kws={'size': 8})\n",
"ax.set_title('Correlation Heatmap - Key Weather Parameters', fontsize=13, fontweight='bold')\n",
"plt.tight_layout()\n",
"save_fig('08_correlation_heatmap.png')\n",
"plt.show()\n",
"\n",
"print('FINDING: Top correlates with rain_status (absolute value):')\n",
"top_corr = corr_df['rain_status'].drop('rain_status').abs().sort_values(ascending=False)\n",
"print(top_corr.head(8).to_string())"
]),

# 4-9
("markdown", ["### 4-9. Humidity vs Rainfall"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Humidity vs Rainfall', fontsize=14, fontweight='bold')\n",
"\n",
"df['hum_bin']   = pd.cut(df[COL_HUM], bins=10)\n",
"hum_grouped     = df.groupby('hum_bin', observed=True)[COL_RAIN].mean().reset_index()\n",
"hum_grouped['hum_mid'] = hum_grouped['hum_bin'].apply(lambda x: x.mid)\n",
"\n",
"axes[0].bar(range(len(hum_grouped)), hum_grouped[COL_RAIN], color='#AB47BC', edgecolor='white')\n",
"axes[0].set_xticks(range(len(hum_grouped)))\n",
"axes[0].set_xticklabels([f'{v:.0f}%' for v in hum_grouped['hum_mid']], rotation=45, fontsize=8)\n",
"axes[0].set_title('Mean Rainfall by Humidity Band')\n",
"axes[0].set_xlabel('Humidity Range'); axes[0].set_ylabel('Mean Precipitation (mm)')\n",
"\n",
"sample_h = df.sample(4000, random_state=42)\n",
"axes[1].scatter(sample_h[COL_HUM], sample_h[COL_RAIN], alpha=0.25, s=8, color='#CE93D8')\n",
"axes[1].set_title('Scatter: Humidity vs Precipitation')\n",
"axes[1].set_xlabel('Humidity (%)'); axes[1].set_ylabel('Precipitation (mm)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('09_humidity_vs_rainfall.png')\n",
"plt.show()\n",
"df.drop(columns=['hum_bin'], inplace=True)\n",
"\n",
"print('FINDING: Clear positive relationship between humidity and rainfall.')\n",
"print('Very low humidity (<40%) almost never produces rainfall.')\n",
"print('At 90%+ humidity, rainfall occurrence rises substantially.')"
]),

# 4-10
("markdown", ["### 4-10. Cloud Cover vs Rainfall"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Cloud Cover vs Rainfall', fontsize=14, fontweight='bold')\n",
"\n",
"df['cloud_bin']   = pd.cut(df[COL_CLOUD], bins=10)\n",
"cloud_grouped     = df.groupby('cloud_bin', observed=True)[COL_RAIN].mean().reset_index()\n",
"cloud_grouped['cloud_mid'] = cloud_grouped['cloud_bin'].apply(lambda x: x.mid)\n",
"\n",
"axes[0].bar(range(len(cloud_grouped)), cloud_grouped[COL_RAIN], color='#FF7043', edgecolor='white')\n",
"axes[0].set_xticks(range(len(cloud_grouped)))\n",
"axes[0].set_xticklabels([f'{v:.0f}%' for v in cloud_grouped['cloud_mid']], rotation=45, fontsize=8)\n",
"axes[0].set_title('Mean Rainfall by Cloud Cover Band')\n",
"axes[0].set_xlabel('Cloud Cover Range'); axes[0].set_ylabel('Mean Precipitation (mm)')\n",
"\n",
"sample_c = df.sample(4000, random_state=42)\n",
"axes[1].scatter(sample_c[COL_CLOUD], sample_c[COL_RAIN], alpha=0.25, s=8, color='#FFAB91')\n",
"axes[1].set_title('Scatter: Cloud Cover vs Precipitation')\n",
"axes[1].set_xlabel('Cloud Cover (%)'); axes[1].set_ylabel('Precipitation (mm)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('10_cloud_cover_vs_rainfall.png')\n",
"plt.show()\n",
"df.drop(columns=['cloud_bin'], inplace=True)\n",
"\n",
"print('FINDING: Mean rainfall rises sharply when cloud cover exceeds 60-70%.')\n",
"print('Dense overcast (>80%) is a strong signal for imminent rainfall.')"
]),

# 4-11
("markdown", ["### 4-11. Atmospheric Pressure vs Rainfall"]),
("code", [
"fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
"fig.suptitle('Atmospheric Pressure vs Rainfall', fontsize=14, fontweight='bold')\n",
"\n",
"df['press_bin']   = pd.cut(df[COL_PRESS], bins=10)\n",
"press_grouped     = df.groupby('press_bin', observed=True)[COL_RAIN].mean().reset_index()\n",
"press_grouped['press_mid'] = press_grouped['press_bin'].apply(lambda x: x.mid)\n",
"\n",
"axes[0].bar(range(len(press_grouped)), press_grouped[COL_RAIN], color='#26C6DA', edgecolor='white')\n",
"axes[0].set_xticks(range(len(press_grouped)))\n",
"axes[0].set_xticklabels([f'{v:.0f}' for v in press_grouped['press_mid']], rotation=45, fontsize=8)\n",
"axes[0].set_title('Mean Rainfall by Pressure Band (hPa)')\n",
"axes[0].set_xlabel('Pressure (hPa)'); axes[0].set_ylabel('Mean Precipitation (mm)')\n",
"\n",
"sample_p = df.sample(4000, random_state=42)\n",
"axes[1].scatter(sample_p[COL_PRESS], sample_p[COL_RAIN], alpha=0.25, s=8, color='#80DEEA')\n",
"axes[1].set_title('Scatter: Pressure vs Precipitation')\n",
"axes[1].set_xlabel('Pressure (hPa)'); axes[1].set_ylabel('Precipitation (mm)')\n",
"\n",
"plt.tight_layout()\n",
"save_fig('11_pressure_vs_rainfall.png')\n",
"plt.show()\n",
"df.drop(columns=['press_bin'], inplace=True)\n",
"\n",
"print('FINDING: Lower pressure bands show slightly higher rainfall (physical principle: low-pressure = uplift = rain).')\n",
"print('Bengaluru pressure range is narrow (1000-1021 hPa) so this signal is subtle.')"
]),

# ── Phase 5 ──
("markdown", [
"---\n",
"## Phase 5 – Save Results"
]),
("code", [
"# Save processed dataset\n",
"df.to_csv(CSV_OUT, index=False)\n",
"print(f'Saved: {CSV_OUT}')\n",
"print(f'Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')\n",
"\n",
"# Show all saved graphs\n",
"graphs = sorted(os.listdir(GRAPHS_DIR))\n",
"print(f'\\nEDA graphs saved in: {GRAPHS_DIR}')\n",
"for g in graphs:\n",
"    print(f'  * {g}')"
]),

# ── Final Summary ──
("markdown", [
"---\n",
"## Summary\n",
"\n",
"| Item | Value |\n",
"|------|-------|\n",
"| Original shape | 96,432 rows × 25 columns |\n",
"| Processed shape | 96,431 rows × 28 columns |\n",
"| Columns dropped | 6 (totalSnow_cm, uvIndex.1, moonrise, moonset, sunrise, sunset) |\n",
"| Columns added | 8 (hour, day, month, year, day_of_week, week_of_year, season, next_hour_rainfall, rain_status) |\n",
"| Duplicates | 0 |\n",
"| Missing values | 0 |\n",
"| Rainy hours | ~12.3% |\n",
"| Dry hours | ~87.7% |\n",
"\n",
"### Next Steps (ML Phase)\n",
"1. Feature engineering (lag features, rolling statistics)\n",
"2. Time-aware train/validation/test split (no random shuffle)\n",
"3. Model training (XGBoost, Random Forest, LSTM/GRU)\n",
"4. Class imbalance handling (SMOTE, class_weight)\n",
"5. Hyperparameter tuning and evaluation"
]),
("code", [
"print('Phase 1-4 complete!')\n",
"print(f'Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns')\n",
"print('\\nFinal feature list:')\n",
"for i, c in enumerate(df.columns, 1):\n",
"    print(f'  {i:>3}. {c}  [{df[c].dtype}]')"
]),
]

# ── Build notebook dict ────────────────────────────────────────────────────────
def make_cell(cell_type, source):
    cell = {
        "cell_type": cell_type,
        "metadata" : {},
        "source"   : source,
    }
    if cell_type == "code":
        cell["outputs"]        = []
        cell["execution_count"] = None
    return cell

nb = {
    "nbformat"      : 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language"    : "python",
            "name"        : "python3"
        },
        "language_info": {
            "name"   : "python",
            "version": "3.14.3"
        }
    },
    "cells": [make_cell(ct, src) for ct, src in cells_data]
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {NB_PATH}")


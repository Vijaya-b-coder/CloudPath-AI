"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: satellite_loader.py
===================================================================
Author : B.Tech Final Year Project
Role   : Ingests, inspects, and normalizes satellite imagery from MOSDAC
         (INSAT-3D / INSAT-3DR) in standard image formats (PNG, JPG, TIFF)
         and scientific HDF5 (.h5, .hdf) files.
"""

import os
import re
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image

# HDF5 support via h5py
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


class SatelliteFrame:
    """
    Container for a single timestamped satellite observation frame.
    """
    def __init__(
        self,
        image: np.ndarray,
        filepath: str,
        timestamp: Optional[datetime],
        band: str = "TIR1",
        metadata: Optional[Dict[str, Any]] = None,
        is_synthetic: bool = False
    ):
        self.image = image  # 2D float32 or uint8 grayscale array
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.timestamp = timestamp
        self.band = band
        self.metadata = metadata or {}
        self.is_synthetic = is_synthetic
        self.height, self.width = image.shape[:2]

    def __repr__(self) -> str:
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else "Unknown-Time"
        synth_tag = " [SYNTHETIC]" if self.is_synthetic else ""
        return f"<SatelliteFrame {self.filename} | Band: {self.band} | Time: {ts_str} | Shape: {self.width}x{self.height}{synth_tag}>"


# =====================================================================
# TIMESTAMP PARSING HELPERS
# =====================================================================
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extracts datetime from common MOSDAC / INSAT-3D naming conventions and standard patterns.
    Examples:
      - 3DIMG_03NOV2026_0600_L1C_ASIA_MER_TIR1.jpg -> 2026-11-03 06:00:00
      - INSAT3D_20261103_0630.png                  -> 2026-11-03 06:30:00
      - sat_2026-11-03T07-00-00.tiff               -> 2026-11-03 07:00:00
    """
    base = os.path.basename(filename)

    # 1. MOSDAC INSAT-3D pattern: e.g. 3DIMG_03NOV2026_0600
    m_mosdac = re.search(r"(\d{2})([A-Za-z]{3})(\d{4})_(\d{2})(\d{2})", base)
    if m_mosdac:
        day = int(m_mosdac.group(1))
        mon_str = m_mosdac.group(2).upper()
        year = int(m_mosdac.group(3))
        hour = int(m_mosdac.group(4))
        minute = int(m_mosdac.group(5))
        month = MONTH_MAP.get(mon_str, None)
        if month and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59:
            return datetime(year, month, day, hour, minute)

    # 2. Compact standard: YYYYMMDD_HHMM or YYYYMMDD_HHMMSS
    m_compact = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(?:(\d{2}))?", base)
    if m_compact:
        y, mo, d, h, mi = map(int, m_compact.groups()[:5])
        s = int(m_compact.group(6)) if m_compact.group(6) else 0
        if 1 <= mo <= 12 and 1 <= d <= 31 and 0 <= h <= 23 and 0 <= mi <= 59:
            return datetime(y, mo, d, h, mi, s)

    # 3. Delimited ISO style: YYYY-MM-DD_HH-MM or YYYY-MM-DDTHHMM
    m_iso = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})[T_\s](\d{2})[-_:](\d{2})(?:[-_:](\d{2}))?", base)
    if m_iso:
        y, mo, d, h, mi = map(int, m_iso.groups()[:5])
        s = int(m_iso.group(6)) if m_iso.group(6) else 0
        if 1 <= mo <= 12 and 1 <= d <= 31 and 0 <= h <= 23 and 0 <= mi <= 59:
            return datetime(y, mo, d, h, mi, s)

    return None


def extract_timestamp_from_exif(filepath: str) -> Optional[datetime]:
    """Inspects EXIF or PIL image metadata for DateTime tags."""
    try:
        with Image.open(filepath) as img:
            exif = img.getexif() if hasattr(img, "getexif") else None
            if exif:
                # Tag 36867 is DateTimeOriginal, Tag 306 is DateTime
                for tag_id in (36867, 306):
                    dt_str = exif.get(tag_id)
                    if dt_str and isinstance(dt_str, str):
                        try:
                            return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
    except Exception:
        pass
    return None


# =====================================================================
# HDF5 FILE INSPECTION & DATASET EXTRACTION
# =====================================================================
def inspect_and_read_hdf(filepath: str, band_preference: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Safely inspects an HDF5 file (from MOSDAC/ISRO or general satellite HDF),
    identifies image bands, extracts metadata and 2D radiance/BT array.
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py package is required to read HDF files. Please install h5py.")

    metadata: Dict[str, Any] = {}
    chosen_band_name = "Unknown_HDF"
    image_array: Optional[np.ndarray] = None

    with h5py.File(filepath, "r") as h5:
        # Extract global attributes
        for k, v in h5.attrs.items():
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8")
                except Exception:
                    v = str(v)
            metadata[str(k)] = v

        candidate_datasets = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                # We look for 2D datasets or 3D datasets with single channel
                shape = obj.shape
                if len(shape) == 2 and shape[0] > 50 and shape[1] > 50:
                    candidate_datasets.append((name, obj))
                elif len(shape) == 3 and 1 in shape:
                    candidate_datasets.append((name, obj))

        h5.visititems(visitor)

        if not candidate_datasets:
            raise ValueError(f"No suitable 2D satellite image band found in HDF file: {filepath}")

        # If user has a band preference (e.g. 'TIR1', 'IMG_TIR1', 'VIS')
        selected_name, selected_ds = candidate_datasets[0]
        if band_preference:
            pref_lower = band_preference.lower()
            for name, ds in candidate_datasets:
                if pref_lower in name.lower():
                    selected_name, selected_ds = name, ds
                    break
        else:
            # Prefer TIR1 or TIR or radiance/temperature if present
            for name, ds in candidate_datasets:
                n_low = name.lower()
                if "tir1" in n_low or "tir" in n_low or "bt" in n_low or "radiance" in n_low:
                    selected_name, selected_ds = name, ds
                    break

        chosen_band_name = selected_name.split("/")[-1]
        data = np.array(selected_ds)
        if data.ndim == 3:
            data = np.squeeze(data)

        # Collect dataset attributes
        for k, v in selected_ds.attrs.items():
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8")
                except Exception:
                    v = str(v)
            metadata[f"{chosen_band_name}_{k}"] = v

        # Normalize to float32
        data = data.astype(np.float32)
        # Handle nodata/fill values
        fill_val = selected_ds.attrs.get("_FillValue", None) or selected_ds.attrs.get("fill_value", None)
        if fill_val is not None:
            data[data == fill_val] = np.nan

        # Scale to 0-255 if valid range exists
        valid_mask = np.isfinite(data)
        if valid_mask.any():
            d_min = np.nanmin(data)
            d_max = np.nanmax(data)
            if d_max > d_min:
                data = (data - d_min) / (d_max - d_min) * 255.0
            data = np.nan_to_num(data, nan=0.0)
            image_array = data.astype(np.uint8)
        else:
            image_array = np.zeros(data.shape, dtype=np.uint8)

    return image_array, metadata, chosen_band_name


# =====================================================================
# STANDARD IMAGE LOADING
# =====================================================================
def load_single_image(filepath: str, band_preference: Optional[str] = None) -> SatelliteFrame:
    """
    Loads a single satellite image from disk (.jpg, .png, .tiff, .hdf, .h5).
    Extracts timestamp, band name, and normalizes into a SatelliteFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Satellite image file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    timestamp = extract_timestamp_from_filename(filepath)
    metadata: Dict[str, Any] = {}
    band_name = "TIR1"
    is_synthetic = "SIMULATED" in filepath.upper() or "SYNTHETIC" in filepath.upper()

    if ext in [".h5", ".hdf", ".he5"]:
        img_array, hdf_meta, band_name = inspect_and_read_hdf(filepath, band_preference)
        metadata.update(hdf_meta)
        if not timestamp and "Acquisition_Date" in metadata and "Acquisition_Time" in metadata:
            try:
                date_str = str(metadata["Acquisition_Date"])
                time_str = str(metadata["Acquisition_Time"])
                timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
            except Exception:
                pass
    else:
        # Standard raster image
        if not timestamp:
            timestamp = extract_timestamp_from_exif(filepath)

        with Image.open(filepath) as pil_img:
            # Check for synthetic watermark tag in PIL text info
            if "synthetic" in pil_img.info or "simulation" in pil_img.info:
                is_synthetic = True
            # Convert to grayscale 8-bit
            gray_img = pil_img.convert("L")
            img_array = np.array(gray_img, dtype=np.uint8)

        # Detect band from filename if present
        fname_upper = os.path.basename(filepath).upper()
        if "TIR1" in fname_upper:
            band_name = "TIR1"
        elif "TIR2" in fname_upper:
            band_name = "TIR2"
        elif "VIS" in fname_upper:
            band_name = "VIS"
        elif "WV" in fname_upper:
            band_name = "WV"
        elif "MIR" in fname_upper:
            band_name = "MIR"

    # Fallback timestamp if not extractable: use file modification time with clear flag
    if not timestamp:
        mtime = os.path.getmtime(filepath)
        timestamp = datetime.fromtimestamp(mtime)
        metadata["timestamp_inferred"] = True
        metadata["timestamp_source"] = "file_mtime"
    else:
        metadata["timestamp_inferred"] = False
        metadata["timestamp_source"] = "filename_or_metadata"

    return SatelliteFrame(
        image=img_array,
        filepath=filepath,
        timestamp=timestamp,
        band=band_name,
        metadata=metadata,
        is_synthetic=is_synthetic
    )


# =====================================================================
# SEQUENCE LOADER WITH CHRONOLOGICAL SORTING
# =====================================================================
def load_satellite_sequence(
    input_source: str,
    band_preference: Optional[str] = None
) -> List[SatelliteFrame]:
    """
    Loads, inspects, and chronologically sorts multiple satellite images
    from a directory or list of file paths.
    """
    file_list = []
    if os.path.isdir(input_source):
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.h5", "*.hdf", "*.bmp")
        for ext in extensions:
            file_list.extend(glob.glob(os.path.join(input_source, ext)))
            file_list.extend(glob.glob(os.path.join(input_source, ext.upper())))
    elif os.path.isfile(input_source):
        file_list = [input_source]
    else:
        # Glob pattern support
        file_list = glob.glob(input_source)

    file_list = sorted(list(set(file_list)))
    if not file_list:
        return []

    frames: List[SatelliteFrame] = []
    for f in file_list:
        try:
            frame = load_single_image(f, band_preference)
            frames.append(frame)
        except Exception as e:
            print(f"[WARNING] Skipping unreadable satellite file {f}: {e}")

    # Chronological sort
    frames.sort(key=lambda fr: fr.timestamp or datetime.min)
    return frames


# =====================================================================
# SYNTHETIC TEST SEQUENCE GENERATOR (FOR TESTING WHEN NO REAL DATA EXISTS)
# =====================================================================
def generate_synthetic_insat_sequence(
    output_dir: str,
    num_frames: int = 5,
    interval_minutes: int = 30,
    base_time: Optional[datetime] = None,
    image_size: Tuple[int, int] = (600, 600)
) -> List[str]:
    """
    Generates a realistic test sequence of timestamped INSAT-3D TIR1-style
    images showing moving cloud formations advecting towards Bengaluru.

    IMPORTANT: Files are clearly watermarked and named [SIMULATED_TEST_DATA]
    to guarantee transparency and avoid presenting synthetic data as actual satellite imagery.
    """
    os.makedirs(output_dir, exist_ok=True)
    if base_time is None:
        base_time = datetime(2026, 11, 3, 14, 0, 0)

    width, height = image_size
    saved_paths = []

    # Bengaluru pixel coordinates in our synthetic 600x600 regional bounding box
    # (roughly south-central India)
    bengaluru_pos = (320, 360)

    # Initial cloud clusters: (cx, cy, radius_x, radius_y, intensity, velocity_x, velocity_y)
    # Velocity is in pixels per frame (~30 min).
    clouds_init = [
        {"cx": 180, "cy": 450, "rx": 65, "ry": 45, "intensity": 230, "vx": 18, "vy": -14, "growth": 1.05},  # Moving NE towards Bengaluru
        {"cx": 420, "cy": 490, "rx": 80, "ry": 55, "intensity": 210, "vx": -10, "vy": -12, "growth": 0.98}, # Bay of Bengal convective cluster
        {"cx": 120, "cy": 220, "rx": 50, "ry": 40, "intensity": 185, "vx": 12, "vy": 8, "growth": 1.02},   # Arabian Sea low cloud
        {"cx": 350, "cy": 160, "rx": 90, "ry": 60, "intensity": 195, "vx": 8, "vy": 6, "growth": 0.95},    # Deccan plateau cloud patch
    ]

    for frame_idx in range(num_frames):
        current_time = base_time + timedelta(minutes=frame_idx * interval_minutes)
        time_str = current_time.strftime("%d%b%Y_%H%M").upper()

        # Canvas: land/ocean background with subtle thermal gradient
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        background = 45 + 15 * np.sin(x_grid / 100.0) * np.cos(y_grid / 120.0)
        # Add slight natural thermal sensor noise
        noise = np.random.normal(0, 3.5, (height, width))
        frame_canvas = np.clip(background + noise, 20, 80).astype(np.float32)

        # Draw moving and evolving cloud clusters
        for cl in clouds_init:
            # Updated position with slight dynamic variation
            cx = cl["cx"] + cl["vx"] * frame_idx + np.random.normal(0, 1.0)
            cy = cl["cy"] + cl["vy"] * frame_idx + np.random.normal(0, 1.0)
            rx = cl["rx"] * (cl["growth"] ** frame_idx)
            ry = cl["ry"] * (cl["growth"] ** frame_idx)

            # Gaussian cloud density profile
            dist_sq = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2
            cloud_profile = cl["intensity"] * np.exp(-0.5 * dist_sq)
            cloud_profile[dist_sq > 4.0] = 0

            # Internal turbulent texture inside clouds
            turb = 1.0 + 0.15 * np.sin(x_grid / 8.0) * np.cos(y_grid / 8.0)
            cloud_profile *= turb

            frame_canvas = np.maximum(frame_canvas, cloud_profile)

        frame_uint8 = np.clip(frame_canvas, 0, 255).astype(np.uint8)

        # Output file name formatted exactly like MOSDAC INSAT-3D TIR1 products
        fname = f"3DIMG_{time_str}_L1C_ASIA_MER_TIR1_SIMULATED.png"
        fpath = os.path.join(output_dir, fname)

        pil_img = Image.fromarray(frame_uint8)
        pil_img.info["synthetic"] = True
        pil_img.save(fpath, format="PNG")
        saved_paths.append(fpath)

    # Also generate one synthetic HDF5 file to test the HDF5 parser
    if H5PY_AVAILABLE:
        h5_name = f"3DIMG_{(base_time + timedelta(minutes=num_frames*interval_minutes)).strftime('%d%b%Y_%H%M').upper()}_L1C_ASIA_MER_TIR1_SIMULATED.h5"
        h5_path = os.path.join(output_dir, h5_name)
        try:
            with h5py.File(h5_path, "w") as h5f:
                h5f.attrs["Satellite_Name"] = "INSAT-3D"
                h5f.attrs["Sensor"] = "IMAGER"
                h5f.attrs["Product_Type"] = "SIMULATED_TEST_DATA"
                h5f.attrs["Acquisition_Date"] = base_time.strftime("%Y%m%d")
                h5f.attrs["Acquisition_Time"] = base_time.strftime("%H%M%S")
                # Create IMG_TIR1 dataset
                h5f.create_dataset("IMG_TIR1", data=frame_uint8, compression="gzip")
            saved_paths.append(h5_path)
        except Exception as e:
            print(f"[INFO] HDF5 synthetic test file generation note: {e}")

    return saved_paths


"""
regional_extractor.py
=====================
CloudPath AI – Phase 3: Satellite Data Processing
Module 2: Open a single NetCDF file and extract the Bengaluru regional domain.

Key facts confirmed from actual metadata inspection:
  - Grid  : 1616 lat × 1618 lon   (0.04° resolution, WGS-84)
  - Lat   : 45.5°N → -10°S  (descending — northern latitudes have low index)
  - Lon   : 44.5°E → 105.5°E (ascending)
  - Encoding: cloud_mask_HHMM  0 = Clear Sky, 1 = Cloud Covered, NaN = Fill/Space
              flag_HHMM         1 = High-level thick, 2 = Low-level thick,
                                3 = Semi-transparent cirrus, 4 = Partial clouds

Author: CloudPath AI Project
"""

import os
import re
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────
# Bengaluru bounding box (degrees)
# ─────────────────────────────────────────────────────────────
BLAT       = 12.97         # Bengaluru centre latitude
BLON       = 77.59         # Bengaluru centre longitude
LAT_MARGIN = 3.0           # ±3° north/south  (~330 km)
LON_MARGIN = 3.0           # ±3° east/west    (~330 km)

CLOUD_MASK_VALUE  = 1.0    # from actual data: 1 = cloudy
CLEAR_MASK_VALUE  = 0.0    # from actual data: 0 = clear sky

# Flag encoding from metadata attributes
FLAG_NAMES = {
    1: "High level thick clouds",
    2: "Low level thick clouds",
    3: "Semi-transparent cirrus clouds",
    4: "Partial clouds",
}


def _parse_date_from_filename(nc_path: str):
    """
    Extract the date from the NetCDF filename.
    Example: INS3S_L3_CF_4km_Ver02_D01JUL2025.nc → 2025-07-01
    """
    basename = os.path.basename(nc_path)
    # Pattern: D<dd><MON><yyyy>
    m = re.search(r"D(\d{2})([A-Z]{3})(\d{4})", basename, re.IGNORECASE)
    if m:
        day   = int(m.group(1))
        month = datetime.strptime(m.group(2), "%b").month
        year  = int(m.group(3))
        return datetime(year, month, day)
    return None


def get_lat_lon_indices(lat_array, lon_array,
                        blat=BLAT, blon=BLON,
                        lat_margin=LAT_MARGIN, lon_margin=LON_MARGIN):
    """
    Return the slice indices [lat_start, lat_end, lon_start, lon_end]
    corresponding to the Bengaluru region bounding box.
    """
    lat_mask = (lat_array >= blat - lat_margin) & (lat_array <= blat + lat_margin)
    lon_mask = (lon_array >= blon - lon_margin) & (lon_array <= blon + lon_margin)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"Bengaluru region not found in the satellite grid. "
            f"Lat range: {float(lat_array.min()):.2f}–{float(lat_array.max()):.2f}, "
            f"Lon range: {float(lon_array.min()):.2f}–{float(lon_array.max()):.2f}"
        )

    return (int(lat_idx[0]), int(lat_idx[-1]) + 1,
            int(lon_idx[0]), int(lon_idx[-1]) + 1)


def list_time_slots(ds) -> list[str]:
    """Return sorted list of HHMM time-slot strings available in this dataset."""
    varlist = list(ds.variables.keys())
    slots = sorted([v.replace("cloud_mask_", "") for v in varlist
                    if v.startswith("cloud_mask_")])
    return slots


def extract_observation(ds, hhmm: str,
                         lat_s, lat_e, lon_s, lon_e) -> dict:
    """
    Extract one cloud-mask observation from an open netCDF4 Dataset.

    Parameters
    ----------
    ds           : open netCDF4.Dataset object
    hhmm         : time-slot string e.g. '0030'
    lat_s, lat_e : lat slice indices
    lon_s, lon_e : lon slice indices

    Returns a dict with:
        hhmm          : time slot string
        cloud_mask    : 2D numpy array (float64), values 0/1/NaN
        flag          : 2D numpy array (float64)
        n_total       : total pixels in region
        n_valid       : pixels with non-NaN cloud_mask values
        n_cloudy      : pixels where cloud_mask == 1
        n_clear       : pixels where cloud_mask == 0
        cloud_pct     : cloud coverage percentage (valid pixels only)
        clear_pct     : clear sky percentage (valid pixels only)
        flag_counts   : dict {flag_value: pixel_count}
    """
    cm_var   = f"cloud_mask_{hhmm}"
    flag_var = f"flag_{hhmm}"

    if cm_var not in ds.variables:
        return None

    # Read only the Bengaluru sub-region — avoids loading the full 1616×1618 array
    cloud_mask = np.array(ds.variables[cm_var][lat_s:lat_e, lon_s:lon_e])
    flag_data  = np.array(ds.variables[flag_var][lat_s:lat_e, lon_s:lon_e]) \
                 if flag_var in ds.variables else None

    # Mask NaN fill values
    valid_mask = ~np.isnan(cloud_mask)
    n_total    = cloud_mask.size
    n_valid    = int(np.sum(valid_mask))
    n_cloudy   = int(np.sum(cloud_mask[valid_mask] == CLOUD_MASK_VALUE))
    n_clear    = int(np.sum(cloud_mask[valid_mask] == CLEAR_MASK_VALUE))

    cloud_pct = (n_cloudy / n_valid * 100) if n_valid > 0 else np.nan
    clear_pct = (n_clear  / n_valid * 100) if n_valid > 0 else np.nan

    # Count flag categories only over cloudy pixels
    flag_counts = {}
    if flag_data is not None:
        for fval in [1, 2, 3, 4]:
            flag_counts[fval] = int(np.sum(
                (flag_data[valid_mask] == fval) & (cloud_mask[valid_mask] == CLOUD_MASK_VALUE)
            ))

    return {
        "hhmm":        hhmm,
        "cloud_mask":  cloud_mask,
        "flag":        flag_data,
        "n_total":     n_total,
        "n_valid":     n_valid,
        "n_cloudy":    n_cloudy,
        "n_clear":     n_clear,
        "cloud_pct":   cloud_pct,
        "clear_pct":   clear_pct,
        "flag_counts": flag_counts,
    }


def extract_all_observations_from_file(nc_path: str) -> list[dict]:
    """
    Open one NetCDF file and extract every time-slot observation for the
    Bengaluru region.  The file is opened once and closed when done —
    the full-disk grids are never loaded into memory.

    Returns a list of observation dicts, each containing:
        timestamp    : datetime object (UTC)
        + all fields from extract_observation()
        + lat_bounds, lon_bounds, lat_s, lat_e, lon_s, lon_e
    """
    observations = []

    file_date = _parse_date_from_filename(nc_path)
    if file_date is None:
        print(f"  WARNING: Could not parse date from filename: {os.path.basename(nc_path)}")

    ds = nc.Dataset(nc_path, "r")
    try:
        lat = np.array(ds.variables["latitude"][:])
        lon = np.array(ds.variables["longitude"][:])

        # Compute Bengaluru indices once per file
        lat_s, lat_e, lon_s, lon_e = get_lat_lon_indices(lat, lon)

        lat_bounds = (float(lat[lat_s:lat_e].min()),
                      float(lat[lat_s:lat_e].max()))
        lon_bounds = (float(lon[lon_s:lon_e].min()),
                      float(lon[lon_s:lon_e].max()))

        slots = list_time_slots(ds)

        for hhmm in slots:
            obs = extract_observation(ds, hhmm, lat_s, lat_e, lon_s, lon_e)
            if obs is None:
                continue

            # Build UTC timestamp from file date + HHMM slot
            if file_date is not None:
                hh = int(hhmm[:2])
                mm = int(hhmm[2:])
                obs["timestamp"]   = file_date + timedelta(hours=hh, minutes=mm)
            else:
                obs["timestamp"]   = None

            obs["nc_file"]     = os.path.basename(nc_path)
            obs["lat_s"]       = lat_s
            obs["lat_e"]       = lat_e
            obs["lon_s"]       = lon_s
            obs["lon_e"]       = lon_e
            obs["lat_bounds"]  = lat_bounds
            obs["lon_bounds"]  = lon_bounds
            obs["region_shape"] = (lat_e - lat_s, lon_e - lon_s)
            observations.append(obs)

    finally:
        ds.close()

    return observations


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nc_dir = r"d:\bengaluru.csv\satellite_nc"
    nc_files = sorted([os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")])
    nc_path = nc_files[0] if nc_files else r"d:\bengaluru.csv\satellite_nc\INS3S_L3_CF_4km_Ver02_D01JUL2025.nc"
    obs_list = extract_all_observations_from_file(nc_path)
    print(f"Observations extracted: {len(obs_list)}")
    if obs_list:
        o = obs_list[0]
        print(f"  First timestamp  : {o['timestamp']}")
        print(f"  Region shape     : {o['region_shape']}")
        print(f"  Lat bounds       : {o['lat_bounds']}")
        print(f"  Lon bounds       : {o['lon_bounds']}")
        print(f"  Cloud coverage   : {o['cloud_pct']:.2f}%")
        print(f"  Clear sky        : {o['clear_pct']:.2f}%")


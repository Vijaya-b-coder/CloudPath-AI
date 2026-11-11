"""
satellite_data_loader.py
========================
CloudPath AI – Phase 3: Satellite Data Processing
Module 1: Locate, verify and extract satellite ZIP archives containing NetCDF files.

Author: CloudPath AI Project
"""

import os
import zipfile
import glob


# ─────────────────────────────────────────────────────────────
# Configuration
# Dedicated directory for July 2025 satellite ZIP files
# ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SATELLITE_DATA_DIR = os.path.join(PROJECT_DIR, "satellite_img")
SATELLITE_NC_DIR = os.path.join(PROJECT_DIR, "satellite_nc")


def locate_zip_files(data_dir: str = SATELLITE_DATA_DIR) -> list[str]:
    """
    Scan the given directory for satellite ZIP files.

    Returns a sorted list of absolute paths to all .zip files found.
    Does not open or modify the files.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Satellite data directory not found: {data_dir}\n"
            f"Please place ZIP files inside: {data_dir}"
        )

    zip_files = sorted(glob.glob(os.path.join(data_dir, "*.zip")))
    return zip_files


def verify_zip_file(zip_path: str) -> dict:
    """
    Verify that a single ZIP file can be opened and list its contents.

    Returns a dict with:
        path        : full path to the ZIP file
        valid       : True if the file is a valid ZIP
        file_size   : file size in bytes
        contents    : list of filenames inside the ZIP
        nc_files    : list of .nc filenames inside the ZIP
        error       : error message string if invalid, else None
    """
    result = {
        "path": zip_path,
        "valid": False,
        "file_size": os.path.getsize(zip_path),
        "contents": [],
        "nc_files": [],
        "error": None,
    }
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            result["contents"] = zf.namelist()
            result["nc_files"] = [f for f in zf.namelist() if f.endswith(".nc")]
            result["valid"] = True
    except zipfile.BadZipFile as exc:
        result["error"] = str(exc)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def extract_zip_file(zip_path: str, extract_dir: str = SATELLITE_NC_DIR,
                     overwrite: bool = False) -> list[str]:
    """
    Extract NetCDF files from a ZIP archive into extract_dir.

    Parameters
    ----------
    zip_path    : path to the ZIP file
    extract_dir : directory where NetCDF files will be saved
    overwrite   : if False, skip files that already exist (avoids re-extraction)

    Returns a list of absolute paths to the extracted NetCDF files.
    """
    os.makedirs(extract_dir, exist_ok=True)
    extracted = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".nc"):
                continue  # only extract NetCDF files
            dest_path = os.path.join(extract_dir, os.path.basename(member))
            if os.path.exists(dest_path) and not overwrite:
                # File already extracted — skip to save time
                extracted.append(dest_path)
                continue
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                while True:
                    buf = src.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            extracted.append(dest_path)

    return extracted


def load_all_satellite_data(data_dir: str = SATELLITE_DATA_DIR,
                             nc_dir: str = SATELLITE_NC_DIR) -> dict:
    """
    High-level function:
      1. Locate all ZIP files in data_dir.
      2. Verify each ZIP.
      3. Extract NetCDF files into nc_dir (skip already-extracted files).
      4. Return a summary dict.
    """
    summary = {
        "zip_files":   [],
        "zip_reports": [],
        "nc_files":    [],
        "errors":      [],
    }

    # Step 1 — Locate ZIP files
    zip_files = locate_zip_files(data_dir)
    summary["zip_files"] = zip_files

    if not zip_files:
        summary["errors"].append(f"No ZIP files found in: {data_dir}")
        return summary

    # Step 2 & 3 — Verify and extract
    for zp in zip_files:
        report = verify_zip_file(zp)
        summary["zip_reports"].append(report)

        if not report["valid"]:
            summary["errors"].append(
                f"Invalid ZIP: {os.path.basename(zp)} — {report['error']}"
            )
            continue

        if not report["nc_files"]:
            summary["errors"].append(
                f"No .nc files inside: {os.path.basename(zp)}"
            )
            continue

        extracted = extract_zip_file(zp, nc_dir, overwrite=False)
        summary["nc_files"].extend(extracted)

    # Sort NetCDF list chronologically by filename
    summary["nc_files"] = sorted(set(summary["nc_files"]))
    return summary


if __name__ == "__main__":
    result = load_all_satellite_data()
    print(f"ZIP directory      : {SATELLITE_DATA_DIR}")
    print(f"ZIP files found    : {len(result['zip_files'])}")
    print(f"NetCDF files ready : {len(result['nc_files'])}")
    if result["errors"]:
        print("ERRORS:")
        for e in result["errors"]:
            print(f"  {e}")

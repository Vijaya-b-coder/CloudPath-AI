"""
cloud_region_detection.py  (Phase 4 – updated)
===============================================
CloudPath AI – Multiple Cloud Detection and Movement Tracking
Module 1: Detect connected cloud regions in each observation frame.

Input  : NPZ cloud-mask stack from Phase 3.
Output : Per-frame list of CloudRegion objects.

Encoding confirmed from Phase 3 metadata:
    0 = Clear Sky
    1 = Cloud Covered
  NaN = Fill / Space

Author: CloudPath AI Project
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Optional


# ─── minimum connected-region size (pixels) to keep ──────────────────
MIN_REGION_PIXELS = 8


@dataclass
class CloudRegion:
    """Stores properties of a single detected cloud region in one frame."""
    frame_idx:   int               # index into the observation stack
    timestamp:   str               # ISO timestamp string
    region_id:   int               # label inside this frame (1-based)
    centroid_r:  float             # row centroid  (pixel coords)
    centroid_c:  float             # column centroid
    area_pixels: int               # number of cloudy pixels
    bbox:        tuple             # (row_min, row_max, col_min, col_max)
    dominant_flag: Optional[int]   # most common flag value (1-4) or None
    flag_counts: dict = field(default_factory=dict)

    # geographic centroid (filled later by speed calculator)
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None


def build_binary_mask(cloud_mask: np.ndarray) -> np.ndarray:
    """
    Convert raw float mask (0/1/NaN) to uint8 binary:
        1 = cloud pixel,  0 = everything else.
    NaN pixels become 0 so they are never labelled as cloud.
    """
    binary = np.zeros(cloud_mask.shape, dtype=np.uint8)
    binary[cloud_mask == 1.0] = 1
    return binary


def detect_regions_in_frame(frame_idx: int,
                             timestamp: str,
                             cloud_mask: np.ndarray,
                             flag_mask: Optional[np.ndarray] = None,
                             min_pixels: int = MIN_REGION_PIXELS
                             ) -> list[CloudRegion]:
    """
    Run connected-component labelling on one cloud-mask frame.

    Returns a list of CloudRegion objects sorted by area (largest first).
    """
    binary = build_binary_mask(cloud_mask)

    # 8-connectivity: diagonal neighbours are connected
    struct  = ndimage.generate_binary_structure(2, 2)
    labelled, n_raw = ndimage.label(binary, structure=struct)

    regions: list[CloudRegion] = []

    for lbl in range(1, n_raw + 1):
        mask = (labelled == lbl)
        area = int(np.sum(mask))
        if area < min_pixels:
            continue

        rows, cols = np.where(mask)
        centroid_r = float(np.mean(rows))
        centroid_c = float(np.mean(cols))
        bbox = (int(rows.min()), int(rows.max()),
                int(cols.min()), int(cols.max()))

        # Flag analysis (cloud type) — only over pixels in this region
        flag_counts = {}
        dominant_flag = None
        if flag_mask is not None:
            fg = flag_mask[mask]
            fg_valid = fg[~np.isnan(fg)].astype(int)
            for fv in [1, 2, 3, 4]:
                cnt = int(np.sum(fg_valid == fv))
                if cnt > 0:
                    flag_counts[fv] = cnt
            if flag_counts:
                dominant_flag = max(flag_counts, key=flag_counts.get)

        regions.append(CloudRegion(
            frame_idx=frame_idx,
            timestamp=timestamp,
            region_id=lbl,
            centroid_r=centroid_r,
            centroid_c=centroid_c,
            area_pixels=area,
            bbox=bbox,
            dominant_flag=dominant_flag,
            flag_counts=flag_counts,
        ))

    # Sort largest first — tracking prefers large stable regions
    regions.sort(key=lambda r: r.area_pixels, reverse=True)
    return regions


def detect_all_frames(cloud_masks: np.ndarray,
                       timestamps: np.ndarray,
                       flags: Optional[np.ndarray] = None,
                       min_pixels: int = MIN_REGION_PIXELS
                       ) -> list[list[CloudRegion]]:
    """
    Run region detection over all frames in the stack.

    cloud_masks : (N, H, W) float32
    timestamps  : (N,)  string array
    flags       : (N, H, W) float32 or None

    Returns list of length N, each element is a list of CloudRegion.
    """
    n_frames = cloud_masks.shape[0]
    all_regions: list[list[CloudRegion]] = []

    for i in range(n_frames):
        cm   = cloud_masks[i]
        flag = flags[i] if flags is not None else None
        ts   = str(timestamps[i])
        frame_regions = detect_regions_in_frame(i, ts, cm, flag, min_pixels)
        all_regions.append(frame_regions)

    return all_regions


def detect_cloud_regions(binary_mask: np.ndarray,
                          min_region_pixels: int = 5) -> dict:
    """
    Label connected cloud regions in a binary mask (for Phase 3 pipeline).
    """
    struct = ndimage.generate_binary_structure(2, 2)
    labelled, n_raw = ndimage.label(binary_mask, structure=struct)

    region_sizes = {}
    region_centroids = {}
    region_bboxes = {}
    valid_labels = []

    for lbl in range(1, n_raw + 1):
        mask = (labelled == lbl)
        size = int(np.sum(mask))
        if size < min_region_pixels:
            labelled[mask] = 0
            continue
        valid_labels.append(lbl)
        region_sizes[lbl] = size

        rows, cols = np.where(mask)
        region_centroids[lbl] = (float(np.mean(rows)), float(np.mean(cols)))
        region_bboxes[lbl] = (int(rows.min()), int(rows.max()),
                               int(cols.min()), int(cols.max()))

    return {
        "labelled_array":   labelled,
        "n_regions":        len(valid_labels),
        "region_labels":    valid_labels,
        "region_sizes":     region_sizes,
        "region_centroids": region_centroids,
        "region_bboxes":    region_bboxes,
    }


def analyse_observation(obs: dict, min_region_pixels: int = 5) -> dict:
    """
    Run full cloud-region analysis on a single observation dict.
    """
    cm = obs.get("cloud_mask")
    if cm is None:
        obs["n_regions"] = 0
        return obs

    binary = build_binary_mask(cm)
    detection = detect_cloud_regions(binary, min_region_pixels=min_region_pixels)

    obs["binary_mask"]       = binary
    obs["labelled_array"]    = detection["labelled_array"]
    obs["n_regions"]         = detection["n_regions"]
    obs["region_labels"]     = detection["region_labels"]
    obs["region_sizes"]      = detection["region_sizes"]
    obs["region_centroids"]  = detection["region_centroids"]
    obs["region_bboxes"]     = detection["region_bboxes"]
    return obs


# ─── standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    NPZ = r"d:\bengaluru.csv\satellite_data_outputs\bengaluru_cloudmask_series.npz"
    data = np.load(NPZ, allow_pickle=True)
    all_r = detect_all_frames(data["cloud_masks"], data["timestamps"], data["flags"])
    total = sum(len(r) for r in all_r)
    print(f"Frames: {len(all_r)},  Total regions: {total}")
    print(f"Frame 0 regions: {len(all_r[0])}, "
          f"largest area: {all_r[0][0].area_pixels if all_r[0] else 0}")

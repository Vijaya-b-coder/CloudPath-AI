"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: cloud_detection.py
===================================================================
Author : B.Tech Final Year Project
Role   : Detects and segments cloud-covered regions from satellite imagery
         using adaptive thresholding, Otsu's method, multi-level density
         classification (Thin, Medium, Deep Convective), and morphological refinement.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2


class CloudDetectionResult:
    """Container for cloud segmentation outputs on a single frame."""
    def __init__(
        self,
        binary_mask: np.ndarray,
        density_mask: np.ndarray,
        cloud_coverage_pct: float,
        contours: List[np.ndarray],
        threshold_used: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.binary_mask = binary_mask        # uint8: 0=clear, 255=cloud
        self.density_mask = density_mask      # uint8: 0=clear, 1=thin, 2=medium, 3=dense/convective
        self.cloud_coverage_pct = cloud_coverage_pct  # float: 0.0 - 100.0%
        self.contours = contours              # list of cv2 contours
        self.threshold_used = threshold_used  # threshold cutoff value
        self.metadata = metadata or {}


def detect_clouds(
    preprocessed_img: np.ndarray,
    method: str = "otsu",
    band_type: str = "TIR1",
    min_cloud_area: int = 150,
    morph_kernel_size: int = 5
) -> CloudDetectionResult:
    """
    Performs robust cloud detection and segmentation:

    Parameters:
      preprocessed_img: Grayscale 8-bit image after contrast & noise reduction
      method: 'otsu' (adaptive global), 'adaptive' (local illumination), or 'multi_threshold'
      band_type: Satellite band ('TIR1', 'TIR2', 'VIS', 'WV')
      min_cloud_area: Minimum area in pixels to filter out minor sensor noise
      morph_kernel_size: Kernel dimension for morphological closing/opening
    """
    height, width = preprocessed_img.shape[:2]
    total_pixels = height * width

    # 1. Base Cloud Segmentation via Thresholding
    if method.lower() == "otsu":
        # Otsu's automated thresholding
        otsu_thresh, raw_mask = cv2.threshold(
            preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_val = float(otsu_thresh)
    elif method.lower() == "adaptive":
        # Adaptive Gaussian thresholding (good for non-uniform sun glint / illumination)
        raw_mask = cv2.adaptiveThreshold(
            preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, -5
        )
        threshold_val = float(np.mean(preprocessed_img))
    else:
        # Default fixed percentage threshold based on band
        threshold_val = 110.0 if band_type.upper().startswith("TIR") else 130.0
        _, raw_mask = cv2.threshold(
            preprocessed_img, threshold_val, 255, cv2.THRESH_BINARY
        )

    # 2. Morphological Refinement
    # Closing fills small internal gaps inside dense cloud decks
    # Opening removes isolated single-pixel sensor noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Filter Small Speckle Contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(cleaned_mask)
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_cloud_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            valid_contours.append(cnt)

    # 4. Multi-Level Cloud Density / Vertical Thickness Estimation
    # In enhanced TIR / Visible imagery:
    # - Level 1 (Thin / Low Clouds): threshold_val to threshold_val + 35
    # - Level 2 (Medium Clouds): threshold_val + 35 to threshold_val + 75
    # - Level 3 (Deep Convective Clouds / Cold Cloud Tops): > threshold_val + 75
    density_mask = np.zeros((height, width), dtype=np.uint8)
    in_cloud = filtered_mask > 0

    t1 = threshold_val
    t2 = min(255.0, threshold_val + 35.0)
    t3 = min(255.0, threshold_val + 75.0)

    density_mask[in_cloud & (preprocessed_img >= t1) & (preprocessed_img < t2)] = 1  # Thin
    density_mask[in_cloud & (preprocessed_img >= t2) & (preprocessed_img < t3)] = 2  # Medium
    density_mask[in_cloud & (preprocessed_img >= t3)] = 3                           # Convective / Dense

    # 5. Compute Coverage Metrics
    cloud_pixels = int(np.count_nonzero(filtered_mask))
    coverage_pct = round((cloud_pixels / total_pixels) * 100.0, 2)

    meta = {
        "method": method,
        "band_type": band_type,
        "total_pixels": total_pixels,
        "cloud_pixels": cloud_pixels,
        "threshold_used": threshold_val,
        "num_cloud_clusters": len(valid_contours),
        "thin_cloud_pct": round(float(np.count_nonzero(density_mask == 1) / total_pixels * 100.0), 2),
        "medium_cloud_pct": round(float(np.count_nonzero(density_mask == 2) / total_pixels * 100.0), 2),
        "dense_cloud_pct": round(float(np.count_nonzero(density_mask == 3) / total_pixels * 100.0), 2),
    }

    return CloudDetectionResult(
        binary_mask=filtered_mask,
        density_mask=density_mask,
        cloud_coverage_pct=coverage_pct,
        contours=valid_contours,
        threshold_used=threshold_val,
        metadata=meta
    )


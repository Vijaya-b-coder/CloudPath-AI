"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: image_preprocessing.py
===================================================================
Author : B.Tech Final Year Project
Role   : Preprocesses raw satellite images via resizing, normalization,
         contrast enhancement (CLAHE/stretching), and boundary-preserving
         noise reduction (bilateral/Gaussian filtering).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = (512, 512),
    interpolation: int = cv2.INTER_AREA
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resizes image to a standardized resolution if requested,
    otherwise retains original dimensions.
    """
    orig_shape = image.shape[:2]
    if target_size is None or orig_shape == target_size:
        return image.copy(), {"resized": False, "original_shape": orig_shape, "final_shape": orig_shape}

    resized = cv2.resize(image, target_size, interpolation=interpolation)
    log = {
        "resized": True,
        "original_shape": orig_shape,
        "final_shape": target_size,
        "interpolation": "INTER_AREA" if interpolation == cv2.INTER_AREA else "OTHER"
    }
    return resized, log


def normalize_image(
    image: np.ndarray,
    min_out: float = 0.0,
    max_out: float = 255.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalizes pixel intensity range to [min_out, max_out] uint8 or float32.
    """
    img_f = image.astype(np.float32)
    img_min = float(np.min(img_f))
    img_max = float(np.max(img_f))

    if img_max > img_min:
        norm = (img_f - img_min) / (img_max - img_min) * (max_out - min_out) + min_out
    else:
        norm = np.zeros_like(img_f)

    norm_uint8 = np.clip(norm, 0, 255).astype(np.uint8)
    log = {
        "min_in": img_min,
        "max_in": img_max,
        "min_out": min_out,
        "max_out": max_out
    }
    return norm_uint8, log


def enhance_contrast(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Enhances satellite imagery contrast:
    - 'clahe': Contrast Limited Adaptive Histogram Equalization (recommended for cloud structures).
    - 'stretch': Min-Max 1%-99% percentile linear stretching.
    - 'equalize': Standard global histogram equalization.
    """
    if method.lower() == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        log = {"method": "clahe", "clip_limit": clip_limit, "tile_grid_size": tile_grid_size}
    elif method.lower() == "stretch":
        p1, p99 = np.percentile(image, (1, 99))
        if p99 > p1:
            stretched = np.clip((image.astype(np.float32) - p1) / (p99 - p1) * 255.0, 0, 255)
            enhanced = stretched.astype(np.uint8)
        else:
            enhanced = image.copy()
        log = {"method": "percentile_stretch", "p1": float(p1), "p99": float(p99)}
    elif method.lower() == "equalize":
        enhanced = cv2.equalizeHist(image)
        log = {"method": "global_equalization"}
    else:
        enhanced = image.copy()
        log = {"method": "none"}

    return enhanced, log


def reduce_noise(
    image: np.ndarray,
    method: str = "bilateral",
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduces atmospheric and sensor noise while preserving cloud boundaries:
    - 'bilateral': Non-linear edge-preserving filter (optimal for satellite cloud boundaries).
    - 'gaussian': Standard linear Gaussian smoothing.
    - 'median': Median filter for salt-and-pepper noise.
    """
    if method.lower() == "bilateral":
        denoised = cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        log = {"method": "bilateral", "diameter": diameter, "sigma_color": sigma_color, "sigma_space": sigma_space}
    elif method.lower() == "gaussian":
        denoised = cv2.GaussianBlur(image, (5, 5), 1.2)
        log = {"method": "gaussian", "kernel": (5, 5), "sigma": 1.2}
    elif method.lower() == "median":
        denoised = cv2.medianBlur(image, 5)
        log = {"method": "median", "ksize": 5}
    else:
        denoised = image.copy()
        log = {"method": "none"}

    return denoised, log


def preprocess_satellite_image(
    raw_image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    contrast_method: str = "clahe",
    denoise_method: str = "bilateral"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    End-to-end preprocessing pipeline for a satellite image.
    Preserves original image and returns preprocessed image along with an audit log.
    """
    logs: Dict[str, Any] = {}

    # 1. Resize if required
    step1, log_resize = resize_image(raw_image, target_size=target_size)
    logs["resize"] = log_resize

    # 2. Normalization to full [0, 255] uint8 dynamic range
    step2, log_norm = normalize_image(step1)
    logs["normalize"] = log_norm

    # 3. Contrast enhancement
    step3, log_contrast = enhance_contrast(step2, method=contrast_method)
    logs["contrast"] = log_contrast

    # 4. Edge-preserving noise reduction
    step4, log_denoise = reduce_noise(step3, method=denoise_method)
    logs["denoise"] = log_denoise

    return step4, logs


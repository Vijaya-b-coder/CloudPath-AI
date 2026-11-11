"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: cloud_motion.py
===================================================================
Author : B.Tech Final Year Project
Role   : Calculates cloud displacement vectors, trajectory direction,
         elapsed time between observations, and speed in pixels/hour.
         Converts to km/h if spatial scale is calibrated without inventing
         geographic speed when calibration is absent.
"""

from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np


def get_compass_direction(degrees: float) -> str:
    """
    Converts angle in degrees [0, 360) into 16-point meteorological compass direction.
    Standard meteorological heading (direction moving TOWARDS):
      0° = North, 90° = East, 180° = South, 270° = West
    """
    compass_sectors = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((degrees + 11.25) / 22.5) % 16
    return compass_sectors[idx]


class CloudMotionVector:
    """Stores velocity and displacement parameters for a single cloud track."""
    def __init__(
        self,
        cloud_id: int,
        prev_centroid: Tuple[float, float],
        curr_centroid: Tuple[float, float],
        dx_pixels: float,
        dy_pixels: float,
        distance_pixels: float,
        direction_deg: float,
        direction_compass: str,
        dt_seconds: Optional[float],
        pixel_speed_per_hour: Optional[float],
        distance_km: Optional[float] = None,
        speed_kmh: Optional[float] = None,
        is_georeferenced: bool = False
    ):
        self.cloud_id = cloud_id
        self.prev_centroid = prev_centroid
        self.curr_centroid = curr_centroid
        self.dx_pixels = dx_pixels
        self.dy_pixels = dy_pixels
        self.distance_pixels = distance_pixels
        self.direction_deg = direction_deg
        self.direction_compass = direction_compass
        self.dt_seconds = dt_seconds
        self.pixel_speed_per_hour = pixel_speed_per_hour
        self.distance_km = distance_km
        self.speed_kmh = speed_kmh
        self.is_georeferenced = is_georeferenced

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cloud_id": self.cloud_id,
            "prev_centroid": self.prev_centroid,
            "curr_centroid": self.curr_centroid,
            "dx_pixels": round(self.dx_pixels, 2),
            "dy_pixels": round(self.dy_pixels, 2),
            "displacement_px": round(self.distance_pixels, 2),
            "direction_deg": round(self.direction_deg, 1),
            "direction_compass": self.direction_compass,
            "elapsed_seconds": round(self.dt_seconds, 1) if self.dt_seconds is not None else None,
            "speed_px_per_hour": round(self.pixel_speed_per_hour, 2) if self.pixel_speed_per_hour is not None else None,
            "displacement_km": round(self.distance_km, 2) if self.distance_km is not None else None,
            "speed_kmh": round(self.speed_kmh, 2) if self.speed_kmh is not None else None,
            "is_georeferenced": self.is_georeferenced
        }


def calculate_cloud_motion(
    prev_centroid: Tuple[float, float],
    curr_centroid: Tuple[float, float],
    prev_time: Optional[datetime],
    curr_time: Optional[datetime],
    cloud_id: int,
    km_per_pixel: Optional[float] = None
) -> CloudMotionVector:
    """
    Calculates displacement vector, compass direction, and velocity between two observations.

    Parameters:
      prev_centroid: (x1, y1) in pixels
      curr_centroid: (x2, y2) in pixels
      prev_time: Timestamp of initial frame
      curr_time: Timestamp of second frame
      cloud_id: Tracking ID
      km_per_pixel: If provided (e.g. 4.0 km/px for INSAT-3D TIR1 full resolution),
                    computes speed in km/h. Otherwise reports pixel velocity without inventing speed.
    """
    x1, y1 = prev_centroid
    x2, y2 = curr_centroid

    dx = x2 - x1
    dy = y2 - y1
    dist_px = float(np.hypot(dx, dy))

    # Angle calculation: in image coordinates, y increases downwards.
    # We map to Cartesian mathematical direction:
    # 0° = East (+x), 90° = North (-y), 180° = West (-x), 270° = South (+y)
    angle_rad = np.arctan2(-dy, dx)
    angle_deg_cartesian = (np.degrees(angle_rad) + 360.0) % 360.0

    # Convert Cartesian angle to meteorological compass heading (0° North, 90° East)
    compass_deg = (450.0 - angle_deg_cartesian) % 360.0
    compass_str = get_compass_direction(compass_deg)

    # Compute elapsed time
    dt_sec: Optional[float] = None
    speed_px_hr: Optional[float] = None
    dist_km: Optional[float] = None
    speed_kmh: Optional[float] = None
    is_georef = False

    if prev_time and curr_time and curr_time > prev_time:
        dt_sec = (curr_time - prev_time).total_seconds()
        if dt_sec > 0:
            speed_px_hr = (dist_px / dt_sec) * 3600.0

    # Georeferencing conversion
    if km_per_pixel is not None and km_per_pixel > 0:
        dist_km = dist_px * km_per_pixel
        is_georef = True
        if dt_sec and dt_sec > 0:
            speed_kmh = (dist_km / dt_sec) * 3600.0

    return CloudMotionVector(
        cloud_id=cloud_id,
        prev_centroid=prev_centroid,
        curr_centroid=curr_centroid,
        dx_pixels=dx,
        dy_pixels=dy,
        distance_pixels=dist_px,
        direction_deg=compass_deg,
        direction_compass=compass_str,
        dt_seconds=dt_sec,
        pixel_speed_per_hour=speed_px_hr,
        distance_km=dist_km,
        speed_kmh=speed_kmh,
        is_georeferenced=is_georef
    )


def compute_frame_motion_summary(motion_vectors: List[CloudMotionVector]) -> Dict[str, Any]:
    """Generates aggregate summary statistics across all moving cloud clusters."""
    if not motion_vectors:
        return {
            "total_tracked_clouds": 0,
            "mean_speed_px_hr": 0.0,
            "mean_speed_kmh": None,
            "predominant_direction": "N/A",
            "is_georeferenced": False
        }

    speeds_px = [mv.pixel_speed_per_hour for mv in motion_vectors if mv.pixel_speed_per_hour is not None]
    speeds_kmh = [mv.speed_kmh for mv in motion_vectors if mv.speed_kmh is not None]
    directions = [mv.direction_compass for mv in motion_vectors if mv.distance_pixels > 0.5]

    # Predominant direction mode
    from collections import Counter
    predom_dir = Counter(directions).most_common(1)[0][0] if directions else "STATIONARY"

    return {
        "total_tracked_clouds": len(motion_vectors),
        "mean_speed_px_hr": round(float(np.mean(speeds_px)), 2) if speeds_px else 0.0,
        "mean_speed_kmh": round(float(np.mean(speeds_kmh)), 2) if speeds_kmh else None,
        "predominant_direction": predom_dir,
        "is_georeferenced": len(speeds_kmh) > 0
    }

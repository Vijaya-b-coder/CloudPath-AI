"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: cloud_prediction.py
===================================================================
Author : B.Tech Final Year Project
Role   : Extrapolates future cloud centroid positions and coverage regions
         for configurable lead times (+30m, +1h, +2h). Computes trajectory
         intersections and Estimated Time of Arrival (ETA) toward Bengaluru.
         Explicitly disclaims all projections as linear extrapolations.
"""

from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np


class ExtrapolatedCloudPosition:
    """Stores a single future extrapolated cloud state."""
    def __init__(
        self,
        cloud_id: int,
        lead_time_minutes: int,
        predicted_timestamp: Optional[datetime],
        predicted_centroid: Tuple[float, float],
        predicted_bbox: Tuple[int, int, int, int],
        uncertainty_radius_px: float,
        will_intersect_target: bool = False,
        eta_minutes_to_target: Optional[float] = None
    ):
        self.cloud_id = cloud_id
        self.lead_time_minutes = lead_time_minutes
        self.predicted_timestamp = predicted_timestamp
        self.predicted_centroid = predicted_centroid  # (px, py)
        self.predicted_bbox = predicted_bbox          # (x, y, w, h)
        self.uncertainty_radius_px = uncertainty_radius_px
        self.will_intersect_target = will_intersect_target
        self.eta_minutes_to_target = eta_minutes_to_target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cloud_id": self.cloud_id,
            "lead_time_min": self.lead_time_minutes,
            "forecast_time": self.predicted_timestamp.strftime("%Y-%m-%d %H:%M") if self.predicted_timestamp else None,
            "predicted_centroid": (round(self.predicted_centroid[0], 1), round(self.predicted_centroid[1], 1)),
            "uncertainty_radius_px": round(self.uncertainty_radius_px, 1),
            "will_intersect_target": self.will_intersect_target,
            "eta_minutes_to_target": round(self.eta_minutes_to_target, 1) if self.eta_minutes_to_target else None,
            "disclaimer": "EXTRAPOLATED_POSITION_NOT_A_GUARANTEED_FORECAST"
        }


def predict_future_cloud_positions(
    cloud_id: int,
    curr_centroid: Tuple[float, float],
    curr_bbox: Tuple[int, int, int, int],
    dx_per_hour: float,
    dy_per_hour: float,
    current_time: Optional[datetime] = None,
    lead_times_minutes: List[int] = [30, 60, 120],
    target_station_coord: Optional[Tuple[float, float]] = None,
    target_alert_radius_px: float = 60.0,
    uncertainty_growth_rate: float = 0.25
) -> List[ExtrapolatedCloudPosition]:
    """
    Extrapolates future positions for an individual tracked cloud based on estimated velocity.

    Parameters:
      cloud_id: Identification number of cloud
      curr_centroid: Current (cx, cy) pixel coordinates
      curr_bbox: (x, y, w, h) bounding box
      dx_per_hour: Horizontal pixel velocity per hour
      dy_per_hour: Vertical pixel velocity per hour
      current_time: Observation timestamp
      lead_times_minutes: Extrapolation intervals (e.g. [30, 60, 120])
      target_station_coord: Optional city/station of interest (e.g. Bengaluru pixel coords)
      target_alert_radius_px: Proximity threshold for target intersection alert
      uncertainty_growth_rate: Expansion factor of uncertainty circle over time
    """
    extrapolations: List[ExtrapolatedCloudPosition] = []
    cx0, cy0 = curr_centroid
    bx, by, bw, bh = curr_bbox

    # Velocity components per minute
    vx_min = dx_per_hour / 60.0
    vy_min = dy_per_hour / 60.0
    speed_per_min = float(np.hypot(vx_min, vy_min))

    for lt in lead_times_minutes:
        # Extrapolated centroid
        pred_cx = cx0 + vx_min * lt
        pred_cy = cy0 + vy_min * lt

        # Extrapolated bounding box
        pred_bx = int(bx + vx_min * lt)
        pred_by = int(by + vy_min * lt)
        pred_bbox = (pred_bx, pred_by, bw, bh)

        # Uncertainty radius expands with time
        initial_radius = max(bw, bh) * 0.25
        uncertainty_r = initial_radius + (uncertainty_growth_rate * speed_per_min * lt)

        # Predicted timestamp
        pred_time = current_time + timedelta(minutes=lt) if current_time else None

        # Check intersection with Target Location
        intersects = False
        eta: Optional[float] = None
        if target_station_coord is not None:
            tx, ty = target_station_coord
            dist_to_target = float(np.hypot(pred_cx - tx, pred_cy - ty))
            if dist_to_target <= (target_alert_radius_px + uncertainty_r):
                intersects = True

            # Calculate ETA along line of motion
            if speed_per_min > 0.05:
                # Vector from cloud to target
                dtx = tx - cx0
                dty = ty - cy0
                # Dot product with velocity vector
                proj = (dtx * vx_min + dty * vy_min) / (speed_per_min ** 2)
                if proj > 0:
                    # Target is ahead of cloud trajectory
                    cross_track_dist = abs(dtx * vy_min - dty * vx_min) / speed_per_min
                    if cross_track_dist <= target_alert_radius_px:
                        eta = proj

        extrapolations.append(ExtrapolatedCloudPosition(
            cloud_id=cloud_id,
            lead_time_minutes=lt,
            predicted_timestamp=pred_time,
            predicted_centroid=(pred_cx, pred_cy),
            predicted_bbox=pred_bbox,
            uncertainty_radius_px=uncertainty_r,
            will_intersect_target=intersects,
            eta_minutes_to_target=eta
        ))

    return extrapolations


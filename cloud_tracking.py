"""
CloudPath AI – Satellite Image Analysis and Cloud Movement Tracking
Module: cloud_tracking.py
===================================================================
Author : B.Tech Final Year Project
Role   : Detects connected cloud formations, extracts spatial moments
         (centroids, bounding boxes, areas), and associates multi-cloud
         entities across consecutive frames with unique persistent IDs,
         handling cloud births, dissipations, splits, and mergers.
"""

from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import cv2

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_ASSIGNMENT = True
except ImportError:
    SCIPY_ASSIGNMENT = False


class CloudObject:
    """Represents an individual cloud entity at a single point in time."""
    def __init__(
        self,
        cloud_id: int,
        frame_idx: int,
        timestamp: Optional[datetime],
        centroid: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        area: float,
        contour: np.ndarray,
        mean_intensity: float = 0.0
    ):
        self.cloud_id = cloud_id
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.centroid = centroid          # (cx, cy)
        self.bbox = bbox                  # (x, y, w, h)
        self.area = area                  # pixel area
        self.contour = contour
        self.mean_intensity = mean_intensity
        self.status = "ACTIVE"            # ACTIVE, APPEARED, DISAPPEARED, SPLIT, MERGED
        self.match_confidence = 1.0       # 0.0 - 1.0

        # Trajectory history
        self.history_centroids: List[Tuple[float, float]] = [centroid]
        self.history_timestamps: List[Optional[datetime]] = [timestamp]
        self.history_areas: List[float] = [area]

    def update(
        self,
        new_centroid: Tuple[float, float],
        new_bbox: Tuple[int, int, int, int],
        new_area: float,
        new_contour: np.ndarray,
        frame_idx: int,
        timestamp: Optional[datetime],
        confidence: float = 1.0,
        status: str = "ACTIVE"
    ):
        self.centroid = new_centroid
        self.bbox = new_bbox
        self.area = new_area
        self.contour = new_contour
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.status = status
        self.match_confidence = confidence

        self.history_centroids.append(new_centroid)
        self.history_timestamps.append(timestamp)
        self.history_areas.append(new_area)


class MultiCloudTracker:
    """
    Associates cloud objects across sequential satellite frames.
    Handles cloud births, deaths, splitting, and merging.
    """
    def __init__(
        self,
        max_match_distance: float = 120.0,
        min_area: float = 200.0,
        max_missed_frames: int = 2
    ):
        self.max_match_distance = max_match_distance
        self.min_area = min_area
        self.max_missed_frames = max_missed_frames

        self.next_id = 1
        self.active_tracks: Dict[int, CloudObject] = {}
        self.dead_tracks: List[CloudObject] = []
        self.missed_counts: Dict[int, int] = {}
        self.prev_gray: Optional[np.ndarray] = None

    def extract_cloud_candidates(
        self,
        binary_mask: np.ndarray,
        gray_image: np.ndarray,
        frame_idx: int,
        timestamp: Optional[datetime]
    ) -> List[CloudObject]:
        """Identifies connected cloud components and extracts geometric moments."""
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[CloudObject] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area:
                continue

            # Compute centroid using spatial moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = float(x + w / 2.0), float(y + h / 2.0)

            x, y, w, h = cv2.boundingRect(cnt)

            # Compute mean pixel intensity inside cloud contour
            mask_single = np.zeros_like(gray_image)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)
            mean_val = float(cv2.mean(gray_image, mask=mask_single)[0])

            candidate = CloudObject(
                cloud_id=-1,  # Assigned during association
                frame_idx=frame_idx,
                timestamp=timestamp,
                centroid=(cx, cy),
                bbox=(x, y, w, h),
                area=area,
                contour=cnt,
                mean_intensity=mean_val
            )
            candidates.append(candidate)

        return candidates

    def compute_optical_flow(self, current_gray: np.ndarray) -> Optional[np.ndarray]:
        """Calculates dense Farneback optical flow as a motion baseline."""
        if self.prev_gray is None or self.prev_gray.shape != current_gray.shape:
            self.prev_gray = current_gray.copy()
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = current_gray.copy()
        return flow

    def update_tracks(
        self,
        binary_mask: np.ndarray,
        gray_image: np.ndarray,
        frame_idx: int,
        timestamp: Optional[datetime]
    ) -> List[CloudObject]:
        """
        Updates cloud tracks with new frame observations.
        Solves the bipartite matching problem between existing tracks and new detections.
        """
        candidates = self.extract_cloud_candidates(binary_mask, gray_image, frame_idx, timestamp)
        flow = self.compute_optical_flow(gray_image)

        # Initial frame: register all candidates as new clouds
        if not self.active_tracks:
            for cand in candidates:
                cand.cloud_id = self.next_id
                self.next_id += 1
                cand.status = "APPEARED"
                self.active_tracks[cand.cloud_id] = cand
                self.missed_counts[cand.cloud_id] = 0
            return list(self.active_tracks.values())

        track_ids = list(self.active_tracks.keys())
        num_tracks = len(track_ids)
        num_candidates = len(candidates)

        if num_candidates == 0:
            # All clouds disappeared in this frame
            for tid in track_ids:
                self.missed_counts[tid] += 1
                self.active_tracks[tid].status = "DISAPPEARED"
                if self.missed_counts[tid] > self.max_missed_frames:
                    self.dead_tracks.append(self.active_tracks.pop(tid))
            return []

        # Build Cost Matrix based on distance and area similarity
        cost_matrix = np.full((num_tracks, num_candidates), fill_value=1e5, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            track = self.active_tracks[tid]
            tx, ty = track.centroid
            for j, cand in enumerate(candidates):
                cx, cy = cand.centroid
                dist = np.hypot(tx - cx, ty - cy)

                # Optical flow prediction bonus if available
                if flow is not None and 0 <= int(ty) < flow.shape[0] and 0 <= int(tx) < flow.shape[1]:
                    fx = flow[int(ty), int(tx), 0]
                    fy = flow[int(ty), int(tx), 1]
                    pred_x = tx + fx
                    pred_y = ty + fy
                    flow_dist = np.hypot(pred_x - cx, pred_y - cy)
                    dist = 0.7 * dist + 0.3 * flow_dist

                # Area similarity ratio (1.0 = identical, 0.0 = completely different)
                area_ratio = min(track.area, cand.area) / max(track.area, cand.area)
                cost = dist + (1.0 - area_ratio) * 40.0

                if dist <= self.max_match_distance:
                    cost_matrix[i, j] = cost

        # Solve matching assignment
        matched_tracks = set()
        matched_candidates = set()

        if SCIPY_ASSIGNMENT and num_tracks > 0 and num_candidates > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.max_match_distance:
                    tid = track_ids[r]
                    matched_tracks.add(tid)
                    matched_candidates.add(c)
                    cand = candidates[c]
                    cand.cloud_id = tid
                    # Compute confidence
                    dist = np.hypot(self.active_tracks[tid].centroid[0] - cand.centroid[0],
                                    self.active_tracks[tid].centroid[1] - cand.centroid[1])
                    conf = max(0.1, round(1.0 - (dist / self.max_match_distance), 2))

                    self.active_tracks[tid].update(
                        new_centroid=cand.centroid,
                        new_bbox=cand.bbox,
                        new_area=cand.area,
                        new_contour=cand.contour,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        confidence=conf,
                        status="ACTIVE"
                    )
                    self.missed_counts[tid] = 0
        else:
            # Greedy nearest neighbor fallback
            for i, tid in enumerate(track_ids):
                best_j = -1
                best_cost = self.max_match_distance
                for j in range(num_candidates):
                    if j not in matched_candidates and cost_matrix[i, j] < best_cost:
                        best_cost = cost_matrix[i, j]
                        best_j = j
                if best_j != -1:
                    matched_tracks.add(tid)
                    matched_candidates.add(best_j)
                    cand = candidates[best_j]
                    cand.cloud_id = tid
                    self.active_tracks[tid].update(
                        new_centroid=cand.centroid,
                        new_bbox=cand.bbox,
                        new_area=cand.area,
                        new_contour=cand.contour,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        confidence=round(1.0 - (best_cost / self.max_match_distance), 2),
                        status="ACTIVE"
                    )
                    self.missed_counts[tid] = 0

        # Handle Unmatched Candidates: New clouds or Splits
        for j, cand in enumerate(candidates):
            if j not in matched_candidates:
                # Check for possible cloud split: was it near a previously large track?
                is_split = False
                for tid in track_ids:
                    t = self.active_tracks[tid]
                    if np.hypot(t.centroid[0] - cand.centroid[0], t.centroid[1] - cand.centroid[1]) < self.max_match_distance * 1.5:
                        cand.status = "SPLIT"
                        is_split = True
                        break

                if not is_split:
                    cand.status = "APPEARED"

                cand.cloud_id = self.next_id
                self.next_id += 1
                self.active_tracks[cand.cloud_id] = cand
                self.missed_counts[cand.cloud_id] = 0

        # Handle Unmatched Tracks: Dissipation or Mergers
        for tid in track_ids:
            if tid not in matched_tracks:
                self.missed_counts[tid] += 1
                # Check for possible merger into an active candidate
                is_merged = False
                for j in matched_candidates:
                    c = candidates[j]
                    if np.hypot(self.active_tracks[tid].centroid[0] - c.centroid[0],
                                self.active_tracks[tid].centroid[1] - c.centroid[1]) < self.max_match_distance:
                        self.active_tracks[tid].status = "MERGED"
                        is_merged = True
                        break

                if not is_merged:
                    self.active_tracks[tid].status = "DISAPPEARED"

                if self.missed_counts[tid] > self.max_missed_frames:
                    self.dead_tracks.append(self.active_tracks.pop(tid))

        return list(self.active_tracks.values())


"""
cloud_matching.py
=================
CloudPath AI – Phase 4: Cloud Movement Tracking
Module 2: Match cloud regions between consecutive observation frames.

Strategy
--------
1.  Centroid distance matrix between all regions in frame t and frame t+1.
2.  Intersection-over-Union (IoU) overlap matrix as a second signal.
3.  Combined cost = alpha * normalised_distance + (1-alpha) * (1 - IoU).
4.  Hungarian algorithm (scipy.optimize.linear_sum_assignment) finds the
    globally optimal 1-to-1 assignment.
5.  Matches are accepted only when cost < MAX_MATCH_COST; others are
    labelled as appear/disappear events.

Confidence label
----------------
    HIGH   : cost < 0.30
    MEDIUM : 0.30 <= cost < 0.55
    LOW    : 0.55 <= cost < MAX_MATCH_COST
    NONE   : cost >= MAX_MATCH_COST  (not matched)

Author: CloudPath AI Project
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Optional

from cloud_region_detection import CloudRegion


# ─── tuning constants ────────────────────────────────────────────────
MAX_CENTROID_DIST_PX = 60     # pixels — maximum plausible displacement per step
MAX_MATCH_COST       = 0.65   # above this: regions are unmatched
ALPHA                = 0.65   # weight of distance vs IoU in combined cost


@dataclass
class CloudMatch:
    """Records one successful region-to-region match between two frames."""
    prev_frame_idx:  int
    curr_frame_idx:  int
    prev_region_id:  int       # region_id inside prev frame
    curr_region_id:  int       # region_id inside curr frame
    prev_centroid:   tuple     # (row, col)
    curr_centroid:   tuple     # (row, col)
    centroid_dist_px: float    # Euclidean pixel distance
    iou:             float     # intersection-over-union
    combined_cost:   float     # lower = better
    confidence:      str       # HIGH / MEDIUM / LOW


def _bbox_to_set(bbox: tuple, shape: tuple) -> np.ndarray:
    """Create a boolean mask for a bounding-box region."""
    r_min, r_max, c_min, c_max = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r_min:r_max + 1, c_min:c_max + 1] = True
    return mask


def _iou(r1: CloudRegion, r2: CloudRegion, shape: tuple) -> float:
    """
    Approximate IoU using bounding-box overlap.
    Cheap proxy — avoids storing per-pixel masks between frames.
    """
    r1_min, r1_max, c1_min, c1_max = r1.bbox
    r2_min, r2_max, c2_min, c2_max = r2.bbox

    inter_r = max(0, min(r1_max, r2_max) - max(r1_min, r2_min) + 1)
    inter_c = max(0, min(c1_max, c2_max) - max(c1_min, c2_min) + 1)
    intersection = inter_r * inter_c

    area1 = (r1_max - r1_min + 1) * (c1_max - c1_min + 1)
    area2 = (r2_max - r2_min + 1) * (c2_max - c2_min + 1)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _confidence_label(cost: float) -> str:
    if cost < 0.30:
        return "HIGH"
    elif cost < 0.55:
        return "MEDIUM"
    else:
        return "LOW"


def match_frames(prev_regions: list[CloudRegion],
                 curr_regions: list[CloudRegion],
                 grid_shape: tuple = (162, 159),
                 max_dist_px: float = MAX_CENTROID_DIST_PX
                 ) -> tuple[list[CloudMatch], list[int], list[int]]:
    """
    Match cloud regions from one frame to the next.

    Returns
    -------
    matches       : list of CloudMatch objects (accepted matches only)
    disappeared   : region_ids in prev_regions with no acceptable match
    appeared      : region_ids in curr_regions with no acceptable match
    """
    if not prev_regions or not curr_regions:
        disappeared = [r.region_id for r in prev_regions]
        appeared    = [r.region_id for r in curr_regions]
        return [], disappeared, appeared

    n_prev = len(prev_regions)
    n_curr = len(curr_regions)

    # ── Build cost matrix ────────────────────────────────────────────
    cost_matrix = np.full((n_prev, n_curr), fill_value=1.0)

    # Normalisation factor: max plausible distance
    norm_dist = max_dist_px

    for i, pr in enumerate(prev_regions):
        for j, cr in enumerate(curr_regions):
            dr = pr.centroid_r - cr.centroid_r
            dc = pr.centroid_c - cr.centroid_c
            dist = np.sqrt(dr * dr + dc * dc)

            # Hard gate: skip pairs that are too far apart
            if dist > max_dist_px:
                continue

            norm_d = dist / norm_dist
            iou_val = _iou(pr, cr, grid_shape)
            cost = ALPHA * norm_d + (1 - ALPHA) * (1 - iou_val)
            cost_matrix[i, j] = cost

    # ── Hungarian assignment ─────────────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches:    list[CloudMatch] = []
    matched_prev = set()
    matched_curr = set()

    for ri, ci in zip(row_ind, col_ind):
        cost = cost_matrix[ri, ci]
        if cost >= MAX_MATCH_COST:
            continue  # reject poor matches

        pr = prev_regions[ri]
        cr = curr_regions[ci]

        dr  = pr.centroid_r - cr.centroid_r
        dc  = pr.centroid_c - cr.centroid_c
        dist = float(np.sqrt(dr * dr + dc * dc))
        iou_val = _iou(pr, cr, grid_shape)

        matches.append(CloudMatch(
            prev_frame_idx  = pr.frame_idx,
            curr_frame_idx  = cr.frame_idx,
            prev_region_id  = pr.region_id,
            curr_region_id  = cr.region_id,
            prev_centroid   = (pr.centroid_r, pr.centroid_c),
            curr_centroid   = (cr.centroid_r, cr.centroid_c),
            centroid_dist_px = dist,
            iou             = iou_val,
            combined_cost   = cost,
            confidence      = _confidence_label(cost),
        ))
        matched_prev.add(ri)
        matched_curr.add(ci)

    disappeared = [prev_regions[i].region_id
                   for i in range(n_prev) if i not in matched_prev]
    appeared    = [curr_regions[j].region_id
                   for j in range(n_curr) if j not in matched_curr]

    return matches, disappeared, appeared


def build_all_matches(all_regions: list[list[CloudRegion]],
                      grid_shape: tuple = (162, 159)
                      ) -> list[tuple[list[CloudMatch], list[int], list[int]]]:
    """
    Match regions across every consecutive pair of frames.

    Returns a list of (matches, disappeared, appeared) tuples,
    one per consecutive frame pair.
    """
    results = []
    for i in range(len(all_regions) - 1):
        m, d, a = match_frames(all_regions[i], all_regions[i + 1], grid_shape)
        results.append((m, d, a))
    return results


# ─── standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    from cloud_region_detection import detect_all_frames

    data = np.load(
        r"d:\bengaluru.csv\satellite_data_outputs\bengaluru_cloudmask_series.npz",
        allow_pickle=True
    )
    all_r = detect_all_frames(data["cloud_masks"], data["timestamps"], data["flags"])
    pair_results = build_all_matches(all_r)

    total_matches = sum(len(m) for m, _, _ in pair_results)
    high_conf     = sum(sum(1 for mm in m if mm.confidence == "HIGH")
                        for m, _, _ in pair_results)
    print(f"Frame pairs:      {len(pair_results)}")
    print(f"Total matches:    {total_matches}")
    print(f"HIGH confidence:  {high_conf}")

    # Sample first pair
    m0, d0, a0 = pair_results[0]
    print(f"\nPair 0: {len(m0)} matches, {len(d0)} disappeared, {len(a0)} appeared")
    for mm in m0[:3]:
        print(f"  {mm.prev_region_id} -> {mm.curr_region_id}  "
              f"dist={mm.centroid_dist_px:.1f}px  "
              f"iou={mm.iou:.3f}  conf={mm.confidence}")


"""
cloud_motion_tracker.py
=======================
CloudPath AI – Phase 4: Cloud Movement Tracking
Module 3: Assign persistent track IDs across all observations and build
          complete cloud trajectories.

Track lifecycle
---------------
    NEW       : region appeared with no predecessor
    ACTIVE    : region matched from previous frame
    LOST      : track was active but found no match this frame
    TERMINATED: track lost for > MAX_LOST_FRAMES

Author: CloudPath AI Project
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from cloud_region_detection import CloudRegion
from cloud_matching         import CloudMatch, build_all_matches


MAX_LOST_FRAMES = 3   # terminate a track after this many consecutive misses


@dataclass
class TrackStep:
    """One timestep in a cloud track."""
    frame_idx:    int
    timestamp:    str
    region_id:    int
    centroid_r:   float
    centroid_c:   float
    area_pixels:  int
    dominant_flag: Optional[int]
    confidence:   str   # match confidence or "NEW"


@dataclass
class CloudTrack:
    """Full trajectory of one tracked cloud object."""
    track_id:   int
    steps:      list[TrackStep] = field(default_factory=list)
    lost_count: int = 0           # consecutive frames with no match
    active:     bool = True

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def first_timestamp(self) -> str:
        return self.steps[0].timestamp if self.steps else ""

    @property
    def last_timestamp(self) -> str:
        return self.steps[-1].timestamp if self.steps else ""


class CloudTracker:
    """
    Stateful tracker: maintains a pool of CloudTrack objects and
    updates them as new frame-pair matches arrive.
    """

    def __init__(self):
        self._tracks: list[CloudTrack] = []
        self._next_id: int = 1
        # Maps (frame_idx, region_id) -> track_id for the CURRENT frame
        self._active_map: dict[tuple[int, int], int] = {}

    # ── public API ────────────────────────────────────────────────────

    def initialise(self, frame0_regions: list[CloudRegion]):
        """Seed tracks from the very first frame."""
        for reg in frame0_regions:
            tid = self._new_track(reg, confidence="NEW")
            self._active_map[(reg.frame_idx, reg.region_id)] = tid

    def update(self,
               matches: list[CloudMatch],
               appeared: list[int],
               curr_regions: list[CloudRegion]):
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        matches      : accepted CloudMatch objects for this frame pair
        appeared     : region_ids in curr frame with no predecessor
        curr_regions : all CloudRegion objects in the current frame
        """
        curr_frame_idx = curr_regions[0].frame_idx if curr_regions else -1
        new_active: dict[tuple[int, int], int] = {}
        matched_curr_ids = set()

        # ── Extend existing tracks with matched regions ────────────
        for match in matches:
            key_prev = (match.prev_frame_idx, match.prev_region_id)
            tid = self._active_map.get(key_prev)
            if tid is None:
                # Prev region was not tracked — start new track from curr
                curr_reg = self._find_region(curr_regions, match.curr_region_id)
                if curr_reg:
                    tid = self._new_track(curr_reg, confidence=match.confidence)
            else:
                curr_reg = self._find_region(curr_regions, match.curr_region_id)
                if curr_reg:
                    track = self._get_track(tid)
                    track.steps.append(TrackStep(
                        frame_idx    = curr_reg.frame_idx,
                        timestamp    = curr_reg.timestamp,
                        region_id    = curr_reg.region_id,
                        centroid_r   = curr_reg.centroid_r,
                        centroid_c   = curr_reg.centroid_c,
                        area_pixels  = curr_reg.area_pixels,
                        dominant_flag= curr_reg.dominant_flag,
                        confidence   = match.confidence,
                    ))
                    track.lost_count = 0

            if tid is not None:
                new_active[(curr_frame_idx, match.curr_region_id)] = tid
                matched_curr_ids.add(match.curr_region_id)

        # ── Mark unmatched previous tracks as LOST ─────────────────
        for (fi, ri), tid in self._active_map.items():
            track = self._get_track(tid)
            if track is None or not track.active:
                continue
            # Check if this track's last frame was not continued
            if (fi, ri) not in {(m.prev_frame_idx, m.prev_region_id)
                                  for m in matches}:
                track.lost_count += 1
                if track.lost_count > MAX_LOST_FRAMES:
                    track.active = False

        # ── Start new tracks for appeared regions ──────────────────
        for rid in appeared:
            if rid in matched_curr_ids:
                continue
            reg = self._find_region(curr_regions, rid)
            if reg:
                tid = self._new_track(reg, confidence="NEW")
                new_active[(curr_frame_idx, rid)] = tid

        self._active_map = new_active

    @property
    def tracks(self) -> list[CloudTrack]:
        return self._tracks

    def tracks_min_length(self, min_len: int = 2) -> list[CloudTrack]:
        """Return only tracks spanning at least min_len frames."""
        return [t for t in self._tracks if t.length >= min_len]

    # ── private helpers ────────────────────────────────────────────────

    def _new_track(self, reg: CloudRegion, confidence: str) -> int:
        tid = self._next_id
        self._next_id += 1
        t = CloudTrack(track_id=tid)
        t.steps.append(TrackStep(
            frame_idx    = reg.frame_idx,
            timestamp    = reg.timestamp,
            region_id    = reg.region_id,
            centroid_r   = reg.centroid_r,
            centroid_c   = reg.centroid_c,
            area_pixels  = reg.area_pixels,
            dominant_flag= reg.dominant_flag,
            confidence   = confidence,
        ))
        self._tracks.append(t)
        return tid

    def _get_track(self, tid: int) -> Optional[CloudTrack]:
        # Tracks are stored in insertion order; most accesses are recent
        for t in reversed(self._tracks):
            if t.track_id == tid:
                return t
        return None

    @staticmethod
    def _find_region(regions: list[CloudRegion],
                     region_id: int) -> Optional[CloudRegion]:
        for r in regions:
            if r.region_id == region_id:
                return r
        return None


def run_tracker(all_regions: list[list[CloudRegion]],
                grid_shape: tuple = (162, 159)) -> CloudTracker:
    """
    Run the full tracking pipeline over all frames.

    Parameters
    ----------
    all_regions : list of per-frame CloudRegion lists (from detect_all_frames)
    grid_shape  : (rows, cols) of the satellite region

    Returns a fully populated CloudTracker instance.
    """
    if not all_regions:
        return CloudTracker()

    # Build all frame-pair matches
    pair_results = build_all_matches(all_regions, grid_shape)

    tracker = CloudTracker()
    tracker.initialise(all_regions[0])

    for i, (matches, disappeared, appeared) in enumerate(pair_results):
        curr_regions = all_regions[i + 1]
        tracker.update(matches, appeared, curr_regions)

    return tracker


# ─── standalone test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    from cloud_region_detection import detect_all_frames

    data = np.load(
        r"d:\bengaluru.csv\satellite_data_outputs\bengaluru_cloudmask_series.npz",
        allow_pickle=True
    )
    all_r = detect_all_frames(data["cloud_masks"], data["timestamps"], data["flags"])
    tracker = run_tracker(all_r)

    all_tracks  = tracker.tracks
    long_tracks = tracker.tracks_min_length(3)
    print(f"Total tracks  : {len(all_tracks)}")
    print(f"Tracks >= 3 frames: {len(long_tracks)}")

    # Show top 5 longest tracks
    sorted_t = sorted(long_tracks, key=lambda t: t.length, reverse=True)
    for t in sorted_t[:5]:
        print(f"  Track {t.track_id:3d}: {t.length} steps, "
              f"{t.first_timestamp} -> {t.last_timestamp}")


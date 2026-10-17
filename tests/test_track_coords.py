"""Focused tests for owner A's route coordinates (``camloc/track_coords.py``), step 4.

The projector wraps the unchanged organizer ``track_frame.py`` helper. Tests cover loop
lengths from loaded geometry, left-positive n, on-line zero residual, the track/pit routes
reported separately, a recovery reset forcing a global search, and the refusal to read GT.
These require the hash-bound static map and are skipped when it is absent.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pytest

from camloc import contracts
from camloc import track_coords
from camloc.contracts import (
    AuxiliaryGeometry,
    Branch,
    Estimate,
    EvidenceKind,
    PoseEvidence,
    PoseSource,
    RecoveryReset,
    Status,
)

ROOT = Path(__file__).resolve().parents[1]
MAP_MCAP = ROOT / "23-00-38_cut_120m_update" / "map.mcap"


def _estimate(x, y, yaw, t_ns):
    return Estimate(
        t_ns=t_ns,
        group_id=1,
        state=np.array([x, y, yaw, 0.0, 0.0, 0.0], dtype=np.float64),
        P=np.eye(6),
        status=Status.TRACKING,
        branch=Branch.TRACK,
        evidence_kinds=(EvidenceKind.PNP,),
        reason="accepted",
        auxiliary=AuxiliaryGeometry.unavailable("none"),
        max_support_exposure_ns=t_ns,
    )


def test_streaming_projection_matches_official_sequence_at_crossing():
    helper = track_coords._load_helper()
    angles = np.linspace(0, 2 * np.pi, 1001)
    line = helper.Line(np.column_stack([100 * np.sin(angles), 100 * np.sin(angles) * np.cos(angles)]))
    projector = track_coords._TrackProjector({"track": line, "pit": line}, "synthetic-map")
    xs, ys = [3.0, 0.2, -0.2], [3.0, -0.1, 0.3]
    stamps = np.array([1_000_000_000, 1_100_000_000, 1_200_000_000], dtype=np.int64)
    expected = line.xy_to_sn(xs, ys, [0, 0, 0], t_ns=stamps)
    for i, (x, y, stamp) in enumerate(zip(xs, ys, stamps)):
        track, _ = projector.project(_estimate(x, y, 0, int(stamp)))
        assert track.segment_index == expected[3][i]
        assert track.s_m == pytest.approx(expected[0][i])
    with pytest.raises(contracts.ContractError, match="timestamps"):
        projector.project(_estimate(0, 0, 0, int(stamps[-1])))


def _pose(camera_id, exposure_ns):
    T_WB = np.eye(4)
    return PoseEvidence(
        camera_id=camera_id,
        exposure_ns=exposure_ns,
        source=PoseSource.TRACKING,
        T_WB=T_WB,
        inlier_count=40,
        inlier_fraction=0.9,
        median_error_px=1.0,
        p90_error_px=2.0,
        occupied_cells=8,
        track_ids=np.arange(40, dtype=np.int64),
        landmark_ids=np.arange(40, dtype=np.int64),
        map_id="map:test",
        calibration_id="calib:test",
        preprocessing_id="prep:test",
        accepted=True,
        reason="accepted",
    )


@unittest.skipUnless(MAP_MCAP.exists(), "static map is not present")
class TrackProjectorTests(unittest.TestCase):
    def setUp(self):
        self.projector = track_coords.load_track_projector(MAP_MCAP)

    def test_loop_lengths_come_from_loaded_geometry(self):
        # The organizer README quotes 2976.875276596 m and 2985.489335790 m.
        self.assertAlmostEqual(self.projector.route_length("track"), 2976.875276596, places=3)
        self.assertAlmostEqual(self.projector.route_length("pit"), 2985.489335790, places=3)

    def test_reference_map_id_is_the_supplied_map_hash(self):
        self.assertEqual(self.projector.reference_map_id, contracts.SUPPLIED_MAP_SHA256)

    def test_on_line_point_has_zero_offset_and_residual(self):
        line = self.projector._lines["track"]
        point = line.p[500]
        yaw = float(line.heading[500])
        track, _pit = self.projector.project(_estimate(float(point[0]), float(point[1]), yaw, 1000))
        self.assertTrue(track.valid)
        self.assertEqual(track.route_id, "track")
        self.assertAlmostEqual(track.n_m, 0.0, places=3)
        self.assertAlmostEqual(abs(track.epsi_rad), 0.0, places=3)
        self.assertLess(track.xy_reconstruction_residual_m, 1e-3)

    def test_offset_to_the_left_is_positive_n(self):
        line = self.projector._lines["track"]
        point = line.p[500]
        yaw = float(line.heading[500])
        left_normal = np.array([-np.sin(yaw), np.cos(yaw)])
        shifted = point + 2.0 * left_normal
        track, _pit = self.projector.project(_estimate(float(shifted[0]), float(shifted[1]), yaw, 1000))
        self.assertAlmostEqual(track.n_m, 2.0, places=2)  # positive to the left

    def test_track_and_pit_are_reported_separately(self):
        line = self.projector._lines["track"]
        point = line.p[800]
        yaw = float(line.heading[800])
        track, pit = self.projector.project(_estimate(float(point[0]), float(point[1]), yaw, 1000))
        self.assertEqual(track.route_id, "track")
        self.assertEqual(pit.route_id, "pit")

    def test_reset_clears_sequence_history(self):
        line = self.projector._lines["track"]
        point = line.p[500]
        yaw = float(line.heading[500])
        self.projector.project(_estimate(float(point[0]), float(point[1]), yaw, 1000))
        self.assertTrue(self.projector._last_t_ns)  # history recorded
        reset = RecoveryReset(
            reset_id=0,
            group_id=2,
            t_ns=1_000_000_000 + contracts.RECOVERY_CONFIRM_MIN_NS,
            candidate=_pose("fr", 1_000_000_000),
            confirmation=_pose("fr", 1_000_000_000 + contracts.RECOVERY_CONFIRM_MIN_NS),
            jump_m=12.0,
            reason="confirmed_recovery",
        )
        self.projector.reset(reset)
        self.assertFalse(self.projector._last_t_ns)  # cleared -> next projection is global

    def test_recovery_jump_beyond_window_reprojects_globally(self):
        line = self.projector._lines["track"]
        # Project near segment 100, then jump across the lap to segment 2000.
        near = line.p[100]
        far = line.p[2000]
        yaw_near = float(line.heading[100])
        yaw_far = float(line.heading[2000])
        first, _ = self.projector.project(_estimate(float(near[0]), float(near[1]), yaw_near, 1000))
        reset = RecoveryReset(
            reset_id=1, group_id=2,
            t_ns=1_000_000_000 + contracts.RECOVERY_CONFIRM_MIN_NS,
            candidate=_pose("fr", 1_000_000_000),
            confirmation=_pose("fr", 1_000_000_000 + contracts.RECOVERY_CONFIRM_MIN_NS),
            jump_m=500.0, reason="confirmed_recovery",
        )
        self.projector.reset(reset)
        jumped, _ = self.projector.project(_estimate(float(far[0]), float(far[1]), yaw_far, 3000))
        # After a global reset the far point projects onto its true segment, not clamped to
        # the +-200 window around segment 100.
        self.assertTrue(jumped.valid)
        self.assertLess(jumped.xy_reconstruction_residual_m, 1e-2)
        self.assertGreater(abs(jumped.segment_index - first.segment_index), 200)


@unittest.skipUnless(MAP_MCAP.exists(), "static map is not present")
class HashGuardTests(unittest.TestCase):
    def test_tampered_map_is_rejected(self):
        with self.assertRaises(contracts.ContractError):
            track_coords.load_track_projector(ROOT / "metadata.yaml")


if __name__ == "__main__":
    unittest.main()

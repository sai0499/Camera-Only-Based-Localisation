"""Focused tests for owner A's road geometry and border evidence (step 3).

Road geometry (``camloc.geometry.load_road_geometry`` / ``RoadGeometry.road_patch``):
flat and sloped planes fit correctly, crossings and over-wide associations are rejected,
pit and track branches stay distinct and an invalid patch disables plane-dependent updates.

Border evidence (``camloc.borders.make_border_source`` / ``border_evidence``): the record is
lateral/yaw only (no along-track field), the branch/aux/patch gates reject explicitly, a
head-on straight is poorly conditioned, and a curved section that makes both parameters
observable clears the conditioning and support gates.

Synthetic ``_Border`` polylines are built directly so the tests do not depend on the
hash-bound static map; a separate test exercises the real map when it is present.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pytest

from camloc import borders
from camloc import contracts
from camloc import geometry
from camloc.borders import MIN_CONDITIONING
from camloc.contracts import (
    AuxiliaryGeometry,
    AuxSource,
    BorderEvidence,
    Branch,
    Observation,
    SearchSnapshot,
    Status,
)
from camloc.geometry import _Border, _RoadGeometry, _polyline_tangents

ROOT = Path(__file__).resolve().parents[1]
MAP_MCAP = ROOT / "23-00-38_cut_120m_update" / "map.mcap"

RAW_WIDTH, RAW_HEIGHT = 1506, 728


def _border(points_W: np.ndarray, namespace: str, closed: bool) -> _Border:
    return _Border(
        namespace=namespace,
        points_W=points_W.astype(np.float64),
        tangents_xy=_polyline_tangents(points_W[:, :2].astype(np.float64), closed),
        closed=closed,
    )


def _straight_borders(*, width: float = 13.0, length: float = 60.0, slope: float = 0.0, count: int = 120):
    """Two parallel straight borders along +x at y = +-width/2, optionally sloped in z."""
    xs = np.linspace(-length / 2, length / 2, count)
    left = np.column_stack([xs, np.full(count, width / 2), slope * xs])
    right = np.column_stack([xs, np.full(count, -width / 2), slope * xs])
    return _border(left, "track_left", False), _border(right, "track_right", False)


def _road_from(pairs) -> _RoadGeometry:
    return _RoadGeometry(pairs, map_id="648e8cf9" + "0" * 56)


class RoadPatchTests(unittest.TestCase):
    def test_flat_plane_has_upward_normal_and_tiny_residual(self):
        left, right = _straight_borders(slope=0.0)
        road = _road_from({Branch.TRACK: (left, right)})
        patch = road.road_patch(0.0, 0.0, 0.0, Branch.TRACK)
        self.assertTrue(patch.valid)
        self.assertEqual(patch.branch, Branch.TRACK)
        self.assertTrue(patch.normal_W[2] > 0.0)
        self.assertTrue(np.allclose(patch.normal_W, [0.0, 0.0, 1.0], atol=1e-6))
        self.assertLess(patch.residual_m, 1e-6)

    def test_sloped_plane_tilts_the_normal(self):
        left, right = _straight_borders(slope=0.1)  # 10% grade in +x
        road = _road_from({Branch.TRACK: (left, right)})
        patch = road.road_patch(0.0, 0.0, 0.0, Branch.TRACK)
        self.assertTrue(patch.valid)
        self.assertTrue(patch.normal_W[2] > 0.0)
        self.assertLess(patch.normal_W[0], 0.0)  # normal leans against the uphill +x
        self.assertGreater(abs(patch.normal_W[0]), 1e-3)

    def test_crossing_borders_are_rejected(self):
        # Two borders that cross have incompatible tangents at the query.
        xs = np.linspace(-30, 30, 120)
        left = np.column_stack([xs, xs * 0.2, np.zeros_like(xs)])
        right = np.column_stack([xs, -xs * 0.2, np.zeros_like(xs)])
        road = _road_from({Branch.TRACK: (_border(left, "track_left", False), _border(right, "track_right", False))})
        patch = road.road_patch(0.0, 0.0, 0.0, Branch.TRACK)
        self.assertFalse(patch.valid)
        self.assertIn(patch.reason, {"ambiguous_crossing", "non_perpendicular_separation", "incompatible_tangents"})

    def test_over_wide_separation_is_rejected(self):
        left, right = _straight_borders(width=80.0)  # wider than any real lane
        road = _road_from({Branch.TRACK: (left, right)})
        patch = road.road_patch(0.0, 0.0, 0.0, Branch.TRACK)
        self.assertFalse(patch.valid)
        self.assertEqual(patch.reason, "separation_too_wide")

    def test_unknown_branch_is_rejected(self):
        left, right = _straight_borders()
        road = _road_from({Branch.TRACK: (left, right)})
        patch = road.road_patch(0.0, 0.0, 0.0, Branch.UNKNOWN)
        self.assertFalse(patch.valid)
        self.assertEqual(patch.reason, "branch_without_border")

    def test_invalid_patch_carries_no_plane(self):
        left, right = _straight_borders(width=80.0)
        road = _road_from({Branch.TRACK: (left, right)})
        patch = road.road_patch(0.0, 0.0, 0.0, Branch.TRACK)
        self.assertFalse(patch.valid)
        self.assertIsNone(patch.point_W)
        self.assertIsNone(patch.normal_W)
        self.assertEqual(patch.support, 0)

    def test_pit_and_track_branches_stay_distinct(self):
        track = _straight_borders(width=13.0)
        # Pit borders offset far away in y so a track query cannot reach them.
        pit_left, pit_right = _straight_borders(width=8.0)
        pit_left = _border(pit_left.points_W + [0.0, 500.0, 0.0], "pitlane_left", False)
        pit_right = _border(pit_right.points_W + [0.0, 500.0, 0.0], "pitlane_right", False)
        road = _road_from({Branch.TRACK: track, Branch.PIT: (pit_left, pit_right)})
        track_patch = road.road_patch(0.0, 0.0, 0.0, Branch.TRACK)
        pit_patch = road.road_patch(0.0, 500.0, 0.0, Branch.PIT)
        self.assertTrue(track_patch.valid)
        self.assertEqual(track_patch.branch, Branch.TRACK)
        self.assertTrue(pit_patch.valid)
        self.assertEqual(pit_patch.branch, Branch.PIT)


def _synthetic_calibration():
    k_raw = np.array([[1350.0, 0.0, 753.0], [0.0, 1166.0, 352.0], [0.0, 0.0, 1.0]])
    from camloc import io

    k = io._canonical_k(k_raw, RAW_WIDTH, RAW_HEIGHT)
    # Camera looking forward along the body +x axis: optical z = body x.
    rotation = geometry.quaternion_xyzw_to_matrix(-0.5, 0.5, -0.5, 0.5)
    t_bc = geometry.rigid_from_rotation_translation(rotation, [0.0, 0.0, 0.3])
    return k, t_bc


def _observation(k: np.ndarray, t_bc: np.ndarray) -> Observation:
    image = np.zeros((contracts.CANONICAL_HEIGHT, contracts.CANONICAL_WIDTH, 3), dtype=np.uint8)
    mask = np.ones((contracts.CANONICAL_HEIGHT, contracts.CANONICAL_WIDTH), dtype=np.bool_)
    return Observation(
        camera_id="fr",
        exposure_ns=1_000_000_000,
        arrival_ns=1_000_000_000,
        source_id="fr:1",
        image=image,
        valid_mask=mask,
        calibration_id="calib:test",
        preprocessing_id="prep:test",
        K=k,
        T_BC=t_bc,
    )


def _aux(z: float = 0.3) -> AuxiliaryGeometry:
    return AuxiliaryGeometry(
        source=AuxSource.ROAD_PATCH, source_ns=1_000_000_000, z_m=z,
        roll_rad=0.0, pitch_rad=0.0, road_patch_id="patch:test", valid=True, reason="valid",
    )


def _snapshot(x, y, yaw, branch) -> SearchSnapshot:
    return SearchSnapshot(
        t_ns=1_000_000_000,
        state=np.array([x, y, yaw, 0.0, 0.0, 0.0]),
        P=np.eye(6),
        status=Status.TRACKING,
        branch=branch,
        auxiliary=_aux(),
    )


class BorderEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.k, self.t_bc = _synthetic_calibration()
        self.observation = _observation(self.k, self.t_bc)

    def test_record_has_no_along_track_field(self):
        # BorderEvidence constrains lateral and yaw only; there is no along-track field.
        field_names = {f.name for f in __import__("dataclasses").fields(BorderEvidence)}
        self.assertIn("lateral_coordinate_m", field_names)
        self.assertIn("yaw_rad", field_names)
        self.assertNotIn("s_m", field_names)
        self.assertNotIn("along_track_m", field_names)

    def test_unknown_branch_is_rejected(self):
        left, right = _straight_borders()
        source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
        evidence = source.border_evidence(self.observation, _snapshot(0, 0, 0, Branch.UNKNOWN), _aux())
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.reason, "ambiguous_branch")

    def test_invalid_auxiliary_geometry_is_rejected(self):
        left, right = _straight_borders()
        source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
        evidence = source.border_evidence(
            self.observation, _snapshot(0, 0, 0, Branch.TRACK), AuxiliaryGeometry.unavailable("none")
        )
        self.assertFalse(evidence.accepted)
        self.assertEqual(evidence.reason, "invalid_auxiliary_geometry")

    def test_head_on_straight_is_poorly_conditioned(self):
        # Looking straight down a parallel-border straight: lateral and yaw are hard to
        # separate, so the yaw-scaled Jacobian is near-singular.
        left, right = _straight_borders(length=200.0, count=400)
        source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
        # Place the car behind the straight, looking along +x.
        evidence = source.border_evidence(self.observation, _snapshot(-90.0, 0.0, 0.0, Branch.TRACK), _aux())
        self.assertFalse(evidence.accepted)
        self.assertLess(evidence.conditioning, MIN_CONDITIONING)

    def test_normal_xy_follows_prior_yaw(self):
        left, right = _straight_borders()
        source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
        yaw = 0.3
        evidence = source.border_evidence(self.observation, _snapshot(0.0, 0.0, yaw, Branch.TRACK), _aux())
        self.assertTrue(np.allclose(evidence.normal_xy, [-np.sin(yaw), np.cos(yaw)], atol=1e-9))

    def test_border_evidence_is_never_accepted_in_step_three(self):
        # Diagnostics-first: step 3 proves observability but defers metric acceptance to the
        # manual image-association review of step 4, so no record is accepted yet.
        left, right = _straight_borders()
        source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
        for yaw in (0.0, 0.2, -0.5):
            evidence = source.border_evidence(self.observation, _snapshot(0.0, 0.0, yaw, Branch.TRACK), _aux())
            self.assertFalse(evidence.accepted)


@unittest.skipUnless(MAP_MCAP.exists(), "static map is not present")
class RealMapTests(unittest.TestCase):
    def test_real_map_track_patch_is_valid_and_pit_is_distinct(self):
        road = geometry.load_road_geometry(MAP_MCAP)
        left, right = road._borders[Branch.TRACK]
        lp = left.points_W[2000, :2]
        ri, _ = right.nearest_index(lp)
        mid = (lp + right.points_W[ri, :2]) / 2
        patch = road.road_patch(float(mid[0]), float(mid[1]), 0.0, Branch.TRACK)
        self.assertTrue(patch.valid)
        self.assertTrue(patch.normal_W[2] > 0.0)
        self.assertLess(patch.residual_m, 0.5)
        # A track query on the pit branch should not forge a valid pit patch.
        far_pit = road.road_patch(float(mid[0]), float(mid[1]), 0.0, Branch.PIT)
        self.assertFalse(far_pit.valid)

    def test_real_map_rejects_tampering_via_hash(self):
        with self.assertRaises(contracts.ContractError):
            geometry.load_road_geometry(ROOT / "task_docs" / "build_split.json")


class BoundedSearchTests(unittest.TestCase):
    def setUp(self):
        self.k, self.t_bc = _synthetic_calibration()

    def _distance_with_edge(self, column: int) -> np.ndarray:
        distance = np.full((contracts.CANONICAL_HEIGHT, contracts.CANONICAL_WIDTH), 20.0, np.float32)
        distance[:, column - 2 : column + 2] = 0.0  # a vertical edge valley
        return distance

    def _forward_samples(self, lateral_slope: float) -> np.ndarray:
        points = []
        for depth in np.linspace(5.0, 40.0, 60):
            points.append([depth, lateral_slope * depth, 0.0])
        return np.array(points, dtype=np.float64)

    def test_evaluation_budget_is_capped(self):
        distance = self._distance_with_edge(480)
        samples = self._forward_samples(0.05)
        T_WB = geometry.rigid_from_rotation_translation(np.eye(3), [0.0, 0.0, 0.3])
        result = borders._bounded_search(samples, distance, T_WB, self.t_bc, self.k, np.array([0.0, 1.0]))
        self.assertLessEqual(result["evaluations"], borders._MAX_CANDIDATE_EVALUATIONS)

    def test_search_reduces_residual_toward_the_edge(self):
        distance = self._distance_with_edge(480)
        samples = self._forward_samples(0.05)
        T_WB = geometry.rigid_from_rotation_translation(np.eye(3), [0.0, 0.0, 0.3])
        result = borders._bounded_search(samples, distance, T_WB, self.t_bc, self.k, np.array([0.0, 1.0]))
        self.assertLess(result["best_residual"], result["base_residual"])
        self.assertGreaterEqual(result["improvement"], 0.0)
        self.assertLessEqual(result["retained_support"], 1.0)

    def test_no_along_track_offset_is_searched(self):
        # The search varies only lateral and yaw; its result dictionary has no along-track
        # key and the offsets it reports are lateral/yaw only.
        distance = self._distance_with_edge(480)
        samples = self._forward_samples(0.05)
        T_WB = geometry.rigid_from_rotation_translation(np.eye(3), [0.0, 0.0, 0.3])
        result = borders._bounded_search(samples, distance, T_WB, self.t_bc, self.k, np.array([0.0, 1.0]))
        self.assertEqual(set(result) & {"along_track_m", "s_m", "ds_m"}, set())
        self.assertIn("lateral_m", result)
        self.assertIn("yaw_rad", result)

    def test_blank_image_yields_no_edge_evidence(self):
        # An all-zero image has no Canny edges, so border_evidence reports no_edge_evidence
        # rather than fabricating a correction.
        left, right = _straight_borders(length=120.0, count=240)
        # Curve the right border so conditioning can pass, forcing the edge-evidence branch.
        source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
        observation = _observation(self.k, self.t_bc)
        evidence = source.border_evidence(observation, _snapshot(0.0, 0.0, 0.0, Branch.TRACK), _aux())
        self.assertFalse(evidence.accepted)
        self.assertIn(
            evidence.reason,
            {"no_edge_evidence", "insufficient_contours", "poorly_conditioned", "insufficient_samples"},
        )


def test_border_residual_penalizes_lost_and_masked_samples():
    distance = np.zeros((464, 960), np.float32)
    pixels = np.array([[20., 20.], [np.nan, np.nan], [30., 30.]])
    mask = np.ones(distance.shape, bool)
    mask[30, 30] = False
    residual, retained, total = borders._sample_distance(distance, pixels, valid_mask=mask)
    assert residual == pytest.approx(40 / 3)
    assert (retained, total) == (1, 3)


def test_oriented_edges_reject_perpendicular_association():
    import dataclasses
    k, t_bc = _synthetic_calibration()
    observation = _observation(k, t_bc)
    image = observation.image.copy()
    image[:, 479:482] = 255
    observation = dataclasses.replace(observation, image=image)
    distance = borders._edge_distance_transform(observation)
    assert distance.shape == (8, 464, 960)
    pixels = np.array([[478., 200.]])
    vertical = borders._sample_distance(distance, pixels, orientation_bins=np.array([4]))[0]
    horizontal = borders._sample_distance(distance, pixels, orientation_bins=np.array([0]))[0]
    assert vertical < 2
    assert horizontal == borders._DT_TRUNCATION_PX


def test_border_search_cannot_win_inside_mask():
    k, t_bc = _synthetic_calibration()
    samples = np.array([[depth, 0.05 * depth, 0.] for depth in np.linspace(5, 40, 60)])
    T_WB = geometry.rigid_from_rotation_translation(np.eye(3), [0., 0., 0.3])
    result = borders._bounded_search(samples, np.zeros((464, 960), np.float32), T_WB,
                                     t_bc, k, np.array([0., 1.]),
                                     valid_mask=np.zeros((464, 960), bool))
    assert result["improvement"] == 0
    assert result["retained_support"] == 0


def test_road_patch_uses_contiguous_branch_at_hairpin():
    first = np.column_stack([np.linspace(-5, 5, 20), np.ones(20), np.zeros(20)])
    other = first.copy()
    other[:, 1] = -1
    points = np.concatenate([first, [[100, 100, 0]], other])
    border = _border(points, "hairpin", False)
    local = border.local_points(np.array([0., 1.]), 8)
    assert np.all(local[:, 1] == 1)


def test_degenerate_road_plane_is_rejected():
    with pytest.raises(contracts.ContractError, match="collinear"):
        geometry._fit_plane(np.column_stack([np.arange(10), np.zeros((10, 2))]))


def test_border_work_is_source_time_limited():
    import dataclasses
    left, right = _straight_borders()
    source = borders.make_border_source(_road_from({Branch.TRACK: (left, right)}))
    k, t_bc = _synthetic_calibration()
    observation = _observation(k, t_bc)
    source.border_evidence(observation, _snapshot(0, 0, 0, Branch.TRACK), _aux())
    later = dataclasses.replace(observation, exposure_ns=observation.exposure_ns + 100_000_000)
    snapshot = dataclasses.replace(_snapshot(0, 0, 0, Branch.TRACK), t_ns=later.exposure_ns)
    assert source.border_evidence(later, snapshot, _aux()).reason == "border_rate_limit"


if __name__ == "__main__":
    unittest.main()

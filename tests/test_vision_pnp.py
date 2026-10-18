"""B's RANSAC PnP gates on synthetic correspondences with a known body pose."""

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import pnp


@pytest.fixture(scope="module")
def scene():
    return vf.Scene(count_u=16, count_v=10)


def _inputs(scene, k=3):
    pixels, visible = scene.project(k)
    return pixels[visible], scene.points_W[visible], scene.pose(k)


def test_clean_correspondences_recover_the_body_pose(scene):
    pixels, points, truth = _inputs(scene)
    solution = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC)
    assert solution.accepted
    np.testing.assert_allclose(solution.T_WB, truth, atol=1e-6)
    assert solution.inliers.size == len(pixels) and solution.occupied_cells == 16


def test_outliers_are_excluded_from_the_inliers(scene):
    pixels, points, truth = _inputs(scene)
    rng = np.random.default_rng(0)
    bad = rng.choice(len(pixels), len(pixels) // 4, replace=False)
    pixels = pixels.copy()
    pixels[bad] += rng.uniform(30, 80, (bad.size, 2)) * rng.choice([-1, 1], (bad.size, 2))
    pixels = np.clip(pixels, 0, [959, 463])
    solution = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC)
    assert solution.accepted
    assert not set(bad.tolist()) & set(solution.inliers.tolist())
    np.testing.assert_allclose(solution.T_WB[:3, 3], truth[:3, 3], atol=1e-3)


def test_gates_reject_weak_geometry(scene):
    pixels, points, _ = _inputs(scene)
    assert pnp.solve_pnp(pixels[:10], points[:10], scene.K, scene.T_BC).reason == \
        "too_few_correspondences"
    clustered = (pixels[:, 0] < 240) & (pixels[:, 1] < 116)  # one grid cell
    rows = np.flatnonzero((pixels[:, 0] < 480) & (pixels[:, 1] < 232))
    reason = pnp.solve_pnp(pixels[rows], points[rows], scene.K, scene.T_BC).reason
    assert reason in ("poor_image_coverage", "too_few_correspondences")
    assert clustered.sum() < 30
    noisy = pixels + np.random.default_rng(1).normal(0, 3.0, pixels.shape)
    assert pnp.solve_pnp(noisy, points, scene.K, scene.T_BC).reason in (
        "high_median_error", "high_p90_error", "low_inlier_fraction")


def test_pose_evidence_records_inliers_and_identity(scene):
    pixels, points, _ = _inputs(scene)
    obs = scene.observation(3)
    ids = np.arange(len(pixels), dtype=np.int64) + 40
    solution = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC)
    evidence = pnp.pose_evidence(solution, observation=obs, source=c.PoseSource.TRACKING,
                                 track_ids=ids, landmark_ids=ids + 1, num_candidates=len(ids),
                                 map_id=f.MAP_ID)
    assert evidence.accepted and evidence.inlier_count == len(ids)
    assert evidence.calibration_id == obs.calibration_id and evidence.map_id == f.MAP_ID
    assert evidence.median_error_px <= evidence.p90_error_px < 1e-3
    rejected = pnp.rejected_pose(obs, c.PoseSource.HANDOFF, f.MAP_ID, "no_active_tracks")
    assert not rejected.accepted and rejected.T_WB is None and rejected.inlier_count == 0


def test_occupied_cells_over_full_image_and_usable_extent():
    pixels = np.array([[10.0, 10.0], [950.0, 10.0], [10.0, 200.0], [950.0, 200.0]])
    assert pnp.occupied_cells(pixels) == 4
    mask = np.zeros((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    mask[:232] = True
    assert pnp.occupied_cells(pixels, mask) == 4
    assert pnp.usable_extent(mask) == (-0.5, 959.5, -0.5, 231.5)

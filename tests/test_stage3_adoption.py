"""Adoption consistency filter (stage3-visual-baseline-gate-fix task 3.3, B's ``_adopt``).

On an accepted handoff the active LK pool becomes the fresh adopted tracks followed by
only those old active tracks whose landmarks reproject within the RANSAC threshold and
lie in front of the camera under the verified adopted pose.
"""

import math

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c, vision


@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


@pytest.fixture(scope="module")
def dense():
    """640 landmarks, so fresh + old exceeds the 512-track cap."""
    return vf.Scene(seed=7, count_u=32, count_v=20)


def _corr(scene, k, map_id, rows, offset=1000):
    pixels, visible = scene.project(k)
    pick = np.flatnonzero(visible)[rows]
    return c.Correspondences(camera_id="fr", exposure_ns=scene.t_ns(k),
                             keypoint_ids=np.arange(pick.size, dtype=np.int64) + offset,
                             pixels=pixels[pick].astype(np.float32),
                             landmark_ids=scene.landmark_ids[pick], map_id=map_id)


def _adopt_handoff(scene, tracker, obs, rows, job_id):
    result = f.make_anchor_result(job_id=job_id, source_exposure_ns=scene.t_ns(0),
                                  map_id=tracker.map.map_id,
                                  correspondences=_corr(scene, 0, tracker.map.map_id, rows),
                                  matched_reference_ids=(1,), remaining_candidate_ids=())
    handoff = tracker.handoff(result, obs)
    assert handoff.accepted, handoff.reason
    tracker.settle(handoff, c.HandoffDisposition.ADOPT)
    return handoff


def _reprojection(scene, tracker, landmark_ids, pixels, T_WB):
    projected, depth = vision.project(tracker.map.points(landmark_ids), T_WB, scene.T_BC, scene.K)
    return np.linalg.norm(projected - pixels, axis=1), depth


def _track_set(scene, rows, track_ids, shift=None):
    pixels, visible = scene.project(0)
    pick = np.flatnonzero(visible)[rows]
    px = pixels[pick].astype(np.float64)
    if shift is not None:
        px = px + shift
    return vision._TrackSet("fr", scene.t_ns(0), np.asarray(track_ids, np.int64),
                            scene.landmark_ids[pick].copy(), px.astype(np.float32))


def _check_pool(pool, fresh_landmarks, expected_old):
    """Fresh first, then the expected old tracks (in their original order), capped."""
    n = min(fresh_landmarks.size, c.MAX_ACTIVE_TRACKS)
    np.testing.assert_array_equal(pool.landmark_ids[:n], fresh_landmarks[:n])
    room = c.MAX_ACTIVE_TRACKS - n
    np.testing.assert_array_equal(pool.track_ids[n:], expected_old.track_ids[:room])
    np.testing.assert_array_equal(pool.landmark_ids[n:], expected_old.landmark_ids[:room])
    np.testing.assert_array_equal(pool.pixels[n:], expected_old.pixels[:room])
    assert len(pool) <= c.MAX_ACTIVE_TRACKS
    assert np.unique(pool.track_ids).size == len(pool)
    assert np.unique(pool.landmark_ids).size == len(pool)


# Feature: stage3-visual-baseline-gate-fix, Property 3: Adoption keeps only tracks consistent with the verified pose
def test_adopt_mixed_stale_old_tracks_dropped_end_to_end(scene, tmp_path):
    """Only part of the old pool is stale; exactly the consistent old tracks survive.

    **Validates: Requirements 2.2, 3.2**
    """
    bank = scene.build_map(tmp_path / "bank")
    visible = int(scene.project(0)[1].sum())
    for iteration in range(6):
        rng = f.seeded_rng(331, iteration)
        tracker = vision.VisionTracker(bank)
        obs = {k: scene.observation(k) for k in (0, 1)}
        for k in (0, 1):
            tracker.observe(obs[k])
        order = rng.permutation(visible)
        rows_a, rows_b = np.sort(order[: visible // 2]), np.sort(order[visible // 2:])
        _adopt_handoff(scene, tracker, obs[1], rows_a, job_id=1)
        active = tracker._active["fr"]
        stale = rng.random(len(active)) < 0.5
        angle = rng.uniform(-math.pi, math.pi, int(stale.sum()))
        active.pixels[stale] += 10.0 * np.column_stack([np.cos(angle), np.sin(angle)])
        before = active.subset(np.arange(len(active)))
        handoff = _adopt_handoff(scene, tracker, obs[1], rows_b, job_id=2)
        fresh = handoff.correspondences.landmark_ids
        error, depth = _reprojection(scene, tracker, before.landmark_ids, before.pixels,
                                     handoff.pose.T_WB)
        consistent = (error <= 4.0) & (depth > 0.5)
        assert not consistent[stale].any()  # 10 px shifts are inconsistent
        expected = before.subset(np.flatnonzero(consistent & ~np.isin(before.landmark_ids, fresh)))
        pool = tracker._active["fr"]
        _check_pool(pool, fresh, expected)
        assert not np.isin(before.track_ids[stale], pool.track_ids).any()
        assert not np.isin(pool.track_ids[:fresh.size], before.track_ids).any()


# Feature: stage3-visual-baseline-gate-fix, Property 3: Adoption keeps only tracks consistent with the verified pose
def test_adopt_cap_unique_ids_and_order(dense, tmp_path):
    """Fresh + consistent old > 512: cap applies to the filtered old tracks, ids unique.

    **Validates: Requirements 2.2, 3.2**
    """
    bank = dense.build_map(tmp_path / "dense")
    visible = int(dense.project(0)[1].sum())
    assert visible > c.MAX_ACTIVE_TRACKS
    T_WB = dense.pose(0)
    for iteration in range(8):
        rng = f.seeded_rng(332, iteration)
        tracker = vision.VisionTracker(bank)
        old_rows = np.sort(rng.permutation(visible)[: int(visible * 0.9)])
        shift = np.zeros((old_rows.size, 2))
        stale = rng.random(old_rows.size) < rng.uniform(0.0, 0.6)
        angle = rng.uniform(-math.pi, math.pi, int(stale.sum()))
        shift[stale] = rng.uniform(5.0, 30.0, (stale.sum(), 1)) * np.column_stack(
            [np.cos(angle), np.sin(angle)])
        old = _track_set(dense, old_rows, 10_000 + np.arange(old_rows.size), shift)
        tracker._active["fr"] = old
        fresh_rows = np.sort(rng.permutation(visible)[: int(rng.integers(100, 400))])
        fresh = _track_set(dense, fresh_rows, np.zeros(fresh_rows.size))
        tracker._adopt(fresh, T_WB)
        pool = tracker._active["fr"]
        keep = ~stale & ~np.isin(old.landmark_ids, fresh.landmark_ids)
        _check_pool(pool, fresh.landmark_ids, old.subset(np.flatnonzero(keep)))
        assert not np.isin(pool.track_ids[:fresh_rows.size], old.track_ids).any()


# Feature: stage3-visual-baseline-gate-fix, Property 3: Adoption keeps only tracks consistent with the verified pose
def test_adopt_drops_old_tracks_behind_camera(scene, tmp_path):
    """Exact old pixels are still dropped when the adopted pose puts them behind the camera.

    **Validates: Requirements 2.2**
    """
    bank = scene.build_map(tmp_path / "bank")
    visible = int(scene.project(0)[1].sum())
    for iteration in range(6):
        rng = f.seeded_rng(333, iteration)
        tracker = vision.VisionTracker(bank)
        old_rows = np.sort(rng.permutation(visible)[: visible // 2])
        old = _track_set(scene, old_rows, 10_000 + np.arange(old_rows.size))
        tracker._active["fr"] = old
        fresh_rows = np.setdiff1d(np.arange(visible), old_rows)
        fresh = _track_set(scene, fresh_rows, np.zeros(fresh_rows.size))
        # Facing away (yaw + pi) or driven past every landmark (> 45 m ahead).
        T_WB = scene.pose(0).copy()
        if iteration % 2:
            T_WB[:3, :3] = vf.rot_z(scene.yaw + math.pi)
        else:
            T_WB[:3, 3] += np.r_[vf.rot_z(scene.yaw)[:2, 0], 0.0] * rng.uniform(60.0, 90.0)
        tracker._adopt(fresh, T_WB)
        pool = tracker._active["fr"]
        np.testing.assert_array_equal(pool.landmark_ids, fresh.landmark_ids)
        assert not np.isin(pool.track_ids, old.track_ids).any()
        assert np.unique(pool.track_ids).size == len(pool)

"""Joint handoff (stage3-gate-fix-round2 task 5, design fallback F2, B's ``handoff``).

When the carried handoff set fails only on support (poor coverage / too few inliers) and the
same image's ``track`` solve also failed, PnP is re-solved once on the handoff set plus the
current active LK tracks of other landmarks, with unique landmark ids and the unchanged gates.
"""

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import pnp, vision

LEFT_MAX_U, RIGHT_MIN_U = 220.0, 740.0  # inside grid columns 0 and 3 of the 4 x 4 cells


@pytest.fixture(scope="module")
def dense():
    return vf.Scene(seed=7, count_u=32, count_v=20)


@pytest.fixture(scope="module")
def bank(dense, tmp_path_factory):
    return dense.build_map(tmp_path_factory.mktemp("joint") / "bank")


def _band_corr(scene, k, map_id, right, rows=None):
    pixels, visible = scene.project(k)
    band = pixels[:, 0] >= RIGHT_MIN_U if right else pixels[:, 0] <= LEFT_MAX_U
    pick = np.flatnonzero(visible & band)
    if rows is not None:
        pick = pick[rows]
    return c.Correspondences(camera_id="fr", exposure_ns=scene.t_ns(k),
                             keypoint_ids=np.arange(pick.size, dtype=np.int64) + 7000,
                             pixels=pixels[pick].astype(np.float32),
                             landmark_ids=scene.landmark_ids[pick], map_id=map_id)


def _result(scene, tracker, corr, job_id):
    return f.make_anchor_result(job_id=job_id, source_exposure_ns=corr.exposure_ns,
                                map_id=tracker.map.map_id, correspondences=corr,
                                matched_reference_ids=(1,), remaining_candidate_ids=())


def _left_tracks_rejected(scene, bank, keep=None):
    """Adopt a full handoff at frame 1, keep only its left-band tracks, track frame 2.

    Returns the tracker, the frame-2 observation and the rejected tracking evidence."""
    tracker = vision.VisionTracker(bank)
    obs = {k: scene.observation(k) for k in (0, 1, 2)}
    for k in (0, 1, 2):
        tracker.observe(obs[k])
    full = tracker.handoff(_result(scene, tracker, scene.correspondences(0, bank.map_id), 1),
                           obs[1])
    assert full.accepted, full.reason
    tracker.settle(full, c.HandoffDisposition.ADOPT)
    active = tracker._active["fr"]
    left = np.flatnonzero(active.pixels[:, 0] <= LEFT_MAX_U)
    if keep is not None:
        left = left[keep(left.size)]
    tracker._active["fr"] = active.subset(left)
    tracked = tracker.track(obs[2], scene.snapshot(2), c.AuxiliaryGeometry.unavailable("none"))
    assert not tracked.accepted and tracked.T_WB is not None  # solved, failed on support
    return tracker, obs[2], tracked


def _joint_events(tracker):
    return [e for e in tracker.events if e.get("event") == "joint_handoff"]


def test_joint_handoff_accepts_when_each_set_alone_fails_coverage(dense, bank):
    tracker, current, tracked = _left_tracks_rejected(dense, bank)
    corr = _band_corr(dense, 0, bank.map_id, right=True)
    handoff = tracker.handoff(_result(dense, tracker, corr, 2), current)
    assert handoff.accepted and handoff.reason == "accepted"
    assert handoff.pose.occupied_cells >= pnp.DEFAULT_GATES.min_cells
    assert handoff.pose.inlier_count >= pnp.DEFAULT_GATES.min_inliers
    used = handoff.correspondences
    assert used.landmark_ids.size <= c.MAX_HANDOFF_POINTS
    assert np.unique(used.landmark_ids).size == used.landmark_ids.size
    assert np.unique(used.keypoint_ids).size == used.keypoint_ids.size
    assert np.isin(tracked.landmark_ids, used.landmark_ids).any()  # current LK tracks joined
    assert np.isin(corr.landmark_ids, used.landmark_ids).any()  # handoff set kept
    error = np.linalg.norm(handoff.pose.T_WB[:3, 3] - dense.pose(2)[:3, 3])
    assert error < 0.2
    events = _joint_events(tracker)
    assert len(events) == 1 and events[0]["reason"] == "accepted" and events[0]["job_id"] == 2
    tracker.settle(handoff, c.HandoffDisposition.ADOPT)
    assert tracker.active_count("fr") >= used.landmark_ids.size


def test_no_joint_without_a_failed_track_solve_on_the_same_image(dense, bank):
    tracker = vision.VisionTracker(bank)
    obs = {k: dense.observation(k) for k in (0, 1, 2)}
    for k in (0, 1, 2):
        tracker.observe(obs[k])
    full = tracker.handoff(_result(dense, tracker, dense.correspondences(0, bank.map_id), 1),
                           obs[1])
    tracker.settle(full, c.HandoffDisposition.ADOPT)
    # No track() on frame 2: the active tracks are not current-image evidence.
    corr = _band_corr(dense, 0, bank.map_id, right=True)
    handoff = tracker.handoff(_result(dense, tracker, corr, 2), obs[2])
    assert not handoff.accepted and handoff.reason == "poor_image_coverage"
    assert set(handoff.correspondences.landmark_ids) <= set(corr.landmark_ids)
    assert _joint_events(tracker) == []
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)


# Feature: stage3-gate-fix-round2, F2: a joint solve passes only the unchanged gates; else the single-set rejection stands
def test_joint_handoff_seeded_property(dense, bank):
    """Random track/handoff subsets and stale current tracks: an accepted joint handoff
    satisfies the unchanged gates, unique ids and the true pose; a failed joint solve leaves
    the single-set rejection and its point set unchanged. Both outcomes occur.

    **Validates: Requirements 2.4**
    """
    outcomes = set()
    for iteration in range(8):
        rng = f.seeded_rng(905, iteration)
        stale = bool(iteration % 2)

        def keep(n, rng=rng):
            return np.sort(rng.choice(n, size=int(rng.integers(max(n // 2, 1), n + 1)),
                                      replace=False))

        tracker, current, _ = _left_tracks_rejected(dense, bank, keep)
        if stale:  # current tracks inconsistent with the true pose by 12 px
            active = tracker._active["fr"]
            angle = rng.uniform(-np.pi, np.pi, len(active))
            active.pixels[:] += 12.0 * np.column_stack([np.cos(angle), np.sin(angle)])
        n_right = int(np.count_nonzero((dense.project(0)[0][:, 0] >= RIGHT_MIN_U)
                                       & dense.project(0)[1]))
        rows = np.sort(rng.choice(n_right, size=int(rng.integers(60, n_right + 1)),
                                  replace=False))
        corr = _band_corr(dense, 0, bank.map_id, right=True, rows=rows)
        handoff = tracker.handoff(_result(dense, tracker, corr, 10 + iteration), current)
        events = _joint_events(tracker)
        assert len(events) == 1
        used = handoff.correspondences
        assert np.unique(used.landmark_ids).size == used.landmark_ids.size
        assert np.unique(used.keypoint_ids).size == used.keypoint_ids.size
        assert used.landmark_ids.size <= c.MAX_HANDOFF_POINTS
        if handoff.accepted:
            outcomes.add("accepted")
            pose = handoff.pose
            assert pose.occupied_cells >= pnp.DEFAULT_GATES.min_cells
            assert pose.inlier_count >= pnp.DEFAULT_GATES.min_inliers
            assert pose.median_error_px <= pnp.DEFAULT_GATES.max_median_px
            if not stale:  # gates bound support, not accuracy, when tracks are corrupt
                assert np.linalg.norm(pose.T_WB[:3, 3] - dense.pose(2)[:3, 3]) < 0.2
            assert events[0]["reason"] == "accepted"
        else:
            outcomes.add("rejected")
            assert handoff.reason in vision.JOINT_HANDOFF_REASONS
            assert events[0]["reason"] != "accepted"
            assert set(used.landmark_ids) <= set(corr.landmark_ids)  # single-set evidence
        tracker.settle(handoff, c.HandoffDisposition.DISCARD)
    assert outcomes == {"accepted", "rejected"}

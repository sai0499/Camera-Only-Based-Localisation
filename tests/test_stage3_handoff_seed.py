"""Seeded handoff carry (stage3-gate-fix-round4, B's ``VisionTracker.handoff``).

Each handoff carry transition whose two exposures both have a logged causal prior seeds LK
with the projected motion between those priors (the same math camera-rate tracking uses);
other transitions carry unseeded, and seeded failures are retried unseeded. Synthetic scenes
only: this is software verification, not localization accuracy.
"""

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import vision


@pytest.fixture(scope="module")
def scene():
    return vf.Scene(seed=4)


@pytest.fixture(scope="module")
def bank(scene, tmp_path_factory):
    return scene.build_map(tmp_path_factory.mktemp("seed") / "bank")


def _result(tracker, corr, job_id=1):
    return f.make_anchor_result(job_id=job_id, source_exposure_ns=corr.exposure_ns,
                                map_id=tracker.map.map_id, correspondences=corr,
                                matched_reference_ids=(1,), remaining_candidate_ids=())


def _tracker(scene, bank, frames, priors=()):
    """Observe ``frames``; log a causal prior (valid aux) at each frame in ``priors``."""
    tracker = vision.VisionTracker(bank)
    obs = {k: scene.observation(k) for k in frames}
    for k in frames:
        tracker.observe(obs[k])
        if k in priors:
            snap = scene.snapshot(k, aux=True)
            assert tracker.track(obs[k], snap, snap.auxiliary).reason == "no_active_tracks"
    return tracker, obs


def _record_guesses(tracker):
    calls = []
    original = tracker._carry

    def carry(tracks, previous, current, guess=None):
        calls.append((previous.exposure_ns, current.exposure_ns, tracks.pixels.copy(),
                      tracks.landmark_ids.copy(), None if guess is None else guess.copy()))
        return original(tracks, previous, current, guess)

    tracker._carry = carry
    return calls


def test_seed_is_projected_motion_between_logged_priors(scene, bank):
    frames = (0, 1, 2, 3)
    tracker, obs = _tracker(scene, bank, frames, priors=frames)
    calls = _record_guesses(tracker)
    handoff = tracker.handoff(_result(tracker, scene.correspondences(0, bank.map_id)), obs[3])
    assert handoff.accepted, handoff.reason
    assert [(a, b) for a, b, *_ in calls] == [(scene.t_ns(k), scene.t_ns(k + 1)) for k in (0, 1, 2)]
    by_ns = {scene.t_ns(k): k for k in frames}
    for before_ns, after_ns, pixels, landmarks, guess in calls:
        points = bank.points(landmarks)
        before, _ = vision.project(points, scene.pose(by_ns[before_ns]), scene.T_BC, scene.K)
        after, _ = vision.project(points, scene.pose(by_ns[after_ns]), scene.T_BC, scene.K)
        assert guess is not None
        np.testing.assert_allclose(guess, pixels + (after - before), atol=1e-3)
    error = np.linalg.norm(handoff.pose.T_WB[:3, 3] - scene.pose(3)[:3, 3])
    assert error < 0.1


def test_transitions_without_both_priors_carry_unseeded(scene, bank):
    tracker, obs = _tracker(scene, bank, (0, 1, 2, 3), priors=(0, 1, 3))
    calls = _record_guesses(tracker)
    tracker.handoff(_result(tracker, scene.correspondences(0, bank.map_id)), obs[3])
    seeded = [guess is not None for *_, guess in calls]
    assert seeded == [True, False, False]  # (0,1) both priors; (1,2), (2,3) miss frame 2

    plain, plain_obs = _tracker(scene, bank, (0, 1, 2, 3))
    plain_calls = _record_guesses(plain)
    plain.handoff(_result(plain, scene.correspondences(0, bank.map_id)), plain_obs[3])
    assert all(guess is None for *_, guess in plain_calls)


def test_failed_seed_is_retried_unseeded_from_previous_pixel(scene, bank):
    tracker, _ = _tracker(scene, bank, (0, 1))
    corr = scene.correspondences(0, bank.map_id)
    tracks = vision._TrackSet("fr", corr.exposure_ns, corr.keypoint_ids, corr.landmark_ids,
                              corr.pixels)
    previous, current = tracker._frame("fr", scene.t_ns(0)), tracker._frame("fr", scene.t_ns(1))
    unseeded = tracker._carry(tracks, previous, current)
    hopeless = (tracks.pixels + 5000.0).astype(np.float32)  # every seeded search fails
    retried = tracker._carry(tracks, previous, current, hopeless)
    assert len(unseeded) > 50
    np.testing.assert_array_equal(retried.track_ids, unseeded.track_ids)
    np.testing.assert_array_equal(retried.pixels, unseeded.pixels)


def test_tracking_guess_math_is_unchanged(scene, bank):
    tracker, obs = _tracker(scene, bank, (0, 1), priors=(0,))
    corr = scene.correspondences(0, bank.map_id)
    tracks = vision._TrackSet("fr", corr.exposure_ns, corr.keypoint_ids, corr.landmark_ids,
                              corr.pixels)
    current_pose = scene.pose(1)
    guess = tracker._guess(tracks, obs[1], current_pose)
    points = bank.points(tracks.landmark_ids)
    before, _ = vision.project(points, tracker._priors["fr"][1], scene.T_BC, scene.K)
    after, _ = vision.project(points, current_pose, scene.T_BC, scene.K)
    np.testing.assert_array_equal(guess, (tracks.pixels + (after - before)).astype(np.float32))
    assert tracker._guess(tracks, obs[1], None) is None
    stale = vision._TrackSet("fr", scene.t_ns(1), tracks.track_ids, tracks.landmark_ids,
                             tracks.pixels)
    assert tracker._guess(stale, obs[1], current_pose) is None  # prior at another exposure


def test_fast_motion_retention_at_least_unseeded(scene, bank):
    corr = scene.correspondences(0, bank.map_id)
    # One 2 m transition (four skipped frames, within HISTORY_MAX_AGE_NS).
    seeded, seeded_obs = _tracker(scene, bank, (0, 4), priors=(0, 4))
    plain, plain_obs = _tracker(scene, bank, (0, 4))
    a = seeded.handoff(_result(seeded, corr), seeded_obs[4])
    b = plain.handoff(_result(plain, corr), plain_obs[4])
    assert a.correspondences is not None and b.correspondences is not None, (a.reason, b.reason)
    carried_seeded = a.correspondences.landmark_ids.size
    carried_plain = b.correspondences.landmark_ids.size
    assert carried_seeded >= carried_plain
    assert carried_seeded > 0


def test_too_few_points_keep_their_rejection_reason(scene, bank):
    corr = scene.correspondences(0, bank.map_id, limit=20)
    seeded, seeded_obs = _tracker(scene, bank, (0, 1, 2), priors=(0, 1, 2))
    plain, plain_obs = _tracker(scene, bank, (0, 1, 2))
    a = seeded.handoff(_result(seeded, corr), seeded_obs[2])
    b = plain.handoff(_result(plain, corr), plain_obs[2])
    assert not a.accepted and not b.accepted
    assert a.reason == b.reason


def test_prior_log_is_bounded_and_cleared_on_calibration_change(scene, bank):
    frames = tuple(range(c.HISTORY_MAX_FRAMES + 3))
    tracker, _ = _tracker(scene, bank, frames, priors=frames)
    log = tracker._prior_log["fr"]
    assert len(log) == c.HISTORY_MAX_FRAMES
    assert log[-1][0] == scene.t_ns(frames[-1])
    k = frames[-1] + 1
    tracker.observe(scene.observation(k, calibration_id="calibration_fr_v2"))
    assert "fr" not in tracker._prior_log

"""B's MappedTracker on a synthetic blob scene: tracking, handoff, settle, candidate, reset."""

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import vision


@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


@pytest.fixture()
def tracker(scene, tmp_path):
    return vision.VisionTracker(scene.build_map(tmp_path / "map"))


def _result(scene, tracker, k, *, job_id=1, limit=None, **overrides):
    fields = dict(job_id=job_id, source_exposure_ns=scene.t_ns(k), map_id=tracker.map.map_id,
                  correspondences=scene.correspondences(k, tracker.map.map_id, limit),
                  matched_reference_ids=(1,), remaining_candidate_ids=())
    fields.update(overrides)
    return f.make_anchor_result(**fields)


def _position_error(T, truth):
    return float(np.linalg.norm(T[:3, 3] - truth[:3, 3]))


def _observe(tracker, scene, frames):
    observations = {k: scene.observation(k) for k in frames}
    for k in frames:
        tracker.observe(observations[k])
    return observations


def test_track_without_tracks_is_rejected_not_invented(scene, tracker):
    obs = _observe(tracker, scene, [0])[0]
    evidence = tracker.track(obs, scene.snapshot(0), c.AuxiliaryGeometry.unavailable("none_yet"))
    assert not evidence.accepted and evidence.reason == "no_active_tracks"
    assert evidence.T_WB is None and evidence.source is c.PoseSource.TRACKING


def test_track_and_handoff_require_observe_first(scene, tracker):
    obs = scene.observation(0)
    with pytest.raises(c.ContractError, match="observe"):
        tracker.track(obs, scene.snapshot(0), c.AuxiliaryGeometry.unavailable("x"))
    tracker.observe(obs)
    with pytest.raises(c.ContractError, match="not after"):
        tracker.observe(obs)  # duplicate exposure


def test_handoff_resolves_current_image_then_tracking_continues(scene, tracker):
    obs = _observe(tracker, scene, [0, 1, 2])
    handoff = tracker.handoff(_result(scene, tracker, 0), obs[2])
    assert handoff.accepted, handoff.reason
    assert handoff.support_ns == (scene.t_ns(0), scene.t_ns(1), scene.t_ns(2))
    assert handoff.measurement_ns == scene.t_ns(2) and handoff.pose.exposure_ns == scene.t_ns(2)
    assert handoff.pose.source is c.PoseSource.HANDOFF
    assert _position_error(handoff.pose.T_WB, scene.pose(2)) < 0.05
    assert handoff.correspondences.landmark_ids.size <= c.MAX_HANDOFF_POINTS
    assert tracker.active_count("fr") == 0  # handoff never modifies track sets
    tracker.settle(handoff, c.HandoffDisposition.ADOPT)
    assert tracker.active_count("fr") == handoff.pose.inlier_count
    for k in (3, 4, 5):
        current = scene.observation(k)
        tracker.observe(current)
        evidence = tracker.track(current, scene.snapshot(k),
                                 c.AuxiliaryGeometry.unavailable("x"))
        assert evidence.accepted, evidence.reason
        assert evidence.exposure_ns == scene.t_ns(k)
        assert _position_error(evidence.T_WB, scene.pose(k)) < 0.05
        assert len(set(evidence.landmark_ids.tolist())) == evidence.inlier_count


def test_tracking_with_projection_seeded_search(scene, tracker):
    obs = _observe(tracker, scene, [0, 1])
    handoff = tracker.handoff(_result(scene, tracker, 0), obs[1])
    tracker.settle(handoff, c.HandoffDisposition.ADOPT)
    # Seed priors at frame 1 via track() with valid auxiliary geometry, then skip a frame.
    tracker.track(obs[1], scene.snapshot(1, aux=True), scene.snapshot(1, aux=True).auxiliary)
    current = scene.observation(3)
    tracker.observe(current)
    snap = scene.snapshot(3, aux=True)
    evidence = tracker.track(current, snap, snap.auxiliary)
    assert evidence.accepted, evidence.reason
    assert _position_error(evidence.T_WB, scene.pose(3)) < 0.05


@pytest.mark.parametrize(
    ("case", "reason"),
    [("stale", "stale_result"), ("missing_chain", "missing_image_chain"),
     ("map", "map_mismatch"), ("not_ok", "anchor_not_ok")],
)
def test_handoff_rejections_keep_a_reason(scene, tracker, case, reason):
    if case == "stale":
        obs = _observe(tracker, scene, list(range(0, 7)))
        result, current = _result(scene, tracker, 0), obs[6]  # 300 ms > 250 ms
    elif case == "missing_chain":
        obs = _observe(tracker, scene, [1, 2])
        result, current = _result(scene, tracker, 0), obs[2]
    elif case == "map":
        obs = _observe(tracker, scene, [0, 1])
        result = _result(scene, tracker, 0, map_id="other_map",
                         correspondences=scene.correspondences(0, "other_map"))
        current = obs[1]
    else:
        obs = _observe(tracker, scene, [0, 1])
        result = _result(scene, tracker, 0, status=c.AnchorStatus.FAILED, correspondences=None,
                         reason="insufficient_matches")
        current = obs[1]
    handoff = tracker.handoff(result, current)
    assert not handoff.accepted and handoff.reason == reason
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)


def test_handoff_rejects_camera_mismatch_and_too_few_points(scene, tracker):
    obs = _observe(tracker, scene, [0, 1])
    few = tracker.handoff(_result(scene, tracker, 0, limit=5), obs[1])
    assert not few.accepted
    rear = scene.observation(1, camera_id="r", calibration_id="calibration_r_v1")
    tracker.observe(rear)
    mismatch = tracker.handoff(_result(scene, tracker, 0), rear)
    assert not mismatch.accepted and mismatch.reason == "camera_mismatch"


def test_settle_is_once_and_adopt_requires_acceptance(scene, tracker):
    obs = _observe(tracker, scene, [0, 1])
    handoff = tracker.handoff(_result(scene, tracker, 0), obs[1])
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)
    with pytest.raises(c.ContractError, match="settled"):
        tracker.settle(handoff, c.HandoffDisposition.ADOPT)


def test_candidate_verification_and_reset_promote_current_tracks(scene, tracker):
    obs = _observe(tracker, scene, [0, 1, 2])
    handoff = tracker.handoff(_result(scene, tracker, 0), obs[2])
    tracker.settle(handoff, c.HandoffDisposition.CANDIDATE)
    assert tracker.has_candidate() and tracker.active_count("fr") == 0
    current = scene.observation(3)
    tracker.observe(current)
    assert tracker.verify_candidate(scene.observation(3, camera_id="r",
                                                      calibration_id="x_r")) is None
    confirmation = tracker.verify_candidate(current)
    assert confirmation.accepted and confirmation.source is c.PoseSource.CANDIDATE
    assert _position_error(confirmation.T_WB, scene.pose(3)) < 0.05
    reset = c.RecoveryReset(reset_id=0, group_id=3, t_ns=scene.t_ns(3), candidate=handoff.pose,
                            confirmation=confirmation, jump_m=12.0, reason="confirmed_recovery")
    tracker.reset(reset)
    assert not tracker.has_candidate()
    assert tracker.active_count("fr") == confirmation.inlier_count
    nxt = scene.observation(4)
    tracker.observe(nxt)
    assert tracker.track(nxt, scene.snapshot(4), c.AuxiliaryGeometry.unavailable("x")).accepted


def test_drop_candidate_keeps_active_tracks(scene, tracker):
    obs = _observe(tracker, scene, [0, 1])
    adopt = tracker.handoff(_result(scene, tracker, 0, job_id=1), obs[1])
    tracker.settle(adopt, c.HandoffDisposition.ADOPT)
    held = tracker.handoff(_result(scene, tracker, 0, job_id=2), obs[1])
    tracker.settle(held, c.HandoffDisposition.CANDIDATE)
    active = tracker.active_count("fr")
    tracker.drop_candidate("candidate_timeout")
    tracker.drop_candidate("already_empty")  # safe when nothing is held
    assert not tracker.has_candidate() and tracker.active_count("fr") == active
    assert tracker.verify_candidate(obs[1]) is None


def test_reset_without_matching_track_set_is_a_contract_error(scene, tracker):
    obs = _observe(tracker, scene, [0, 1, 2])
    handoff = tracker.handoff(_result(scene, tracker, 0), obs[2])
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)
    pose = handoff.pose
    later = f.make_pose_evidence(camera_id="fr", exposure_ns=scene.t_ns(4),
                                 source=c.PoseSource.CANDIDATE, map_id=tracker.map.map_id,
                                 calibration_id=pose.calibration_id,
                                 preprocessing_id=pose.preprocessing_id)
    reset = c.RecoveryReset(reset_id=1, group_id=4, t_ns=scene.t_ns(4), candidate=pose,
                            confirmation=later, jump_m=1.0, reason="confirmed_recovery")
    with pytest.raises(c.ContractError, match="no current track set"):
        tracker.reset(reset)


def test_history_is_bounded_by_age_and_count(scene, tracker):
    for k in range(40):
        tracker.observe(scene.observation(k, image=scene.image(k, blank=True)))
    history = tracker._history["fr"]
    assert len(history) <= c.HISTORY_MAX_FRAMES
    assert history[-1].exposure_ns - history[0].exposure_ns <= c.HISTORY_MAX_AGE_NS


def test_incompatible_intrinsics_are_rejected(scene, tracker):
    K = scene.K.copy()
    K[0, 0] += 5.0
    obs = scene.observation(0, K=K)
    tracker.observe(obs)
    evidence = tracker.track(obs, scene.snapshot(0), c.AuxiliaryGeometry.unavailable("x"))
    assert evidence.reason == "map_intrinsics_mismatch"


def test_spread_subset_is_deterministic_and_spatial():
    rng = np.random.default_rng(3)
    pixels = np.column_stack([rng.uniform(0, 959, 600), rng.uniform(0, 463, 600)])
    keys = rng.permutation(600)
    first = vision.spread_subset(pixels, keys, 256)
    assert np.array_equal(first, vision.spread_subset(pixels, keys, 256))
    assert first.size == 256 and np.unique(first).size == 256
    cells = (np.clip(pixels[first, 0] * 4 // 960, 0, 3) + 4 * np.clip(pixels[first, 1] * 4 // 464,
                                                                       0, 3))
    assert np.unique(cells).size == 16

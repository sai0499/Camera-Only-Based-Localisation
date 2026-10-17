"""Production filter math and exposure-timing fixtures; no localization score claims."""

import math

import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c
from camloc import filter as module
from camloc.filter import CameraFilter, transition

T = f.EXPOSURE_NS


def initial(yaw=0.0):
    return c.InitialPose(timestamp_ns=T, x=0.0, y=0.0, yaw=yaw)


def road_aux(stamp=T):
    return c.AuxiliaryGeometry(source=c.AuxSource.ROAD_PATCH, source_ns=stamp,
                               z_m=0.3, roll_rad=0, pitch_rad=0,
                               road_patch_id="road_patch_0", valid=True, reason="valid")


def group(index, offset_ns=0, **slots):
    frame = f.make_frame_set(
        group_id=index, selected_camera_ids=("fr",),
        images=(f.make_image_message(exposure_ns=T + offset_ns),),
    )
    return f.make_group_evidence(frame_set=frame, **slots)


def pose(stamp=T, x=0.0, yaw=0.0, **fields):
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return f.make_pose_evidence(
        exposure_ns=stamp, T_WB=f.rigid(R, [x, 0, 0.3]),
        inlier_count=64, track_ids=np.arange(64), landmark_ids=np.arange(64) + 100,
        inlier_fraction=0.8, occupied_cells=8, median_error_px=0.5, p90_error_px=1,
        **fields,
    )


@pytest.mark.parametrize("rate", [10, 20, 30, 60, 120])
def test_production_F_Q_compose_over_equal_elapsed_time(rate):
    F_total, Q_total = np.eye(6), np.zeros((6, 6))
    for i in range(rate):
        dt_ns = (i + 1) * c.NS_PER_S // rate - i * c.NS_PER_S // rate
        F, Q = transition(dt_ns)
        Q_total = F @ Q_total @ F.T + Q
        F_total = F @ F_total
    expected_F, expected_Q = transition(c.NS_PER_S)
    np.testing.assert_allclose(F_total, expected_F, atol=1e-13)
    np.testing.assert_allclose(Q_total, expected_Q, atol=1e-13)


@pytest.mark.parametrize("rate", [10, 20, 30, 60, 120])
def test_prediction_only_filter_covariance_is_rate_consistent(rate):
    estimator = CameraFilter(initial())
    expected = estimator.snapshot(T + c.NS_PER_S)
    for i in range(rate + 1):
        result = estimator.step(group(i, i * c.NS_PER_S // rate))
    np.testing.assert_allclose(result.P, expected.P, rtol=1e-13, atol=1e-12)
    np.testing.assert_array_equal(result.state, expected.state)


@pytest.mark.parametrize("bad", [-1, 1.0, True])
def test_prediction_rejects_bad_dt(bad):
    with pytest.raises(c.ContractError):
        transition(bad)


def test_zero_dt_and_initial_uncertainty():
    F, Q = transition(0)
    np.testing.assert_array_equal(F, np.eye(6))
    assert not Q.any()
    estimator = CameraFilter(initial())
    np.testing.assert_allclose(np.sqrt(np.diag(estimator.estimate.P)), module.INITIAL_STD)
    np.testing.assert_array_equal(estimator.estimate.state[3:], 0)
    assert estimator.estimate.status is c.Status.LOST
    assert isinstance(estimator, c.Estimator)


def test_snapshot_is_nonmutating_and_cannot_go_backwards():
    estimator = CameraFilter(initial())
    before = estimator.estimate
    future = estimator.snapshot(T + 100_000_000)
    assert estimator.estimate is before and future.P[0, 0] > before.P[0, 0]
    with pytest.raises(c.ContractError):
        estimator.snapshot(T - 1)


def test_pnp_wraps_innovation_uses_joseph_and_does_not_add_velocity_measurement():
    estimator = CameraFilter(initial(yaw=math.pi - 0.01))
    result = estimator.step(group(0, pose=pose(yaw=-math.pi + 0.01)))
    assert abs(c.wrap_angle(result.state[2] - math.pi)) < 0.02
    np.testing.assert_array_equal(result.state[3:], 0)
    np.testing.assert_allclose(result.P, result.P.T, atol=1e-14)
    assert np.linalg.eigvalsh(result.P).min() >= 0
    assert result.status is c.Status.TRACKING and result.auxiliary.source is c.AuxSource.PNP
    assert result.max_support_exposure_ns == T


def test_pnp_information_cap_uses_exposure_time(monkeypatch):
    noises = []
    update = module._update

    def record(state, covariance, value, H, R):
        noises.append(R.copy())
        return update(state, covariance, value, H, R)

    monkeypatch.setattr(module, "_update", record)
    estimator = CameraFilter(initial())
    estimator.step(group(0, pose=pose()))
    estimator.step(group(1, 10_000_000, pose=pose(T + 10_000_000)))
    np.testing.assert_allclose(noises[0], np.diag(module.POSE_STD**2))
    np.testing.assert_allclose(noises[1], noises[0] * 10)


def test_outlier_predicts_and_duplicate_step_does_not_mutate_state():
    estimator = CameraFilter(initial())
    result = estimator.step(group(0, 100_000_000, pose=pose(T + 100_000_000, x=100)))
    assert result.reason == "innovation_gate"
    assert result.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
    assert result.max_support_exposure_ns is None and result.state[0] == 0
    with pytest.raises(c.ContractError):
        estimator.step(group(0, 100_000_000))
    assert estimator.estimate is result


def test_border_updates_only_lateral_and_yaw_and_is_source_time_limited():
    estimator = CameraFilter(initial())
    border = f.make_border_evidence(exposure_ns=T, lateral_coordinate_m=1, yaw_rad=0)
    first = estimator.step(group(0, border=border, auxiliary=road_aux()))
    assert first.state[0] == 0 and first.P[0, 0] == 0.25
    assert first.state[1] == pytest.approx(0.5)
    assert first.status is c.Status.DEGRADED
    later = estimator.step(group(1, 100_000_000, auxiliary=road_aux(T + 100_000_000),
                                 border=f.make_border_evidence(exposure_ns=T + 100_000_000)))
    assert later.reason == "border_rate_limit"
    assert later.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)


def test_border_support_does_not_reset_full_position_timer_or_branch():
    estimator = CameraFilter(initial())
    for i in range(17):
        stamp = T + i * 200_000_000
        result = estimator.step(group(i, stamp - T, auxiliary=road_aux(stamp),
                                      border=f.make_border_evidence(exposure_ns=stamp)))
    assert result.status is c.Status.LOST
    stamp += 200_000_000
    result = estimator.step(group(17, stamp - T, auxiliary=road_aux(stamp),
                                  border=f.make_border_evidence(exposure_ns=stamp, branch=c.Branch.PIT)))
    assert result.reason == "branch_mismatch" and result.branch is c.Branch.TRACK


@pytest.mark.parametrize("rate", [10, 20, 30, 60, 120])
def test_motion_composes_without_extra_CV_displacement_and_EMA_is_time_based(rate):
    estimator = CameraFilter(initial())
    estimator.step(group(0, auxiliary=road_aux()))
    for i in range(1, rate + 1):
        previous, current = T + (i - 1) * c.NS_PER_S // rate, T + i * c.NS_PER_S // rate
        displacement = (current - previous) / c.NS_PER_S
        motion = f.make_motion_evidence(
            previous_ns=previous, current_ns=current,
            T_Bprev_Bcurr=f.rigid(np.eye(3), [displacement, 0, 0]),
        )
        result = estimator.step(group(i, current - T, motion=motion, auxiliary=road_aux(current)))
    assert result.state[0] == pytest.approx(1.0)
    assert result.state[3] == pytest.approx(1 - math.exp(-1 / 0.2))
    assert np.linalg.eigvalsh(result.P).min() >= 0
    assert len(estimator._history["fr"]) <= c.HISTORY_MAX_FRAMES
    assert min(estimator._history["fr"]) >= result.t_ns - c.HISTORY_MAX_AGE_NS


def test_missing_motion_history_is_rejected_and_bad_auxiliary_cannot_be_consumed():
    estimator = CameraFilter(initial())
    result = estimator.step(group(0, 50_000_000, motion=f.make_motion_evidence(current_ns=T + 50_000_000)))
    assert result.reason == "missing_motion_history"
    assert result.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
    with pytest.raises(c.ContractError):
        group(0, border=f.make_border_evidence(), auxiliary=c.AuxiliaryGeometry.unavailable("off_road"))
    with pytest.raises(c.ContractError):
        group(0, auxiliary=road_aux(T + 1))
    with pytest.raises(c.ContractError):
        group(0, auxiliary=c.AuxiliaryGeometry(
            source=c.AuxSource.PNP, source_ns=T, z_m=0, roll_rad=0, pitch_rad=0,
            road_patch_id=None, valid=True, reason="valid"))
    with pytest.raises(c.ContractError, match="road patch differs"):
        group(0, 50_000_000, auxiliary=road_aux(),
              motion=f.make_motion_evidence(current_ns=T + 50_000_000, road_patch_id="other_patch"))


def test_ground_motion_rotation_direction_and_slope_geometry():
    estimator = CameraFilter(initial(yaw=math.pi / 2))
    estimator.step(group(0, auxiliary=road_aux()))
    delta = f.rigid([[math.cos(0.1), -math.sin(0.1), 0],
                     [math.sin(0.1), math.cos(0.1), 0], [0, 0, 1]], [1, 0, 0])
    result = estimator.step(group(1, 100_000_000, motion=f.make_motion_evidence(
        previous_ns=T, current_ns=T + 100_000_000, T_Bprev_Bcurr=delta)))
    np.testing.assert_allclose(result.state[:3], [0, 1, math.pi / 2 + 0.1], atol=1e-12)
    estimator = CameraFilter(initial())
    tilted = c.AuxiliaryGeometry(
        source=c.AuxSource.ROAD_PATCH, source_ns=T, z_m=0.3, roll_rad=0.05, pitch_rad=-0.1,
        road_patch_id="road_patch_0", valid=True, reason="valid")
    estimator.step(group(0, auxiliary=tilted))
    result = estimator.step(group(1, 100_000_000, motion=f.make_motion_evidence(
        previous_ns=T, current_ns=T + 100_000_000,
        T_Bprev_Bcurr=f.rigid(np.eye(3), [1, 0, 0]))))
    assert result.state[0] == pytest.approx(math.cos(0.1))
    assert result.auxiliary.z_m == pytest.approx(0.3 + math.sin(0.1))


def test_motion_cannot_overwrite_a_newer_fused_pose():
    estimator = CameraFilter(initial())
    rear_ns = T + f.GROUP_EXPOSURE_STEP_NS
    estimator.step(f.make_group_evidence(pose=pose(rear_ns, x=1, camera_id="r")))
    expected = estimator.snapshot(T + 100_000_000)
    result = estimator.step(group(1, 100_000_000, motion=f.make_motion_evidence(
        previous_ns=T, current_ns=T + 100_000_000)))
    assert result.reason == "intervening_update"
    np.testing.assert_allclose(result.state, expected.state)


@pytest.mark.parametrize(("geometry", "interval", "reason"), [
    (False, 50_000_000, "missing_motion_geometry"),
    (True, 300_000_000, "stale_motion"),
])
def test_unusable_motion_geometry_or_age_predicts_with_a_reason(geometry, interval, reason):
    estimator = CameraFilter(initial())
    estimator.step(group(0, auxiliary=road_aux() if geometry else None))
    result = estimator.step(group(1, interval, motion=f.make_motion_evidence(
        previous_ns=T, current_ns=T + interval)))
    assert result.reason == reason
    assert result.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
    assert result.max_support_exposure_ns is None


def test_earlier_border_and_later_motion_share_one_causal_group_update():
    estimator = CameraFilter(initial())
    estimator.step(f.make_group_evidence(auxiliary=road_aux()))
    front_ns, rear_ns = T + 100_000_000, T + 100_500_000
    frame = f.make_frame_set(group_id=1, images=(
        f.make_image_message(exposure_ns=front_ns),
        f.make_image_message(camera_id="r", exposure_ns=rear_ns),
    ))
    result = estimator.step(f.make_group_evidence(
        frame_set=frame, auxiliary=road_aux(front_ns),
        motion=f.make_motion_evidence(
            camera_id="r", previous_ns=T + f.GROUP_EXPOSURE_STEP_NS, current_ns=rear_ns,
            T_Bprev_Bcurr=f.rigid(np.eye(3), [1, 0, 0])),
        border=f.make_border_evidence(exposure_ns=front_ns, lateral_coordinate_m=0, yaw_rad=0),
    ))
    assert result.evidence_kinds == (c.EvidenceKind.GROUND_MOTION, c.EvidenceKind.BORDER)
    assert result.t_ns == result.max_support_exposure_ns == rear_ns
    assert result.state[0] == pytest.approx(1.0)
    assert np.linalg.eigvalsh(result.P).min() >= 0


def test_later_camera_evidence_cannot_rewrite_an_earlier_exposure_cache():
    estimator = CameraFilter(initial())
    frame = f.make_frame_set()
    rear_ns = T + f.GROUP_EXPOSURE_STEP_NS
    result = estimator.step(f.make_group_evidence(frame_set=frame, pose=pose(rear_ns, x=1, camera_id="r")))
    front = estimator.exposure_snapshot("fr", T)
    assert front.state[0] == 0 and not front.auxiliary.valid
    assert result.state[0] > 0 and result.auxiliary.source_ns == rear_ns
    before = result.state.copy()
    estimator.step(group(1, 100_000_000))
    np.testing.assert_array_equal(result.state, before)


def test_earlier_border_branch_is_present_in_its_own_exposure_cache():
    estimator = CameraFilter(initial())
    estimator.step(f.make_group_evidence(
        border=f.make_border_evidence(exposure_ns=T), auxiliary=road_aux()))
    assert estimator.exposure_snapshot("fr", T).branch is c.Branch.TRACK


# ---------------------------------------------------------------------------------------
# Stage 4: status machine, recovery_search_active flag and the reset path.
# ---------------------------------------------------------------------------------------

MS = 1_000_000
FULL_KINDS = {c.EvidenceKind.PNP, c.EvidenceKind.HANDOFF_PNP, c.EvidenceKind.RECOVERY_RESET}


def initial_at(stamp, yaw=0.0):
    return c.InitialPose(timestamp_ns=stamp, x=0.0, y=0.0, yaw=yaw)


def planar(x, y, yaw):
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return f.rigid(R, [x, y, 0.3])


def fr_frame(index, stamp):
    return f.make_frame_set(group_id=index, selected_camera_ids=("fr",),
                            images=(f.make_image_message(exposure_ns=stamp),))


def two_camera_frame(index, front_ns, rear_ns):
    return f.make_frame_set(group_id=index, images=(
        f.make_image_message(exposure_ns=front_ns),
        f.make_image_message(camera_id="r", exposure_ns=rear_ns)))


def gated_pose(estimator, stamp, camera="fr", offset_m=0.0):
    """A tracking pose at the prior's own x/y/yaw (plus ``offset_m``), so the gate decides."""
    prior = estimator.snapshot(stamp).state
    return f.make_pose_evidence(camera_id=camera, exposure_ns=stamp,
                                T_WB=planar(prior[0] + offset_m, prior[1], prior[2]))


def reset_evidence(frame, camera="fr", T_WB=None):
    m = next(image.exposure_ns for image in frame.images if image.camera_id == camera)
    T_WB = np.eye(4) if T_WB is None else T_WB
    confirmation = f.make_pose_evidence(camera_id=camera, exposure_ns=m,
                                        source=c.PoseSource.CANDIDATE, T_WB=T_WB)
    candidate = f.make_pose_evidence(camera_id=camera, exposure_ns=m - c.RECOVERY_CONFIRM_MIN_NS,
                                     source=c.PoseSource.CANDIDATE, T_WB=T_WB)
    reset = f.make_recovery_reset(group_id=frame.group_id, t_ns=frame.t_ns,
                                  confirmation=confirmation, candidate=candidate)
    return f.make_group_evidence(frame_set=frame, reset=reset)


def fingerprint(estimator):
    """Byte-level view of everything a rejected step must leave untouched."""
    e = estimator.estimate
    history = {camera: [(k, id(s), s.state.tobytes(), s.P.tobytes()) for k, s in h.items()]
               for camera, h in estimator._history.items()}
    return (id(e), e.t_ns, e.status, e.state.tobytes(), e.P.tobytes(), history,
            dict(estimator._last_exposure),
            (estimator._last_pnp_ns, estimator._last_visual_ns, estimator._last_border_ns,
             estimator._last_fused_ns, estimator._reset_ns))


def reference_status(aux_valid, now, last_visual, last_full, initial_ns, full_consumed, flag):
    """Independent restatement of the design's status precedence (integer ns ages)."""
    visual_age = now - (initial_ns if last_visual is None else last_visual)
    full_age = now - (initial_ns if last_full is None else last_full)
    if not aux_valid or visual_age > 1_000_000_000 or full_age > 3_000_000_000:
        return c.Status.LOST
    if full_consumed or (last_full is not None and full_age <= 500_000_000):
        return c.Status.TRACKING
    return c.Status.RECOVERING if flag else c.Status.DEGRADED


def run_blackout(last_dark_ms, flag):
    """10 Hz fr stream: PnP at T, prediction only through ``last_dark_ms``, then PnP."""
    estimator = CameraFilter(initial())
    statuses = {0: estimator.step(group(0, pose=pose()), recovery_search_active=flag).status}
    for i in range(1, last_dark_ms // 100 + 1):
        statuses[i * 100] = estimator.step(group(i, i * 100 * MS),
                                           recovery_search_active=flag).status
    i = last_dark_ms // 100 + 1
    statuses[i * 100] = estimator.step(group(i, i * 100 * MS, pose=pose(T + i * 100 * MS)),
                                       recovery_search_active=flag).status
    return statuses


@pytest.mark.parametrize("flag", [False, True])
def test_short_blackout_stays_tracking(flag):
    statuses = run_blackout(500, flag)
    assert set(statuses) == {0, 100, 200, 300, 400, 500, 600}
    assert all(status is c.Status.TRACKING for status in statuses.values())


@pytest.mark.parametrize("flag", [False, True])
def test_one_second_blackout_degrades_or_recovers_but_is_not_lost(flag):
    statuses = run_blackout(1000, flag)
    fallback = c.Status.RECOVERING if flag else c.Status.DEGRADED
    assert all(statuses[ms] is c.Status.TRACKING for ms in range(0, 600, 100))
    assert all(statuses[ms] is fallback for ms in range(600, 1100, 100))
    assert statuses[1000] is not c.Status.LOST
    assert statuses[1100] is c.Status.TRACKING


@pytest.mark.parametrize("flag", [False, True])
def test_blackout_beyond_one_second_is_lost_for_either_flag(flag):
    statuses = run_blackout(1100, flag)
    assert statuses[1100] is c.Status.LOST
    assert statuses[1200] is c.Status.TRACKING


@pytest.mark.parametrize("bad", [1, 0, np.True_, np.False_, None])
def test_non_bool_search_flag_is_rejected_without_mutation(bad):
    estimator = CameraFilter(initial())
    estimator.step(group(0, pose=pose()))
    before = fingerprint(estimator)
    with pytest.raises(c.ContractError, match="recovery_search_active"):
        estimator.step(group(1, 100 * MS), recovery_search_active=bad)
    assert fingerprint(estimator) == before
    assert estimator.step(group(1, 100 * MS)).group_id == 1


# Feature: stage4-scheduler, Property 12: Status precedence and process noise
def test_property_status_precedence_and_process_noise(monkeypatch):
    """**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.8**"""
    scales = []

    def record(dt_ns, noise_scale=1.0):
        scales.append(noise_scale)
        return transition(dt_ns, noise_scale)

    monkeypatch.setattr(module, "transition", record)
    kinds = ("pnp", "handoff", "reset", "motion", "border", "gated", "nothing")
    gaps_ms = (20, 50, 100, 150, 300, 600, 1050, 1600)
    seen = set()
    for iteration in range(120):
        rng = f.seeded_rng(12, iteration)
        initial_ns = T - int(rng.integers(0, 2_000)) * MS
        estimator = CameraFilter(initial_at(initial_ns))
        last_visual = last_full = None
        aux_valid, t, previous_t = False, T, None
        for index in range(int(rng.integers(8, 20))):
            if index:
                previous_t, t = t, t + int(rng.choice(gaps_ms)) * MS
            kind = str(rng.choice(kinds))
            flag = bool(rng.integers(2))
            frame = fr_frame(index, t)
            aux_choice = int(rng.integers(3))
            fresh = (None, road_aux(t), c.AuxiliaryGeometry.unavailable("off_road"))[aux_choice]
            slots = {}
            if kind == "pnp":
                slots["pose"] = gated_pose(estimator, t)
            elif kind == "gated":
                slots["pose"] = gated_pose(estimator, t, offset_m=1e7)
            elif kind == "handoff":
                tracking = gated_pose(estimator, t)
                slots["pose"] = f.make_handoff_evidence(
                    measurement_ns=t, pose=f.make_pose_evidence(
                        exposure_ns=t, source=c.PoseSource.HANDOFF, T_WB=tracking.T_WB))
            elif kind == "motion" and previous_t is not None:
                slots["motion"] = f.make_motion_evidence(
                    previous_ns=previous_t, current_ns=t,
                    T_Bprev_Bcurr=f.rigid(np.eye(3), [0.2, 0, 0]))
                fresh = None if aux_choice == 2 else fresh
            elif kind == "border":
                slots["border"] = f.make_border_evidence(exposure_ns=t)
                fresh = road_aux(t)
            if kind == "reset":
                evidence = reset_evidence(frame, T_WB=planar(*rng.uniform(-20, 20, 2),
                                                             rng.uniform(-math.pi, math.pi)))
                fresh = None
            else:
                evidence = f.make_group_evidence(frame_set=frame, auxiliary=fresh, **slots)
            old = estimator.estimate
            allowed = {module._NOISE_SCALE[old.status]}
            if "motion" in slots:
                snap = estimator._history.get("fr", {}).get(previous_t)
                if snap is not None:
                    allowed.add(max(allowed | {module._NOISE_SCALE[snap.status]}))
            scales.clear()
            result = estimator.step(evidence, recovery_search_active=flag)
            assert scales and set(scales) <= allowed
            applied = set(result.evidence_kinds)
            expected_applied = {"pnp": {c.EvidenceKind.PNP},
                                "handoff": {c.EvidenceKind.HANDOFF_PNP},
                                "reset": {c.EvidenceKind.RECOVERY_RESET},
                                "gated": {c.EvidenceKind.PREDICTION_ONLY}}.get(kind)
            if expected_applied is not None:
                assert applied == expected_applied
            full_consumed = bool(applied & FULL_KINDS)
            if full_consumed:
                last_full = last_visual = t
                aux_valid = True
            else:
                if applied & {c.EvidenceKind.GROUND_MOTION, c.EvidenceKind.BORDER}:
                    last_visual = t
                if fresh is not None:
                    aux_valid = fresh.valid
                elif c.EvidenceKind.GROUND_MOTION in applied:
                    aux_valid = True
            assert result.auxiliary.valid is aux_valid
            expected = reference_status(aux_valid, t, last_visual, last_full, initial_ns,
                                        full_consumed, flag)
            assert result.status is expected, (iteration, index, kind)
            assert isinstance(estimator._last_fused_ns, int)
            seen.add(expected)
    assert seen == set(c.Status)


def test_reset_initializes_state_covariance_kinds_status_and_ages():
    estimator = CameraFilter(initial_at(T - 200 * MS))
    evidence = f.make_group_evidence(reset=f.make_recovery_reset(
        confirmation=f.make_pose_evidence(source=c.PoseSource.CANDIDATE,
                                          T_WB=planar(3.0, -2.0, 0.4))))
    m, now = T, evidence.t_ns
    result = estimator.step(evidence)
    F, Q = transition(now - m, module._NOISE_SCALE[c.Status.LOST])
    P0 = np.diag([0.5**2, 0.5**2, math.radians(1)**2, 30**2, 30**2, math.radians(30)**2])
    np.testing.assert_allclose(result.state, F @ [3.0, -2.0, 0.4, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(result.P, F @ P0 @ F.T + Q, rtol=1e-12, atol=1e-15)
    assert result.evidence_kinds == (c.EvidenceKind.RECOVERY_RESET,)
    assert result.status is c.Status.TRACKING and result.branch is c.Branch.UNKNOWN
    assert result.reason == "confirmed_recovery" and result.max_support_exposure_ns == m
    assert result.auxiliary.source is c.AuxSource.PNP and result.auxiliary.source_ns == m
    assert estimator._last_pnp_ns == estimator._last_visual_ns == estimator._last_fused_ns == m
    assert estimator._reset_ns == m and now - estimator._last_pnp_ns == now - m
    np.testing.assert_allclose(estimator.exposure_snapshot("fr", m).state, [3.0, -2.0, 0.4, 0, 0, 0])
    assert set(estimator._history["r"]) == {now}


def test_reset_outside_the_exposure_window_raises_without_mutation():
    # The default fixture confirms at T, the default InitialPose time: not after last fused.
    estimator = CameraFilter(initial())
    before = fingerprint(estimator)
    with pytest.raises(c.ContractError, match="last fused"):
        estimator.step(f.make_group_evidence(reset=f.make_recovery_reset()))
    assert fingerprint(estimator) == before

    estimator = CameraFilter(initial_at(T - 200 * MS))
    estimator.step(group(0, -100 * MS, auxiliary=road_aux(T - 100 * MS)))
    evidence = reset_evidence(two_camera_frame(1, T, T + MS))
    confirmation = evidence.reset.confirmation
    before = fingerprint(estimator)
    object.__setattr__(confirmation, "exposure_ns", evidence.t_ns + 1)  # bypass record checks
    with pytest.raises(c.ContractError, match="after the group time"):
        estimator.step(evidence)
    assert fingerprint(estimator) == before
    object.__setattr__(confirmation, "exposure_ns", T)
    assert estimator.step(evidence).evidence_kinds == (c.EvidenceKind.RECOVERY_RESET,)


def test_motion_delta_spanning_a_reset_is_declined_as_intervening_update():
    estimator = CameraFilter(initial_at(T - 200 * MS))
    estimator.step(group(0, -100 * MS, auxiliary=road_aux(T - 100 * MS)))
    estimator.step(reset_evidence(fr_frame(1, T)))
    assert min(estimator._history["fr"]) == T
    expected = estimator.snapshot(T + 50 * MS)
    result = estimator.step(group(2, 50 * MS, motion=f.make_motion_evidence(
        previous_ns=T - 100 * MS, current_ns=T + 50 * MS)))
    assert result.reason == "intervening_update"
    assert result.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
    np.testing.assert_array_equal(result.state[:3], expected.state[:3])


def test_next_pnp_information_interval_starts_at_the_reset_exposure(monkeypatch):
    noises = []
    update = module._update

    def record(state, covariance, value, H, R):
        noises.append(R.copy())
        return update(state, covariance, value, H, R)

    monkeypatch.setattr(module, "_update", record)
    estimator = CameraFilter(initial_at(T - 200 * MS))
    estimator.step(reset_evidence(fr_frame(0, T)))
    estimator.step(group(1, 10 * MS, pose=pose(T + 10 * MS)))
    np.testing.assert_allclose(noises[-1], np.diag(module.POSE_STD**2) * 10)


def random_prior_history(estimator, rng, start_ns, groups):
    """Commit ``groups`` random two-camera groups; return the last committed fr exposure."""
    t, previous = start_ns, None
    for index in range(groups):
        frame = two_camera_frame(index, t, t + MS // 2)
        kind = int(rng.integers(4))
        slots = {"auxiliary": road_aux(t)}
        if kind == 1:
            slots["pose"] = gated_pose(estimator, t)
        elif kind == 2:
            slots["border"] = f.make_border_evidence(exposure_ns=t)
        elif kind == 3 and previous is not None:
            slots["motion"] = f.make_motion_evidence(
                previous_ns=previous, current_ns=t, T_Bprev_Bcurr=f.rigid(np.eye(3), [0.3, 0, 0]))
        estimator.step(f.make_group_evidence(frame_set=frame, **slots),
                       recovery_search_active=bool(rng.integers(2)))
        previous, t = t, t + int(rng.integers(30, 200)) * MS
    return previous


# Feature: stage4-scheduler, Property 15: Reset state, covariance and history invalidation
def test_property_reset_state_covariance_and_history_invalidation():
    """**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**"""
    P0 = np.diag([0.5**2, 0.5**2, math.radians(1)**2, 30**2, 30**2, math.radians(30)**2])
    for iteration in range(120):
        rng = f.seeded_rng(15, iteration)
        estimator = CameraFilter(initial_at(T - 2 * c.NS_PER_S))
        groups = int(rng.integers(1, 7))
        last_front = random_prior_history(estimator, rng, T - c.NS_PER_S, groups)
        old = estimator.estimate
        front_ns = old.t_ns + int(rng.integers(20, 150)) * MS
        rear_ns = front_ns + int(rng.integers(0, 5 * MS))
        camera = str(rng.choice(["fr", "r"]))
        x, y, yaw = *rng.uniform(-50, 50, 2), rng.uniform(-math.pi, math.pi)
        evidence = reset_evidence(two_camera_frame(groups, front_ns, rear_ns), camera,
                                  planar(x, y, yaw))
        m, now = (front_ns if camera == "fr" else rear_ns), evidence.t_ns
        result = estimator.step(evidence, recovery_search_active=bool(rng.integers(2)))
        F, Q = transition(now - m, module._NOISE_SCALE[old.status])
        expected = F @ [x, y, yaw, 0, 0, 0]
        expected[2] = c.wrap_angle(expected[2])
        np.testing.assert_allclose(result.state, expected, atol=1e-9)
        np.testing.assert_allclose(result.P, F @ P0 @ F.T + Q, rtol=1e-12, atol=1e-15)
        assert result.branch is c.Branch.UNKNOWN and result.status is c.Status.TRACKING
        assert result.evidence_kinds == (c.EvidenceKind.RECOVERY_RESET,)
        assert result.max_support_exposure_ns == m
        assert result.auxiliary.source is c.AuxSource.PNP and result.auxiliary.source_ns == m
        assert now - estimator._last_pnp_ns == now - estimator._last_visual_ns == now - m
        assert all(stamp >= m for h in estimator._history.values() for stamp in h)
        assert (front_ns in estimator._history.get("fr", {})) is (front_ns >= m)
        assert rear_ns in estimator._history["r"]
        next_ns = rear_ns + int(rng.integers(10, 100)) * MS
        prior = estimator.snapshot(next_ns)
        after = estimator.step(f.make_group_evidence(
            frame_set=fr_frame(groups + 1, next_ns),
            motion=f.make_motion_evidence(previous_ns=last_front, current_ns=next_ns)))
        assert after.reason == "intervening_update"
        np.testing.assert_array_equal(after.state[:3], prior.state[:3])


# Feature: stage4-scheduler, Property 7: Invalid inputs are rejected atomically
def test_property_invalid_flag_or_reset_exposure_is_rejected_atomically():
    """**Validates: Requirements 8.9, 10.6** (filter part)"""
    bad_flags = (1, 0, np.True_, np.False_, None, 1.0, "true")
    for iteration in range(120):
        rng = f.seeded_rng(7, iteration)
        estimator = CameraFilter(initial_at(T - c.NS_PER_S))
        groups = int(rng.integers(0, 5))
        if groups:
            random_prior_history(estimator, rng, T - 900 * MS, groups)
        now = estimator.estimate.t_ns + int(rng.integers(10, 200)) * MS
        case = int(rng.integers(3))
        before = fingerprint(estimator)
        if case == 0:
            evidence = f.make_group_evidence(frame_set=fr_frame(groups, now), auxiliary=road_aux(now))
            with pytest.raises(c.ContractError):
                estimator.step(evidence, recovery_search_active=bad_flags[iteration % len(bad_flags)])
            assert fingerprint(estimator) == before
            restore = None
        else:
            evidence = reset_evidence(fr_frame(groups, now))
            confirmation = evidence.reset.confirmation
            if case == 1:
                bad = now + int(rng.integers(1, c.NS_PER_S))
            else:
                bad = estimator._last_fused_ns - int(rng.integers(0, 100 * MS))
            object.__setattr__(confirmation, "exposure_ns", bad)  # bypass record checks
            with pytest.raises(c.ContractError):
                estimator.step(evidence, recovery_search_active=bool(rng.integers(2)))
            assert fingerprint(estimator) == before
            restore = (confirmation, now)
        if restore is not None:
            object.__setattr__(restore[0], "exposure_ns", restore[1])
        assert estimator.step(evidence).group_id == groups

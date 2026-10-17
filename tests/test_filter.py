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


def test_stage_four_reset_is_explicitly_unsupported():
    estimator = CameraFilter(initial())
    with pytest.raises(c.ContractError, match="Stage 4"):
        estimator.step(f.make_group_evidence(reset=f.make_recovery_reset()))

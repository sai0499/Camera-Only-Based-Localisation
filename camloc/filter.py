"""Team C's six-state causal filter. No reader, visual backend or worker lives here.

Noise densities and measurement floors are build-plan defaults, not calibrated uncertainty.
Ground motion uses a bounded exposure-state cache. A delta crossing an intervening fused
update (or a recovery reset) is declined rather than replacing that newer information with
an older state. A confirmed RecoveryReset re-initializes the state at its confirmation
exposure; the recovery search itself lives outside the filter and only reaches it as the
``recovery_search_active`` status flag.
"""

import math
from collections import OrderedDict

import numpy as np
from scipy.spatial.transform import Rotation

from camloc import contracts as c

INITIAL_STD = np.array([0.5, 0.5, math.radians(2), 30, 30, math.radians(30)])
POSE_STD = np.array([0.5, 0.5, math.radians(1)])
BORDER_STD = POSE_STD[[0, 2]]
_NOISE_SCALE = {c.Status.TRACKING: 1.0, c.Status.DEGRADED: 4.0,
                c.Status.RECOVERING: 4.0, c.Status.LOST: 10.0}


def transition(dt_ns: int, noise_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """CV F and continuous white-acceleration Q for an exact, nonnegative ns interval."""
    dt = c.as_ns(dt_ns, "prediction dt_ns") / c.NS_PER_S
    if isinstance(noise_scale, bool) or not math.isfinite(noise_scale) or noise_scale <= 0:
        raise c.ContractError("noise_scale must be finite and positive")
    F, Q = np.eye(6), np.zeros((6, 6))
    block = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    for position, velocity, density in ((0, 3, 3.6), (1, 4, 3.6),
                                        (2, 5, (math.pi / 3)**2 * 0.1)):
        F[position, velocity] = dt
        Q[np.ix_([position, velocity], [position, velocity])] = density * noise_scale * block
    return F, Q


def _predict(state, covariance, dt_ns, scale):
    F, Q = transition(dt_ns, scale)
    predicted = F @ state
    predicted[2] = c.wrap_angle(predicted[2])
    return predicted, F @ covariance @ F.T + Q


def _update(state, covariance, value, H, R):
    innovation = value - H @ state
    innovation[-1] = c.wrap_angle(innovation[-1])
    gain = np.linalg.solve(H @ covariance @ H.T + R, H @ covariance).T
    updated = state + gain @ innovation
    updated[2] = c.wrap_angle(updated[2])
    residual = np.eye(6) - gain @ H
    P = residual @ covariance @ residual.T + gain @ R @ gain.T
    return updated, (P + P.T) * 0.5


def _pose_values(transform):
    return np.array([transform[0, 3], transform[1, 3],
                     math.atan2(transform[1, 0], transform[0, 0])])


def _status(aux_valid, visual_age_ns, full_age_ns, full_consumed, has_full_pose, search_active):
    """Status precedence: Lost, then Tracking, then Recovering, then Degraded (integer ns ages).

    Ages fall back to the InitialPose time, but the InitialPose is not a Full_Pose: Tracking
    by age needs an applied PnP, handoff PnP or reset (``has_full_pose``).
    """
    if not aux_valid or visual_age_ns > c.NS_PER_S or full_age_ns > 3 * c.NS_PER_S:
        return c.Status.LOST
    if full_consumed or (has_full_pose and full_age_ns <= c.NS_PER_S // 2):
        return c.Status.TRACKING
    if search_active:
        return c.Status.RECOVERING
    return c.Status.DEGRADED


def _auxiliary(transform, stamp, source, patch_id=None):
    R = transform[:3, :3]
    return c.AuxiliaryGeometry(
        source=source, source_ns=stamp, z_m=float(transform[2, 3]),
        roll_rad=math.atan2(R[2, 1], R[2, 2]),
        pitch_rad=math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])),
        road_patch_id=patch_id, valid=True, reason="valid",
    )


class CameraFilter:
    """One InitialPose, snapshot() without mutation, and one committed step per FrameSet.

    ``exposure_snapshot`` supplies C's preceding-exposure cache for ground geometry.
    Geometry from A enters only as GroupEvidence.auxiliary; a consumed PnP supplies its
    own full geometry. Neither mechanism injects z/roll/pitch into the six-state covariance.
    """

    def __init__(self, initial_pose: c.InitialPose) -> None:
        if not isinstance(initial_pose, c.InitialPose):
            raise c.ContractError("CameraFilter requires one InitialPose")
        self.initial_pose = initial_pose
        self.estimate = c.Estimate(
            t_ns=initial_pose.timestamp_ns, group_id=None,
            state=[initial_pose.x, initial_pose.y, initial_pose.yaw, 0, 0, 0],
            P=np.diag(INITIAL_STD**2), status=c.Status.LOST, branch=c.Branch.UNKNOWN,
            evidence_kinds=(c.EvidenceKind.INITIAL_POSE,), reason="geometry_unavailable",
            auxiliary=c.AuxiliaryGeometry.unavailable("not_estimated"),
            max_support_exposure_ns=None,
        )
        self._history: dict[str, OrderedDict[int, c.SearchSnapshot]] = {}
        self._last_exposure: dict[str, int] = {}
        self._last_pnp_ns: int | None = None
        self._last_visual_ns: int | None = None
        self._last_border_ns: int | None = None
        self._last_fused_ns = initial_pose.timestamp_ns
        self._reset_ns: int | None = None

    def snapshot(self, t_ns: int) -> c.SearchSnapshot:
        stamp = c.as_ns(t_ns, "snapshot t_ns")
        previous = self.estimate
        if stamp < previous.t_ns:
            raise c.ContractError("snapshot cannot precede the last committed time")
        state, P = _predict(previous.state, previous.P, stamp - previous.t_ns,
                            _NOISE_SCALE[previous.status])
        return c.SearchSnapshot(t_ns=stamp, state=state, P=P, status=previous.status,
                                branch=previous.branch, auxiliary=previous.auxiliary)

    def exposure_snapshot(self, camera_id: str, exposure_ns: int) -> c.SearchSnapshot:
        camera = c.check_camera_id(camera_id)
        stamp = c.as_ns(exposure_ns)
        if stamp >= self.estimate.t_ns:
            return self.snapshot(stamp)
        try:
            return self._history[camera][stamp]
        except KeyError:
            raise c.ContractError("missing preceding-exposure state") from None

    def pose_in_gate(self, pose: c.PoseEvidence) -> bool:
        """C's innovation gate at the measurement exposure; frontend calls this before selection."""
        c.require_accepted(pose)
        prior = self.snapshot(pose.exposure_ns)
        value = _pose_values(pose.T_WB)
        position_limit = max(5.0, 3 * math.sqrt(float(np.linalg.eigvalsh(prior.P[:2, :2])[-1])))
        yaw_limit = max(math.radians(10), 3 * math.sqrt(float(prior.P[2, 2])))
        return (float(np.linalg.norm(value[:2] - prior.state[:2])) <= position_limit
                and abs(c.wrap_angle(value[2] - prior.state[2])) <= yaw_limit)

    def _motion_state(self, motion: c.MotionEvidence):
        if self._reset_ns is not None and motion.previous_ns < self._reset_ns:
            return None, "intervening_update"
        history = self._history.get(motion.camera_id, {})
        previous = history.get(motion.previous_ns)
        if previous is None:
            return None, "missing_motion_history"
        if motion.current_ns - motion.previous_ns > c.HISTORY_MAX_AGE_NS:
            return None, "stale_motion"
        if self._last_fused_ns > motion.previous_ns:
            return None, "intervening_update"
        if not previous.auxiliary.valid:
            return None, "missing_motion_geometry"
        aux = previous.auxiliary
        R = Rotation.from_euler("ZYX", [previous.state[2], aux.pitch_rad, aux.roll_rad]).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = R @ motion.T_Bprev_Bcurr[:3, :3]
        displacement = R @ motion.T_Bprev_Bcurr[:3, 3]
        transform[:3, 3] = np.array([previous.state[0], previous.state[1], aux.z_m]) + displacement
        dt_ns = motion.current_ns - motion.previous_ns
        dt = dt_ns / c.NS_PER_S
        alpha = -math.expm1(-dt / 0.2)
        state = previous.state.copy()
        state[:3] = _pose_values(transform)
        rates = np.r_[displacement[:2], c.wrap_angle(state[2] - previous.state[2])] / dt
        state[3:] = (1 - alpha) * state[3:] + alpha * rates
        J = np.eye(6)
        J[0, 2], J[1, 2] = -displacement[1], displacement[0]
        G = np.zeros((6, 3))
        G[:2, :2] = R[:2, :2]
        heading = transform[:2, 0]
        denominator = float(heading @ heading)
        if denominator < 1e-8:
            return None, "degenerate_motion_heading"
        delta_yaw = math.atan2(motion.T_Bprev_Bcurr[1, 0], motion.T_Bprev_Bcurr[0, 0])
        derivative = (R @ [-math.sin(delta_yaw), math.cos(delta_yaw), 0])[:2]
        G[2, 2] = (heading[0] * derivative[1] - heading[1] * derivative[0]) / denominator
        G[3:, :] = alpha * G[:3, :] / dt
        _, Q = transition(dt_ns, max(_NOISE_SCALE[previous.status], _NOISE_SCALE[self.estimate.status]))
        P = J @ previous.P @ J.T + Q + G @ motion.covariance @ G.T
        geometry = _auxiliary(transform, motion.current_ns, c.AuxSource.ROAD_PATCH,
                              motion.road_patch_id)
        return (state, P, geometry), "accepted"

    def step(self, evidence: c.GroupEvidence, *, recovery_search_active: bool = False) -> c.Estimate:
        """Commit one group. ``recovery_search_active`` only selects Recovering over Degraded."""
        if not isinstance(recovery_search_active, bool):
            raise c.ContractError("recovery_search_active must be a bool")
        if not isinstance(evidence, c.GroupEvidence):
            raise c.ContractError("step requires GroupEvidence")
        old = self.estimate
        expected_group = 0 if old.group_id is None else old.group_id + 1
        if evidence.group_id != expected_group:
            raise c.ContractError("group ids must start at zero and increase by one")
        if evidence.t_ns < old.t_ns or (old.group_id is not None and evidence.t_ns == old.t_ns):
            raise c.ContractError("group time must increase after initialization")
        for image in evidence.frame_set.images:
            if image.exposure_ns < old.t_ns:
                raise c.ContractError("current image precedes the last committed time")
            if image.exposure_ns <= self._last_exposure.get(image.camera_id, -1):
                raise c.ContractError("duplicate or decreasing camera exposure")
        if evidence.reset is not None:
            return self._reset_step(evidence, old)

        scale = _NOISE_SCALE[old.status]
        state, P, stamp, aux = old.state.copy(), old.P.copy(), old.t_ns, old.auxiliary
        anchors = [(stamp, state.copy(), P.copy(), aux, old.status, old.branch)]
        kinds, support = [], []
        reason = "no_evidence"
        last_pnp, last_visual, last_border = self._last_pnp_ns, self._last_visual_ns, self._last_border_ns
        branch = old.branch
        fused_ns = self._last_fused_ns

        if evidence.pose is not None:
            pose = evidence.pose.pose if isinstance(evidence.pose, c.HandoffEvidence) else evidence.pose
            if self.pose_in_gate(pose):
                stamp = pose.exposure_ns
                state, P = _predict(state, P, stamp - old.t_ns, scale)
                H = np.eye(6)[:3]
                quality = max(1.0, 40 / pose.inlier_count, 8 / pose.occupied_cells,
                              pose.median_error_px / 2, pose.p90_error_px / 4)
                interval = 0.1 if last_pnp is None else (stamp - last_pnp) / c.NS_PER_S
                if interval <= 0:
                    raise c.ContractError("PnP measurement times must increase")
                R = np.diag(POSE_STD**2) * quality * max(1.0, 0.1 / interval)
                state, P = _update(state, P, _pose_values(pose.T_WB), H, R)
                aux = _auxiliary(pose.T_WB, stamp, c.AuxSource.PNP)
                kinds.append(c.EvidenceKind.HANDOFF_PNP if isinstance(evidence.pose, c.HandoffEvidence)
                             else c.EvidenceKind.PNP)
                support.append(stamp)
                last_pnp = last_visual = fused_ns = stamp
                anchors.append((stamp, state.copy(), P.copy(), aux, c.Status.TRACKING, branch))
                reason = "accepted"
            else:
                reason = "innovation_gate"
        else:
            if evidence.motion is not None:
                motion = evidence.motion
                outcome, reason = self._motion_state(motion)
                if outcome is not None:
                    state, P, aux = outcome
                    stamp = motion.current_ns
                    kinds.append(c.EvidenceKind.GROUND_MOTION)
                    support.append(stamp)
                    last_visual = fused_ns = stamp
                    anchors.append((stamp, state.copy(), P.copy(), aux, c.Status.DEGRADED, branch))
            if evidence.border is not None:
                border = evidence.border
                if last_border is not None and border.exposure_ns - last_border < c.NS_PER_S // 5:
                    reason = "border_rate_limit"
                elif branch not in (c.Branch.UNKNOWN, border.branch):
                    reason = "branch_mismatch"
                else:
                    measurement_time = max(stamp, border.exposure_ns)
                    state, P = _predict(state, P, measurement_time - stamp, scale)
                    stamp = measurement_time
                    H = np.zeros((2, 6))
                    H[0, :2], H[1, 2] = border.normal_xy, 1
                    value = np.array([border.lateral_coordinate_m, border.yaw_rad])
                    R = border.R.copy()
                    R += np.diag(np.maximum(0, BORDER_STD**2 - np.diag(R)))
                    lag_ns = stamp - border.exposure_ns
                    if lag_ns:
                        prior = self.snapshot(border.exposure_ns)
                        velocity_H = np.array([[*border.normal_xy, 0], [0, 0, 1]])
                        lag = lag_ns / c.NS_PER_S
                        value += velocity_H @ prior.state[3:] * lag
                        _, Q = transition(lag_ns, scale)
                        R += H @ Q @ H.T + lag**2 * velocity_H @ prior.P[3:, 3:] @ velocity_H.T
                    state, P = _update(state, P, value, H, R)
                    kinds.append(c.EvidenceKind.BORDER)
                    support.append(border.exposure_ns)
                    last_border = border.exposure_ns
                    last_visual = max(last_visual or 0, border.exposure_ns)
                    fused_ns = stamp
                    branch = border.branch
                    anchors.append((stamp, state.copy(), P.copy(), aux, c.Status.DEGRADED, branch))
                    reason = "accepted"

        state, P = _predict(state, P, evidence.t_ns - stamp, scale)
        if evidence.auxiliary is not None and not any(k in kinds for k in (c.EvidenceKind.PNP,
                                                                         c.EvidenceKind.HANDOFF_PNP)):
            aux = evidence.auxiliary
        now = evidence.t_ns
        visual_age = now - (last_visual if last_visual is not None else self.initial_pose.timestamp_ns)
        full_age = now - (last_pnp if last_pnp is not None else self.initial_pose.timestamp_ns)
        full_consumed = any(k in kinds for k in (c.EvidenceKind.PNP, c.EvidenceKind.HANDOFF_PNP))
        status = _status(aux.valid, visual_age, full_age, full_consumed, last_pnp is not None,
                         recovery_search_active)
        result = c.Estimate(
            t_ns=now, group_id=evidence.group_id, state=state, P=(P + P.T) * 0.5,
            status=status, branch=branch, evidence_kinds=tuple(kinds) or (c.EvidenceKind.PREDICTION_ONLY,),
            reason=reason, auxiliary=aux, max_support_exposure_ns=max(support) if support else None,
        )
        cached = []
        for image in evidence.frame_set.images:
            anchor = next(a for a in reversed(anchors) if a[0] <= image.exposure_ns)
            when, previous_state, previous_P, geometry, previous_status, previous_branch = anchor
            cached_state, cached_P = _predict(previous_state, previous_P,
                                               image.exposure_ns - when, scale)
            fresh = evidence.auxiliary
            if fresh is not None and not any(k in kinds for k in (c.EvidenceKind.PNP,
                                                                  c.EvidenceKind.HANDOFF_PNP)):
                available_ns = fresh.source_ns if fresh.source_ns is not None else now
                if available_ns <= image.exposure_ns:
                    geometry = fresh
            cached.append((image, c.SearchSnapshot(
                t_ns=image.exposure_ns, state=cached_state, P=cached_P, status=previous_status,
                branch=previous_branch, auxiliary=geometry,
            )))
        # Commit only after every output/cache record validates; emitted records are immutable.
        self.estimate = result
        self._last_pnp_ns, self._last_visual_ns, self._last_border_ns = last_pnp, last_visual, last_border
        self._last_fused_ns = fused_ns
        self._store_snapshots(evidence, cached, now)
        return result

    def _reset_step(self, evidence: c.GroupEvidence, old: c.Estimate) -> c.Estimate:
        """Re-initialize at the confirmation exposure ``m``; computes locals, commits at the end."""
        reset = evidence.reset
        confirmation = reset.confirmation
        m, now = confirmation.exposure_ns, evidence.t_ns
        if m > now:
            raise c.ContractError("reset confirmation exposure is after the group time")
        if m <= self._last_fused_ns:
            raise c.ContractError("reset confirmation exposure must follow the last fused update")
        scale = _NOISE_SCALE[old.status]
        state0 = np.r_[_pose_values(confirmation.T_WB), 0.0, 0.0, 0.0]
        P0 = np.diag(np.r_[POSE_STD, INITIAL_STD[3:]] ** 2)
        aux = _auxiliary(confirmation.T_WB, m, c.AuxSource.PNP)
        state, P = _predict(state0, P0, now - m, scale)
        status = _status(aux.valid, now - m, now - m, True, True, False)
        result = c.Estimate(
            t_ns=now, group_id=evidence.group_id, state=state, P=(P + P.T) * 0.5,
            status=status, branch=c.Branch.UNKNOWN,
            evidence_kinds=(c.EvidenceKind.RECOVERY_RESET,), reason=reset.reason,
            auxiliary=aux, max_support_exposure_ns=m,
        )
        histories = {camera: OrderedDict((stamp, snap) for stamp, snap in history.items()
                                         if stamp >= m)
                     for camera, history in self._history.items()}
        cached = []
        for image in evidence.frame_set.images:
            if image.exposure_ns < m:
                continue
            cached_state, cached_P = _predict(state0, P0, image.exposure_ns - m, scale)
            cached.append((image, c.SearchSnapshot(
                t_ns=image.exposure_ns, state=cached_state, P=cached_P, status=c.Status.TRACKING,
                branch=c.Branch.UNKNOWN, auxiliary=aux,
            )))
        # Commit only after every output/cache record validates.
        self.estimate = result
        self._last_pnp_ns = self._last_visual_ns = self._last_fused_ns = m
        self._reset_ns = m
        self._history = histories
        self._store_snapshots(evidence, cached, now)
        return result

    def _store_snapshots(self, evidence, cached, now):
        """Record every image exposure, cache the given snapshots and prune the history."""
        for image in evidence.frame_set.images:
            self._last_exposure[image.camera_id] = image.exposure_ns
        for image, snapshot in cached:
            history = self._history.setdefault(image.camera_id, OrderedDict())
            history[image.exposure_ns] = snapshot
        for history in self._history.values():
            while history and (len(history) > c.HISTORY_MAX_FRAMES
                               or next(iter(history)) < now - c.HISTORY_MAX_AGE_NS):
                history.popitem(last=False)

"""Recovery manager: candidate lifecycle, confirmation bounds and reset construction.

The frontend is simulated by building GroupEvidence directly, as the design describes:
gated-out tracking poses and handoffs are unused with ``innovation_gate``, accepted
``verify_candidate`` output is unused with ``candidate_verification`` and rejected output
goes to ``rejected``. The estimator is a fixed-prior stub, except one test that uses the
real ``CameraFilter`` for ``jump_m``. Synthetic software checks only.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c
from camloc.recovery import RecoveryManager, RecoveryOutcome, RecoveryPolicy

F0 = f.EXPOSURE_NS  # fr exposure of the candidate's group
MS = 1_000_000
R_OFFSET_NS = 500_000  # the r image is exposed 0.5 ms after fr; t_ns is the r exposure
PRIOR_XY = (10.0, -4.0)


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


class FixedPrior:
    """Minimal c.Estimator: a constant read-only prior; the manager must never step it."""

    def __init__(self, x: float = PRIOR_XY[0], y: float = PRIOR_XY[1]) -> None:
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        self.calls: list[int] = []

    def snapshot(self, t_ns: int) -> c.SearchSnapshot:
        self.calls.append(t_ns)
        return f.make_search_snapshot(t_ns=t_ns, state=self.state)

    def step(self, evidence: c.GroupEvidence) -> c.Estimate:
        raise AssertionError("the recovery manager must not step the estimator")


class FakeVerifier:
    """Local stand-in for MappedTracker.verify_candidate: an accepted or rejected pose."""

    def __init__(self, accept: bool, xy: tuple[float, float] = (0.0, 0.0)) -> None:
        self.accept = accept
        self.xy = xy
        self.calls: list[int] = []

    def verify_candidate(self, observation: c.Observation) -> c.PoseEvidence | None:
        self.calls.append(observation.exposure_ns)
        extra: dict[str, Any] = {} if self.accept else {"accepted": False, "reason": "too_few_inliers"}
        return pose(observation.exposure_ns, *self.xy, camera=observation.camera_id,
                    source=c.PoseSource.CANDIDATE, **extra)


def planar(x: float, y: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[0, 3], T[1, 3] = x, y
    return T


def pose(ns: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, *, camera: str = "fr",
         source: c.PoseSource = c.PoseSource.TRACKING, **overrides: Any) -> c.PoseEvidence:
    return f.make_pose_evidence(camera_id=camera, exposure_ns=ns, source=source,
                                T_WB=planar(x, y, yaw), **overrides)


def handoff(ns: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, *,
            camera: str = "fr") -> c.HandoffEvidence:
    nested = pose(ns, x, y, yaw, camera=camera, source=c.PoseSource.HANDOFF)
    return f.make_handoff_evidence(camera_id=camera, measurement_ns=ns, pose=nested)


def frame(group_id: int, fr_ns: int, *, fr_calibration: str | None = None) -> c.FrameSet:
    fr_extra = {} if fr_calibration is None else {"calibration_id": fr_calibration}
    images = (f.make_image_message(camera_id="fr", exposure_ns=fr_ns, **fr_extra),
              f.make_image_message(camera_id="r", exposure_ns=fr_ns + R_OFFSET_NS))
    return f.make_frame_set(group_id=group_id, images=images)


def group(frame_set: c.FrameSet, *, unused: tuple = (), **slots: Any) -> c.GroupEvidence:
    fields: dict[str, Any] = {"pose": None, "motion": None, "border": None, "reset": None,
                              "rejected": (), "auxiliary": None}
    fields.update(slots)
    return c.GroupEvidence(frame_set=frame_set, unused=unused, **fields)


def gated(record: c.EvidenceRecord) -> c.UnusedEvidence:
    return c.UnusedEvidence(reason="innovation_gate", evidence=record)


def verified(record: c.PoseEvidence) -> c.UnusedEvidence:
    return c.UnusedEvidence(reason="candidate_verification", evidence=record)


def held_manager(policy: RecoveryPolicy | None = None, *, xy=(0.0, 0.0), yaw=0.0,
                 source: str = "tracking") -> tuple[RecoveryManager, c.PoseEvidence]:
    """A manager holding a candidate created in group 0 from the fr image at F0."""
    manager = RecoveryManager(policy)
    record = pose(F0, *xy, yaw) if source == "tracking" else handoff(F0, *xy, yaw)
    _, outcome = manager.process(group(frame(0, F0), unused=(gated(record),)), FixedPrior())
    assert manager.search_active()
    held = record if source == "tracking" else record.pose
    assert outcome.candidate_handoff is (record if source == "handoff" else None)
    return manager, held


def kinds(events: list[dict]) -> list[tuple[str, str]]:
    return [(e["kind"], e["reason"]) for e in events]


# ---------------------------------------------------------------------------------------
# Example tests (task 5.2)
# ---------------------------------------------------------------------------------------


def test_policy_defaults_and_validation() -> None:
    policy = RecoveryPolicy()
    assert (policy.confirm_min_ns, policy.timeout_ns) == (c.RECOVERY_CONFIRM_MIN_NS, 500_000_000)
    assert (policy.max_speed_mps, policy.position_margin_m) == (90.0, 2.0)
    assert (policy.max_yaw_rate_degps, policy.yaw_margin_deg) == (90.0, 5.0)
    with pytest.raises(c.ContractError, match="confirm_min_ns"):
        RecoveryPolicy(confirm_min_ns=c.RECOVERY_CONFIRM_MIN_NS - 1)
    with pytest.raises(c.ContractError, match="timeout_ns"):
        RecoveryPolicy(timeout_ns=0)
    with pytest.raises(c.ContractError, match="max_speed_mps"):
        RecoveryPolicy(max_speed_mps=math.nan)
    assert RecoveryPolicy(confirm_min_ns=c.RECOVERY_CONFIRM_MIN_NS + 1).confirm_min_ns > 50 * MS


def test_creation_event_and_counts() -> None:
    manager, held = held_manager()
    (event,) = manager.drain_diagnostics()
    assert event["record"] == "RecoveryEvent" and event["kind"] == "candidate_created"
    assert (event["group_id"], event["candidate_exposure_ns"]) == (0, F0)
    assert event["reason"] == "innovation_gate" and event["candidate_source"] == "tracking"
    assert manager.counts()["candidate_created"] == 1
    assert manager.drain_diagnostics() == []


def test_gap_of_exactly_50_ms_confirms_with_reset_identity_and_jump() -> None:
    manager, held = held_manager()
    manager.drain_diagnostics()
    m = F0 + c.RECOVERY_CONFIRM_MIN_NS
    confirmation = pose(m, 1.0, 2.0, source=c.PoseSource.CANDIDATE)
    frame_set = frame(1, m)
    prior = FixedPrior()
    converted, outcome = manager.process(group(frame_set, unused=(verified(confirmation),)), prior)

    reset = outcome.reset
    assert isinstance(outcome, RecoveryOutcome) and outcome.candidate_handoff is None
    assert reset is not None and converted.reset is reset
    assert (reset.group_id, reset.t_ns) == (1, frame_set.t_ns)
    assert reset.candidate is held and reset.confirmation is confirmation
    assert reset.reason == "confirmed_recovery" and reset.reset_id == 0
    assert prior.calls == [m]
    assert reset.jump_m == pytest.approx(math.hypot(1.0 - PRIOR_XY[0], 2.0 - PRIOR_XY[1]))
    assert converted.consumed_kinds() == (c.EvidenceKind.RECOVERY_RESET,)
    assert converted.unused == () and not manager.search_active()
    (event,) = manager.drain_diagnostics()
    assert event["kind"] == "candidate_confirmed"
    assert (event["group_id"], event["candidate_exposure_ns"], event["confirmation_exposure_ns"]) == (1, F0, m)
    assert event["jump_m"] == reset.jump_m and event["reset_id"] == 0
    json.dumps(event)
    assert manager.counts()["candidate_confirmations"] == 1


def test_gap_of_50_ms_minus_1_ns_keeps_the_candidate() -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    m = F0 + c.RECOVERY_CONFIRM_MIN_NS - 1
    evidence = group(frame(1, m), unused=(verified(pose(m, source=c.PoseSource.CANDIDATE)),))
    out, outcome = manager.process(evidence, FixedPrior())
    assert out is evidence and outcome.reset is None and manager.search_active()
    assert kinds(manager.drain_diagnostics()) == [("verification_rejected", "confirmation_too_early")]


def test_jump_m_uses_the_real_camera_filter_snapshot() -> None:
    from camloc.filter import CameraFilter

    estimator = CameraFilter(c.InitialPose(timestamp_ns=F0 - c.NS_PER_S, x=3.0, y=4.0, yaw=0.0))
    manager, _ = held_manager()
    m = F0 + 100 * MS
    _, outcome = manager.process(
        group(frame(1, m), unused=(verified(pose(m, 6.0, 8.0, source=c.PoseSource.CANDIDATE)),)),
        estimator,
    )
    expected = estimator.snapshot(m).state
    assert outcome.reset is not None
    assert outcome.reset.jump_m == pytest.approx(math.hypot(6.0 - expected[0], 8.0 - expected[1]))


@pytest.mark.parametrize(("age_ns", "held"), [(500_000_000, True), (500_000_001, False)])
def test_timeout_boundary(age_ns: int, held: bool) -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    fr_ns = F0 + age_ns - R_OFFSET_NS  # group t_ns (the r exposure) is F0 + age_ns
    evidence = group(frame(5, fr_ns))
    assert evidence.t_ns - F0 == age_ns
    manager.process(evidence, FixedPrior())
    assert manager.search_active() is held
    expected = [] if held else [("candidate_discarded", "timeout")]
    assert kinds(manager.drain_diagnostics()) == expected
    assert manager.counts()["candidate_discards"] == (0 if held else 1)


def test_timed_out_candidate_is_not_confirmed_and_a_new_one_is_created() -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    fr_ns = F0 + 501 * MS
    fresh = pose(fr_ns, 0.5, 0.0)
    out, outcome = manager.process(group(frame(6, fr_ns), unused=(gated(fresh),)), FixedPrior())
    assert outcome.reset is None and manager.search_active()
    events = manager.drain_diagnostics()
    assert kinds(events) == [("candidate_discarded", "timeout"), ("candidate_created", "innovation_gate")]
    assert events[1]["candidate_exposure_ns"] == fr_ns


def test_confirming_handoff_is_removed_and_fallback_slots_are_superseded() -> None:
    manager, held = held_manager()
    manager.drain_diagnostics()
    m = F0 + 100 * MS
    confirming = handoff(m, 2.0, 0.0)
    other = c.UnusedEvidence(reason="superseded_by_pose",
                             evidence=pose(m + R_OFFSET_NS, 2.0, 0.0, camera="r"))
    motion = f.make_motion_evidence(current_ns=m)
    border = f.make_border_evidence(exposure_ns=m)
    evidence = group(frame(1, m), unused=(other, gated(confirming)), motion=motion, border=border)

    converted, outcome = manager.process(evidence, FixedPrior())
    assert outcome.reset is not None and outcome.candidate_handoff is None
    assert outcome.reset.confirmation is confirming.pose and outcome.reset.candidate is held
    assert converted.pose is None and converted.motion is None and converted.border is None
    assert converted.frame_set is evidence.frame_set and converted.rejected == evidence.rejected
    assert [item.evidence for item in converted.unused] == [other.evidence, motion, border]
    assert [item.reason for item in converted.unused] == [
        "superseded_by_pose", "superseded_by_reset", "superseded_by_reset"]
    assert all(item.evidence is not confirming for item in converted.unused)
    (event,) = manager.drain_diagnostics()
    assert event["kind"] == "candidate_confirmed"
    assert event["removed_unused"] == evidence.unused[1].summary()


def test_calibration_mismatch_keeps_the_candidate() -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    m = F0 + 100 * MS
    # The verification pose carries the old fr calibration; the current fr image has a new one.
    evidence = group(frame(1, m, fr_calibration="calibration_fr_v2"),
                     unused=(verified(pose(m, source=c.PoseSource.CANDIDATE)),))
    _, outcome = manager.process(evidence, FixedPrior())
    assert outcome.reset is None and manager.search_active()
    assert kinds(manager.drain_diagnostics()) == [("verification_rejected", "calibration_mismatch")]


@pytest.mark.parametrize("field", ["map_id", "preprocessing_id"])
def test_identity_mismatch_keeps_the_candidate(field: str) -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    m = F0 + 100 * MS
    mismatched = pose(m, source=c.PoseSource.CANDIDATE, **{field: "other_v9"})
    _, outcome = manager.process(group(frame(1, m), unused=(verified(mismatched),)), FixedPrior())
    assert outcome.reset is None and manager.search_active()
    (event,) = manager.drain_diagnostics()
    assert (event["kind"], event["reason"]) == ("verification_rejected", "identity_mismatch")
    assert (event["group_id"], event["candidate_exposure_ns"], event["verification_exposure_ns"]) == (1, F0, m)


def test_other_camera_may_differ_in_preprocessing_but_not_map() -> None:
    manager, _ = held_manager()
    m = F0 + 100 * MS
    r_pose = pose(m + R_OFFSET_NS, camera="r", source=c.PoseSource.CANDIDATE, preprocessing_id="rear_v2")
    _, outcome = manager.process(group(frame(1, m), unused=(verified(r_pose),)), FixedPrior())
    assert outcome.reset is not None and outcome.reset.confirmation is r_pose


def test_rejected_verify_candidate_keeps_candidate_and_retries_until_confirmed() -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    rejecting, accepting = FakeVerifier(accept=False), FakeVerifier(accept=True, xy=(1.0, 0.0))

    m1 = F0 + 100 * MS
    result = rejecting.verify_candidate(f.make_observation(exposure_ns=m1))
    assert result is not None and not result.accepted
    _, outcome = manager.process(group(frame(1, m1), rejected=(result,)), FixedPrior())
    assert outcome.reset is None and manager.search_active()
    (event,) = manager.drain_diagnostics()
    assert (event["kind"], event["reason"], event["candidate_exposure_ns"]) == (
        "verification_rejected", "too_few_inliers", F0)

    m2 = F0 + 200 * MS
    result = accepting.verify_candidate(f.make_observation(exposure_ns=m2))
    assert result is not None and result.accepted
    _, outcome = manager.process(group(frame(2, m2), unused=(verified(result),)), FixedPrior())
    assert outcome.reset is not None and outcome.reset.group_id == 2
    assert manager.counts()["verification_rejected"] == 1


def test_verification_without_a_held_candidate_is_logged() -> None:
    manager = RecoveryManager()
    m = F0
    accepted = pose(m, source=c.PoseSource.CANDIDATE)
    rejected = pose(m, source=c.PoseSource.CANDIDATE, accepted=False, reason="too_few_inliers")
    evidence = group(frame(0, m), unused=(verified(accepted),), rejected=(rejected,))
    out, outcome = manager.process(evidence, FixedPrior())
    assert out is evidence and outcome == RecoveryOutcome(reset=None, candidate_handoff=None)
    assert not manager.search_active()  # candidate-source poses never become candidates
    events = manager.drain_diagnostics()
    assert kinds(events) == [("verification_rejected", "no_candidate_held")] * 2
    assert all(e["candidate_exposure_ns"] is None for e in events)


def test_tracking_reacquired_discards_without_reset_or_creation() -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    m = F0 + 100 * MS
    evidence = group(frame(1, m), pose=pose(m, 0.1, 0.0),
                     unused=(gated(pose(m, 40.0, 0.0)), verified(pose(m, source=c.PoseSource.CANDIDATE))))
    out, outcome = manager.process(evidence, FixedPrior())
    assert out is evidence and outcome.reset is None and outcome.candidate_handoff is None
    assert not manager.search_active()
    assert kinds(manager.drain_diagnostics()) == [
        ("candidate_discarded", "tracking_reacquired"), ("verification_rejected", "no_candidate_held")]


def test_motion_inconsistent_replaces_with_the_new_source() -> None:
    manager, _ = held_manager()
    manager.drain_diagnostics()
    m = F0 + 100 * MS
    far = handoff(m, 200.0, 0.0)  # 200 m in 0.1 s fails 90 m/s * 0.1 s + 2 m
    _, outcome = manager.process(group(frame(1, m), unused=(gated(far),)), FixedPrior())
    assert outcome.reset is None and outcome.candidate_handoff is far and manager.search_active()
    events = manager.drain_diagnostics()
    assert kinds(events) == [("candidate_discarded", "motion_inconsistent"),
                             ("candidate_replaced", "motion_inconsistent")]
    assert (events[1]["candidate_exposure_ns"], events[1]["new_candidate_exposure_ns"]) == (F0, m)
    assert events[1]["new_candidate_source"] == "handoff"


def test_process_rejects_a_supplied_reset_and_wrong_types() -> None:
    manager = RecoveryManager()
    with pytest.raises(c.ContractError):
        manager.process(f.make_group_evidence(reset=f.make_recovery_reset()), FixedPrior())
    with pytest.raises(c.ContractError):
        manager.process("evidence", FixedPrior())  # type: ignore[arg-type]
    assert manager.drain_diagnostics() == [] and not manager.search_active()


# ---------------------------------------------------------------------------------------
# Property 13 (task 5.3)
# ---------------------------------------------------------------------------------------


def _yaw_of(record: c.PoseEvidence) -> float:
    return math.atan2(float(record.T_WB[1, 0]), float(record.T_WB[0, 0]))


# Feature: stage4-scheduler, Property 13: Candidate confirmation rules
def test_property_13_candidate_confirmation_rules() -> None:
    """**Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.9, 9.10**"""
    policy = RecoveryPolicy()
    outcomes: dict[str, int] = {}
    for k in range(300):
        rng = f.seeded_rng(1313, k)
        cx, cy, cyaw = rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-math.pi, math.pi)
        manager, held = held_manager(xy=(cx, cy), yaw=cyaw,
                                     source=str(rng.choice(["tracking", "handoff"])))
        manager.drain_diagnostics()

        # Group fr exposure offset: near the 50 ms boundary, anywhere, or near the timeout.
        mode = rng.integers(3)
        if mode == 0:
            d_fr = c.RECOVERY_CONFIRM_MIN_NS + int(rng.integers(-2, 3))
        elif mode == 1:
            d_fr = int(rng.integers(0, 450 * MS))
        else:
            d_fr = 500 * MS - R_OFFSET_NS + int(rng.integers(-2, 3))
        camera = str(rng.choice(["fr", "r"]))
        exposure = F0 + d_fr + (R_OFFSET_NS if camera == "r" else 0)
        gap = exposure - F0
        dt = gap / 1e9

        # Displacement and yaw change around their inclusive bounds.
        bound_r = policy.max_speed_mps * dt + policy.position_margin_m
        bound_e = math.radians(policy.max_yaw_rate_degps * dt + policy.yaw_margin_deg)
        r = bound_r * float(rng.choice([rng.uniform(0, 1), rng.uniform(1, 1.5), 1.0]))
        e = bound_e * float(rng.choice([rng.uniform(0, 1), rng.uniform(1, 1.5), 1.0]))
        e = min(e, math.pi - 1e-3) * float(rng.choice([-1.0, 1.0]))
        heading = rng.uniform(-math.pi, math.pi)
        px, py = cx + r * math.cos(heading), cy + r * math.sin(heading)

        identity = str(rng.choice(["match", "map", "pre", "cal", "match"]))
        ids: dict[str, Any] = {}
        if identity == "map":
            ids["map_id"] = "other_map"
        elif identity == "pre":
            ids["preprocessing_id"] = "other_pre"
        kind = str(rng.choice(["verification", "tracking", "handoff", "rejected"]))
        fr_calibration = "calibration_fr_v2" if identity == "cal" and camera == "fr" else None
        if identity == "cal" and camera == "r":
            ids["calibration_id"] = "calibration_r_v2"  # differs from the current r image

        if kind == "handoff":
            nested = pose(exposure, px, py, cyaw + e, camera=camera, source=c.PoseSource.HANDOFF, **ids)
            record = f.make_handoff_evidence(camera_id=camera, measurement_ns=exposure, pose=nested, **ids)
            p, unused, rejected = nested, (gated(record),), ()
        elif kind == "tracking":
            p = pose(exposure, px, py, cyaw + e, camera=camera, **ids)
            unused, rejected = (gated(p),), ()
        elif kind == "verification":
            p = pose(exposure, px, py, cyaw + e, camera=camera, source=c.PoseSource.CANDIDATE, **ids)
            unused, rejected = (verified(p),), ()
        else:
            p = pose(exposure, px, py, cyaw + e, camera=camera, source=c.PoseSource.CANDIDATE,
                     accepted=False, reason="too_few_inliers", **ids)
            unused, rejected = (), (p,)
        frame_set = frame(1, F0 + d_fr, fr_calibration=fr_calibration)
        evidence = group(frame_set, unused=unused, rejected=rejected)
        prior = FixedPrior(*rng.uniform(-100, 100, 2))
        converted, outcome = manager.process(evidence, prior)
        events = manager.drain_diagnostics()

        # Reference decision with the inclusive bounds.
        same_camera = camera == held.camera_id
        identity_ok = identity in ("match", "cal") or (identity == "pre" and not same_camera)
        timed_out = frame_set.t_ns - held.exposure_ns > policy.timeout_ns
        dxy = math.hypot(float(p.T_WB[0, 3]) - float(held.T_WB[0, 3]),
                         float(p.T_WB[1, 3]) - float(held.T_WB[1, 3]))
        dyaw = abs(c.wrap_angle(_yaw_of(p) - _yaw_of(held)))
        motion_ok = dxy <= bound_r and dyaw <= bound_e
        new_source = kind in ("tracking", "handoff")

        if timed_out:
            expected = "timeout"
        elif kind == "rejected":
            expected = "rejected"
        elif gap < c.RECOVERY_CONFIRM_MIN_NS:
            expected = "too_early"
        elif not identity_ok:
            expected = "identity"
        elif identity == "cal":
            expected = "calibration"
        elif not motion_ok:
            expected = "motion"
        else:
            expected = "reset"
        outcomes[expected] = outcomes.get(expected, 0) + 1
        where = f"(seed 1313, k={k}, expected {expected}, events {kinds(events)})"

        if expected == "reset":
            reset = outcome.reset
            assert reset is not None and converted.reset is reset, where
            assert (reset.group_id, reset.t_ns) == (1, frame_set.t_ns), where
            assert reset.candidate is held and reset.confirmation is p, where
            assert reset.jump_m == pytest.approx(
                math.hypot(px - prior.state[0], py - prior.state[1]), abs=1e-6), where
            assert prior.calls == [exposure], where
            assert kinds(events) == [("candidate_confirmed", "confirmed_recovery")], where
            assert not manager.search_active(), where
            continue
        assert outcome.reset is None and converted is evidence, where
        if expected == "timeout":
            assert events[0]["kind"] == "candidate_discarded" and events[0]["reason"] == "timeout", where
            assert manager.search_active() is new_source, where
        elif expected == "motion":
            assert kinds(events)[0] == ("candidate_discarded", "motion_inconsistent"), where
            assert manager.search_active() is new_source, where
            if new_source:
                assert events[1]["kind"] == "candidate_replaced", where
                assert events[1]["new_candidate_exposure_ns"] == exposure, where
        else:
            reason = {"too_early": "confirmation_too_early", "identity": "identity_mismatch",
                      "calibration": "calibration_mismatch", "rejected": "too_few_inliers"}[expected]
            assert kinds(events) == [("verification_rejected", reason)], where
            assert events[0]["candidate_exposure_ns"] == held.exposure_ns, where
            assert manager.search_active(), where
    assert set(outcomes) == {"timeout", "rejected", "too_early", "identity", "calibration",
                             "motion", "reset"}, outcomes


# ---------------------------------------------------------------------------------------
# Property 14, manager part (task 5.4)
# ---------------------------------------------------------------------------------------

_REQUIRED = ("group_id", "candidate_exposure_ns", "reason")


def _random_group(rng: np.random.Generator, group_id: int, fr_ns: int,
                  truth: tuple[float, float]) -> c.GroupEvidence:
    """Random frontend output around a true position, with outliers and identity changes."""
    frame_set = frame(group_id, fr_ns)

    def near() -> tuple[float, float, float]:
        if rng.random() < 0.25:
            return truth[0] + rng.uniform(100, 300), truth[1], 0.0
        return truth[0] + rng.normal(0, 0.5), truth[1] + rng.normal(0, 0.5), rng.normal(0, 0.02)

    def ids() -> dict[str, Any]:
        return {"map_id": "other_map"} if rng.random() < 0.1 else {}

    unused: list[c.UnusedEvidence] = []
    rejected: list[c.EvidenceRecord] = []
    for _ in range(rng.integers(0, 3)):
        camera = str(rng.choice(["fr", "r"]))
        ns = fr_ns + (R_OFFSET_NS if camera == "r" else 0)
        choice = rng.integers(5)
        if choice == 0:
            unused.append(gated(pose(ns, *near(), camera=camera, **ids())))
        elif choice == 1:
            unused.append(gated(handoff(ns, *near(), camera=camera)))
        elif choice == 2:
            unused.append(verified(pose(ns, *near(), camera=camera, source=c.PoseSource.CANDIDATE, **ids())))
        elif choice == 3:
            rejected.append(pose(ns, camera=camera, source=c.PoseSource.CANDIDATE,
                                 accepted=False, reason="too_few_inliers"))
        else:  # untaken evidence that is never a candidate source
            unused.append(c.UnusedEvidence(reason="camera_not_selected",
                                           evidence=pose(ns, *near(), camera=camera)))
    slot = rng.random()
    slots: dict[str, Any] = {}
    if slot < 0.15:
        slots["pose"] = pose(fr_ns, *truth)
    elif slot < 0.35:
        slots["motion"] = f.make_motion_evidence(current_ns=fr_ns)
        if rng.random() < 0.5:
            slots["border"] = f.make_border_evidence(exposure_ns=fr_ns)
    return group(frame_set, unused=tuple(unused), rejected=tuple(rejected), **slots)


def _first_new_source(evidence: c.GroupEvidence) -> tuple[c.PoseEvidence, c.HandoffEvidence | None] | None:
    for item in evidence.unused:
        if item.reason != "innovation_gate":
            continue
        if isinstance(item.evidence, c.HandoffEvidence):
            return item.evidence.pose, item.evidence
        if item.evidence.source is c.PoseSource.TRACKING:
            return item.evidence, None
    return None


# Feature: stage4-scheduler, Property 14: Candidate lifecycle and settle mapping
def test_property_14_candidate_lifecycle_manager_part() -> None:
    """**Validates: Requirements 9.1, 9.6, 9.7, 9.8, 11.6**"""
    seen: dict[str, int] = {}
    for k in range(120):
        rng = f.seeded_rng(1414, k)
        manager = RecoveryManager()
        prior = FixedPrior()
        model: tuple[int, str] | None = None  # (exposure, camera) of the held candidate
        tallies = {"candidate_created": 0, "candidate_replaced": 0, "candidate_discarded": 0,
                   "verification_rejected": 0, "candidate_confirmed": 0}
        fr_ns = F0
        for g in range(int(rng.integers(5, 25))):
            fr_ns += int(rng.integers(20, 200)) * MS
            truth = (20.0 * (fr_ns - F0) / 1e9, 0.0)
            evidence = _random_group(rng, g, fr_ns, truth)
            before = model
            last_discard: tuple[int, str] | None = None  # candidate discarded in this group
            converted, outcome = manager.process(evidence, prior)
            events = manager.drain_diagnostics()
            where = f"(seed 1414, k={k}, group {g}, events {kinds(events)})"
            json.dumps(events)

            for event in events:
                assert all(key in event for key in _REQUIRED), where
                assert event["group_id"] == g and isinstance(event["reason"], str), where
                assert event["record"] == "RecoveryEvent", where
                tallies[event["kind"]] += 1
                seen[event["kind"]] = seen.get(event["kind"], 0) + 1
                if event["reason"] == "no_candidate_held":
                    assert model is None and event["candidate_exposure_ns"] is None, where
                    continue
                assert isinstance(event["candidate_exposure_ns"], int), where
                current = (event["candidate_exposure_ns"], event["candidate_camera_id"])
                if event["kind"] == "candidate_created":
                    assert model is None, where  # at most one candidate held
                    model = current
                elif event["kind"] == "candidate_replaced":
                    assert model is None and isinstance(event["new_candidate_exposure_ns"], int), where
                    assert current == last_discard, where  # names the replaced candidate
                    model = (event["new_candidate_exposure_ns"], event["new_candidate_camera_id"])
                else:
                    # A rejected verification after a same-group motion discard names the
                    # candidate that was held when confirmation started.
                    assert current == (model if model is not None else last_discard), where
                    if event["kind"] in ("candidate_discarded", "candidate_confirmed"):
                        last_discard, model = model, None

            assert manager.search_active() is (model is not None), where
            taken = [e for e in events if e["kind"] in ("candidate_created", "candidate_replaced")]
            assert len(taken) <= 1, where
            if evidence.pose is not None:  # reacquired: unchanged, no reset, no creation
                assert converted is evidence and outcome.reset is None and not taken, where
                if before is not None and not any(e["reason"] == "timeout" for e in events):
                    assert ("candidate_discarded", "tracking_reacquired") in kinds(events), where
                assert model is None, where
            if taken:  # a valid candidate source: the first gated-out tracking pose or handoff
                source = _first_new_source(evidence)
                assert source is not None, where
                stamp_key = ("candidate_exposure_ns" if taken[0]["kind"] == "candidate_created"
                             else "new_candidate_exposure_ns")
                assert taken[0][stamp_key] == source[0].exposure_ns, where
                assert outcome.candidate_handoff is source[1], where
            else:
                assert outcome.candidate_handoff is None, where
            if outcome.reset is not None:
                assert converted.reset is outcome.reset and converted.pose is None, where
                assert converted.motion is None and converted.border is None, where
                assert (outcome.reset.group_id, outcome.reset.t_ns) == (g, evidence.t_ns), where
                assert kinds(events)[-1] == ("candidate_confirmed", "confirmed_recovery"), where
            else:
                assert converted is evidence, where

        counts = manager.counts()
        assert counts == {
            "candidate_created": tallies["candidate_created"],
            "candidate_replaced": tallies["candidate_replaced"],
            "candidate_discards": tallies["candidate_discarded"],
            "verification_rejected": tallies["verification_rejected"],
            "candidate_confirmations": tallies["candidate_confirmed"],
        }
    assert {"candidate_created", "candidate_replaced", "candidate_discarded",
            "verification_rejected", "candidate_confirmed"} <= set(seen), seen

"""C's recovery manager: one uncommitted candidate, confirmed into a RecoveryReset.

The manager reads the GroupEvidence the frontend returned for a group and either passes it
through unchanged or converts it into a group that carries a confirmed ``RecoveryReset``.
It holds at most one candidate: an accepted ``tracking`` pose, or the nested pose of an
accepted HandoffEvidence, that the frontend logged as unused with reason
``innovation_gate``. A later Confirmation_Pose confirms it only when it is at least
``confirm_min_ns`` later, has matching identities, matches its image's calibration and is
motion-consistent with inclusive bounds. The estimator is used only for its read-only
``snapshot``; the manager never steps it and never returns a pose of its own.

Every lifecycle step is recorded as one ``RecoveryEvent`` diagnostic carrying
``group_id``, ``candidate_exposure_ns`` and ``reason``. Records are appended only.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from camloc import contracts as c

__all__ = ["RecoveryManager", "RecoveryOutcome", "RecoveryPolicy"]

_NEW_SOURCE_REASON = "innovation_gate"
_VERIFICATION_REASON = "candidate_verification"
_SUPERSEDED_REASON = "superseded_by_reset"
_RESET_REASON = "confirmed_recovery"

COUNT_KEYS = (
    "candidate_created",
    "candidate_replaced",
    "candidate_discards",
    "verification_rejected",
    "candidate_confirmations",
)


def _check_ns_field(value: object, name: str, *, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise c.ContractError(f"RecoveryPolicy.{name}: expected an int, got {value!r}")
    if value < lo:
        raise c.ContractError(f"RecoveryPolicy.{name}: {value} is below {lo}")
    return value


def _check_float_field(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise c.ContractError(f"RecoveryPolicy.{name}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise c.ContractError(f"RecoveryPolicy.{name}: {value!r} must be finite and >= 0")
    return number


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RecoveryPolicy:
    """Tunable recovery bounds; the motion bounds are inclusive."""

    confirm_min_ns: int = c.RECOVERY_CONFIRM_MIN_NS
    timeout_ns: int = 500_000_000
    max_speed_mps: float = 90.0
    position_margin_m: float = 2.0
    max_yaw_rate_degps: float = 90.0
    yaw_margin_deg: float = 5.0

    def __post_init__(self) -> None:
        _check_ns_field(self.confirm_min_ns, "confirm_min_ns", lo=c.RECOVERY_CONFIRM_MIN_NS)
        _check_ns_field(self.timeout_ns, "timeout_ns", lo=1)
        for name in ("max_speed_mps", "position_margin_m", "max_yaw_rate_degps", "yaw_margin_deg"):
            object.__setattr__(self, name, _check_float_field(getattr(self, name), name))


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RecoveryOutcome:
    """What ``process`` decided for one group, for the runner's settle mapping."""

    reset: c.RecoveryReset | None
    candidate_handoff: c.HandoffEvidence | None  # handoff newly taken as the candidate


@dataclasses.dataclass(frozen=True, slots=True)
class _Held:
    pose: c.PoseEvidence
    group_id: int
    handoff: c.HandoffEvidence | None


@dataclasses.dataclass(frozen=True, slots=True)
class _Source:
    index: int  # position in GroupEvidence.unused
    pose: c.PoseEvidence
    handoff: c.HandoffEvidence | None


def _yaw(pose: c.PoseEvidence) -> float:
    T = pose.T_WB
    return math.atan2(float(T[1, 0]), float(T[0, 0]))


def _new_source(item: c.UnusedEvidence, index: int) -> _Source | None:
    """A gated-out accepted tracking pose or accepted handoff (its nested pose)."""
    if item.reason != _NEW_SOURCE_REASON:
        return None
    record = item.evidence
    if (
        isinstance(record, c.PoseEvidence)
        and record.accepted
        and record.source is c.PoseSource.TRACKING
    ):
        return _Source(index, record, None)
    if isinstance(record, c.HandoffEvidence) and record.accepted and record.pose is not None:
        return _Source(index, record.pose, record)
    return None


def _verification_source(item: c.UnusedEvidence, index: int) -> _Source | None:
    """An accepted candidate-source pose the frontend got from ``verify_candidate``."""
    record = item.evidence
    if (
        item.reason == _VERIFICATION_REASON
        and isinstance(record, c.PoseEvidence)
        and record.accepted
        and record.source is c.PoseSource.CANDIDATE
    ):
        return _Source(index, record, None)
    return None


class RecoveryManager:
    """Holds at most one recovery candidate and confirms it into a RecoveryReset."""

    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        if policy is None:
            policy = RecoveryPolicy()
        if not isinstance(policy, RecoveryPolicy):
            raise c.ContractError(f"RecoveryManager: policy must be a RecoveryPolicy, got {policy!r}")
        self.policy = policy
        self._held: _Held | None = None
        self._next_reset_id = 0
        self._counts = dict.fromkeys(COUNT_KEYS, 0)
        self._events: list[dict] = []

    # ---------------------------------------------------------------- public queries

    def search_active(self) -> bool:
        """True while a candidate is held."""
        return self._held is not None

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def drain_diagnostics(self) -> list[dict]:
        events, self._events = self._events, []
        return events

    # ---------------------------------------------------------------- per-group decision

    def process(
        self, evidence: c.GroupEvidence, estimator: c.Estimator
    ) -> tuple[c.GroupEvidence, RecoveryOutcome]:
        """Decide one group: timeout, reacquire, confirmation, verifications, creation, reset.

        All decisions are computed into locals; the held candidate, counters and events
        change only after every check (including the estimator snapshot) has succeeded.
        """
        if not isinstance(evidence, c.GroupEvidence):
            raise c.ContractError(f"RecoveryManager.process: expected GroupEvidence, got {evidence!r}")
        if evidence.reset is not None:
            raise c.ContractError("RecoveryManager.process: the frontend's evidence must not carry a reset")
        if not isinstance(estimator, c.Estimator):
            raise c.ContractError("RecoveryManager.process: estimator must implement c.Estimator")

        policy = self.policy
        t = evidence.t_ns
        g = evidence.group_id
        events: list[dict] = []
        held = self._held

        def event(kind: str, cand: _Held | None, reason: str, **extra: object) -> None:
            events.append({
                "record": "RecoveryEvent",
                "kind": kind,
                "group_id": g,
                "candidate_exposure_ns": None if cand is None else cand.pose.exposure_ns,
                "candidate_camera_id": None if cand is None else cand.pose.camera_id,
                "candidate_group_id": None if cand is None else cand.group_id,
                "reason": reason,
                **extra,
            })

        def verification(cand: _Held | None, pose: c.PoseEvidence, reason: str) -> None:
            event("verification_rejected", cand, reason,
                  verification_exposure_ns=pose.exposure_ns,
                  verification_camera_id=pose.camera_id)

        new_sources: list[_Source] = []
        confirmations: list[_Source] = []
        verifications: list[_Source] = []
        current_images = {(image.camera_id, image.exposure_ns): image.calibration_id
                          for image in evidence.frame_set.images}
        for index, item in enumerate(evidence.unused):
            source = _new_source(item, index)
            if source is not None:
                new_sources.append(source)
                confirmations.append(source)
                continue
            source = _verification_source(item, index)
            if source is not None:
                confirmations.append(source)
                verifications.append(source)
        rejected_verifications = [
            record for record in evidence.rejected
            if isinstance(record, c.PoseEvidence) and record.source is c.PoseSource.CANDIDATE
        ]

        def no_candidate_held() -> None:
            for source in verifications:
                verification(None, source.pose, "no_candidate_held")
            for record in rejected_verifications:
                verification(None, record, "no_candidate_held")

        # 1. Timeout (inclusive: exactly timeout_ns old stays held).
        if held is not None and t - held.pose.exposure_ns > policy.timeout_ns:
            event("candidate_discarded", held, "timeout")
            held = None

        # 2. A filled pose slot: tracking reacquired, evidence unchanged, nothing created.
        if evidence.pose is not None:
            if held is not None:
                event("candidate_discarded", held, "tracking_reacquired")
                held = None
            no_candidate_held()
            self._commit(None, events)
            return evidence, RecoveryOutcome(reset=None, candidate_handoff=None)

        # 3. Confirmation, stopping at the first decisive outcome.
        confirming: _Source | None = None
        motion_discarded: _Held | None = None
        if held is not None:
            calibration = {image.camera_id: image.calibration_id for image in evidence.frame_set.images}
            candidate = held
            for source in confirmations:
                p = source.pose
                gap = p.exposure_ns - candidate.pose.exposure_ns
                if gap < policy.confirm_min_ns:
                    verification(candidate, p, "confirmation_too_early")
                    continue
                mismatch = self._mismatch(candidate.pose, p, calibration)
                if mismatch is not None:
                    verification(candidate, p, mismatch)
                    continue
                if not self._motion_consistent(candidate.pose, p, gap):
                    event("candidate_discarded", candidate, "motion_inconsistent",
                          verification_exposure_ns=p.exposure_ns,
                          verification_camera_id=p.camera_id)
                    motion_discarded = candidate
                    held = None
                    break
                confirming = source
                break
            for record in rejected_verifications:
                verification(candidate, record, record.reason)
        else:
            no_candidate_held()

        # 5 (built before committing anything). Reset construction.
        if confirming is not None:
            assert held is not None
            p = confirming.pose
            prior = estimator.snapshot(p.exposure_ns)
            jump_m = math.hypot(float(p.T_WB[0, 3]) - float(prior.state[0]),
                                float(p.T_WB[1, 3]) - float(prior.state[1]))
            reset = c.RecoveryReset(
                reset_id=self._next_reset_id, group_id=g, t_ns=t, candidate=held.pose,
                confirmation=p, jump_m=jump_m, reason=_RESET_REASON,
            )
            removed = evidence.unused[confirming.index]
            unused = tuple(item for index, item in enumerate(evidence.unused)
                           if index != confirming.index)
            unused += tuple(
                c.UnusedEvidence(reason=_SUPERSEDED_REASON, evidence=record)
                for record in (evidence.motion, evidence.border) if record is not None
            )
            converted = c.GroupEvidence(
                frame_set=evidence.frame_set, pose=None, motion=None, border=None, reset=reset,
                rejected=evidence.rejected, unused=unused, auxiliary=evidence.auxiliary,
            )
            event("candidate_confirmed", held, _RESET_REASON,
                  confirmation_exposure_ns=p.exposure_ns,
                  confirmation_camera_id=p.camera_id,
                  jump_m=jump_m, reset_id=reset.reset_id,
                  removed_unused=removed.summary())
            self._next_reset_id += 1
            self._commit(None, events)
            return converted, RecoveryOutcome(reset=reset, candidate_handoff=None)

        # 4. Candidate creation or replacement (the pose slot is empty here).
        candidate_handoff: c.HandoffEvidence | None = None
        if held is None:
            valid_sources = []
            for source in new_sources:
                if current_images.get((source.pose.camera_id, source.pose.exposure_ns)) != source.pose.calibration_id:
                    verification(None, source.pose, "calibration_mismatch")
                else:
                    valid_sources.append(source)
            new_sources = valid_sources
        if held is None and new_sources:
            source = new_sources[0]
            held = _Held(source.pose, g, source.handoff)
            candidate_handoff = source.handoff
            origin = "handoff" if source.handoff is not None else "tracking"
            if motion_discarded is not None:
                event("candidate_replaced", motion_discarded, "motion_inconsistent",
                      new_candidate_exposure_ns=source.pose.exposure_ns,
                      new_candidate_camera_id=source.pose.camera_id,
                      new_candidate_group_id=g, new_candidate_source=origin)
            else:
                event("candidate_created", held, _NEW_SOURCE_REASON, candidate_source=origin)
        self._commit(held, events)
        return evidence, RecoveryOutcome(reset=None, candidate_handoff=candidate_handoff)

    # ---------------------------------------------------------------- helpers

    def _mismatch(
        self, candidate: c.PoseEvidence, p: c.PoseEvidence, calibration: dict[str, str]
    ) -> str | None:
        identities = ("map_id",)
        if candidate.camera_id == p.camera_id:
            identities += ("calibration_id", "preprocessing_id")
        if any(getattr(candidate, name) != getattr(p, name) for name in identities):
            return "identity_mismatch"
        if calibration.get(p.camera_id) != p.calibration_id:
            return "calibration_mismatch"
        return None

    def _motion_consistent(self, candidate: c.PoseEvidence, p: c.PoseEvidence, gap_ns: int) -> bool:
        policy = self.policy
        dt = gap_ns / 1e9
        dxy = np.asarray(p.T_WB[:2, 3], dtype=np.float64) - candidate.T_WB[:2, 3]
        displacement = math.hypot(float(dxy[0]), float(dxy[1]))
        dyaw = abs(c.wrap_angle(_yaw(p) - _yaw(candidate)))
        return (
            displacement <= policy.max_speed_mps * dt + policy.position_margin_m
            and dyaw <= math.radians(policy.max_yaw_rate_degps * dt + policy.yaw_margin_deg)
        )

    def _commit(self, held: _Held | None, events: list[dict]) -> None:
        self._held = held
        for item in events:
            kind = item["kind"]
            if kind == "candidate_created":
                self._counts["candidate_created"] += 1
            elif kind == "candidate_replaced":
                self._counts["candidate_replaced"] += 1
            elif kind == "candidate_discarded":
                self._counts["candidate_discards"] += 1
            elif kind == "verification_rejected":
                self._counts["verification_rejected"] += 1
            elif kind == "candidate_confirmed":
                self._counts["candidate_confirmations"] += 1
        self._events.extend(events)

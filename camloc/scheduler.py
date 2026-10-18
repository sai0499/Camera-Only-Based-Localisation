"""Deterministic bounded anchor scheduler (owner C, Stage 4).

``AnchorScheduler`` keeps at most one In_Flight_Job and one latest Pending_Request. Every
creation, replacement, expiry, rate and trigger decision uses source time only (integer
nanosecond exposures and group times); the injected ``clock`` is read only for the
worker watchdog and the Runtime_Fields ``stall_ns`` / ``worker_duration_ns``.

Per group the runner calls ``begin_group`` (delivery at the Delivery_Group, the first group
with ``t_ns >= delivery_not_before_ns``), lets the frontend ``propose`` through the returned
``GroupContext`` and then calls ``end_group`` (the Group_Boundary: release, creation,
replacement, expiry and dispatch). ``AnchorWorker.run`` is called only inside ``end_group``,
synchronously (default) or in the background, using one daemon worker thread with a bounded
wait in either mode. A job's outcome is inspected only
at its Delivery_Group, so both modes give the same logical schedule. A Backend_Fault
disables the scheduler for the rest of the run; nothing is raised for it.

Passing the scheduler tests is synthetic timing verification only. It establishes no
localization accuracy, runtime performance or native high-rate support.
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from camloc import contracts as c

__all__ = ["AnchorScheduler", "GroupContext", "Proposal", "SchedulerPolicy"]

# Candidate order at a Group_Boundary: at most one request is created per group.
PURPOSE_ORDER = (
    c.AnchorPurpose.RECOVERY,
    c.AnchorPurpose.FRONT_ANCHOR,
    c.AnchorPurpose.REAR_ANCHOR,
    c.AnchorPurpose.WARM_RETRIEVAL,
)
_PURPOSE_CAMERAS = {
    c.AnchorPurpose.FRONT_ANCHOR: c.PRIMARY_CAMERA,
    c.AnchorPurpose.REAR_ANCHOR: c.SECONDARY_CAMERA,
}
COUNT_KEYS = (
    "anchor_created",
    "anchor_dispatched",
    "anchor_delivered",
    "anchor_failed",
    "anchor_rejected",
    "anchor_replaced",
    "anchor_expired",
    "anchor_cancelled",
    "anchor_stalled",
    "anchor_faulted",
    "anchor_rate_limited",
    "anchor_proposals_discarded",
)


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _policy_int(name: str, value: object, lo: int, hi: int | None = None) -> None:
    if not _is_int(value) or value < lo or (hi is not None and value > hi):
        bound = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
        raise c.ContractError(f"SchedulerPolicy.{name}: expected an int {bound}, got {value!r}")


def _policy_float(name: str, value: object) -> None:
    ok = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    if not ok or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise c.ContractError(
            f"SchedulerPolicy.{name}: expected a finite number > 0, got {value!r}"
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerPolicy:
    """Scheduler limits; times are integer nanoseconds of source time except the watchdog.

    ``watchdog_ns`` bounds one worker ``run`` call in clock (wall) time. The rate caps
    count dispatched requests whose source exposure lies in ``(s - window_ns, s]``.
    """

    watchdog_ns: int = 2_000_000_000
    window_ns: int = 1_000_000_000
    normal_front: int = 2
    normal_rear: int = 1
    normal_warm: int = 1
    normal_total: int = c.NORMAL_JOBS_PER_S
    recovery_warm: int = 2
    recovery_total: int = c.RECOVERY_JOBS_PER_S
    refresh_interval_ns: int = 500_000_000
    min_track_count: int = 40
    refresh_travel_m: float = 8.0
    refresh_yaw_deg: float = 8.0

    def __post_init__(self) -> None:
        _policy_int("watchdog_ns", self.watchdog_ns, 1)
        _policy_int("window_ns", self.window_ns, 1)
        _policy_int("normal_front", self.normal_front, 0)
        _policy_int("normal_rear", self.normal_rear, 0)
        _policy_int("normal_warm", self.normal_warm, 0)
        _policy_int("normal_total", self.normal_total, 1, c.NORMAL_JOBS_PER_S)
        _policy_int("recovery_warm", self.recovery_warm, 0)
        _policy_int("recovery_total", self.recovery_total, 1, c.RECOVERY_JOBS_PER_S)
        _policy_int("refresh_interval_ns", self.refresh_interval_ns, 1)
        _policy_int("min_track_count", self.min_track_count, 0)
        _policy_float("refresh_travel_m", self.refresh_travel_m)
        _policy_float("refresh_yaw_deg", self.refresh_yaw_deg)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Proposal:
    """One validated frontend proposal; the source camera is ``observation.camera_id``."""

    purpose: c.AnchorPurpose
    observation: c.Observation
    reference_ids: tuple[int, ...]  # <= MAX_REFERENCES_PER_JOB; () = retrieval in the job
    track_count: int  # >= 0; used by the front refresh trigger


class GroupContext:
    """Per-group view the frontend receives: due results, ``rear_in_flight`` and ``propose``.

    ``results`` are the checked ``ok`` AnchorResults whose Delivery_Group is this group, in
    ascending job id order. ``settled`` (read-only) holds every checked ``ok`` and ``failed``
    result settled at this group, in job order, unchanged (the frontend reads the remaining
    candidate ids of failed jobs from it); ``results`` is its ``ok`` subset. ``rear_in_flight``
    is True when a rear-camera job, including recovery, is in flight (through its
    Delivery_Group). While the scheduler is faulted all three are empty and ``propose``
    validates its input and then ignores it.
    """

    __slots__ = (
        "_frame_set", "_max_prior_ns", "_open", "_proposals", "_settled", "group_id",
        "rear_in_flight", "results", "t_ns",
    )

    def __init__(
        self,
        frame_set: c.FrameSet,
        results: tuple[c.AnchorResult, ...],
        rear_in_flight: bool,
        max_prior_ns: int | None,
        settled: tuple[c.AnchorResult, ...] | None = None,
    ) -> None:
        self.group_id: int = frame_set.group_id
        self.t_ns: int = frame_set.t_ns
        self.results: tuple[c.AnchorResult, ...] = results
        self._settled: tuple[c.AnchorResult, ...] = results if settled is None else tuple(settled)
        self.rear_in_flight: bool = rear_in_flight
        self._frame_set = frame_set
        self._max_prior_ns = max_prior_ns
        self._proposals: list[Proposal] = []
        self._open = True

    @property
    def settled(self) -> tuple[c.AnchorResult, ...]:
        """Every ``ok`` and ``failed`` result settled at this group (read-only)."""
        return self._settled

    def propose(
        self,
        *,
        purpose: c.AnchorPurpose,
        observation: c.Observation,
        reference_ids: tuple[int, ...] = (),
        track_count: int,
    ) -> None:
        """Validate and record one proposal; raise ``ContractError`` naming a bad input.

        On error nothing is recorded, so the scheduler state is unchanged.
        """
        where = f"GroupContext.propose(group {self.group_id})"
        if not self._open:
            raise c.ContractError(f"{where}: the context is closed (end_group already ran)")
        if not isinstance(purpose, c.AnchorPurpose):
            raise c.ContractError(f"{where}.purpose: expected an AnchorPurpose, got {purpose!r}")
        if not isinstance(observation, c.Observation):
            raise c.ContractError(
                f"{where}.observation: expected an Observation, got {type(observation).__name__}"
            )
        image = (observation.camera_id, observation.exposure_ns, observation.calibration_id)
        images = {(m.camera_id, m.exposure_ns, m.calibration_id) for m in self._frame_set.images}
        if image not in images:
            raise c.ContractError(
                f"{where}.observation: (camera_id, exposure_ns, calibration_id) {image} is not "
                "an image of this frame set"
            )
        source = next(m for m in self._frame_set.images if m.camera_id == observation.camera_id)
        if observation.source_id != source.source_id:
            raise c.ContractError(f"{where}.observation: source_id differs from this group's image")
        required = _PURPOSE_CAMERAS.get(purpose)
        if required is not None and observation.camera_id != required:
            raise c.ContractError(
                f"{where}.observation: purpose {purpose.value!r} runs on camera {required!r}, "
                f"got {observation.camera_id!r}"
            )
        if not isinstance(reference_ids, (tuple, list)):
            raise c.ContractError(
                f"{where}.reference_ids: expected a tuple of ids, got {type(reference_ids).__name__}"
            )
        if len(reference_ids) > c.MAX_REFERENCES_PER_JOB:
            raise c.ContractError(
                f"{where}.reference_ids: at most MAX_REFERENCES_PER_JOB = "
                f"{c.MAX_REFERENCES_PER_JOB} ids, got {len(reference_ids)}"
            )
        if any(not _is_int(ref) or ref < 0 for ref in reference_ids):
            raise c.ContractError(
                f"{where}.reference_ids: ids must be non-negative ints, got {list(reference_ids)!r}"
            )
        refs = tuple(int(ref) for ref in reference_ids)
        if len(set(refs)) != len(refs):
            raise c.ContractError(f"{where}.reference_ids: ids repeat in {refs}")
        if not _is_int(track_count) or track_count < 0:
            raise c.ContractError(
                f"{where}.track_count: expected a non-negative int, got {track_count!r}"
            )
        if self._max_prior_ns is not None and observation.exposure_ns < self._max_prior_ns:
            raise c.ContractError(
                f"{where}.observation: exposure_ns {observation.exposure_ns} is earlier than an "
                f"earlier group's proposal exposure {self._max_prior_ns}"
            )
        self._proposals.append(
            Proposal(
                purpose=purpose,
                observation=observation,
                reference_ids=refs,
                track_count=int(track_count),
            )
        )


@dataclasses.dataclass(slots=True)
class _Pending:
    request: c.AnchorRequest
    odo: tuple[float, float]  # odometers at the source group (front trigger base)


@dataclasses.dataclass(slots=True)
class _Job:
    request: c.AnchorRequest
    future: concurrent.futures.Future | None = None
    sync_outcome: tuple[str, Any, int] | None = None  # (kind, value, duration_ns)
    dispatch_clock_ns: int = 0
    delivered_at: int | None = None


def _call_worker(
    worker: c.AnchorWorker, request: c.AnchorRequest, clock: Callable[[], int]
) -> tuple[str, Any, int]:
    """Run one job and capture ("result", value, duration) or ("exception", exc, duration)."""
    t0 = clock()
    try:
        value: Any = worker.run(request)
        kind = "result"
    except Exception as exc:  # every worker exception is a Backend_Fault, classified later
        value, kind = exc, "exception"
    return kind, value, clock() - t0


class AnchorScheduler:
    """Bounded, source-time anchor scheduler; see the module docstring."""

    def __init__(
        self,
        worker: c.AnchorWorker,
        *,
        map_id: str,
        policy: SchedulerPolicy | None = None,
        threaded: bool = False,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not callable(getattr(worker, "run", None)):
            raise c.ContractError("AnchorScheduler.worker: expected an AnchorWorker with run()")
        if not isinstance(map_id, str) or not map_id:
            raise c.ContractError(f"AnchorScheduler.map_id: expected a non-empty str, got {map_id!r}")
        policy = SchedulerPolicy() if policy is None else policy
        if not isinstance(policy, SchedulerPolicy):
            raise c.ContractError(
                f"AnchorScheduler.policy: expected a SchedulerPolicy, got {type(policy).__name__}"
            )
        if not isinstance(threaded, bool):
            raise c.ContractError(f"AnchorScheduler.threaded: expected a bool, got {threaded!r}")
        if not callable(clock):
            raise c.ContractError("AnchorScheduler.clock: expected a callable returning int ns")
        self._worker = worker
        self.map_id = map_id
        self.policy = policy
        self.threaded = threaded
        self._clock = clock

        self._started = False
        self._finished = False
        self._closed = False
        self._next_job_id = 0
        self._in_flight: _Job | None = None
        self._pending: _Pending | None = None
        self._window: collections.deque[tuple[int, c.AnchorPurpose]] = collections.deque()
        self._last_front_ns = 0
        self._front_base: tuple[float, float] = (0.0, 0.0)
        self._odo: tuple[float, float] = (0.0, 0.0)
        self._prev_pose: tuple[float, float, float] | None = None
        self._max_proposal_ns: int | None = None
        self._last_group: tuple[int, int] | None = None  # (group_id, t_ns)
        self._faulted = False
        self._counts = dict.fromkeys(COUNT_KEYS, 0)
        self._events: list[dict[str, Any]] = []
        self._open: GroupContext | None = None
        self._snapshot: c.SearchSnapshot | None = None

    # -- public state -----------------------------------------------------------------

    @property
    def faulted(self) -> bool:
        """True once a Backend_Fault occurred; no job is dispatched afterwards."""
        return self._faulted

    def recovery_work_active(self) -> bool:
        """True when not faulted and the in-flight or pending request is a ``recovery`` job."""
        if self._faulted:
            return False
        jobs = []
        if self._in_flight is not None:
            jobs.append(self._in_flight.request)
        if self._pending is not None:
            jobs.append(self._pending.request)
        return any(r.purpose is c.AnchorPurpose.RECOVERY for r in jobs)

    def counts(self) -> dict[str, int]:
        """Event tallies (``anchor_stalled`` is a Runtime_Field)."""
        return dict(self._counts)

    def drain_diagnostics(self) -> list[dict[str, Any]]:
        """Return and clear the SchedulerEvent records accumulated so far, in order."""
        events, self._events = self._events, []
        return events

    # -- lifecycle --------------------------------------------------------------------

    def start(self, initial_pose: c.InitialPose) -> None:
        """Set the front refresh clock from the one InitialPose; once, before any group."""
        if self._started or self._last_group is not None or self._finished or self._closed:
            raise c.ContractError("AnchorScheduler.start: already started or closed")
        if not isinstance(initial_pose, c.InitialPose):
            raise c.ContractError(
                f"AnchorScheduler.start: expected an InitialPose, got {type(initial_pose).__name__}"
            )
        self._started = True
        self._last_front_ns = initial_pose.timestamp_ns

    def begin_group(
        self, frame_set: c.FrameSet, snapshot: c.SearchSnapshot, status: c.Status
    ) -> GroupContext:
        """Open group ``frame_set``: deliver the in-flight job if this is its Delivery_Group."""
        where = "AnchorScheduler.begin_group"
        if not self._started:
            raise c.ContractError(f"{where}: start() was not called")
        if self._finished or self._closed:
            raise c.ContractError(f"{where}: the scheduler is finished or closed")
        if self._open is not None:
            raise c.ContractError(f"{where}: group {self._open.group_id} is still open")
        if not isinstance(frame_set, c.FrameSet):
            raise c.ContractError(f"{where}.frame_set: expected a FrameSet")
        if not isinstance(snapshot, c.SearchSnapshot):
            raise c.ContractError(f"{where}.snapshot: expected a SearchSnapshot")
        if not isinstance(status, c.Status):
            raise c.ContractError(f"{where}.status: expected a Status, got {status!r}")
        if snapshot.t_ns != frame_set.t_ns:
            raise c.ContractError(
                f"{where}.snapshot: t_ns {snapshot.t_ns} differs from the group t_ns {frame_set.t_ns}"
            )
        if self._last_group is not None and (
            frame_set.group_id <= self._last_group[0] or frame_set.t_ns <= self._last_group[1]
        ):
            raise c.ContractError(
                f"{where}.frame_set: group {frame_set.group_id} at {frame_set.t_ns} does not follow "
                f"group {self._last_group[0]} at {self._last_group[1]}"
            )

        # Odometers from consecutive group snapshots; the first group adds nothing.
        pose = (float(snapshot.state[0]), float(snapshot.state[1]), float(snapshot.state[2]))
        if self._prev_pose is not None:
            dist = math.hypot(pose[0] - self._prev_pose[0], pose[1] - self._prev_pose[1])
            dyaw = abs(c.wrap_angle(pose[2] - self._prev_pose[2]))
            self._odo = (self._odo[0] + dist, self._odo[1] + dyaw)
        self._prev_pose = pose
        self._last_group = (frame_set.group_id, frame_set.t_ns)
        self._snapshot = snapshot

        settled: tuple[c.AnchorResult, ...] = ()
        job = self._in_flight
        if (
            job is not None
            and job.delivered_at is None
            and frame_set.t_ns >= job.request.delivery_not_before_ns
        ):
            job.delivered_at = frame_set.group_id
            result = self._resolve(job, frame_set)
            if result is not None:
                settled = (result,)
        results = tuple(r for r in settled if r.status is c.AnchorStatus.OK)

        if self._faulted:
            context = GroupContext(frame_set, (), False, self._max_proposal_ns, ())
        else:
            rear = (
                self._in_flight is not None
                and self._in_flight.request.camera_id == c.SECONDARY_CAMERA
            )
            context = GroupContext(frame_set, results, rear, self._max_proposal_ns, settled)
        self._open = context
        return context

    def end_group(self, estimate: c.Estimate, *, frontend_failed: bool) -> None:
        """Group_Boundary: release, create/replace, expire and dispatch."""
        where = "AnchorScheduler.end_group"
        context = self._open
        if context is None:
            raise c.ContractError(f"{where}: no group is open")
        if not isinstance(estimate, c.Estimate):
            raise c.ContractError(f"{where}.estimate: expected an Estimate")
        if estimate.group_id != context.group_id or estimate.t_ns != context.t_ns:
            raise c.ContractError(
                f"{where}.estimate: group {estimate.group_id} at {estimate.t_ns} is not the open "
                f"group {context.group_id} at {context.t_ns}"
            )
        if not isinstance(frontend_failed, bool):
            raise c.ContractError(f"{where}.frontend_failed: expected a bool")
        group_id, t_ns = context.group_id, context.t_ns
        proposals = list(context._proposals)
        context._open = False
        self._open = None
        if proposals:
            latest = max(p.observation.exposure_ns for p in proposals)
            if self._max_proposal_ns is None or latest > self._max_proposal_ns:
                self._max_proposal_ns = latest

        # 1. Release the in-flight slot at its Delivery_Group.
        if self._in_flight is not None and self._in_flight.delivered_at == group_id:
            self._in_flight = None
        # 2. A faulted scheduler ignores proposals.
        if self._faulted:
            return
        recovery_mode = estimate.status is not c.Status.TRACKING
        # 3. Proposals: at most one request per group.
        if frontend_failed:
            if proposals:
                self._counts["anchor_proposals_discarded"] += len(proposals)
                self._event("proposals_discarded", group_id, reason="frontend_failed",
                            count=len(proposals))
        else:
            self._create(proposals, group_id, recovery_mode)
        # 4. Expiry: a pending request whose Delivery_Group is now committed.
        if self._pending is not None and self._pending.request.delivery_not_before_ns <= t_ns:
            request = self._pending.request
            self._pending = None
            self._counts["anchor_expired"] += 1
            self._event("expired", group_id, request, reason="delivery_group_committed")
        # 5. Dispatch into a free slot.
        if self._in_flight is None and self._pending is not None:
            self._dispatch(group_id, recovery_mode)

    def finish(self) -> None:
        """End of input: cancel the undelivered in-flight job and the pending request."""
        if self._open is not None:
            raise c.ContractError(
                f"AnchorScheduler.finish: group {self._open.group_id} is still open"
            )
        if self._finished:
            raise c.ContractError("AnchorScheduler.finish: already finished")
        self._finished = True
        group_id = None if self._last_group is None else self._last_group[0]
        job = self._in_flight
        if job is not None and job.delivered_at is None:
            if job.future is not None:
                job.future.cancel()  # any result is ignored
            self._in_flight = None
            self._counts["anchor_cancelled"] += 1
            self._event("cancelled", group_id, job.request, reason="end_of_input")
        if self._pending is not None:
            request = self._pending.request
            self._pending = None
            self._counts["anchor_cancelled"] += 1
            self._event("cancelled", group_id, request, reason="end_of_input")
        self.close()

    def close(self) -> None:
        """Ignore unfinished work without joining a potentially stuck backend; idempotent."""
        if self._closed:
            return
        self._closed = True
        group_id = None if self._last_group is None else self._last_group[0]
        if self._open is not None:
            self._open._open = False
            self._open = None
        if self._in_flight is not None:
            job, self._in_flight = self._in_flight, None
            if job.future is not None:
                job.future.cancel()
            if job.delivered_at is None:
                self._counts["anchor_cancelled"] += 1
                self._event("cancelled", group_id, job.request, reason="run_aborted")
        if self._pending is not None:
            request, self._pending = self._pending.request, None
            self._counts["anchor_cancelled"] += 1
            self._event("cancelled", group_id, request, reason="run_aborted")

    # -- rate limits and trigger ------------------------------------------------------

    def rate_allows(self, purpose: c.AnchorPurpose, s: int, recovery_mode: bool) -> bool:
        """True if dispatching ``purpose`` at source exposure ``s`` keeps every cap."""
        return self._rate_block(purpose, s, recovery_mode) is None

    def _rate_block(self, purpose: c.AnchorPurpose, s: int, recovery_mode: bool) -> str | None:
        """Name of the first cap a dispatch at ``s`` would exceed, or None."""
        lo = s - self.policy.window_ns
        inside = [p for ns, p in self._window if lo < ns <= s]
        inside.append(purpose)
        policy = self.policy
        if recovery_mode:
            caps = (
                ("recovery_warm", c.AnchorPurpose.WARM_RETRIEVAL, policy.recovery_warm),
                ("recovery_total", None, policy.recovery_total),
            )
        else:
            caps = (
                ("normal_front", c.AnchorPurpose.FRONT_ANCHOR, policy.normal_front),
                ("normal_rear", c.AnchorPurpose.REAR_ANCHOR, policy.normal_rear),
                ("normal_warm", c.AnchorPurpose.WARM_RETRIEVAL, policy.normal_warm),
                ("normal_total", None, policy.normal_total),
            )
        for name, which, cap in caps:
            if which is not None and which is not purpose:
                continue
            used = len(inside) if which is None else sum(p is which for p in inside)
            if used > cap:
                return name
        return None

    def _front_trigger(self, proposal: Proposal) -> bool:
        policy = self.policy
        s = proposal.observation.exposure_ns
        return (
            s - self._last_front_ns >= policy.refresh_interval_ns
            or proposal.track_count < policy.min_track_count
            or self._odo[0] - self._front_base[0] >= policy.refresh_travel_m
            or math.degrees(self._odo[1] - self._front_base[1]) >= policy.refresh_yaw_deg
        )

    # -- boundary steps ---------------------------------------------------------------

    def _create(self, proposals: list[Proposal], group_id: int, recovery_mode: bool) -> None:
        chosen: Proposal | None = None
        for purpose in PURPOSE_ORDER:
            for proposal in proposals:
                if proposal.purpose is not purpose:
                    continue
                if purpose is c.AnchorPurpose.FRONT_ANCHOR and not self._front_trigger(proposal):
                    continue
                cap = self._rate_block(purpose, proposal.observation.exposure_ns, recovery_mode)
                if cap is not None:
                    self._counts["anchor_rate_limited"] += 1
                    self._event("rate_limited", group_id, reason="rate_limited", stage="create",
                                cap=cap, purpose=purpose.value,
                                camera_id=proposal.observation.camera_id, source_group_id=group_id,
                                source_exposure_ns=proposal.observation.exposure_ns)
                    continue
                chosen = proposal
                break
            if chosen is not None:
                break
        if chosen is None:
            return
        s = chosen.observation.exposure_ns
        assert self._snapshot is not None
        request = c.AnchorRequest(
            job_id=self._next_job_id,
            purpose=chosen.purpose,
            camera_id=chosen.observation.camera_id,
            source_group_id=group_id,
            source_exposure_ns=s,
            delivery_not_before_ns=s + c.DELIVERY_MIN_DELAY_NS,
            observation=chosen.observation,
            snapshot=self._snapshot,
            reference_ids=chosen.reference_ids,
            map_id=self.map_id,
            preprocessing_id=chosen.observation.preprocessing_id,
        )
        self._next_job_id += 1
        self._counts["anchor_created"] += 1
        self._event("created", group_id, request, reason="created")
        if self._pending is not None:
            old = self._pending.request
            self._counts["anchor_replaced"] += 1
            self._event("replaced", group_id, request, reason="replaced",
                        replaced_job_id=old.job_id)
        self._pending = _Pending(request=request, odo=self._odo)

    def _dispatch(self, group_id: int, recovery_mode: bool) -> None:
        assert self._pending is not None
        pending, self._pending = self._pending, None
        request = pending.request
        s = request.source_exposure_ns
        cap = self._rate_block(request.purpose, s, recovery_mode)
        if cap is not None:
            self._counts["anchor_rate_limited"] += 1
            self._counts["anchor_expired"] += 1
            self._event("rate_limited", group_id, request, reason="rate_limited_at_dispatch",
                        stage="dispatch", cap=cap)
            return
        # Dispatched exposures are nondecreasing, so pruning from the left is exact.
        while self._window and self._window[0][0] <= s - self.policy.window_ns:
            self._window.popleft()
        self._window.append((s, request.purpose))
        if request.purpose is c.AnchorPurpose.FRONT_ANCHOR:
            self._last_front_ns = s
            self._front_base = pending.odo
        job = _Job(request=request)
        self._in_flight = job
        self._counts["anchor_dispatched"] += 1
        self._event("dispatched", group_id, request, reason="dispatched")
        future = concurrent.futures.Future()
        job.dispatch_clock_ns = self._clock()
        job.future = future

        def work():
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(_call_worker(self._worker, request, self._clock))
                except BaseException as exc:
                    future.set_exception(exc)

        # A timed-out native/GPU call cannot be killed safely. Disable subsequent jobs;
        # a daemon also prevents that one abandoned call from hanging process shutdown.
        threading.Thread(target=work, name="camloc-anchor", daemon=True).start()
        if not self.threaded:
            try:
                job.sync_outcome = self._future_outcome(future, self.policy.watchdog_ns / 1e9)
            except concurrent.futures.TimeoutError:
                job.sync_outcome = ("timeout", None, None)
            job.future = None

    # -- delivery ---------------------------------------------------------------------

    def _resolve(self, job: _Job, frame_set: c.FrameSet) -> c.AnchorResult | None:
        """Inspect a job's outcome at its Delivery_Group; return a checked ``ok`` or
        ``failed`` (settled) result, else None."""
        group_id = frame_set.group_id
        request = job.request
        watchdog = self.policy.watchdog_ns
        if job.future is None:
            assert job.sync_outcome is not None
            kind, value, duration = job.sync_outcome
            job.sync_outcome = None
        else:
            future = job.future
            if future.done():
                kind, value, duration = self._future_outcome(future)
            else:
                stall_start = self._clock()
                remaining = max(0, watchdog - (stall_start - job.dispatch_clock_ns))
                try:
                    kind, value, duration = self._future_outcome(future, remaining / 1e9)
                except concurrent.futures.TimeoutError:
                    kind, value, duration = "timeout", None, None
                stall_ns = self._clock() - stall_start
                self._counts["anchor_stalled"] += 1
                self._event("stall", group_id, request, reason="worker_unfinished",
                            stall_ns=stall_ns)
            job.future = None  # an abandoned result is never read

        fault: tuple[str, str] | None = None
        if kind == "timeout" or (duration is not None and duration > watchdog):
            fault = ("watchdog", f"worker run exceeded watchdog_ns = {watchdog}")
        elif kind == "exception":
            fault = ("exception", f"{type(value).__name__}: {value}")
        elif not isinstance(value, c.AnchorResult):
            fault = ("bad_return", f"run returned {type(value).__name__}, not AnchorResult")
        elif value.status is c.AnchorStatus.BACKEND_FAULT:
            fault = ("backend_fault_status", f"result reason {value.reason!r}")
        if fault is not None:
            self._fault(group_id, request, *fault, duration)
            return None

        result: c.AnchorResult = value
        extra: dict[str, Any] = {"delivery_group_id": group_id, "delivery_t_ns": frame_set.t_ns,
                                 "worker_duration_ns": duration}
        try:
            result.check_against(request)
        except c.ContractError as exc:
            self._counts["anchor_rejected"] += 1
            self._event("rejected", group_id, request, reason="check_against_mismatch",
                        detail=str(exc), result=result.summary(), **extra)
            return None
        if result.status is c.AnchorStatus.CANCELLED:
            self._counts["anchor_rejected"] += 1
            self._event("rejected", group_id, request, reason="unexpected_cancelled",
                        detail=result.reason, result=result.summary(), **extra)
            return None
        if result.status is c.AnchorStatus.FAILED:
            self._counts["anchor_failed"] += 1
            self._event("failed", group_id, request, reason=result.reason,
                        result=result.summary(), **extra)
            return result
        self._counts["anchor_delivered"] += 1
        self._event("delivered", group_id, request, reason="ok", result=result.summary(), **extra)
        return result

    @staticmethod
    def _future_outcome(
        future: concurrent.futures.Future, timeout: float | None = None
    ) -> tuple[str, Any, int | None]:
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise
        except BaseException as exc:  # includes SystemExit raised in the isolated worker thread
            return "exception", exc, None

    def _fault(
        self, group_id: int, request: c.AnchorRequest, fault_kind: str, detail: str,
        duration: int | None,
    ) -> None:
        self._faulted = True
        self._counts["anchor_faulted"] += 1
        self._event("backend_fault", group_id, request, reason="backend_fault",
                    fault_kind=fault_kind, detail=detail, worker_duration_ns=duration)
        if self._pending is not None:
            pending = self._pending.request
            self._pending = None
            self._counts["anchor_cancelled"] += 1
            self._event("cancelled", group_id, pending, reason="backend_fault")

    # -- diagnostics ------------------------------------------------------------------

    def _event(
        self, kind: str, group_id: int | None, request: c.AnchorRequest | None = None,
        **fields: Any,
    ) -> None:
        record: dict[str, Any] = {
            "record": "SchedulerEvent",
            "kind": kind,
            "group_id": group_id,
            "job_id": None,
            "purpose": None,
            "camera_id": None,
            "source_group_id": None,
            "source_exposure_ns": None,
            "delivery_not_before_ns": None,
            "reason": None,
        }
        if request is not None:
            record.update(
                job_id=request.job_id,
                purpose=request.purpose.value,
                camera_id=request.camera_id,
                source_group_id=request.source_group_id,
                source_exposure_ns=request.source_exposure_ns,
                delivery_not_before_ns=request.delivery_not_before_ns,
            )
        record.update(fields)
        self._events.append(record)

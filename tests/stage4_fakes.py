"""Test-only Stage 4 fakes: a synthetic stream, its GT, an anchor worker and tracker fakes.

Nothing here is imported by ``camloc``. The synthetic trajectory is known only to tests; no
fake returns a pose to the estimator of its own accord. The helpers write no files and read
no recording, GT CSV or ``evaluation/reference*`` data. Passing tests built on them is
synthetic software and timing verification, not a localization result.
"""

import copy
import math
import threading
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

import numpy as np

import contract_factories as f
from camloc import contracts as c

T0 = f.EXPOSURE_NS  # 19-digit first `fr` exposure of every synthetic stream
CAMERA_STEP_NS = f.GROUP_EXPOSURE_STEP_NS  # `r` exposure = `fr` exposure + 500_000 ns
RADIUS_M = 200.0
SPEED_MPS = 20.0
YAW_RATE_RADPS = SPEED_MPS / RADIUS_M  # 0.1 rad/s, counter-clockwise
DEFAULT_PAYLOAD = f.JPEG_PAYLOAD
MAX_RANDOM_DELAY_S = 0.05
FAULT_KINDS = ("exception", "bad_return", "backend_fault_status", "watchdog")
WORKER_MODES = ("sync", "event_before", "event_after", "random_delay")
RUNTIME_KEYS = frozenset({"stall_ns", "worker_duration_ns"})

# One shared read-only canonical image and mask. Observation stores (never copies) and
# freezes its buffers, so sharing them keeps fake observations cheap.
SHARED_IMAGE = f.canonical_image()
SHARED_IMAGE.setflags(write=False)
SHARED_MASK = f.canonical_mask()
SHARED_MASK.setflags(write=False)


# ---------------------------------------------------------------------------------------
# Synthetic stream and its test-only ground truth
# ---------------------------------------------------------------------------------------


def gt_pose(t_ns: int) -> tuple[float, float, float]:
    """Return the synthetic (x, y, yaw) at ``t_ns``: a 200 m circle at 20 m/s from the origin.

    Test only. At ``T0`` the body is at x = y = yaw = 0, heading along +x and turning left.
    """
    angle = YAW_RATE_RADPS * (c.as_ns(t_ns, "gt_pose t_ns") - T0) / c.NS_PER_S
    return (RADIUS_M * math.sin(angle), RADIUS_M * (1.0 - math.cos(angle)),
            c.wrap_angle(angle))


def pose_matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """Return the planar rigid T_WB of (x, y, yaw) at height ``z``."""
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                         [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    return f.rigid(rotation, (x, y, z))


def gt_transform(t_ns: int, z: float = 0.0) -> np.ndarray:
    """Return the synthetic T_WB at ``t_ns`` (test only)."""
    return pose_matrix(*gt_pose(t_ns), z=z)


def synthetic_stream(
    n: int,
    rate_hz: float,
    *,
    cameras: Sequence[str] = (c.PRIMARY_CAMERA, c.SECONDARY_CAMERA),
    jitter_ns: int = 0,
    seed: int = 0,
    payload: bytes = DEFAULT_PAYLOAD,
) -> list[c.FrameSet]:
    """Return ``n`` complete FrameSets at ``rate_hz`` along the ``gt_pose`` trajectory.

    Group ``k``'s first camera is exposed at ``T0 + k * T`` (integer ns, ``T = 1 s / rate``)
    and each later camera ``CAMERA_STEP_NS`` after the previous one, so the default
    ``("fr", "r")`` stream has ``r = fr + 500_000`` and ``t_ns`` equals the last camera's
    exposure. ``jitter_ns`` (at most 1 ms) adds a seeded integer offset in
    ``[-jitter_ns, jitter_ns]`` to every group except group 0, whose first exposure stays
    exactly ``T0``. ``payload`` is the opaque JPEG bytes of every image.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative int")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    cameras = tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("cameras must be a non-empty tuple of unique ids")
    if isinstance(jitter_ns, bool) or not isinstance(jitter_ns, int) or not 0 <= jitter_ns <= 1_000_000:
        raise ValueError("jitter_ns must be an int in [0, 1_000_000]")
    period_ns = round(c.NS_PER_S / rate_hz)
    spread_ns = (len(cameras) - 1) * CAMERA_STEP_NS
    if 2 * jitter_ns + spread_ns >= period_ns:
        raise ValueError("jitter and camera spread must stay below the group period")
    rng = np.random.default_rng([seed, 4])
    offsets = rng.integers(-jitter_ns, jitter_ns, size=n, endpoint=True) if jitter_ns else np.zeros(n)
    frames = []
    for k in range(n):
        base = T0 + k * period_ns + (int(offsets[k]) if k else 0)
        images = tuple(
            f.make_image_message(camera_id=camera, exposure_ns=base + j * CAMERA_STEP_NS,
                                 data=payload)
            for j, camera in enumerate(cameras)
        )
        frames.append(f.make_frame_set(group_id=k, selected_camera_ids=cameras, images=images))
    return frames


def observation(image: c.ImageMessage) -> c.Observation:
    """Return a cheap Observation of ``image`` on the shared read-only image and mask."""
    return f.make_observation(
        camera_id=image.camera_id, exposure_ns=image.exposure_ns, arrival_ns=image.arrival_ns,
        source_id=image.source_id, calibration_id=image.calibration_id,
        image=SHARED_IMAGE, valid_mask=SHARED_MASK,
    )


def image_of(frame_set: c.FrameSet, camera_id: str) -> c.ImageMessage | None:
    """Return ``frame_set``'s image of ``camera_id``, or None."""
    return next((image for image in frame_set.images if image.camera_id == camera_id), None)


# ---------------------------------------------------------------------------------------
# Anchor worker fake
# ---------------------------------------------------------------------------------------


class FakeClock:
    """Injected nanosecond clock: ``clock()`` reads, ``advance(ns)`` moves simulated time."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now = c.as_ns(start_ns)
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now

    def advance(self, ns: int) -> None:
        step = c.as_ns(ns, "clock advance")
        with self._lock:
            self._now += step


class FakeWorker:
    """``AnchorWorker`` fake whose AnchorResult depends only on the AnchorRequest.

    Every result field except the Runtime_Field ``worker_duration_ns`` is a function of the
    request alone. Modes:

    - ``sync``: return at once (plus ``duration_ns`` on an injected ``clock``).
    - ``event_before``: return at once and set the job's ``finished_event``, so a test
      source can wait for completion before yielding the Delivery_Group.
    - ``event_after``: block until ``release_through(t_ns)`` covers the request's
      ``delivery_not_before_ns`` (or ``release_all``), forcing a stall at delivery.
    - ``random_delay``: sleep a seeded per-job delay in ``[0, max_delay_s]`` (<= 50 ms).

    ``duration_ns`` advances an injected ``FakeClock`` on each call (simulated work, no
    sleep). ``fault`` (one of ``FAULT_KINDS``) applies to the job ids in ``fault_jobs``, or
    to every job when ``fault_jobs`` is None. ``variant(request)`` may return "ok",
    "failed", "cancelled" or "mismatch" to produce non-fault outcomes. Concurrent ``run``
    calls are counted; ``max_concurrency`` must stay 1 under C's scheduler.
    """

    def __init__(
        self,
        mode: str = "sync",
        *,
        seed: int = 0,
        max_delay_s: float = MAX_RANDOM_DELAY_S,
        clock: FakeClock | None = None,
        duration_ns: int = 0,
        fault: str | None = None,
        fault_jobs: Collection[int] | None = None,
        watchdog_duration_ns: int = 2_500_000_000,
        hang_timeout_s: float = 2.0,
        variant: Callable[[c.AnchorRequest], str] | None = None,
    ) -> None:
        if mode not in WORKER_MODES:
            raise ValueError(f"unknown FakeWorker mode {mode!r}")
        if fault is not None and fault not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {fault!r}")
        if not 0 <= max_delay_s <= MAX_RANDOM_DELAY_S:
            raise ValueError("max_delay_s must lie in [0, 0.05]")
        if duration_ns and clock is None:
            raise ValueError("a simulated duration needs an injected clock")
        self.mode, self.seed, self.max_delay_s = mode, seed, max_delay_s
        self.clock, self.duration_ns = clock, c.as_ns(duration_ns, "duration_ns")
        self.fault = fault
        self.fault_jobs = None if fault_jobs is None else frozenset(fault_jobs)
        self.watchdog_duration_ns = c.as_ns(watchdog_duration_ns, "watchdog_duration_ns")
        self.hang_timeout_s = hang_timeout_s
        self.variant = variant
        self.requests: list[c.AnchorRequest] = []  # in call-start order
        self.active = 0
        self.max_concurrency = 0
        self._lock = threading.Lock()
        self._release = threading.Condition(self._lock)
        self._released_through_ns = -1
        self._released_all = False
        self._finished: dict[int, threading.Event] = {}

    # -- test controls ------------------------------------------------------------------

    @property
    def job_ids(self) -> list[int]:
        with self._lock:
            return [request.job_id for request in self.requests]

    def finished_event(self, job_id: int) -> threading.Event:
        """Event set when job ``job_id`` returns or raises (created on first use)."""
        with self._lock:
            return self._finished.setdefault(job_id, threading.Event())

    def release_through(self, t_ns: int, delay_s: float = 0.0) -> None:
        """Let ``event_after`` jobs with ``delivery_not_before_ns <= t_ns`` finish.

        With ``delay_s > 0`` the release happens on a ``threading.Timer``, so the runner is
        already waiting at the Delivery_Group when the job completes.
        """
        def release():
            with self._release:
                self._released_through_ns = max(self._released_through_ns, t_ns)
                self._release.notify_all()

        if delay_s > 0:
            timer = threading.Timer(delay_s, release)
            timer.daemon = True
            timer.start()
        else:
            release()

    def release_all(self) -> None:
        """Release every gated or hanging job (test cleanup)."""
        with self._release:
            self._released_all = True
            self._release.notify_all()

    # -- AnchorWorker -------------------------------------------------------------------

    def run(self, request: c.AnchorRequest) -> c.AnchorResult:
        with self._lock:
            self.requests.append(request)
            self.active += 1
            self.max_concurrency = max(self.max_concurrency, self.active)
            done = self._finished.setdefault(request.job_id, threading.Event())
        start = time.perf_counter_ns() if self.clock is None else self.clock()
        try:
            faulty = self.fault is not None and (
                self.fault_jobs is None or request.job_id in self.fault_jobs)
            self._wait(request, faulty)
            if self.clock is not None:
                self.clock.advance(self.duration_ns)
                if faulty and self.fault == "watchdog":
                    self.clock.advance(self.watchdog_duration_ns)
            if faulty and self.fault == "exception":
                raise RuntimeError(f"fake backend exception in job {request.job_id}")
            if faulty and self.fault == "bad_return":
                return {"job_id": request.job_id, "status": "ok"}  # type: ignore[return-value]
            end = time.perf_counter_ns() if self.clock is None else self.clock()
            status = "backend_fault_status" if faulty and self.fault == "backend_fault_status" else (
                self.variant(request) if self.variant is not None else "ok")
            return result_for(request, status, worker_duration_ns=end - start)
        finally:
            with self._lock:
                self.active -= 1
            done.set()

    def _wait(self, request: c.AnchorRequest, faulty: bool) -> None:
        if faulty and self.fault == "watchdog" and self.clock is None:
            with self._release:  # a real hang: only release_all (or the safety timeout) ends it
                self._release.wait_for(lambda: self._released_all, timeout=self.hang_timeout_s)
            return
        if self.mode == "event_after":
            with self._release:
                self._release.wait_for(
                    lambda: self._released_all
                    or request.delivery_not_before_ns <= self._released_through_ns,
                    timeout=self.hang_timeout_s,
                )
        elif self.mode == "random_delay":
            rng = np.random.default_rng([self.seed, request.job_id])
            time.sleep(float(rng.uniform(0.0, self.max_delay_s)))


def result_for(request: c.AnchorRequest, status: str = "ok", *,
               worker_duration_ns: int | None = None) -> c.AnchorResult:
    """Return the deterministic AnchorResult of ``request`` for one fake outcome.

    ``status`` is "ok", "failed", "cancelled", "backend_fault_status" or "mismatch" (an
    ``ok`` result answering a different job id, which ``check_against`` rejects).
    """
    job = request.job_id
    references = request.reference_ids or (job % 5,)
    matched = references[:1]
    remaining = tuple(i for i in range(job % 7 + 10, job % 7 + 13) if i not in matched)
    fields: dict[str, Any] = {
        "job_id": job + 1_000_000 if status == "mismatch" else job,
        "purpose": request.purpose,
        "camera_id": request.camera_id,
        "source_group_id": request.source_group_id,
        "source_exposure_ns": request.source_exposure_ns,
        "matched_reference_ids": matched,
        "remaining_candidate_ids": remaining,
        "map_id": request.map_id,
        "preprocessing_id": request.preprocessing_id,
        "model_ids": ("fake_disk", "fake_lightglue"),
        "worker_duration_ns": worker_duration_ns,
    }
    if status in ("ok", "mismatch"):
        rows = 3 + job % 4
        fields.update(status=c.AnchorStatus.OK, reason="ok", correspondences=f.make_correspondences(
            camera_id=request.camera_id, exposure_ns=request.source_exposure_ns,
            keypoint_ids=np.arange(rows), landmark_ids=np.arange(rows) + 100 * (job % 50),
            pixels=np.column_stack([np.linspace(50.0, 900.0, rows), np.linspace(40.0, 420.0, rows)]),
            map_id=request.map_id,
        ))
    else:
        fault_status = {"failed": (c.AnchorStatus.FAILED, "insufficient_matches"),
                        "cancelled": (c.AnchorStatus.CANCELLED, "fake_cancelled"),
                        "backend_fault_status": (c.AnchorStatus.BACKEND_FAULT, "fake_backend_fault")}
        if status not in fault_status:
            raise ValueError(f"unknown fake result status {status!r}")
        result_status, reason = fault_status[status]
        fields.update(status=result_status, reason=reason, correspondences=None)
    return c.AnchorResult(**fields)


# ---------------------------------------------------------------------------------------
# MappedTracker / TrackProjector fakes
# ---------------------------------------------------------------------------------------


class FakeTracker:
    """``MappedTracker`` fake that logs ``(method, group_id, argument)`` tuples.

    The frontend sets ``current_group`` before calling it; ``reset`` logs the reset's own
    group id. ``track``/``handoff``/``verify_candidate`` delegate to optional callables;
    without one, ``track``/``handoff`` raise and ``verify_candidate`` returns None, so the
    fake never invents a pose.
    """

    def __init__(self, *, track: Callable[..., c.PoseEvidence] | None = None,
                 handoff: Callable[..., c.HandoffEvidence] | None = None,
                 verify: Callable[[c.Observation], c.PoseEvidence | None] | None = None) -> None:
        self._track, self._handoff, self._verify = track, handoff, verify
        self.current_group: int | None = None
        self.calls: list[tuple[str, int | None, Any]] = []

    def _log(self, name: str, argument: Any, group_id: int | None = None) -> None:
        self.calls.append((name, self.current_group if group_id is None else group_id, argument))

    def observe(self, observation: c.Observation) -> None:
        self._log("observe", observation)

    def track(self, observation, snapshot, aux) -> c.PoseEvidence:
        self._log("track", observation)
        if self._track is None:
            raise NotImplementedError("FakeTracker.track has no configured outcome")
        return self._track(observation, snapshot, aux)

    def handoff(self, result: c.AnchorResult, current: c.Observation) -> c.HandoffEvidence:
        self._log("handoff", result)
        if self._handoff is None:
            raise NotImplementedError("FakeTracker.handoff has no configured outcome")
        return self._handoff(result, current)

    def settle(self, handoff: c.HandoffEvidence, disposition: c.HandoffDisposition) -> None:
        self._log("settle", (handoff, c.HandoffDisposition(disposition)))

    def verify_candidate(self, observation: c.Observation) -> c.PoseEvidence | None:
        self._log("verify_candidate", observation)
        return None if self._verify is None else self._verify(observation)

    def reset(self, reset: c.RecoveryReset) -> None:
        self._log("reset", reset, reset.group_id)

    def names(self) -> list[str]:
        return [name for name, _, _ in self.calls]


class FakeProjector:
    """``TrackProjector`` fake that logs ``reset`` calls; ``project`` needs a callable."""

    def __init__(self, project: Callable[[c.Estimate], tuple] | None = None) -> None:
        self._project = project
        self.calls: list[tuple[str, int | None, Any]] = []

    def project(self, estimate: c.Estimate):
        self.calls.append(("project", estimate.group_id, estimate))
        if self._project is None:
            raise NotImplementedError("FakeProjector.project has no configured outcome")
        return self._project(estimate)

    def reset(self, reset: c.RecoveryReset) -> None:
        self.calls.append(("reset", reset.group_id, reset))


# ---------------------------------------------------------------------------------------
# Runtime-field removal
# ---------------------------------------------------------------------------------------


def _strip_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_keys(item) for key, item in value.items() if key not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [_strip_keys(item) for item in value]
    return copy.deepcopy(value)


def _is_stall(record: Any) -> bool:
    return (isinstance(record, dict) and record.get("record") == "SchedulerEvent"
            and record.get("kind") == "stall")


def strip_runtime(records: Iterable[dict] | dict) -> list[dict] | dict:
    """Return a copy without Runtime_Fields, and with nothing else removed.

    Drops every ``stall`` SchedulerEvent, the keys ``stall_ns`` and ``worker_duration_ns``
    at any depth and the top-level ``anchor_stalled`` key (RunSummary record or the
    runner's returned counts). A single mapping is treated as one record.
    """
    def one(record: dict) -> dict:
        stripped = _strip_keys(record)
        stripped.pop("anchor_stalled", None)
        return stripped

    if isinstance(records, dict):
        return one(records)
    return [one(record) for record in records if not _is_stall(record)]

"""Stage 4 AnchorScheduler tests: bounded queue, delivery, expiry, EOF, rate limits, trigger,
atomic rejection, result checking, backend faults and threaded mode.

Property tests are seeded numpy loops (Hypothesis is not installed) that drive the scheduler
directly, in sync mode unless stated, over the test-only ``stage4_fakes`` streams. Each
property is checked against expectations computed in this file from the specification, not
from the scheduler's internals. Passing these tests is synthetic timing verification only;
it establishes no localization accuracy, runtime performance or native high-rate support.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import contract_factories as f
import stage4_fakes as s4
from camloc import contracts as c
from camloc.scheduler import AnchorScheduler, SchedulerPolicy

P = c.AnchorPurpose
S = c.Status
FR, R = c.PRIMARY_CAMERA, c.SECONDARY_CAMERA
RATES = (10, 20, 30, 60)
GROUPS_TO_DELIVERY = {10: 1, 20: 2, 30: 3, 60: 6}  # ceil(90 ms / T)
DELAY = c.DELIVERY_MIN_DELAY_NS
WINDOW_NS = 1_000_000_000
PURPOSES = tuple(P)
STATUSES = tuple(S)
CANDIDATE_ORDER = (P.RECOVERY, P.FRONT_ANCHOR, P.REAR_ANCHOR, P.WARM_RETRIEVAL)
NORMAL_CAPS = {P.FRONT_ANCHOR: 2, P.REAR_ANCHOR: 1, P.WARM_RETRIEVAL: 1}
RECOVERY_CAPS = {P.WARM_RETRIEVAL: 2}
NORMAL_TOTAL, RECOVERY_TOTAL = 3, 4
REFRESH_NS, MIN_TRACKS, TRAVEL_M, YAW_DEG = 500_000_000, 40, 8.0, 8.0
OUTCOME_KINDS = ("delivered", "failed", "rejected")
SEQUENCE_KINDS = ("dispatched", "delivered", "replaced", "expired", "cancelled")
QUEUE_KINDS = SEQUENCE_KINDS + ("created", "rate_limited", "proposals_discarded")
BASE_GROUPS = 90
ITERATIONS = 100


@pytest.mark.parametrize("threaded", [False, True])
def test_stuck_worker_watchdog_allows_run_and_process_to_finish(threaded):
    # A subprocess proves shutdown too: executor non-daemon threads otherwise hang here.
    code = f'''
import sys, threading
sys.path.insert(0, "tests")
from test_scheduler import Run, exact_stream
from camloc.scheduler import SchedulerPolicy
from camloc import contracts as c
class Stuck:
    def run(self, request):
        threading.Event().wait()
run = Run(exact_stream(10, 3), Stuck(), threaded={threaded!r},
          policy=SchedulerPolicy(watchdog_ns=50_000_000))
ctx = run.begin(0)
run.propose(ctx, 0, c.AnchorPurpose.FRONT_ANCHOR, track_count=0)
run.end(0)
assert run.begin(1).results == ()
assert run.scheduler.faulted
run.end(1)
run.begin(2)
run.end(2)
run.finish()
assert run.scheduler.counts()["anchor_dispatched"] == 1
print("bounded")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, timeout=10, check=True)
    assert result.stdout.strip() == "bounded"


# ---------------------------------------------------------------------------------------
# Streams, inputs and the driving harness
# ---------------------------------------------------------------------------------------


class Stream:
    """FrameSets with cached observations (per camera), snapshots and estimates."""

    def __init__(self, frames: list[c.FrameSet]) -> None:
        self.frames = list(frames)
        self.obs = [{im.camera_id: s4.observation(im) for im in fs.images} for fs in self.frames]
        self._snapshots: dict = {}
        self._estimates: dict = {}

    def snapshot(self, k: int, status: c.Status) -> c.SearchSnapshot:
        key = (k, status)
        if key not in self._snapshots:
            t_ns = self.frames[k].t_ns
            x, y, yaw = s4.gt_pose(t_ns)
            self._snapshots[key] = f.make_search_snapshot(
                t_ns=t_ns, state=np.array([x, y, yaw, s4.SPEED_MPS, 0.0, 0.0]), status=status)
        return self._snapshots[key]

    def pose(self, k: int) -> tuple[float, float, float]:
        state = self.snapshot(k, S.TRACKING).state
        return float(state[0]), float(state[1]), float(state[2])

    def estimate(self, k: int, status: c.Status) -> c.Estimate:
        key = (k, status)
        if key not in self._estimates:
            fs, snap = self.frames[k], self.snapshot(k, status)
            self._estimates[key] = c.Estimate(
                t_ns=fs.t_ns, group_id=fs.group_id, state=np.array(snap.state),
                P=np.array(snap.P), status=status, branch=c.Branch.TRACK,
                evidence_kinds=(c.EvidenceKind.PREDICTION_ONLY,), reason="no_evidence",
                auxiliary=c.AuxiliaryGeometry.unavailable("not_estimated"),
                max_support_exposure_ns=None)
        return self._estimates[key]


@functools.cache
def base_stream(rate: int, seed: int) -> Stream:
    return Stream(s4.synthetic_stream(BASE_GROUPS, rate, jitter_ns=1_000_000, seed=seed))


@functools.cache
def exact_stream(rate: int, n: int) -> Stream:
    return Stream(s4.synthetic_stream(n, rate))


def frame(group_id: int, t_ns: int) -> c.FrameSet:
    """A FrameSet whose ``r`` image is exposed at ``t_ns`` and ``fr`` 0.5 ms earlier."""
    images = (f.make_image_message(camera_id=FR, exposure_ns=t_ns - s4.CAMERA_STEP_NS),
              f.make_image_message(camera_id=R, exposure_ns=t_ns))
    return f.make_frame_set(group_id=group_id, images=images)


@dataclasses.dataclass(frozen=True)
class Group:
    index: int  # frame index in the stream
    status: c.Status
    proposals: tuple  # (purpose, camera, reference_ids, track_count)
    frontend_failed: bool = False


def random_groups(rng: np.random.Generator, n: int, *, drop: float = 0.15,
                  density: float = 0.35, tc_low: int = 20, switch: float = 0.15,
                  failed: float = 0.05) -> list[Group]:
    """Random groups: dropped frames, mode switches and at most one proposal per purpose."""
    indices, k = [], 0
    while len(indices) < n and k < BASE_GROUPS:
        indices.append(k)
        k += 1 + (int(rng.integers(1, 4)) if rng.random() < drop else 0)
    status, groups = S.TRACKING, []
    for k in indices:
        if rng.random() < switch:
            status = STATUSES[int(rng.integers(len(STATUSES)))]
        proposals = []
        for j in rng.permutation(len(PURPOSES)):
            purpose = PURPOSES[int(j)]
            if rng.random() >= density:
                continue
            camera = {P.FRONT_ANCHOR: FR, P.REAR_ANCHOR: R}.get(purpose) or (FR, R)[int(rng.integers(2))]
            refs = tuple(int(v) for v in rng.choice(30, size=int(rng.integers(0, 3)), replace=False))
            proposals.append((purpose, camera, refs, int(rng.integers(tc_low, 200))))
        groups.append(Group(k, status, tuple(proposals), bool(rng.random() < failed)))
    return groups


class Run:
    """Drives one AnchorScheduler group by group and keeps every event and context."""

    def __init__(self, stream: Stream, worker=None, *, threaded: bool = False,
                 policy: SchedulerPolicy | None = None, clock=None) -> None:
        self.stream = stream
        self.worker = s4.FakeWorker() if worker is None else worker
        kwargs = {} if clock is None else {"clock": clock}
        self.scheduler = AnchorScheduler(self.worker, map_id=f.MAP_ID, policy=policy,
                                         threaded=threaded, **kwargs)
        self.initial_ns = stream.frames[0].t_ns
        self.scheduler.start(c.InitialPose(timestamp_ns=self.initial_ns, x=0.0, y=0.0, yaw=0.0))
        self.events: list[dict] = []
        self.contexts: dict = {}
        self.order: list[int] = []  # frame indices in run order
        self.t_of: dict = {}
        self.status_of: dict = {}
        self.finish_at: int | None = None
        self.in_end = False

    def begin(self, k: int, status: c.Status = S.TRACKING):
        fs = self.stream.frames[k]
        ctx = self.scheduler.begin_group(fs, self.stream.snapshot(k, status), status)
        self.order.append(k)
        self.contexts[fs.group_id] = ctx
        self.t_of[fs.group_id] = fs.t_ns
        self.status_of[fs.group_id] = status
        return ctx

    def propose(self, ctx, k: int, purpose, camera=None, refs=(), track_count=100) -> None:
        camera = (R if purpose is P.REAR_ANCHOR else FR) if camera is None else camera
        ctx.propose(purpose=purpose, observation=self.stream.obs[k][camera],
                    reference_ids=refs, track_count=track_count)

    def end(self, k: int, status: c.Status = S.TRACKING, frontend_failed: bool = False) -> None:
        self.in_end = True
        try:
            self.scheduler.end_group(self.stream.estimate(k, status),
                                     frontend_failed=frontend_failed)
        finally:
            self.in_end = False
        self.events.extend(self.scheduler.drain_diagnostics())

    def step(self, group: Group):
        ctx = self.begin(group.index, group.status)
        for purpose, camera, refs, tc in group.proposals:
            self.propose(ctx, group.index, purpose, camera, refs, tc)
        self.end(group.index, group.status, group.frontend_failed)
        return ctx

    def drive(self, groups) -> Run:
        for group in groups:
            self.step(group)
        return self

    def finish(self) -> Run:
        self.finish_at = len(self.events)
        self.scheduler.finish()
        self.events.extend(self.scheduler.drain_diagnostics())
        return self

    def of(self, *kinds: str) -> list[dict]:
        return [e for e in self.events if e["kind"] in kinds]

    def ids(self, *kinds: str) -> list[int]:
        return [e["job_id"] for e in self.of(*kinds)]

    def delivered_ids(self, group_id: int) -> list[int]:
        return [r.job_id for r in self.contexts[group_id].results]


class SpyWorker(s4.FakeWorker):
    """FakeWorker that records whether each ``run`` call happens inside ``end_group``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[int, bool]] = []
        self.owner: Run | None = None

    def run(self, request):
        self.calls.append((request.job_id, self.owner.in_end))
        return super().run(request)


def blocked(window: list[tuple[int, str]], purpose: str, s: int, recovery: bool) -> bool:
    """Spec rate check (Req 5.1/5.3) over earlier dispatched (s, purpose value) pairs."""
    inside = [p for ns, p in window if s - WINDOW_NS < ns <= s] + [purpose]
    caps, total = (RECOVERY_CAPS, RECOVERY_TOTAL) if recovery else (NORMAL_CAPS, NORMAL_TOTAL)
    return len(inside) > total or inside.count(purpose) > caps.get(P(purpose), len(inside))


def odometers(run: Run) -> dict[int, tuple[float, float]]:
    """Summed planar distance and absolute wrapped yaw over consecutive group snapshots."""
    odo, prev, out = (0.0, 0.0), None, {}
    for k in run.order:
        pose = run.stream.pose(k)
        if prev is not None:
            odo = (odo[0] + math.hypot(pose[0] - prev[0], pose[1] - prev[1]),
                   odo[1] + abs(c.wrap_angle(pose[2] - prev[2])))
        prev = pose
        out[run.stream.frames[k].group_id] = odo
    return out


# ---------------------------------------------------------------------------------------
# Spec reference model and event replay
# ---------------------------------------------------------------------------------------


class QueueModel:
    """Reference model of Requirements 3-5 and the design's boundary steps (always-ok worker)."""

    def __init__(self, initial_ns: int) -> None:
        self.next_id = 0
        self.flight: dict | None = None
        self.pending: dict | None = None
        self.window: list[tuple[int, str]] = []
        self.last_front = initial_ns
        self.odo = (0.0, 0.0)
        self.base = (0.0, 0.0)
        self.prev = None
        self.group = None
        self.events: list[tuple] = []

    def begin(self, gid: int, t_ns: int, pose) -> None:
        if self.prev is not None:
            self.odo = (self.odo[0] + math.hypot(pose[0] - self.prev[0], pose[1] - self.prev[1]),
                        self.odo[1] + abs(c.wrap_angle(pose[2] - self.prev[2])))
        self.prev, self.group = pose, gid
        if self.flight and self.flight["at"] is None and t_ns >= self.flight["dnb"]:
            self.flight["at"] = gid
            self.events.append(("delivered", gid, self.flight["id"], None))

    def triggered(self, s: int, track_count: int) -> bool:
        return (s - self.last_front >= REFRESH_NS or track_count < MIN_TRACKS
                or self.odo[0] - self.base[0] >= TRAVEL_M
                or math.degrees(self.odo[1] - self.base[1]) >= YAW_DEG)

    def end(self, gid: int, t_ns: int, proposals, recovery: bool, failed: bool) -> None:
        if self.flight and self.flight["at"] == gid:
            self.flight = None
        if failed:
            if proposals:
                self.events.append(("proposals_discarded", gid, None, len(proposals)))
        else:
            self._create(gid, proposals, recovery)
        if self.pending and self.pending["dnb"] <= t_ns:
            self.events.append(("expired", gid, self.pending["id"], None))
            self.pending = None
        if self.flight is None and self.pending is not None:
            req, self.pending = self.pending, None
            if blocked(self.window, req["purpose"], req["s"], recovery):
                self.events.append(("rate_limited", gid, req["id"], ("dispatch", req["purpose"])))
                return
            self.window.append((req["s"], req["purpose"]))
            if req["purpose"] == P.FRONT_ANCHOR.value:
                self.last_front, self.base = req["s"], req["odo"]
            self.flight = {"id": req["id"], "dnb": req["dnb"], "at": None}
            self.events.append(("dispatched", gid, req["id"], None))

    def _create(self, gid: int, proposals, recovery: bool) -> None:
        order = sorted(range(len(proposals)),
                       key=lambda j: (CANDIDATE_ORDER.index(proposals[j][0]), j))
        for j in order:
            purpose, s, track_count = proposals[j]
            if purpose is P.FRONT_ANCHOR and not self.triggered(s, track_count):
                continue
            if blocked(self.window, purpose.value, s, recovery):
                self.events.append(("rate_limited", gid, None, ("create", purpose.value)))
                continue
            job, self.next_id = self.next_id, self.next_id + 1
            self.events.append(("created", gid, job, None))
            if self.pending is not None:
                self.events.append(("replaced", gid, job, self.pending["id"]))
            self.pending = {"id": job, "purpose": purpose.value, "s": s, "dnb": s + DELAY,
                            "odo": self.odo}
            return

    def finish(self) -> None:
        if self.flight and self.flight["at"] is None:
            self.events.append(("cancelled", self.group, self.flight["id"], "end_of_input"))
        if self.pending:
            self.events.append(("cancelled", self.group, self.pending["id"], "end_of_input"))


def queue_key(e: dict) -> tuple:
    aux = {"replaced": e.get("replaced_job_id"),
           "rate_limited": (e.get("stage"), e.get("purpose")),
           "proposals_discarded": e.get("count"),
           "cancelled": e.get("reason")}.get(e["kind"])
    return e["kind"], e["group_id"], e["job_id"], aux


@dataclasses.dataclass
class Replay:
    flight: int | None = None
    flight_done: bool = False
    pending: int | None = None
    created: dict = dataclasses.field(default_factory=dict)
    replaced: set = dataclasses.field(default_factory=set)
    dispatched: dict = dataclasses.field(default_factory=dict)  # id -> dispatch group
    outcome: dict = dataclasses.field(default_factory=dict)  # id -> (kind, group)
    expired: set = dataclasses.field(default_factory=set)
    cancelled: dict = dataclasses.field(default_factory=dict)  # id -> reason


def replay(events: list[dict]) -> Replay:
    """Replay events asserting the Property 1 queue invariants; return the final state."""
    st, expect, last_id = Replay(), None, -1
    for e in events:
        kind, job = e["kind"], e["job_id"]
        if expect is not None:  # a creation over a pending request is followed by `replaced`
            assert kind == "replaced" and (e["replaced_job_id"], job) == expect, e
            st.replaced.add(expect[0])
            expect = None
            continue
        if kind == "created":
            assert job > last_id, "created job ids must strictly increase"
            assert e["delivery_not_before_ns"] == e["source_exposure_ns"] + DELAY
            last_id, st.created[job] = job, e
            if st.pending is not None:
                expect = (st.pending, job)
            st.pending = job
        elif kind == "replaced":
            raise AssertionError(f"replaced without a creation over a pending request: {e}")
        elif kind == "expired" or (kind == "rate_limited" and e["stage"] == "dispatch"):
            assert job == st.pending and job not in st.expired
            st.expired.add(job)
            st.pending = None
        elif kind == "rate_limited":
            assert job is None
        elif kind == "dispatched":
            assert job == st.pending and job not in st.replaced
            assert st.flight is None or st.flight_done, "two jobs in flight"
            st.flight, st.flight_done, st.pending = job, False, None
            st.dispatched[job] = e["group_id"]
        elif kind in OUTCOME_KINDS or kind == "backend_fault":
            assert job == st.flight and not st.flight_done and job not in st.outcome
            st.flight_done = True
            st.outcome[job] = (kind, e["group_id"])
        elif kind == "cancelled":
            assert job not in st.cancelled
            if job == st.pending:
                st.pending = None
            else:
                assert job == st.flight and not st.flight_done
                assert e["reason"] == "end_of_input", "in-flight cancelled before delivery"
                st.flight = None
            st.cancelled[job] = e["reason"]
    assert expect is None
    return st


def check_rear_in_flight(run: Run, st: Replay) -> None:
    """``rear_in_flight`` is True exactly while a rear job is in flight (through delivery)."""
    spans = []
    for job, dispatch_group in st.dispatched.items():
        if st.created[job]["camera_id"] == R:
            spans.append((dispatch_group, st.outcome.get(job, (None, math.inf))[1]))
    for gid, ctx in run.contexts.items():
        assert ctx.rear_in_flight == any(lo < gid <= hi for lo, hi in spans), gid


# ---------------------------------------------------------------------------------------
# Property 1: bounded queue and replacement
# ---------------------------------------------------------------------------------------


def test_property_1_bounded_queue_and_replacement():
    # Feature: stage4-scheduler, Property 1: Bounded queue and replacement semantics
    replacements = empty_refs = 0
    for i in range(ITERATIONS):
        rng = np.random.default_rng([1, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 30, density=0.45)
        run, model = Run(stream), QueueModel(stream.frames[0].t_ns)
        for g in groups:
            fs = stream.frames[g.index]
            ctx = run.begin(g.index, g.status)
            model.begin(fs.group_id, fs.t_ns, stream.pose(g.index))
            for purpose, camera, refs, tc in g.proposals:
                run.propose(ctx, g.index, purpose, camera, refs, tc)
            run.end(g.index, g.status, g.frontend_failed)
            model.end(fs.group_id, fs.t_ns,
                      [(p, stream.obs[g.index][cam].exposure_ns, tc) for p, cam, _, tc in g.proposals],
                      g.status is not S.TRACKING, g.frontend_failed)
        run.finish()
        model.finish()
        assert [queue_key(e) for e in run.of(*QUEUE_KINDS)] == model.events, (i, run.events)
        st = replay(run.events)
        check_rear_in_flight(run, st)
        assert run.worker.max_concurrency <= 1
        assert st.replaced.isdisjoint(st.dispatched) and st.replaced.isdisjoint(st.outcome)
        replacements += len(st.replaced)
        empty_refs += sum(not r.reference_ids for r in run.worker.requests)
    assert replacements > 0 and empty_refs > 0


# ---------------------------------------------------------------------------------------
# Property 2: delivery group rule
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", RATES)
def test_property_2_delivery_k_groups_later(rate):
    # Feature: stage4-scheduler, Property 2: Delivery group rule
    k = GROUPS_TO_DELIVERY[rate]
    stream = exact_stream(rate, k + 3)
    run = Run(stream)
    ctx = run.begin(0)
    run.propose(ctx, 0, P.RECOVERY, R)  # source exposure == source group's t_ns
    run.end(0)
    for j in range(1, k + 3):
        run.begin(j)
        run.end(j)
    run.finish()
    (created,) = run.of("created")
    assert created["delivery_not_before_ns"] == stream.frames[0].t_ns + DELAY
    assert [e["group_id"] for e in run.of("delivered")] == [k]
    assert {g: run.delivered_ids(g) for g in run.contexts} == {
        g: ([0] if g == k else []) for g in run.contexts}


@pytest.mark.parametrize("offset, delivered_at", [(0, 1), (-1, 2)])
def test_property_2_delivery_boundary_dnb(offset, delivered_at):
    # Feature: stage4-scheduler, Property 2: Delivery group rule
    dnb = s4.T0 + DELAY
    stream = Stream([frame(0, s4.T0), frame(1, dnb + offset), frame(2, dnb + 50_000_000)])
    run = Run(stream)
    ctx = run.begin(0)
    run.propose(ctx, 0, P.RECOVERY, R)
    run.end(0)
    for j in (1, 2):
        run.begin(j)
        run.end(j)
    run.finish()
    assert run.of("created")[0]["delivery_not_before_ns"] == dnb
    assert [e["group_id"] for e in run.of("delivered")] == [delivered_at]
    assert run.delivered_ids(delivered_at) == [0]
    assert run.delivered_ids(3 - delivered_at) == []


def test_property_2_delivery_group_rule_random():
    # Feature: stage4-scheduler, Property 2: Delivery group rule
    for i in range(ITERATIONS):
        rng = np.random.default_rng([2, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 30, drop=0.25)
        worker = SpyWorker()
        run = Run(stream, worker)
        worker.owner = run
        run.drive(groups).finish()
        times = [(stream.frames[g.index].group_id, stream.frames[g.index].t_ns) for g in groups]
        for e in run.of("created"):
            assert e["delivery_not_before_ns"] == e["source_exposure_ns"] + DELAY
        delivered = {e["job_id"]: e["group_id"] for e in run.of("delivered")}
        cancelled = set(run.ids("cancelled"))
        for e in run.of("dispatched"):
            expected = next((gid for gid, t in times
                             if gid > e["group_id"] and t >= e["delivery_not_before_ns"]), None)
            assert delivered.get(e["job_id"]) == expected
            if expected is None:
                assert e["job_id"] in cancelled
        for gid in run.contexts:
            assert run.delivered_ids(gid) == [j for j, at in delivered.items() if at == gid]
        assert all(inside for _, inside in worker.calls), "worker.run outside end_group"
        assert [j for j, _ in worker.calls] == run.ids("dispatched")


# ---------------------------------------------------------------------------------------
# Property 3: pending dispatch and expiry
# ---------------------------------------------------------------------------------------


def test_property_3_pending_dispatch_and_expiry():
    # Feature: stage4-scheduler, Property 3: Pending dispatch and expiry
    tally = {"dispatched": 0, "expired": 0}
    for i in range(ITERATIONS):
        rng = np.random.default_rng([3, i])
        stream = base_stream(RATES[i % 4], i % 3)
        run = Run(stream).drive(random_groups(rng, 30, drop=0.35, density=0.5)).finish()
        dnb = {e["job_id"]: e["delivery_not_before_ns"] for e in run.of("created")}
        boundary: dict = {}
        pending = None
        for e in run.events[:run.finish_at]:
            kind, job = e["kind"], e["job_id"]
            b = boundary.setdefault(e["group_id"], {"released": False, "pending": pending,
                                                    "outcome": None})
            if kind in OUTCOME_KINDS:
                b["released"] = True
            elif kind == "created":
                pending = b["pending"] = job
            elif kind in ("expired", "dispatched") or (kind == "rate_limited" and e["stage"] == "dispatch"):
                assert job == pending and b["outcome"] is None
                b["outcome"], pending = (kind, job), None
        for gid, b in boundary.items():
            if not b["released"] or b["pending"] is None:
                continue
            kind, job = b["outcome"]
            assert job == b["pending"]
            if dnb[job] > run.t_of[gid]:
                assert kind in ("dispatched", "rate_limited")
                tally["dispatched"] += kind == "dispatched"
            else:
                assert kind == "expired"
                tally["expired"] += 1
        for e in run.of("dispatched"):
            assert e["delivery_not_before_ns"] > run.t_of[e["group_id"]]
        expired = run.of("expired")
        for e in expired:
            assert e["delivery_not_before_ns"] <= run.t_of[e["group_id"]]
            assert e["source_group_id"] is not None
        assert len({e["job_id"] for e in expired}) == len(expired)
        assert {e["job_id"] for e in expired}.isdisjoint(run.ids("dispatched"))
    assert tally["dispatched"] > 0 and tally["expired"] > 0, tally


# ---------------------------------------------------------------------------------------
# Property 4: EOF cancellation
# ---------------------------------------------------------------------------------------


def test_property_4_eof_cancellation():
    # Feature: stage4-scheduler, Property 4: EOF cancellation
    runs = cancelled_total = 0
    for i in range(8):
        rng = np.random.default_rng([4, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 25, drop=0.25, density=0.5)
        for m in range(1, len(groups) + 1):
            run = Run(stream).drive(groups[:m])
            before = replay(run.events)
            assert not run.of("cancelled")
            expected = set() if before.pending is None else {before.pending}
            if before.flight is not None and not before.flight_done:
                expected.add(before.flight)
            run.finish()
            cancels = run.events[run.finish_at:]
            assert all(e["kind"] == "cancelled" and e["reason"] == "end_of_input" for e in cancels)
            ids = [e["job_id"] for e in cancels]
            assert len(ids) == len(set(ids)) and set(ids) == expected
            assert not set(ids) & (set(before.outcome) | before.expired | before.replaced)
            assert not any(set(ids) & set(run.delivered_ids(g)) for g in run.contexts)
            replay(run.events)
            runs += 1
            cancelled_total += len(ids)
    assert runs >= ITERATIONS and cancelled_total > 0


# ---------------------------------------------------------------------------------------
# Property 5: source-time rate limits
# ---------------------------------------------------------------------------------------


def test_property_5_source_time_rate_limits():
    # Feature: stage4-scheduler, Property 5: Source-time rate limits
    at_cap = {False: 0, True: 0}
    for i in range(ITERATIONS):
        rng = np.random.default_rng([5, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 40, density=0.6, tc_low=0, switch=0.25)
        run = Run(stream).drive(groups).finish()
        dispatched = run.of("dispatched")
        for n, e in enumerate(dispatched):
            s = e["source_exposure_ns"]
            inside = [d["purpose"] for d in dispatched[:n + 1]
                      if s - WINDOW_NS < d["source_exposure_ns"] <= s]
            recovery = run.status_of[e["group_id"]] is not S.TRACKING
            caps, total = (RECOVERY_CAPS, RECOVERY_TOTAL) if recovery else (NORMAL_CAPS, NORMAL_TOTAL)
            assert len(inside) <= total, (i, e)
            for purpose, cap in caps.items():
                assert inside.count(purpose.value) <= cap, (i, e)
            at_cap[recovery] += len(inside) == total
        created_groups = [e["group_id"] for e in run.of("created")]
        assert len(created_groups) == len(set(created_groups)), "more than one request per group"
        assert all(len(r.reference_ids) <= c.MAX_REFERENCES_PER_JOB for r in run.worker.requests)
    assert at_cap[False] > 0 and at_cap[True] > 0, at_cap


# ---------------------------------------------------------------------------------------
# Property 6: refresh trigger and no-burst deferral
# ---------------------------------------------------------------------------------------


def test_property_6_refresh_trigger_and_deferral():
    # Feature: stage4-scheduler, Property 6: Refresh trigger and no-burst deferral
    tally = {"created": 0, "deferred": 0, "not_triggered": 0}
    for i in range(ITERATIONS):
        rng = np.random.default_rng([6, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 40, density=0.55, tc_low=30, switch=0.1, failed=0.0)
        run = Run(stream).drive(groups).finish()
        odo = odometers(run)
        events: dict = {}
        for e in run.events[:run.finish_at]:
            events.setdefault(e["group_id"], []).append(e)
        last_s, base, window = run.initial_ns, (0.0, 0.0), []
        for g in groups:
            gid = stream.frames[g.index].group_id
            evs = events.get(gid, [])
            created = [e for e in evs if e["kind"] == "created"]
            assert len(created) <= 1
            proposals = {p: (cam, tc) for p, cam, _, tc in g.proposals}
            recovery = g.status is not S.TRACKING
            if P.FRONT_ANCHOR in proposals:
                s = stream.obs[g.index][FR].exposure_ns
                tc = proposals[P.FRONT_ANCHOR][1]
                trig = (s - last_s >= REFRESH_NS or tc < MIN_TRACKS
                        or odo[gid][0] - base[0] >= TRAVEL_M
                        or math.degrees(odo[gid][1] - base[1]) >= YAW_DEG)
                front_created = any(e["purpose"] == P.FRONT_ANCHOR.value for e in created)
                assert trig or not front_created
                if trig and P.RECOVERY not in proposals:
                    limited = [e for e in evs if e["kind"] == "rate_limited"
                               and e["stage"] == "create"]
                    if blocked(window, P.FRONT_ANCHOR.value, s, recovery):
                        assert not front_created and limited
                        assert limited[0]["purpose"] == P.FRONT_ANCHOR.value
                        for event in created:
                            assert not blocked(window, event["purpose"], event["source_exposure_ns"], recovery)
                        tally["deferred"] += 1
                    else:
                        assert front_created and not limited
                        tally["created"] += 1
                    if not recovery and front_created:
                        assert all(e["purpose"] != P.REAR_ANCHOR.value for e in created)
                elif not trig:
                    tally["not_triggered"] += 1
            for e in evs:
                if e["kind"] == "dispatched":
                    window.append((e["source_exposure_ns"], e["purpose"]))
                    if e["purpose"] == P.FRONT_ANCHOR.value:
                        last_s, base = e["source_exposure_ns"], odo[e["source_group_id"]]
    assert min(tally.values()) > 0, tally


# ---------------------------------------------------------------------------------------
# Property 7 (scheduler part): invalid proposals are rejected atomically
# ---------------------------------------------------------------------------------------


def _invalid_proposals(stream: Stream, k: int) -> list[tuple[str, dict]]:
    fr_obs, r_obs = stream.obs[k][FR], stream.obs[k][R]
    foreign_cal = f.make_observation(camera_id=FR, exposure_ns=fr_obs.exposure_ns,
                                     calibration_id="calibration_fr_v2",
                                     image=s4.SHARED_IMAGE, valid_mask=s4.SHARED_MASK)
    good = {"purpose": P.WARM_RETRIEVAL, "observation": fr_obs, "reference_ids": (),
            "track_count": 10}
    cases = [
        ("purpose", {"purpose": "front_anchor"}),
        ("purpose", {"purpose": None}),
        ("observation", {"observation": stream.frames[k].images[0]}),
        ("observation", {"observation": foreign_cal}),
        ("observation", {"observation": stream.obs[k + 1][FR]}),
        ("observation", {"purpose": P.FRONT_ANCHOR, "observation": r_obs}),
        ("observation", {"purpose": P.REAR_ANCHOR, "observation": fr_obs}),
        ("reference_ids", {"reference_ids": (1, 2, 3)}),
        ("reference_ids", {"reference_ids": (-1,)}),
        ("reference_ids", {"reference_ids": (4, 4)}),
        ("reference_ids", {"reference_ids": (1.5,)}),
        ("reference_ids", {"reference_ids": (True,)}),
        ("reference_ids", {"reference_ids": "12"}),
        ("track_count", {"track_count": -1}),
        ("track_count", {"track_count": 40.0}),
        ("track_count", {"track_count": True}),
        ("track_count", {"track_count": None}),
    ]
    return [(name, {**good, **bad}) for name, bad in cases]


def _probe(scheduler: AnchorScheduler, t_ns: int) -> tuple:
    rates = tuple(scheduler.rate_allows(p, t_ns + d, rec) for p in PURPOSES
                  for rec in (False, True) for d in (0, 300_000_000, 900_000_000))
    return scheduler.counts(), scheduler.recovery_work_active(), scheduler.faulted, rates


def test_property_7_invalid_proposals_rejected_atomically():
    # Feature: stage4-scheduler, Property 7: Invalid inputs are rejected atomically
    for i in range(ITERATIONS):
        rng = np.random.default_rng([7, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 16, density=0.5, failed=0.0)
        m = int(rng.integers(len(groups)))
        a, b = Run(stream), Run(stream)
        a.drive(groups[:m])
        b.drive(groups[:m])
        g = groups[m]
        ctx_a, ctx_b = a.begin(g.index, g.status), b.begin(g.index, g.status)
        cases = _invalid_proposals(stream, g.index)
        name, bad = cases[int(rng.integers(len(cases)))]
        split = int(rng.integers(len(g.proposals) + 1))
        for purpose, camera, refs, tc in g.proposals[:split]:
            a.propose(ctx_a, g.index, purpose, camera, refs, tc)
            b.propose(ctx_b, g.index, purpose, camera, refs, tc)
        a.events.extend(a.scheduler.drain_diagnostics())  # this group's delivery, if any
        before = _probe(a.scheduler, stream.frames[g.index].t_ns)
        with pytest.raises(c.ContractError, match=rf"\.{name}"):
            ctx_a.propose(**bad)
        assert _probe(a.scheduler, stream.frames[g.index].t_ns) == before
        assert a.scheduler.drain_diagnostics() == []
        for purpose, camera, refs, tc in g.proposals[split:]:
            a.propose(ctx_a, g.index, purpose, camera, refs, tc)
            b.propose(ctx_b, g.index, purpose, camera, refs, tc)
        a.end(g.index, g.status)
        b.end(g.index, g.status)
        a.drive(groups[m + 1:]).finish()
        b.drive(groups[m + 1:]).finish()
        assert s4.strip_runtime(a.events) == s4.strip_runtime(b.events)
        assert a.scheduler.counts() == b.scheduler.counts()
        assert a.worker.job_ids == b.worker.job_ids


def test_property_7_earlier_exposure_rejected():
    # Feature: stage4-scheduler, Property 7: Invalid inputs are rejected atomically
    # Group 1's `fr` image is exposed before group 0's proposed `r` image.
    stream = Stream([frame(0, s4.T0), f.make_frame_set(group_id=1, images=(
        f.make_image_message(camera_id=FR, exposure_ns=s4.T0 - 100_000),
        f.make_image_message(camera_id=R, exposure_ns=s4.T0 + 400_000)))])
    run = Run(stream)
    ctx = run.begin(0)
    run.propose(ctx, 0, P.RECOVERY, R)
    run.end(0)
    ctx = run.begin(1)
    before = _probe(run.scheduler, stream.frames[1].t_ns)
    with pytest.raises(c.ContractError, match=r"\.observation.*earlier"):
        run.propose(ctx, 1, P.WARM_RETRIEVAL, FR)
    assert _probe(run.scheduler, stream.frames[1].t_ns) == before
    run.propose(ctx, 1, P.WARM_RETRIEVAL, R)  # the later image is still accepted
    run.end(1)
    assert [e["kind"] for e in run.events] == ["created", "dispatched", "created"]


# ---------------------------------------------------------------------------------------
# Property 8: result checking
# ---------------------------------------------------------------------------------------

MIX = ("ok", "failed", "cancelled", "job_id", "purpose", "camera_id", "source_group_id",
       "source_exposure_ns", "map_id", "preprocessing_id", "out_of_reference")


def mixed_result(request: c.AnchorRequest, variant: str) -> c.AnchorResult:
    fields = {"job_id": request.job_id, "purpose": request.purpose,
              "camera_id": request.camera_id, "source_group_id": request.source_group_id,
              "source_exposure_ns": request.source_exposure_ns, "map_id": request.map_id,
              "preprocessing_id": request.preprocessing_id,
              "matched_reference_ids": request.reference_ids[:1],
              "remaining_candidate_ids": (), "worker_duration_ns": None}
    if variant == "failed":
        fields["status"] = c.AnchorStatus.FAILED
    elif variant == "cancelled":
        fields.update(status=c.AnchorStatus.CANCELLED, reason="worker_cancelled")
    elif variant == "out_of_reference":
        fields["matched_reference_ids"] = (max(request.reference_ids) + 1,)
    elif variant != "ok":
        current = fields[variant]
        fields[variant] = {
            "job_id": lambda v: v + 1_000_000,
            "purpose": lambda v: PURPOSES[(PURPOSES.index(v) + 1) % len(PURPOSES)],
            "camera_id": lambda v: R if v == FR else FR,
            "source_group_id": lambda v: v + 1,
            "source_exposure_ns": lambda v: v + 1,
            "map_id": lambda v: "other_map_v1",
            "preprocessing_id": lambda v: "other_preprocessing_v1",
        }[variant](current)
    return f.make_anchor_result(**fields)


class MixWorker:
    """AnchorWorker returning a seeded per-job mix of good and withheld results."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.requests: list[c.AnchorRequest] = []
        self.variant: dict[int, str] = {}

    def run(self, request: c.AnchorRequest) -> c.AnchorResult:
        self.requests.append(request)
        variant = MIX[int(np.random.default_rng([self.seed, request.job_id]).integers(len(MIX)))]
        if variant == "out_of_reference" and not request.reference_ids:
            variant = "ok"
        self.variant[request.job_id] = variant
        return mixed_result(request, variant)


def test_property_8_result_checking():
    # Feature: stage4-scheduler, Property 8: Result checking
    seen: set = set()
    for i in range(ITERATIONS):
        rng = np.random.default_rng([8, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 30, density=0.5)
        worker = MixWorker(i)
        run = Run(stream, worker).drive(groups).finish()
        requests = {r.job_id: r for r in worker.requests}
        outcomes = run.of(*OUTCOME_KINDS)
        assert len({e["job_id"] for e in outcomes}) == len(outcomes)
        for e in outcomes:
            variant = worker.variant[e["job_id"]]
            seen.add(variant)
            expected = {"ok": "delivered", "failed": "failed"}.get(variant, "rejected")
            assert e["kind"] == expected and e["reason"], (variant, e)
            assert e["group_id"] == e["delivery_group_id"]
            if expected == "failed":
                assert e["reason"] == "insufficient_matches"
        delivered = set(run.ids("delivered"))
        for gid, ctx in run.contexts.items():
            for result in ctx.results:
                assert result.status is c.AnchorStatus.OK and result.job_id in delivered
                result.check_against(requests[result.job_id])
        counts = run.scheduler.counts()
        assert counts["anchor_rejected"] == len(run.of("rejected"))
        assert counts["anchor_failed"] == len(run.of("failed"))
        assert counts["anchor_delivered"] == len(delivered)
        assert not run.scheduler.faulted
        # Scheduling continues: the same inputs with an always-ok worker schedule alike.
        ok = Run(stream).drive(groups).finish()
        kinds = ("created", "replaced", "expired", "dispatched", "rate_limited", "cancelled")
        assert [queue_key(e) for e in run.of(*kinds)] == [queue_key(e) for e in ok.of(*kinds)]
    assert seen == set(MIX), seen


# ---------------------------------------------------------------------------------------
# Backend faults (Requirements 6.4, 6.5)
# ---------------------------------------------------------------------------------------


class ContractErrorWorker(s4.FakeWorker):
    def run(self, request):
        self.requests.append(request)
        raise c.ContractError("fake worker contract violation")


@pytest.mark.parametrize("kind, fault_kind", [
    ("exception", "exception"), ("contract_error", "exception"), ("bad_return", "bad_return"),
    ("backend_fault_status", "backend_fault_status"), ("watchdog", "watchdog"),
])
def test_backend_fault(kind, fault_kind):
    clock = s4.FakeClock()
    if kind == "contract_error":
        worker = ContractErrorWorker()
    else:
        worker = s4.FakeWorker(clock=clock, fault=kind, fault_jobs={0})
    stream = exact_stream(30, 12)
    run = Run(stream, worker, clock=clock)
    for k in range(3):  # job 0 dispatched; job 1 pending, then replaced by job 2
        ctx = run.begin(k)
        run.propose(ctx, k, P.RECOVERY, R)
        run.end(k)
    assert run.ids("dispatched") == [0] and run.ids("replaced") == [2]
    for k in range(3, 12):  # group 3 is job 0's Delivery_Group
        ctx = run.begin(k)
        assert ctx.results == () and not ctx.rear_in_flight
        for purpose in PURPOSES:
            run.propose(ctx, k, purpose, track_count=0)
        run.end(k)
        assert run.scheduler.faulted
    run.finish()
    (fault,) = run.of("backend_fault")
    assert (fault["job_id"], fault["fault_kind"], fault["group_id"]) == (0, fault_kind, 3)
    if kind == "contract_error":
        assert "ContractError" in fault["detail"]
    assert [(e["job_id"], e["reason"]) for e in run.of("cancelled")] == [(2, "backend_fault")]
    assert run.ids("dispatched") == [0] and not run.of(*OUTCOME_KINDS)
    assert all(e["group_id"] < 3 for e in run.of("created"))
    assert [r.job_id for r in worker.requests] == [0]
    assert run.scheduler.counts()["anchor_faulted"] == 1
    assert not run.scheduler.recovery_work_active()


# ---------------------------------------------------------------------------------------
# Threaded mode (Requirements 3.1, 4.6, 6.4, 7.2, 7.3, 7.7)
# ---------------------------------------------------------------------------------------


def _stall_run(worker, threaded: bool) -> tuple[Run, list]:
    stream = exact_stream(10, 4)
    run = Run(stream, worker, threaded=threaded)
    try:
        ctx = run.begin(0)
        run.propose(ctx, 0, P.FRONT_ANCHOR, track_count=0)
        run.end(0)
        if threaded:  # release only once the scheduler is already waiting at group 1
            worker.release_through(stream.obs[0][FR].exposure_ns + DELAY, delay_s=0.02)
        results = []
        for k in range(1, 4):
            results.extend(run.begin(k).results)
            run.end(k)
        run.finish()
    finally:
        worker.release_all()
        run.scheduler.close()
    return run, results


def test_threaded_stall_delivers_sync_result():
    sync, sync_results = _stall_run(s4.FakeWorker(), False)
    worker = s4.FakeWorker("event_after")
    threaded, results = _stall_run(worker, True)
    (stall,) = threaded.of("stall")
    assert stall["job_id"] == 0 and stall["group_id"] == 1 and stall["stall_ns"] >= 0
    assert threaded.scheduler.counts()["anchor_stalled"] == 1
    assert [s4.strip_runtime(r.summary()) for r in results] == [
        s4.strip_runtime(r.summary()) for r in sync_results]
    assert len(results) == 1 and results[0].job_id == 0
    assert s4.strip_runtime(threaded.events) == s4.strip_runtime(sync.events)
    assert worker.max_concurrency == 1


def test_threaded_real_watchdog_faults():
    worker = s4.FakeWorker("event_after")
    run = Run(exact_stream(10, 4), worker, threaded=True,
              policy=SchedulerPolicy(watchdog_ns=50_000_000))
    try:
        ctx = run.begin(0)
        run.propose(ctx, 0, P.FRONT_ANCHOR, track_count=0)
        run.end(0)
        ctx = run.begin(1)  # waits at most the 50 ms watchdog for the gated job
        assert ctx.results == () and run.scheduler.faulted
        run.propose(ctx, 1, P.RECOVERY, R)
        run.end(1)
        worker.release_all()
        assert worker.finished_event(0).wait(2.0)
        for k in (2, 3):  # the abandoned job's late result is never read
            ctx = run.begin(k)
            assert ctx.results == ()
            run.propose(ctx, k, P.RECOVERY, R)
            run.end(k)
        run.finish()
    finally:
        worker.release_all()
        run.scheduler.close()
    (fault,) = run.of("backend_fault")
    assert fault["fault_kind"] == "watchdog" and fault["job_id"] == 0
    assert len(run.of("stall")) == 1
    assert run.ids("dispatched") == [0] and not run.of(*OUTCOME_KINDS)
    assert worker.max_concurrency == 1


def _sequence(run: Run) -> list[tuple]:
    return [(e["kind"], e["job_id"], e["group_id"], run.t_of[e["group_id"]],
             e["source_exposure_ns"], e["delivery_not_before_ns"]) for e in run.of(*SEQUENCE_KINDS)]


def test_threaded_matches_sync_sequences():
    for seed in range(4):
        rng = np.random.default_rng([15, seed])
        stream = base_stream(RATES[seed % 4], seed % 3)
        groups = random_groups(rng, 30, drop=0.2, density=0.5)
        sync = Run(stream).drive(groups).finish()
        assert sync.of("dispatched")
        workers = [s4.FakeWorker("random_delay", seed=seed, max_delay_s=0.005)]
        if seed == 0:
            workers.append(s4.FakeWorker())
        for worker in workers:
            threaded = Run(stream, worker, threaded=True)
            try:
                threaded.drive(groups).finish()
            finally:
                worker.release_all()
                threaded.scheduler.close()
            assert _sequence(threaded) == _sequence(sync)
            assert s4.strip_runtime(threaded.events) == s4.strip_runtime(sync.events)
            assert worker.max_concurrency == 1


# ---------------------------------------------------------------------------------------
# Stage 3 gate fix round 2: GroupContext.settled (OK and FAILED results, read-only)
# ---------------------------------------------------------------------------------------


class KeepingMixWorker(MixWorker):
    """MixWorker that also keeps every returned AnchorResult by job id."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.returned: dict[int, c.AnchorResult] = {}

    def run(self, request: c.AnchorRequest) -> c.AnchorResult:
        result = super().run(request)
        self.returned[request.job_id] = result
        return result


def test_settled_holds_ok_and_failed_results_and_events_are_unchanged():
    # Feature: stage3-gate-fix-round2, Property 5: Preservation - Supported, no-index and named-reference paths
    """``settled`` is exactly the delivered and failed results of each group, unchanged and
    in job order; ``results`` is its ok subset; rejected results never appear; the event
    log equals the log of the same inputs with an always-ok worker for queue events.

    **Validates: Requirements 2.3, 3.4, 3.5**
    """
    seen = {"ok": 0, "failed": 0, "withheld": 0}
    for i in range(ITERATIONS):
        rng = np.random.default_rng([31, i])
        stream = base_stream(RATES[i % 4], i % 3)
        groups = random_groups(rng, 30, density=0.5)
        worker = KeepingMixWorker(i)
        run = Run(stream, worker).drive(groups).finish()
        outcome_at = {}
        for e in run.of(*OUTCOME_KINDS):
            outcome_at.setdefault(e["group_id"], []).append(e)
        for gid, ctx in run.contexts.items():
            expected = [e["job_id"] for e in outcome_at.get(gid, [])
                        if e["kind"] in ("delivered", "failed")]
            assert [r.job_id for r in ctx.settled] == sorted(expected), (i, gid)
            assert ctx.results == tuple(r for r in ctx.settled if r.status is c.AnchorStatus.OK)
            for result in ctx.settled:
                assert result is worker.returned[result.job_id]  # the worker's own record
                assert result.status in (c.AnchorStatus.OK, c.AnchorStatus.FAILED)
                seen["ok" if result.status is c.AnchorStatus.OK else "failed"] += 1
            seen["withheld"] += sum(e["kind"] == "rejected" for e in outcome_at.get(gid, []))
            with pytest.raises(AttributeError):
                ctx.settled = ()
        ok = Run(stream).drive(groups).finish()
        kinds = ("created", "replaced", "expired", "dispatched", "rate_limited", "cancelled")
        assert [queue_key(e) for e in run.of(*kinds)] == [queue_key(e) for e in ok.of(*kinds)]
    assert all(seen.values()), seen


def test_faulted_scheduler_settles_nothing():
    stream = exact_stream(10, 6)
    run = Run(stream, s4.FakeWorker(fault="exception", fault_jobs=(0,)))
    ctx = run.begin(0, S.LOST)
    run.propose(ctx, 0, P.RECOVERY)
    run.end(0, S.LOST)
    for k in range(1, 6):
        ctx = run.begin(k, S.LOST)
        assert ctx.settled == () and ctx.results == ()
        run.end(k, S.LOST)
    assert run.scheduler.faulted

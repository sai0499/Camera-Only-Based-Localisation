"""Stage 4 runner integration: optional scheduler, recovery manager, tracker and projector.

Every frontend, worker, tracker and projector here is a test fake built on the synthetic
``stage4_fakes`` trajectory. Passing these tests is synthetic software and timing
verification of the runner wiring only; it establishes no localization accuracy, runtime
or native high-rate support. Files are written only under pytest's ``tmp_path``.
"""

import inspect
import itertools
import json
import math
import time

import numpy as np
import pytest

import contract_factories as f
import stage4_fakes as s4
from camloc import contracts as c
from camloc.filter import CameraFilter
from camloc.recovery import RecoveryManager
from camloc.scheduler import AnchorScheduler
from run import _run_basic, run_stream

FRONT, REAR, WARM = (c.AnchorPurpose.FRONT_ANCHOR, c.AnchorPurpose.REAR_ANCHOR,
                     c.AnchorPurpose.WARM_RETRIEVAL)
BASE_KEYS = {"groups", "frontend_errors", "prediction_only"}
EXTENDED_KEYS = BASE_KEYS | {
    "anchor_dispatched", "anchor_delivered", "anchor_replaced", "anchor_expired",
    "anchor_rejected", "anchor_cancelled", "anchor_stalled", "anchor_faulted",
    "candidate_confirmations", "candidate_discards", "anchor_disabled",
}
JOB_KINDS = ("dispatched", "delivered", "replaced", "expired", "cancelled")
_names = itertools.count()


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def initial_pose():
    return c.InitialPose(timestamp_ns=s4.T0, x=0.0, y=0.0, yaw=0.0)


def pnp(stamp, rng, *, camera="fr", x_offset=0.0):
    x, y, yaw = s4.gt_pose(stamp)
    noise = rng.uniform(-0.05, 0.05, size=2)
    return f.make_pose_evidence(
        camera_id=camera, exposure_ns=stamp,
        T_WB=s4.pose_matrix(x + noise[0] + x_offset, y + noise[1],
                            yaw + math.radians(rng.uniform(-0.1, 0.1)), z=0.3),
        inlier_count=64, track_ids=np.arange(64), landmark_ids=np.arange(64) + 100,
        inlier_fraction=0.8, occupied_cells=8, median_error_px=0.5, p90_error_px=1.0,
    )


def handoff(job_id, image, source_ns, *, x_offset=0.0, dx=0.0, accepted=True):
    """A HandoffEvidence of job ``job_id`` measured on ``image`` (pose at GT + offsets)."""
    if not accepted:
        return f.make_handoff_evidence(
            job_id=job_id, camera_id=image.camera_id, measurement_ns=image.exposure_ns,
            source_exposure_ns=source_ns, support_ns=(), correspondences=None, pose=None,
            accepted=False, reason="too_few_inliers")
    x, y, yaw = s4.gt_pose(image.exposure_ns)
    pose = f.make_pose_evidence(
        camera_id=image.camera_id, exposure_ns=image.exposure_ns, source=c.PoseSource.HANDOFF,
        T_WB=s4.pose_matrix(x + x_offset + dx, y, yaw, z=0.3))
    return f.make_handoff_evidence(
        job_id=job_id, camera_id=image.camera_id, measurement_ns=image.exposure_ns,
        source_exposure_ns=source_ns, support_ns=(source_ns, image.exposure_ns), pose=pose)


def anchor_frontend(*, noise_seed=0, changed_after=None, log=None, on_call=None):
    """Group-indexed frontend: consumes a due AnchorResult as handoff PnP, else plain PnP.

    With a context it proposes a front, rear and warm job every group, so the scheduler's
    trigger and rate rules decide what is created. Results change the committed pose
    (through a job-dependent offset), so delivery timing is visible in the CSV.
    """
    def frontend(frame, prior, *context):
        assert len(context) <= 1 and prior.t_ns == frame.t_ns
        context = context[0] if context else None
        g = frame.group_id
        if on_call is not None:
            on_call(frame)
        seed = noise_seed + (1 if changed_after is not None and g > changed_after else 0)
        rng = np.random.default_rng([seed, g])
        fr = s4.image_of(frame, "fr")
        if log is not None:
            log.append((g, None if context is None else context.rear_in_flight,
                        () if context is None else tuple(r.job_id for r in context.results)))
        if context is not None and context.results:
            result = context.results[0]
            image = s4.image_of(frame, result.camera_id)
            pose = handoff(result.job_id, image, result.source_exposure_ns,
                           dx=0.01 * (result.job_id % 4 + len(result.matched_reference_ids)))
        else:
            pose = pnp(fr.exposure_ns, rng)
        if context is not None:
            # A low track count every other group triggers an early front refresh.
            context.propose(purpose=FRONT, observation=s4.observation(fr),
                            reference_ids=(g % 3,), track_count=30 if g % 2 == 1 else 50)
            rear = s4.image_of(frame, "r")
            if rear is not None:
                context.propose(purpose=REAR, observation=s4.observation(rear), track_count=50)
            context.propose(purpose=WARM, observation=s4.observation(fr), track_count=50)
        return f.make_group_evidence(frame_set=frame, pose=pose)

    return frontend


def run(tmp_path, frames, frontend, *, basic=False, **components):
    name = f"run{next(_names)}"
    output, diagnostics = tmp_path / f"{name}.csv", tmp_path / f"{name}.jsonl"
    runner = _run_basic if basic else run_stream
    counts = runner(frames, CameraFilter(initial_pose()), frontend, output=output,
                    diagnostics=diagnostics, **components)
    text = diagnostics.read_text(encoding="utf-8")
    return {"csv": output.read_bytes(), "jsonl": diagnostics.read_bytes(), "counts": counts,
            "records": [json.loads(line) for line in text.splitlines()], "output": output}


def scheduler_for(worker, **kwargs):
    return AnchorScheduler(worker, map_id=f.MAP_ID, **kwargs)


def events(records, *kinds):
    return [r for r in records if r["record"] == "SchedulerEvent" and (not kinds or r["kind"] in kinds)]


def job_sequence(records):
    return [(r["kind"], r["job_id"], r.get("replaced_job_id"), r["source_exposure_ns"],
             r["group_id"], r.get("delivery_t_ns")) for r in events(records, *JOB_KINDS)]


def estimates(records):
    return [r for r in records if r["record"] == "Estimate" and r["group_id"] is not None]


# ---------------------------------------------------------------------------------------
# Examples (task 7.3)
# ---------------------------------------------------------------------------------------


def test_frontend_gets_two_arguments_without_scheduler_and_three_with_one(tmp_path):
    frames = s4.synthetic_stream(6, 10)
    calls = []

    def frontend(*args):
        calls.append(len(args))
        return f.make_group_evidence(frame_set=args[0])

    run(tmp_path, frames, frontend, tracker=s4.FakeTracker())
    run(tmp_path, frames, frontend, recovery=RecoveryManager())
    assert calls == [2] * 12
    calls.clear()
    out = run(tmp_path, frames, frontend, scheduler=scheduler_for(s4.FakeWorker()))
    assert calls == [3] * 6 and out["counts"]["groups"] == 6


def test_rear_in_flight_is_true_at_the_rear_jobs_delivery_group(tmp_path):
    frames = s4.synthetic_stream(8, 10)
    log = []

    def frontend(frame, prior, context):
        log.append((frame.group_id, context.rear_in_flight, [r.job_id for r in context.results]))
        if frame.group_id == 0:  # front is not triggered at the InitialPose time
            context.propose(purpose=REAR, observation=s4.observation(s4.image_of(frame, "r")),
                            track_count=50)
        return f.make_group_evidence(frame_set=frame)

    worker = s4.FakeWorker()
    out = run(tmp_path, frames, frontend, scheduler=scheduler_for(worker))
    assert [request.purpose for request in worker.requests] == [REAR]
    # dnb = r exposure + 90 ms, so group 1 (t_ns + 100 ms) is the Delivery_Group.
    assert log[1] == (1, True, [0])
    assert [flag for _, flag, _ in log] == [False, True] + [False] * 6
    assert out["counts"]["anchor_delivered"] == 1


def _settle_frontend(tracker):
    """No scheduler: PnP, an adopted and a rejected handoff, a candidate, then its reset."""
    def frontend(frame, prior):
        g = frame.group_id
        tracker.current_group = g
        tracker.calls.append(("frontend", g, None))
        fr, r = s4.image_of(frame, "fr"), s4.image_of(frame, "r")
        source = fr.exposure_ns - 100_000_000
        rng = np.random.default_rng([0, g])
        if g < 3:
            return f.make_group_evidence(frame_set=frame, pose=pnp(fr.exposure_ns, rng))
        if g == 3:
            return f.make_group_evidence(
                frame_set=frame, pose=handoff(1, fr, source),
                rejected=(handoff(2, fr, source, accepted=False),))
        if g == 4:  # two gated-out handoffs: the first becomes the candidate
            return f.make_group_evidence(frame_set=frame, unused=(
                c.UnusedEvidence(reason="innovation_gate",
                                 evidence=handoff(3, fr, source, x_offset=30.0)),
                c.UnusedEvidence(reason="innovation_gate",
                                 evidence=handoff(4, r, source, x_offset=30.0)),
            ))
        if g == 5:  # 100 ms later and motion-consistent: confirms the reset
            return f.make_group_evidence(frame_set=frame, unused=(
                c.UnusedEvidence(reason="innovation_gate",
                                 evidence=handoff(5, fr, source, x_offset=30.0)),))
        return f.make_group_evidence(frame_set=frame)

    return frontend


class SharedLogProjector(s4.FakeProjector):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def reset(self, reset):
        super().reset(reset)
        self.shared.append(("projector_reset", reset.group_id, reset))


def test_settle_dispositions_and_reset_calls_follow_the_committed_group(tmp_path):
    # Properties 14 and 15, runner parts: settle mapping and reset-call order.
    frames = s4.synthetic_stream(8, 10)
    tracker = s4.FakeTracker()
    projector = SharedLogProjector(tracker.calls)
    out = run(tmp_path, frames, _settle_frontend(tracker), recovery=RecoveryManager(),
              tracker=tracker, projector=projector)
    kinds = [r["evidence_kinds"] for r in estimates(out["records"])]
    assert kinds[3] == ["handoff_pnp"] and kinds[4] == ["prediction_only"]
    assert kinds[5] == ["recovery_reset"]
    sequence = [(name, g, arg[0].job_id, arg[1]) if name == "settle" else (name, g)
                for name, g, arg in tracker.calls if name != "frontend" or g >= 3]
    D = c.HandoffDisposition
    assert sequence == [
        ("frontend", 3), ("settle", 3, 1, D.ADOPT), ("settle", 3, 2, D.DISCARD),
        ("frontend", 4), ("settle", 4, 3, D.CANDIDATE), ("settle", 4, 4, D.DISCARD),
        ("frontend", 5), ("settle", 5, 5, D.CANDIDATE), ("reset", 5), ("projector_reset", 5),
        ("frontend", 6), ("frontend", 7),
    ]
    resets = [arg for name, _, arg in tracker.calls if name in ("reset", "projector_reset")]
    assert resets[0] is resets[1] and resets[0].group_id == 5
    assert [call[0] for call in projector.calls].count("reset") == 1
    assert [call[1] for call in projector.calls if call[0] == "project"] == list(range(len(frames)))
    # Recovery manager without a scheduler: events are written, counts are extended.
    recovery = [r["kind"] for r in out["records"] if r["record"] == "RecoveryEvent"]
    assert recovery == ["candidate_created", "candidate_confirmed"]
    assert out["counts"]["candidate_confirmations"] == 1
    assert out["counts"]["candidate_discards"] == 0
    assert out["counts"]["anchor_dispatched"] == 0 and out["counts"]["anchor_disabled"] == 0
    assert all(r["status"] != "recovering" for r in estimates(out["records"]))


def test_run_summary_and_returned_key_sets(tmp_path):
    frames = s4.synthetic_stream(12, 10)
    frontend = anchor_frontend()
    with_scheduler = run(tmp_path, frames, frontend,
                         scheduler=scheduler_for(s4.FakeWorker()), tracker=s4.FakeTracker())
    recovery_only = run(tmp_path, frames, frontend, recovery=RecoveryManager())
    tracker_only = run(tmp_path, frames, frontend, tracker=s4.FakeTracker(),
                       projector=s4.FakeProjector())
    for out, keys in ((with_scheduler, EXTENDED_KEYS), (recovery_only, EXTENDED_KEYS),
                      (tracker_only, BASE_KEYS)):
        summary = out["records"][-1]
        assert set(out["counts"]) == keys
        assert summary == {"record": "RunSummary", **out["counts"]}
        assert sum(r["record"] == "RunSummary" for r in out["records"]) == 1
    assert with_scheduler["counts"]["anchor_dispatched"] > 0
    assert with_scheduler["counts"]["anchor_delivered"] > 0
    assert all(value == 0 for key, value in recovery_only["counts"].items()
               if key.startswith(("anchor_", "candidate_")))


def test_signature_and_estimator_protocol_are_unchanged():
    parameters = list(inspect.signature(run_stream).parameters.values())
    P = inspect.Parameter
    original = [(p.name, p.kind, p.default) for p in parameters[:6]]
    assert original == [
        ("source", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("estimator", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("build_evidence", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("output", P.KEYWORD_ONLY, P.empty),
        ("diagnostics", P.KEYWORD_ONLY, P.empty),
        ("episode", P.KEYWORD_ONLY, None),
    ]
    assert [(p.name, p.kind, p.default) for p in parameters[6:]] == [
        (name, P.KEYWORD_ONLY, None) for name in ("scheduler", "recovery", "tracker", "projector")]
    assert [(p.name, p.kind, p.default) for p in inspect.signature(_run_basic).parameters.values()] \
        == original
    assert list(inspect.signature(c.Estimator.step).parameters) == ["self", "evidence"]
    assert list(inspect.signature(c.Estimator.snapshot).parameters) == ["self", "t_ns"]


def test_failed_frontend_proposals_are_discarded(tmp_path):
    frames = s4.synthetic_stream(4, 10)

    def frontend(frame, prior, context):
        context.propose(purpose=REAR, observation=s4.observation(s4.image_of(frame, "r")),
                        track_count=50)
        raise c.InputError("synthetic decode failure")

    worker = s4.FakeWorker()
    out = run(tmp_path, frames, frontend, scheduler=scheduler_for(worker))
    assert worker.requests == [] and out["counts"]["frontend_errors"] == 4
    assert [r["count"] for r in events(out["records"], "proposals_discarded")] == [1] * 4


# ---------------------------------------------------------------------------------------
# Property 16 (task 7.4)
# ---------------------------------------------------------------------------------------


def random_frontend(seed):
    """Seeded two-argument frontend mixing PnP, gated-out, rejected, failed and empty groups."""
    def frontend(*args):
        frontend.arg_counts.append(len(args))  # asserted by the caller (errors are swallowed)
        frame, prior = args[:2]
        rng = np.random.default_rng([seed, frame.group_id])
        stamp = s4.image_of(frame, "fr").exposure_ns
        kind = rng.choice(["pnp", "pnp", "gated", "rejected", "error", "empty"])
        if kind == "pnp":
            return f.make_group_evidence(frame_set=frame, pose=pnp(stamp, rng))
        if kind == "gated":
            return f.make_group_evidence(frame_set=frame, unused=(
                c.UnusedEvidence(reason="innovation_gate", evidence=pnp(stamp, rng, x_offset=50.0)),))
        if kind == "rejected":
            bad = f.make_pose_evidence(exposure_ns=stamp, accepted=False, reason="too_few_inliers")
            return f.make_group_evidence(frame_set=frame, rejected=(bad,))
        if kind == "error":
            raise c.InputError(f"synthetic failure {frame.group_id}")
        return f.make_group_evidence(frame_set=frame)

    frontend.arg_counts = []
    return frontend


def test_default_path_identity(tmp_path):
    # Feature: stage4-scheduler, Property 16: Default-path identity
    for seed in range(4):
        rng = np.random.default_rng([seed, 16])
        rate = int(rng.choice([10, 20, 30, 60]))
        cameras = ("fr", "r") if rng.random() < 0.7 else ("fr",)
        frames = s4.synthetic_stream(int(rng.integers(5, 25)), rate, cameras=cameras,
                                     jitter_ns=int(rng.integers(0, 1_000_001)), seed=seed)
        frontend = random_frontend(seed)
        basic = run(tmp_path, frames, frontend, basic=True)
        default = run(tmp_path, frames, frontend)
        tracker = s4.FakeTracker()
        tracker_only = run(tmp_path, frames, frontend, tracker=tracker)
        projector = s4.FakeProjector()
        projector_only = run(tmp_path, frames, frontend, projector=projector)
        for out in (default, tracker_only):
            assert out["csv"] == basic["csv"] and out["jsonl"] == basic["jsonl"]
            assert out["counts"] == basic["counts"] and set(out["counts"]) == BASE_KEYS
        assert projector_only["csv"] == basic["csv"] and projector_only["counts"] == basic["counts"]
        assert [r for r in projector_only["records"] if r["record"] != "ProjectionError"] == basic["records"]
        assert sum(r["record"] == "ProjectionError" for r in projector_only["records"]) == len(frames)
        assert tracker.calls == []
        assert all(call[0] == "project" for call in projector.calls)
        assert frontend.arg_counts == [2] * (4 * len(frames))


# ---------------------------------------------------------------------------------------
# Property 9 (task 7.5)
# ---------------------------------------------------------------------------------------


def test_backend_fault_disables_anchors_while_output_continues(tmp_path):
    # Feature: stage4-scheduler, Property 9: Backend fault disables anchors while output continues
    frames = s4.synthetic_stream(45, 30)
    probe = run(tmp_path, frames, anchor_frontend(), scheduler=scheduler_for(s4.FakeWorker()))
    delivered_jobs = [r["job_id"] for r in events(probe["records"], "delivered")]
    assert len(delivered_jobs) >= 4
    for index, kind in enumerate(s4.FAULT_KINDS):
        rng = np.random.default_rng([index, 9])
        fault_job = delivered_jobs[int(rng.integers(0, 4))]  # a job that reaches delivery
        clock = s4.FakeClock()
        worker = s4.FakeWorker(clock=clock, fault=kind, fault_jobs={fault_job})
        log = []
        out = run(tmp_path, frames, anchor_frontend(log=log),
                  scheduler=scheduler_for(worker, clock=clock))
        records = out["records"]
        assert out["csv"].count(b"\n") == len(frames) + 1
        faults = events(records, "backend_fault")
        assert len(faults) == 1 and faults[0]["job_id"] == fault_job
        assert faults[0]["fault_kind"] == kind
        fault_group = faults[0]["group_id"]
        assert not [r for r in events(records, "delivered") if r["job_id"] == fault_job]
        assert all(fault_job not in jobs for _, _, jobs in log)
        assert all(not jobs and flag is False for g, flag, jobs in log if g >= fault_group)
        # No dispatch after the fault; the faulted job is the worker's last request.
        assert worker.job_ids[-1] == fault_job
        assert not [r for r in events(records, "dispatched", "created")
                    if r["group_id"] >= fault_group]
        # Every created job is resolved exactly once; a pending one is cancelled by the fault.
        created = [r["job_id"] for r in events(records, "created")]
        resolved = ([r["job_id"] for r in events(records, "dispatched", "expired", "cancelled")]
                    + [r["replaced_job_id"] for r in events(records, "replaced")]
                    + [r["job_id"] for r in events(records, "rate_limited") if r["job_id"] is not None])
        assert sorted(created) == sorted(resolved)
        cancelled = events(records, "cancelled")
        assert all(r["reason"] == "backend_fault" and r["group_id"] == fault_group
                   for r in cancelled)
        disabled = [r["group_id"] for r in events(records, "anchor_disabled")]
        assert disabled == [frame.group_id for frame in frames if frame.group_id >= fault_group]
        counts = out["counts"]
        assert counts["frontend_errors"] == 0
        assert counts["anchor_disabled"] == counts["anchor_faulted"] == 1
        assert records[-1]["anchor_disabled"] == records[-1]["anchor_faulted"] == 1
        assert counts["groups"] == len(frames)


def test_backend_fault_cancels_a_pending_request(tmp_path):
    # Example backing Property 9's pending-cancel clause: at 30 Hz job 0 is still in flight
    # when later proposals become pending, so its fault cancels one.
    frames = s4.synthetic_stream(20, 30)
    worker = s4.FakeWorker(fault="exception", fault_jobs={0})
    out = run(tmp_path, frames, anchor_frontend(), scheduler=scheduler_for(worker))
    cancelled = events(out["records"], "cancelled")
    assert len(cancelled) == 1 and cancelled[0]["reason"] == "backend_fault"
    assert cancelled[0]["group_id"] == events(out["records"], "backend_fault")[0]["group_id"]
    assert worker.job_ids == [0]


# ---------------------------------------------------------------------------------------
# Property 10 (task 7.6)
# ---------------------------------------------------------------------------------------


def wait_idle(worker, scheduler, timeout_s=1.0):
    """Block until every dispatched job has returned (event-gated finish before delivery)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with worker._lock:
            idle = worker.active == 0 and len(worker.requests) >= scheduler.counts()["anchor_dispatched"]
        if idle:
            return
        time.sleep(0.0005)
    raise AssertionError("worker did not finish in time")


def variant_run(tmp_path, frames, variant, seed=0):
    clock = None
    threaded = variant not in ("sync", "repeat", "sim_300ms", "sim_1500ms")
    if variant in ("sim_300ms", "sim_1500ms"):
        clock = s4.FakeClock()
        duration = 300_000_000 if variant == "sim_300ms" else 1_500_000_000
        worker = s4.FakeWorker(clock=clock, duration_ns=duration)
    elif variant == "random_delay":
        worker = s4.FakeWorker("random_delay", seed=seed)
    elif variant in ("event_before", "event_after"):
        worker = s4.FakeWorker(variant)
    else:
        worker = s4.FakeWorker()
    kwargs = {"threaded": threaded} | ({} if clock is None else {"clock": clock})
    scheduler = scheduler_for(worker, **kwargs)

    def source():
        for frame in frames:
            if variant == "event_before":
                wait_idle(worker, scheduler)
            elif variant == "event_after":
                worker.release_through(frame.t_ns, delay_s=0.02)
            yield frame

    try:
        out = run(tmp_path, source(), anchor_frontend(), scheduler=scheduler)
    finally:
        worker.release_all()
    assert worker.max_concurrency <= 1
    out["worker_jobs"] = worker.job_ids
    return out


def test_determinism_independent_of_worker_execution(tmp_path):
    # Feature: stage4-scheduler, Property 10: Determinism independent of worker execution
    long_frames = s4.synthetic_stream(101, 10)  # 10 s of 10 Hz source time
    short_frames = s4.synthetic_stream(45, 30, jitter_ns=1_000_000, seed=3)
    # End the 30 Hz stream at a dispatch group, so a job is in flight at end of input.
    probe = variant_run(tmp_path, short_frames, "sync")["records"]
    last_dispatch = [r["group_id"] for r in events(probe, "dispatched")][-1]
    short_frames = short_frames[: last_dispatch + 1]
    # At 10 Hz every job is delivered one group later; the jittered 30 Hz stream also has
    # pending requests that are replaced and an in-flight job cancelled at end of input.
    plans = [
        (long_frames, ["repeat", "threaded", "event_before", "event_after", "sim_300ms",
                       "sim_1500ms", ("random_delay", 0)],
         {"dispatched", "delivered"}),
        (short_frames, ["repeat", "threaded", "event_before", "event_after"]
         + [("random_delay", s) for s in range(3)],
         {"dispatched", "delivered", "replaced", "cancelled"}),
    ]
    for frames, variants, required in plans:
        base = variant_run(tmp_path, frames, "sync")
        assert base["counts"]["anchor_delivered"] > 0 and base["counts"]["anchor_faulted"] == 0
        expected = s4.strip_runtime(base["records"])
        for entry in variants:
            variant, seed = entry if isinstance(entry, tuple) else (entry, 0)
            out = variant_run(tmp_path, frames, variant, seed)
            assert out["csv"] == base["csv"], (variant, seed)
            assert s4.strip_runtime(out["records"]) == expected, (variant, seed)
            assert job_sequence(out["records"]) == job_sequence(base["records"])
            # A job dispatched at the last group may be cancelled before a thread starts it.
            eof = {r["job_id"] for r in events(base["records"], "cancelled")}
            assert ([j for j in out["worker_jobs"] if j not in eof]
                    == [j for j in base["worker_jobs"] if j not in eof])
            assert s4.strip_runtime(out["counts"]) == s4.strip_runtime(base["counts"])
        kinds = {kind for kind, *_ in job_sequence(base["records"])}
        assert required <= kinds, kinds
        assert base["counts"]["frontend_errors"] == 0


# ---------------------------------------------------------------------------------------
# Property 11 (task 7.7)
# ---------------------------------------------------------------------------------------


def test_causal_prefix_and_append_only_output(tmp_path):
    # Feature: stage4-scheduler, Property 11: Causal prefix and append-only output
    frames = s4.synthetic_stream(30, 30)
    changed_images = s4.synthetic_stream(30, 30, payload=b"\xff\xd8changed_future\xff\xd9")
    holder = {}
    snapshots = []

    def check_flushed(frame):
        # Group g's frontend runs after group g - 1's Group_Boundary. The runner swallows
        # frontend exceptions, so the flushed file is recorded here and checked afterwards.
        if frame.group_id:
            snapshots.append((frame.group_id, holder["output"].read_bytes().splitlines()))

    name = f"prefix{next(_names)}"
    holder["output"] = tmp_path / f"{name}.csv"
    counts = run_stream(frames, CameraFilter(initial_pose()), anchor_frontend(on_call=check_flushed),
                        output=holder["output"], diagnostics=tmp_path / f"{name}.jsonl",
                        scheduler=scheduler_for(s4.FakeWorker()))
    base = holder["output"].read_bytes().splitlines()
    base_records = [json.loads(line) for line in
                    (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    delivered = events(base_records, "delivered")
    assert counts["frontend_errors"] == 0 and counts["anchor_delivered"] > 0
    assert [g for g, _ in snapshots] == list(range(1, len(frames)))
    for g, lines in snapshots:  # boundary k = g - 1 has exactly k + 1 rows
        assert lines[0] == b"timestamp_ns,x,y,yaw" and len(lines) == g + 1
        assert lines == base[: g + 1]  # earlier rows never change
    assert len(base) == len(frames) + 1
    stamps = [int(line.split(b",")[0]) for line in base[1:]]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))
    assert stamps == [frame.t_ns for frame in frames]

    for k in (0, len(frames) // 2, len(frames) - 2):
        def late_failed(request, k=k):
            return "failed" if request.source_group_id > k else "ok"

        variants = [
            (frames[: k + 1] + changed_images[k + 1:], anchor_frontend(), s4.FakeWorker()),
            (frames, anchor_frontend(changed_after=k), s4.FakeWorker()),
            (frames, anchor_frontend(), s4.FakeWorker(variant=late_failed)),
            (frames[: k + 1], anchor_frontend(), s4.FakeWorker()),
        ]
        for index, (stream, frontend, worker) in enumerate(variants):
            out = run(tmp_path, stream, frontend, scheduler=scheduler_for(worker))
            lines = out["csv"].splitlines()
            assert lines[: k + 2] == base[: k + 2], (k, index)
            assert len(lines) == len(stream) + 1
            visible = (k < len(frames) - 5 if index == 1 else
                       index == 2 and any(r["source_group_id"] > k for r in delivered))
            if visible:
                assert lines != base, (k, index)  # the later change does reach the output


@pytest.mark.parametrize("bad", ["before_initial", "changed_calibration"])
def test_stage4_path_repeats_input_checks(tmp_path, bad):
    frames = s4.synthetic_stream(2, 10)
    if bad == "before_initial":
        frames = [f.make_frame_set(group_id=0, selected_camera_ids=("fr",), images=(
            f.make_image_message(exposure_ns=s4.T0 - 1),))]
    else:
        frames[1] = f.make_frame_set(group_id=1, images=tuple(
            f.make_image_message(camera_id=i.camera_id, exposure_ns=i.exposure_ns,
                                 calibration_id="other") for i in frames[1].images))
    worker = s4.FakeWorker()
    scheduler = scheduler_for(worker)
    with pytest.raises(c.ContractError):
        run(tmp_path, frames, anchor_frontend(), scheduler=scheduler)
    assert scheduler._closed  # closed in the runner's finally block

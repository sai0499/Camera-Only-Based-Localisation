"""Regression cases for the resolved Stage 4 lifecycle and evaluator decisions."""
import dataclasses
import json

import numpy as np
import pytest

import contract_factories as f
import stage4_fakes as s4
from camloc import contracts as c
from camloc.filter import CameraFilter
from camloc.recovery import RecoveryManager
from run import run_stream
from test_runner_stage4 import _settle_frontend, handoff, run, scheduler_for
from test_scheduler import Run, exact_stream
from test_submission import _reference_eval


class StatefulTracker(s4.FakeTracker):
    def __init__(self):
        super().__init__()
        self.candidate = None
        self.active = None

    def settle(self, evidence, disposition):
        super().settle(evidence, disposition)
        if disposition is c.HandoffDisposition.CANDIDATE:
            self.candidate = evidence

    def drop_candidate(self, reason):
        super().drop_candidate(reason)
        self.candidate = None

    def reset(self, reset):
        assert self.candidate is not None and self.candidate.pose is reset.confirmation
        self.active, self.candidate = self.candidate, None
        super().reset(reset)


def test_confirming_handoff_tracks_are_promoted_instead_of_old_candidate(tmp_path):
    tracker = StatefulTracker()
    run(tmp_path, s4.synthetic_stream(8, 10), _settle_frontend(tracker),
        recovery=RecoveryManager(), tracker=tracker)
    assert tracker.active.job_id == 5
    assert tracker.candidate is None


@pytest.mark.parametrize("mode", ["timeout", "tracking_reacquired", "motion_inconsistent"])
def test_old_candidate_is_dropped_before_replacement_settle(tmp_path, mode):
    tracker = StatefulTracker()
    frames = s4.synthetic_stream(9, 10)

    def frontend(frame, prior):
        tracker.current_group = frame.group_id
        image = s4.image_of(frame, "fr")
        if frame.group_id == 0 or (frame.group_id == 1 and mode == "motion_inconsistent"):
            evidence = handoff(frame.group_id, image, image.exposure_ns - 100_000_000,
                               x_offset=30 if frame.group_id == 0 else 300)
            return f.make_group_evidence(frame_set=frame, unused=(
                c.UnusedEvidence(reason="innovation_gate", evidence=evidence),))
        if frame.group_id == 1 and mode == "tracking_reacquired":
            return f.make_group_evidence(frame_set=frame,
                                        pose=f.make_pose_evidence(exposure_ns=image.exposure_ns))
        return f.make_group_evidence(frame_set=frame)

    run(tmp_path, frames, frontend, recovery=RecoveryManager(), tracker=tracker)
    drops = [(g, reason) for name, g, reason in tracker.calls if name == "drop_candidate"]
    assert drops[0][1] == mode
    if mode == "motion_inconsistent":
        names = [name for name, g, _ in tracker.calls if g == 1]
        assert names == ["drop_candidate", "settle"]
    assert tracker.candidate is None


def test_old_tracker_protocol_is_rejected_before_output_creation(tmp_path):
    tracker = s4.FakeTracker()
    tracker.drop_candidate = None
    output = tmp_path / "output.csv"
    initial = c.InitialPose(timestamp_ns=s4.T0, x=0., y=0., yaw=0.)
    with pytest.raises(c.ContractError, match="drop_candidate"):
        run_stream([], CameraFilter(initial), lambda *args: None, tracker=tracker,
                   output=output, diagnostics=tmp_path / "diagnostics.jsonl")
    assert not output.exists()


def test_selected_out_of_gate_pose_cannot_falsely_discard_recovery(tmp_path):
    frames = s4.synthetic_stream(2, 10)

    def frontend(frame, prior):
        image = s4.image_of(frame, "fr")
        T_WB = np.eye(4)
        T_WB[0, 3] = 30
        pose = f.make_pose_evidence(exposure_ns=image.exposure_ns, T_WB=T_WB)
        if frame.group_id == 0:
            return f.make_group_evidence(frame_set=frame, unused=(
                c.UnusedEvidence(reason="innovation_gate", evidence=pose),))
        return f.make_group_evidence(frame_set=frame, pose=pose)

    result = run(tmp_path, frames, frontend, recovery=RecoveryManager())
    assert result["counts"]["candidate_confirmations"] == 1
    assert result["counts"]["candidate_discards"] == 0
    estimates = [r for r in result["records"] if r["record"] == "Estimate" and r["group_id"] is not None]
    assert estimates[-1]["evidence_kinds"] == ["recovery_reset"]


def test_basic_source_failure_is_recorded_with_written_prefix(tmp_path):
    frame = s4.synthetic_stream(1, 10)[0]

    def source():
        yield frame
        raise c.InputError("injected source failure")

    output, diagnostics = tmp_path / "output.csv", tmp_path / "diagnostics.jsonl"
    with pytest.raises(c.InputError, match="injected"):
        run_stream(source(), CameraFilter(c.InitialPose(timestamp_ns=s4.T0, x=0., y=0., yaw=0.)),
                   lambda frame, prior: f.make_group_evidence(frame_set=frame),
                   output=output, diagnostics=diagnostics)
    records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert records[-1]["record"] == "RunAborted"
    assert records[-1]["last_committed_group_id"] == 0
    assert len(output.read_text().splitlines()) == 2


def test_first_gt_boundary_exclusion_never_changes_camera_output(tmp_path):
    reference_eval = _reference_eval()
    frames = s4.synthetic_stream(3, 10, cameras=("fr",))
    initial = c.InitialPose(timestamp_ns=s4.T0, x=0., y=0., yaw=0.)
    output = tmp_path / "submission.csv"
    run_stream(frames, CameraFilter(initial), lambda frame, prior: f.make_group_evidence(frame_set=frame),
               output=output, diagnostics=tmp_path / "diagnostics.jsonl")
    before = output.read_bytes()
    submission = reference_eval.read_csv(output)
    stamps = np.array([frame.t_ns for frame in frames], dtype=np.int64)
    gt = {"timestamp_ns": np.arange(s4.T0, s4.T0 + 200_000_001, 20_000_000, dtype=np.int64)}
    validation = reference_eval.validate_submission(submission, gt, stamps)
    assert validation["submitted"] == 3 and validation["evaluated"] == 2
    np.testing.assert_array_equal(submission["timestamp_ns"], stamps)
    assert output.read_bytes() == before


def test_capped_front_proposals_do_not_starve_eligible_rear_job():
    replay = Run(exact_stream(10, 4))
    try:
        for k in range(4):
            context = replay.begin(k)
            replay.propose(context, k, c.AnchorPurpose.FRONT_ANCHOR, track_count=0)
            replay.propose(context, k, c.AnchorPurpose.REAR_ANCHOR)
            replay.end(k)
        replay.finish()
        assert [r.purpose for r in replay.worker.requests] == [
            c.AnchorPurpose.FRONT_ANCHOR, c.AnchorPurpose.FRONT_ANCHOR,
            c.AnchorPurpose.REAR_ANCHOR]
    finally:
        replay.scheduler.close()


def test_proposal_source_identity_must_match_its_compressed_image():
    replay = Run(exact_stream(10, 2))
    try:
        context = replay.begin(0)
        wrong = dataclasses.replace(replay.stream.obs[0]["fr"], source_id="another-image")
        with pytest.raises(c.ContractError, match="source_id"):
            context.propose(purpose=c.AnchorPurpose.FRONT_ANCHOR,
                            observation=wrong, track_count=0)
        assert replay.scheduler.counts()["anchor_created"] == 0
    finally:
        replay.scheduler.close()


@pytest.mark.parametrize("failure", ["filter", "source", "tracker"])
def test_fatal_paths_preserve_diagnostics_and_cancel_pending_work(tmp_path, failure):
    frames = s4.synthetic_stream(4, 60)
    estimator = CameraFilter(c.InitialPose(timestamp_ns=s4.T0, x=0., y=0., yaw=0.))
    scheduler = scheduler_for(s4.FakeWorker())
    tracker = s4.FakeTracker()
    original_step = estimator.step

    def step(evidence, **kwargs):
        if evidence.group_id == 1 and failure == "filter":
            raise c.ContractError("injected filter failure")
        return original_step(evidence, **kwargs)

    estimator.step = step

    def source():
        yield frames[0]
        if failure == "source":
            raise c.InputError("injected source failure")
        yield from frames[1:]

    def frontend(frame, prior, context):
        if frame.group_id == 0:
            context.propose(purpose=c.AnchorPurpose.FRONT_ANCHOR,
                            observation=s4.observation(frame.images[0]), track_count=0)
        if frame.group_id == 1 and failure == "tracker":
            return f.make_group_evidence(frame_set=frame, pose=handoff(
                10, frame.images[0], frame.images[0].exposure_ns - 100_000_000))
        return f.make_group_evidence(frame_set=frame)

    def fail_settle(*args):
        raise c.ContractError("injected tracker failure")

    if failure == "tracker":
        tracker.settle = fail_settle
    output, diagnostics = tmp_path / "output.csv", tmp_path / "diagnostics.jsonl"
    with pytest.raises((c.ContractError, c.InputError), match="injected"):
        run_stream(source(), estimator, frontend, scheduler=scheduler, tracker=tracker,
                   output=output, diagnostics=diagnostics)
    records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert records[-1]["record"] == "RunAborted"
    assert not any(r["record"] == "RunSummary" for r in records)
    assert any(r.get("kind") == "cancelled" and r["reason"] == "run_aborted" for r in records)
    assert len(output.read_text().splitlines()) == (3 if failure == "tracker" else 2)
    assert scheduler.counts()["anchor_cancelled"] == 1

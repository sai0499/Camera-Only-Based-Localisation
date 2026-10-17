"""Stage 2 streaming core plus optional Stage 4 wiring. Raw-MCAP frontend wiring waits for A/B.

``build_evidence(frame_set, prior)`` is C's main-thread frontend assembly function. It
calls A/B through their contracts and returns the selected GroupEvidence for that exact
FrameSet. When an ``AnchorScheduler`` is supplied it is called as
``build_evidence(frame_set, prior, context)`` with that group's ``GroupContext``. Test fakes
live only under tests; this module supplies no invented visual poses.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from camloc import contracts as c
from camloc.filter import CameraFilter
from camloc.recovery import RecoveryManager
from camloc.scheduler import AnchorScheduler
from camloc.submission import SubmissionWriter

# Extra RunSummary / returned-count keys when a scheduler or recovery manager is supplied.
SCHEDULER_SUMMARY_KEYS = (
    "anchor_dispatched",
    "anchor_delivered",
    "anchor_replaced",
    "anchor_expired",
    "anchor_rejected",
    "anchor_cancelled",
    "anchor_stalled",
    "anchor_faulted",
)
RECOVERY_SUMMARY_KEYS = ("candidate_confirmations", "candidate_discards")


def run_stream(
    source: Iterable[c.FrameSet],
    estimator: CameraFilter,
    build_evidence: Callable[..., c.GroupEvidence],
    *,
    output: str | Path,
    diagnostics: str | Path,
    episode: c.EpisodeBounds | None = None,
    scheduler: AnchorScheduler | None = None,
    recovery: RecoveryManager | None = None,
    tracker: c.MappedTracker | None = None,
    projector: c.TrackProjector | None = None,
) -> dict[str, int]:
    """Write one row per supplied group, including frontend failures and weak evidence.

    With none of ``scheduler``, ``recovery``, ``tracker`` and ``projector`` supplied this is
    the unchanged Stage 2 loop (``_run_basic``). Otherwise ``_run_stage4`` runs the same
    input checks and records, adds the scheduler's GroupContext to the frontend call,
    passes the frontend's evidence through the recovery manager, settles handoffs, resets
    the tracker/projector after a RecoveryReset and writes the scheduler and recovery
    diagnostics. Worker faults never raise; filter and recovery failures are recorded and
    re-raised.
    """
    if scheduler is None and recovery is None and tracker is None and projector is None:
        return _run_basic(source, estimator, build_evidence, output=output,
                          diagnostics=diagnostics, episode=episode)
    return _run_stage4(source, estimator, build_evidence, output=output, diagnostics=diagnostics,
                       episode=episode, scheduler=scheduler, recovery=recovery, tracker=tracker,
                       projector=projector)


def _run_basic(
    source: Iterable[c.FrameSet],
    estimator: CameraFilter,
    build_evidence: Callable[[c.FrameSet, c.SearchSnapshot], c.GroupEvidence],
    *,
    output: str | Path,
    diagnostics: str | Path,
    episode: c.EpisodeBounds | None = None,
) -> dict[str, int]:
    """Write one row per supplied group, including frontend failures and weak evidence.

    Construct the estimator once from the single allowed InitialPose. Paths are created
    exclusively; a frontend exception is recorded and that group is prediction-only.
    Structural input/filter failures stop the run explicitly. No group buffering, worker
    scheduling, reference access or alternate initial pose is hidden in this function.
    """
    if not isinstance(estimator, CameraFilter) or estimator.estimate.group_id is not None:
        raise c.ContractError("run_stream requires a freshly initialized CameraFilter")
    if episode is not None:
        episode.check_initial_pose(estimator.initial_pose)
    output, diagnostics = Path(output), Path(diagnostics)
    if output.resolve() == diagnostics.resolve():
        raise c.ContractError("submission and diagnostics paths must differ")
    for path in (output, diagnostics):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    counts = {"groups": 0, "frontend_errors": 0, "prediction_only": 0}
    calibration: dict[str, str] = {}
    selected = None
    with SubmissionWriter(output) as writer, open(diagnostics, "x", encoding="utf-8", newline="") as log:
        def record(value):
            log.write(json.dumps(value, allow_nan=False) + "\n")

        record(estimator.estimate.summary())
        for frame_set in source:
            if not isinstance(frame_set, c.FrameSet):
                raise c.ContractError("source must yield FrameSet records")
            if selected is not None and selected != frame_set.selected_camera_ids:
                raise c.ContractError("selected cameras cannot change during a run")
            selected = frame_set.selected_camera_ids
            record(frame_set.summary())
            for image in frame_set.images:
                if image.exposure_ns < estimator.initial_pose.timestamp_ns:
                    raise c.ContractError("image precedes the initial pose")
                if episode is not None and not episode.contains(image.exposure_ns):
                    raise c.ContractError("image is outside the declared episode")
                expected = calibration.setdefault(image.camera_id, image.calibration_id)
                if expected != image.calibration_id:
                    raise c.ContractError("camera calibration changed during a run")
            prior = estimator.snapshot(frame_set.t_ns)
            try:
                evidence = build_evidence(frame_set, prior)
                if not isinstance(evidence, c.GroupEvidence) or evidence.frame_set is not frame_set:
                    raise c.ContractError("frontend must return evidence for this exact FrameSet")
            except Exception as exc:
                counts["frontend_errors"] += 1
                record({"record": "FrontendError", "group_id": frame_set.group_id,
                        "exception": type(exc).__name__, "detail": str(exc)})
                evidence = c.GroupEvidence(frame_set=frame_set, pose=None, motion=None,
                                           border=None, reset=None, rejected=(), unused=())
            record(evidence.summary())
            try:
                estimate = estimator.step(evidence)
            except Exception as exc:
                record({"record": "FilterError", "group_id": frame_set.group_id,
                        "exception": type(exc).__name__, "detail": str(exc)})
                raise
            writer.write(estimate)
            record(estimate.summary())
            counts["groups"] += 1
            counts["prediction_only"] += estimate.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
        if isinstance(source, c.FrameSetSource):
            for event in source.trailing_events():
                record(event.summary())
        record({"record": "RunSummary", **counts})
    return counts


def _check_frame_set(
    frame_set: object,
    selected: tuple[str, ...] | None,
    calibration: dict[str, str],
    *,
    initial_ns: int,
    episode: c.EpisodeBounds | None,
    record: Callable[[dict[str, Any]], None],
) -> tuple[str, ...]:
    """The ``_run_basic`` input checks, in the same order around the FrameSet record.

    Returns the selected camera ids; ``calibration`` is updated in place.
    """
    if not isinstance(frame_set, c.FrameSet):
        raise c.ContractError("source must yield FrameSet records")
    if selected is not None and selected != frame_set.selected_camera_ids:
        raise c.ContractError("selected cameras cannot change during a run")
    record(frame_set.summary())
    for image in frame_set.images:
        if image.exposure_ns < initial_ns:
            raise c.ContractError("image precedes the initial pose")
        if episode is not None and not episode.contains(image.exposure_ns):
            raise c.ContractError("image is outside the declared episode")
        expected = calibration.setdefault(image.camera_id, image.calibration_id)
        if expected != image.calibration_id:
            raise c.ContractError("camera calibration changed during a run")
    return frame_set.selected_camera_ids


def _handoffs(evidence: c.GroupEvidence) -> list[c.HandoffEvidence]:
    """HandoffEvidence records of ``evidence`` in slot order: pose, unused, rejected."""
    found = []
    if isinstance(evidence.pose, c.HandoffEvidence):
        found.append(evidence.pose)
    found.extend(item.evidence for item in evidence.unused
                 if isinstance(item.evidence, c.HandoffEvidence))
    found.extend(item for item in evidence.rejected if isinstance(item, c.HandoffEvidence))
    return found


def _disabled_event(group_id: int) -> dict[str, Any]:
    """The per-group ``anchor_disabled`` SchedulerEvent written while the scheduler is faulted."""
    return {"record": "SchedulerEvent", "kind": "anchor_disabled", "group_id": group_id,
            "job_id": None, "purpose": None, "camera_id": None, "source_group_id": None,
            "source_exposure_ns": None, "delivery_not_before_ns": None,
            "reason": "backend_fault"}


def _run_stage4(
    source: Iterable[c.FrameSet],
    estimator: CameraFilter,
    build_evidence: Callable[..., c.GroupEvidence],
    *,
    output: str | Path,
    diagnostics: str | Path,
    episode: c.EpisodeBounds | None,
    scheduler: AnchorScheduler | None,
    recovery: RecoveryManager | None,
    tracker: c.MappedTracker | None,
    projector: c.TrackProjector | None,
) -> dict[str, int]:
    """The Stage 4 loop; see design.md "Per-group order (Stage 4 path)"."""
    try:
        if not isinstance(estimator, CameraFilter) or estimator.estimate.group_id is not None:
            raise c.ContractError("run_stream requires a freshly initialized CameraFilter")
        if episode is not None:
            episode.check_initial_pose(estimator.initial_pose)
        output, diagnostics = Path(output), Path(diagnostics)
        if output.resolve() == diagnostics.resolve():
            raise c.ContractError("submission and diagnostics paths must differ")
        for path in (output, diagnostics):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        extended = scheduler is not None or recovery is not None
        counts = {"groups": 0, "frontend_errors": 0, "prediction_only": 0}
        calibration: dict[str, str] = {}
        selected = None
        initial_ns = estimator.initial_pose.timestamp_ns
        with (SubmissionWriter(output) as writer,
              open(diagnostics, "x", encoding="utf-8", newline="") as log):
            def record(value):
                log.write(json.dumps(value, allow_nan=False) + "\n")

            record(estimator.estimate.summary())
            if scheduler is not None:
                scheduler.start(estimator.initial_pose)
            for frame_set in source:
                selected = _check_frame_set(frame_set, selected, calibration,
                                            initial_ns=initial_ns, episode=episode, record=record)
                group_id = frame_set.group_id
                # 1-2. Prior, then the scheduler's delivery for this group.
                prior = estimator.snapshot(frame_set.t_ns)
                context = (None if scheduler is None
                           else scheduler.begin_group(frame_set, prior, estimator.estimate.status))
                # 3. Frontend: two arguments without a scheduler, three with one.
                frontend_failed = False
                try:
                    if context is None:
                        evidence = build_evidence(frame_set, prior)
                    else:
                        evidence = build_evidence(frame_set, prior, context)
                    if not isinstance(evidence, c.GroupEvidence) or evidence.frame_set is not frame_set:
                        raise c.ContractError("frontend must return evidence for this exact FrameSet")
                except Exception as exc:
                    frontend_failed = True
                    counts["frontend_errors"] += 1
                    record({"record": "FrontendError", "group_id": group_id,
                            "exception": type(exc).__name__, "detail": str(exc)})
                    evidence = c.GroupEvidence(frame_set=frame_set, pose=None, motion=None,
                                               border=None, reset=None, rejected=(), unused=())
                # 4. Recovery manager: unchanged evidence or a converted reset group.
                outcome = None
                stepped = evidence
                if recovery is not None:
                    try:
                        stepped, outcome = recovery.process(evidence, estimator)
                    except Exception as exc:
                        record({"record": "RecoveryError", "group_id": group_id,
                                "exception": type(exc).__name__, "detail": str(exc)})
                        raise
                # 5. Recovery_Search_Active (needs a working scheduler).
                active = (
                    scheduler is not None
                    and not scheduler.faulted
                    and ((recovery is not None and recovery.search_active())
                         or scheduler.recovery_work_active())
                )
                # 6-8. Step, CSV row (flushed), Estimate record.
                record(stepped.summary())
                try:
                    estimate = estimator.step(stepped, recovery_search_active=active)
                except Exception as exc:
                    record({"record": "FilterError", "group_id": group_id,
                            "exception": type(exc).__name__, "detail": str(exc)})
                    raise
                writer.write(estimate)
                writer.flush()
                record(estimate.summary())
                counts["groups"] += 1
                counts["prediction_only"] += (
                    estimate.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,))
                # 9. Settle every handoff of the frontend's evidence, in slot order.
                if tracker is not None:
                    for handoff in _handoffs(evidence):
                        if (stepped.pose is handoff
                                and estimate.evidence_kinds == (c.EvidenceKind.HANDOFF_PNP,)):
                            disposition = c.HandoffDisposition.ADOPT
                        elif outcome is not None and outcome.candidate_handoff is handoff:
                            disposition = c.HandoffDisposition.CANDIDATE
                        else:
                            disposition = c.HandoffDisposition.DISCARD
                        tracker.settle(handoff, disposition)
                # 10. Reset the tracker and projector after an applied RecoveryReset.
                if estimate.evidence_kinds == (c.EvidenceKind.RECOVERY_RESET,):
                    for target in (tracker, projector):
                        if target is not None:
                            target.reset(stepped.reset)
                if projector is not None:
                    try:
                        projections = projector.project(estimate)
                        if (not isinstance(projections, tuple) or len(projections) != 2
                                or any(not isinstance(p, c.TrackProjection) for p in projections)
                                or tuple(p.route_id for p in projections) != c.ROUTE_IDS
                                or any(p.t_ns != estimate.t_ns for p in projections)
                                or projections[0].reference_map_id != projections[1].reference_map_id):
                            raise c.ContractError("projector must return current track/pit projections")
                        for projection in projections:
                            record(projection.summary())
                    except Exception as exc:
                        record({"record": "ProjectionError", "group_id": group_id,
                                "exception": type(exc).__name__, "detail": str(exc)})
                # 11-12. Group_Boundary, then this group's diagnostics.
                if scheduler is not None:
                    scheduler.end_group(estimate, frontend_failed=frontend_failed)
                    for event in scheduler.drain_diagnostics():
                        record(event)
                if recovery is not None:
                    for event in recovery.drain_diagnostics():
                        record(event)
                if scheduler is not None and scheduler.faulted:
                    record(_disabled_event(group_id))
            if isinstance(source, c.FrameSetSource):
                for event in source.trailing_events():
                    record(event.summary())
            if scheduler is not None:
                scheduler.finish()
                for event in scheduler.drain_diagnostics():
                    record(event)
            if extended:
                anchor = {} if scheduler is None else scheduler.counts()
                rec = {} if recovery is None else recovery.counts()
                counts.update({key: anchor.get(key, 0) for key in SCHEDULER_SUMMARY_KEYS})
                counts.update({key: rec.get(key, 0) for key in RECOVERY_SUMMARY_KEYS})
                counts["anchor_disabled"] = int(scheduler is not None and scheduler.faulted)
            record({"record": "RunSummary", **counts})
        return counts
    finally:
        if scheduler is not None:
            scheduler.close()


if __name__ == "__main__":
    raise SystemExit(
        "run_stream is the tested Stage 2 API. Raw-MCAP CLI wiring requires A/B's "
        "reader and visual frontend at Stage 3; no localization run was started."
    )

"""Causal raw-camera localizer and streaming API over the shared A/B/C components.

``build_evidence(frame_set, prior)`` is C's main-thread frontend assembly function. It
calls A/B through their contracts and returns the selected GroupEvidence for that exact
FrameSet. When an ``AnchorScheduler`` is supplied it is called as
``build_evidence(frame_set, prior, context)`` with that group's ``GroupContext``. Test fakes
live only under tests; this module supplies no invented visual poses.
"""

import dataclasses
import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterable
from contextlib import ExitStack
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


def _cleanup_run(record, estimator, scheduler, recovery):
    def cleanup(exc_type, exc, traceback):
        if scheduler is not None:
            scheduler.close()
            for event in scheduler.drain_diagnostics():
                record(event)
        if recovery is not None:
            for event in recovery.drain_diagnostics():
                record(event)
        if exc is not None:
            record({"record": "RunAborted", "last_committed_group_id": estimator.estimate.group_id,
                    "exception": exc_type.__name__, "detail": str(exc)})
        return False
    return cleanup


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
    with (SubmissionWriter(output) as writer,
          open(diagnostics, "x", encoding="utf-8", newline="") as log,
          ExitStack() as lifecycle):
        def record(value):
            log.write(json.dumps(value, allow_nan=False) + "\n")

        lifecycle.push(_cleanup_run(record, estimator, None, None))
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
        if tracker is not None and not callable(getattr(tracker, "drop_candidate", None)):
            raise c.ContractError("tracker must implement drop_candidate(reason) for recovery cleanup")
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
              open(diagnostics, "x", encoding="utf-8", newline="") as log,
              ExitStack() as lifecycle):
            def record(value):
                log.write(json.dumps(value, allow_nan=False) + "\n")

            lifecycle.push(_cleanup_run(record, estimator, scheduler, recovery))
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
                        if evidence.pose is not None:
                            pose = (evidence.pose.pose if isinstance(evidence.pose, c.HandoffEvidence)
                                    else evidence.pose)
                            if not estimator.pose_in_gate(pose):
                                evidence = dataclasses.replace(evidence, pose=None, unused=(
                                    c.UnusedEvidence(reason="innovation_gate", evidence=evidence.pose),
                                    *evidence.unused))
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
                recovery_events = [] if recovery is None else recovery.drain_diagnostics()
                for event in recovery_events:
                    record(event)
                # 9. Settle every handoff of the frontend's evidence, in slot order.
                if tracker is not None:
                    discarded = next((event for event in recovery_events
                                      if event["kind"] in ("candidate_discarded", "candidate_replaced")), None)
                    if discarded is not None:
                        tracker.drop_candidate(discarded["reason"])
                    for handoff in _handoffs(evidence):
                        if (stepped.pose is handoff
                                and estimate.evidence_kinds == (c.EvidenceKind.HANDOFF_PNP,)):
                            disposition = c.HandoffDisposition.ADOPT
                        elif (stepped.pose is not None
                              and estimate.evidence_kinds in ((c.EvidenceKind.PNP,), (c.EvidenceKind.HANDOFF_PNP,))
                              and any(item.evidence is handoff and item.reason == "compatible_refresh"
                                      for item in stepped.unused)):
                            disposition = c.HandoffDisposition.ADOPT
                        elif outcome is not None and outcome.candidate_handoff is handoff:
                            disposition = c.HandoffDisposition.CANDIDATE
                        elif (stepped.reset is not None and handoff.pose is stepped.reset.confirmation
                              and estimate.evidence_kinds == (c.EvidenceKind.RECOVERY_RESET,)):
                            # reset() promotes this current set, never the older candidate's set.
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
                if scheduler is not None and scheduler.faulted:
                    record(_disabled_event(group_id))
                drain = getattr(build_evidence, "drain_diagnostics", None)
                if drain is not None:
                    for event in drain():
                        record(event)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bag", type=Path, default=c.PROJECT_ROOT /
                        "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--split", choices=("tune", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh result directory")
    args = parser.parse_args(argv)
    initial = c.read_initial_pose(args.initial_pose)
    episode = c.load_split()[args.split]
    episode.check_initial_pose(initial)
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    try:
        from camloc import borders, features, geometry, io, landmark_map, track_coords, vision
        from camloc.frontend import VisualFrontend

        bank = landmark_map.load_landmark_map(args.artifacts)
        manifest = bank.manifest
        fit = c.load_split()["fit"]
        if (manifest.get("split_sha256") != c.BUILD_SPLIT_SHA256
                or manifest.get("supplied_map_sha256") != c.SUPPLIED_MAP_SHA256
                or manifest.get("feature_backend") != features.backend_identity()
                or manifest.get("fitting_interval_ns") != [str(fit.start_ns), str(fit.end_ns)]
                or bool(((bank.keyframe_exposure_ns < fit.start_ns)
                         | (bank.keyframe_exposure_ns >= fit.end_ns)).any())):
            raise c.ContractError("map fitting provenance or feature configuration is incompatible")
        calibrations = io.read_calibrations(args.bag)
        tracker = vision.load_tracker(bank, calibrations)
        if any(reason != "valid" for reason in tracker.calibration_compatibility.values()):
            raise c.ContractError(f"incompatible map/rig: {tracker.calibration_compatibility}")
        worker = vision.load_anchor_worker(bank)
        if tuple(manifest.get("model_ids", ())) != worker.backend.model_ids:
            raise c.ContractError("map checkpoints differ from the installed pinned backend")
        supplied_map = c.PROJECT_ROOT / "23-00-38_cut_120m_update" / "map.mcap"
        road = geometry.load_road_geometry(supplied_map)
        estimator = CameraFilter(initial)
        frontend = VisualFrontend(geometry.make_canonicalizer(calibrations), tracker, estimator,
            road=road, borders=borders.make_border_source(road), motion=vision.make_ground_motion())
        source = io.open_frame_sets(args.bag, initial_pose=initial, episode=episode,
                                   calibrations=calibrations)
        metadata = {"split": args.split, "initial_pose": initial.summary(),
                    "episode": episode.summary(), "map": bank.summary(),
                    "models": list(worker.backend.model_ids),
                    "query_bag": args.bag.name,
                    "calibration_compatibility": tracker.calibration_compatibility,
                    "source_sha256": {str(path.relative_to(c.PROJECT_ROOT)): hashlib.sha256(
                        path.read_bytes()).hexdigest() for path in [
                            c.PROJECT_ROOT / "run.py", *sorted((c.PROJECT_ROOT / "camloc").glob("*.py"))]},
                    "border_corrections": False, "ground_motion_corrections": False,
                    "global_retrieval": False, "startup_seconds": time.perf_counter() - began}
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        replay_began = time.perf_counter()
        counts = run_stream(source, estimator, frontend, output=args.output / "submission.csv",
            diagnostics=args.output / "diagnostics.jsonl", episode=episode,
            scheduler=AnchorScheduler(worker, map_id=bank.map_id, threaded=True),
            recovery=RecoveryManager(), tracker=tracker,
            projector=track_coords.load_track_projector(supplied_map))
        metadata.update(counts)
        metadata["replay_seconds"] = time.perf_counter() - replay_began
        metadata["total_seconds"] = time.perf_counter() - began
        metadata["outputs_per_second"] = counts["groups"] / metadata["replay_seconds"]
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(counts))
        return 0
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"exception": type(exc).__name__,
            "detail": str(exc), "elapsed_seconds": time.perf_counter() - began}) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 2 streaming core. Raw-MCAP frontend wiring waits for A/B's Stage 3 integration.

``build_evidence(frame_set, prior)`` is C's main-thread frontend assembly function. It
calls A/B through their contracts and returns the selected GroupEvidence for that exact
FrameSet. Test fakes live only under tests; this module supplies no invented visual poses.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from camloc import contracts as c
from camloc.filter import CameraFilter
from camloc.submission import SubmissionWriter


def run_stream(
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


if __name__ == "__main__":
    raise SystemExit(
        "run_stream is the tested Stage 2 API. Raw-MCAP CLI wiring requires A/B's "
        "reader and visual frontend at Stage 3; no localization run was started."
    )

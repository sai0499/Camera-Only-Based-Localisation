"""Validate a complete exposure-stamped submission against allowed camera input and logs."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from camloc import contracts as c


def _support(value, group_ns):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("exposure_ns", "measurement_ns", "current_ns", "source_ns",
                       "source_exposure_ns", "max_support_exposure_ns") and item is not None:
                if c.as_ns(item) > group_ns:
                    raise c.ContractError("future image support in diagnostics")
            elif key == "support_ns":
                if any(c.as_ns(stamp) > group_ns for stamp in item):
                    raise c.ContractError("future handoff support in diagnostics")
            else:
                _support(item, group_ns)
    elif isinstance(value, list):
        for item in value:
            _support(item, group_ns)


def _current_image(record, images):
    if record is None:
        return
    kind = record["record"]
    field = {"PoseEvidence": "exposure_ns", "HandoffEvidence": "measurement_ns",
             "MotionEvidence": "current_ns", "BorderEvidence": "exposure_ns"}[kind]
    image = images.get(record["camera_id"])
    if image is None or record[field] != image.exposure_ns:
        raise c.ContractError("evidence is not from this group's own camera exposure")
    if "calibration_id" in record and record["calibration_id"] != image.calibration_id:
        raise c.ContractError("evidence calibration differs from the current image")
    if kind == "HandoffEvidence" and record["pose"] is not None:
        _current_image(record["pose"], images)


def check_metrics(directory: Path, submission: Path | None = None):
    for name, sections in (("original_xy.txt", 1), ("updated_xy.txt", 1),
                           ("reconciled_supplied_map.txt", 2)):
        text = (directory / name).read_text(encoding="utf-8")
        if text.count("<- SCORE") < sections or re.search(r"\b(?:nan|inf)\b", text.lower()):
            raise c.ContractError(f"missing or non-finite metric section in {name}")
    if not (directory / "validation.json").is_file():
        raise c.ContractError("scorer validation.json is missing")
    if submission is not None:
        report = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        with submission.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if report.get("submission_sha256") != digest:
            raise c.ContractError("scorer report belongs to a different submission")


def validate(submission: Path, diagnostics: Path, expected_groups, episode: c.EpisodeBounds):
    groups, estimates, evidence = {}, {}, {}
    complete = None
    with diagnostics.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("record")
            if kind == "RunAborted":
                raise c.ContractError("run aborted")
            target = {"FrameSet": groups, "Estimate": estimates, "GroupEvidence": evidence}.get(kind)
            key = (record["frame_set"]["group_id"] if kind == "GroupEvidence"
                   else record.get("group_id"))
            if target is not None and key is not None:
                if key in target:
                    raise c.ContractError(f"duplicate {kind} diagnostic")
                target[key] = record
            if kind == "RunSummary":
                if complete is not None:
                    raise c.ContractError("duplicate RunSummary")
                complete = record
    if complete is None:
        raise c.ContractError("RunSummary is missing")
    stamps, visual, full_pose_stamps = [], 0, []
    lost = 0
    with submission.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["timestamp_ns", "x", "y", "yaw"]:
            raise c.ContractError("submission header must be timestamp_ns,x,y,yaw")
        for frame in expected_groups:
            row = next(reader, None)
            if row is None or len(row) != 4:
                raise c.ContractError("missing group output or malformed row")
            stamp = c.parse_decimal_ns(row[0], "submission timestamp")
            if stamp != frame.t_ns or (stamps and stamp <= stamps[-1]) or not episode.contains(stamp):
                raise c.ContractError("timestamp differs from the complete camera-group sequence")
            values = [float(value) for value in row[1:]]
            if not all(math.isfinite(value) for value in values) or not -math.pi <= values[2] <= math.pi:
                raise c.ContractError("pose must be finite with yaw in radians [-pi,pi]")
            logged = groups.pop(frame.group_id, None)
            estimate = estimates.pop(frame.group_id, None)
            selected = evidence.pop(frame.group_id, None)
            if logged != frame.summary() or estimate is None or selected is None:
                raise c.ContractError("missing or mismatched group/estimate/evidence diagnostic")
            if estimate["t_ns"] != stamp or selected["frame_set"] != logged:
                raise c.ContractError("diagnostic timestamp/group mismatch")
            if any(abs(a - b) > 1e-9 for a, b in zip(values, estimate["state"][:3], strict=True)):
                raise c.ContractError("CSV values differ from the emitted estimate")
            _support(estimate, stamp)
            _support(selected, stamp)
            images = {image.camera_id: image for image in frame.images}
            for slot in ("pose", "motion", "border"):
                _current_image(selected[slot], images)
            if selected["reset"] is not None:
                _current_image(selected["reset"]["confirmation"], images)
            support = estimate["max_support_exposure_ns"]
            if support is not None and support not in {image.exposure_ns for image in frame.images}:
                raise c.ContractError("estimate support is not a current camera exposure")
            visual += estimate["evidence_kinds"] != ["prediction_only"]
            lost += estimate["status"] == c.Status.LOST
            if any(kind in estimate["evidence_kinds"] for kind in (
                    c.EvidenceKind.PNP, c.EvidenceKind.HANDOFF_PNP, c.EvidenceKind.RECOVERY_RESET)):
                full = (selected["reset"]["confirmation"] if selected["reset"] is not None
                        else selected["pose"])
                if full is not None and full["record"] == "HandoffEvidence":
                    full = full["pose"]
                if full is None or not full["accepted"] or support != full["exposure_ns"]:
                    raise c.ContractError("full-pose update lacks accepted current-image evidence")
                full_pose_stamps.append(support)
            stamps.append(stamp)
        if next(reader, None) is not None:
            raise c.ContractError("extra submission rows")
    if not stamps or groups or estimates or evidence or complete["groups"] != len(stamps):
        raise c.ContractError("incomplete or extra diagnostics")
    gaps = sorted(b - a for a, b in zip(stamps[:-1], stamps[1:]))
    cadence = gaps[len(gaps) // 2] if gaps else c.NS_PER_S
    coverage = len(stamps) * cadence / (episode.end_ns - episode.start_ns)
    if (coverage < 0.9 or stamps[0] - episode.start_ns > 2 * cadence
            or episode.end_ns - stamps[-1] > 2 * cadence):
        raise c.ContractError("incomplete episode coverage or unexpected output rate")
    # Full-pose gaps beyond the filter's 3 s loss boundary cannot certify a continuous baseline.
    anchors = [episode.start_ns, *full_pose_stamps, episode.end_ns]
    max_full_pose_gap = max(b - a for a, b in zip(anchors[:-1], anchors[1:]))
    return {"valid": True, "rows": len(stamps), "visual_updates": visual,
            "full_pose_updates": len(full_pose_stamps), "lost_rows": lost,
            "max_full_pose_gap_ns": max_full_pose_gap,
            "visual_baseline_gate_passed": bool(full_pose_stamps) and max_full_pose_gap <= 3 * c.NS_PER_S,
            "first_timestamp_ns": str(stamps[0]), "last_timestamp_ns": str(stamps[-1]),
            "median_cadence_ns": cadence, "max_gap_ns": max(gaps, default=0),
            "nominal_coverage": min(coverage, 1.0), "exposure_sequence_matches_input": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--split", choices=("tune", "holdout"), required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from camloc.io import open_frame_sets

    episode = c.load_split()[args.split]
    source = open_frame_sets(args.bag, initial_pose=c.read_initial_pose(args.initial_pose), episode=episode)
    result = validate(args.submission, args.diagnostics, source, episode)
    if args.evaluation is not None:
        check_metrics(args.evaluation, args.submission)
        result["metric_sections_complete"] = True
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

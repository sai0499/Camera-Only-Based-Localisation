"""Offline map/camera utilization and visual-gap report; reads no query ground truth."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from camloc import landmark_map as lm, pnp  # noqa: E402


def run_support(folder):
    folder = Path(folder)
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    cameras, full, outputs, lost, errors = defaultdict(Counter), [], [], 0, Counter()
    start = int(meta["initial_pose"]["timestamp_ns"])
    with (folder / "diagnostics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind, camera = record["record"], record.get("camera_id")
            if kind == "CameraSelection":
                for camera in record["decoded"]:
                    cameras[camera]["decoded"] += 1
            elif kind == "SchedulerEvent" and record.get("kind") == "dispatched":
                cameras[camera]["anchor_jobs"] += 1
            elif kind == "GroupEvidence":
                value = record["pose"] or ((record.get("reset") or {}).get("confirmation"))
                if value is not None:
                    cameras[value["camera_id"]]["selected_visual_poses"] += 1
            elif kind == "Estimate" and record.get("group_id") is not None:
                stamp = int(record["t_ns"])
                outputs.append(stamp)
                lost += record["status"] == "lost"
                if any(k in record["evidence_kinds"] for k in ("pnp", "handoff_pnp", "recovery_reset")):
                    full.append(stamp)
            elif kind.endswith("Error") or kind == "RunAborted":
                errors[kind] += 1
    if not outputs or "groups" not in meta:
        raise ValueError("report requires a completed nonempty run")
    end = meta["episode"].get("end_ns") or outputs[-1] + int(np.median(np.diff(outputs)))
    anchors = [start, *full, end]
    gaps = sorted(({"start_s": (a-start)/1e9, "end_s": (b-start)/1e9,
                    "duration_s": (b-a)/1e9} for a,b in zip(anchors,anchors[1:])),
                  key=lambda gap: -gap["duration_s"])
    return {"run": str(folder), "outputs": len(outputs), "full_visual_updates": len(full),
        "lost_outputs": lost, "processing_errors": dict(errors),
        "camera_policy": meta.get("camera_policy", "legacy_front_rear"),
        "camera_utilization": {cam: dict(value) for cam,value in cameras.items()},
        "longest_visual_gap_s": gaps[0]["duration_s"],
        "gaps_over_3s": sum(gap["duration_s"] > 3 for gap in gaps),
        "longest_gaps": gaps[:12]}


def map_support(bank):
    cameras = defaultdict(lambda: {"landmarks": [], "cells": [], "depths": []})
    for frame_id in bank.keyframe_ids:
        frame = bank.keyframe(int(frame_id))
        linked = frame.landmark_ids >= 0
        entry = cameras[frame.camera_id]
        entry["landmarks"].append(int(linked.sum()))
        entry["cells"].append(pnp.occupied_cells(frame.keypoints[linked], bank.mask(frame.camera_id)))
        if linked.any():
            depth = (bank.points(frame.landmark_ids[linked]) - frame.T_WC[:3, 3]) @ frame.T_WC[:3, 2]
            entry["depths"].extend(depth.tolist())
    return {cam: {"references": len(value["landmarks"]),
        "landmarks_per_reference_min_median_max": [min(value["landmarks"]), float(np.median(value["landmarks"])), max(value["landmarks"])],
        "mapped_cells_median": float(np.median(value["cells"])),
        "references_with_30_landmarks_and_6_cells": sum(n >= 30 and cells >= 6
            for n,cells in zip(value["landmarks"],value["cells"],strict=True)),
        "observation_depth_median_m": float(np.median(value["depths"])) if value["depths"] else None}
        for cam,value in cameras.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = lm.load_landmark_map(args.artifacts)
    report = {"scope": "Frozen fitting geometry and logged current-image support; no GT accuracy inference",
              "map": bank.summary(), "camera_map_support": map_support(bank), "run": run_support(args.run)}
    if args.baseline:
        report["baseline"] = run_support(args.baseline)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["run"], indent=2))

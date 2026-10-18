"""B review: does the usable-extent coverage grid admit worse poses than the full-image grid?

Offline measurement (owner: B). For anchor jobs on episode stretches (retrieval-ranked
references, as in recovery) it solves PnP on (a) the worker's source correspondences and
(b) the same points carried one delivery later through ``VisionTracker.handoff``, with the
cell gate disabled, then classifies every solve that passes all other gates by
``occupied_cells`` over the full image (F) and over the usable extent (U):

    both   F >= 6 and U >= 6     usable_only  U >= 6 > F     full_only  F >= 6 > U

and reports count, median / p90 planar error and the wrong-pose rate (> 2 m) per class.
Production keeps the usable grid unless usable-only poses are measurably worse. GT is read
only for scoring.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import build_map  # noqa: E402
from camloc import contracts as c  # noqa: E402
from camloc import geometry as geo  # noqa: E402
from camloc import io as cio  # noqa: E402
from camloc import landmark_map as lm  # noqa: E402
from camloc import pnp, vision  # noqa: E402
from camloc import retrieval as rt  # noqa: E402

WRONG_M = 2.0
NO_CELL_GATE = dataclasses.replace(pnp.DEFAULT_GATES, min_cells=0)


def classify(full: int, usable: int, min_cells: int = pnp.DEFAULT_GATES.min_cells) -> str:
    if full >= min_cells and usable >= min_cells:
        return "both"
    if usable >= min_cells:
        return "usable_only"
    if full >= min_cells:
        return "full_only"
    return "neither"


def table(records: list[dict]) -> dict:
    out = {}
    for name in ("both", "usable_only", "full_only", "neither"):
        errors = np.array([r["error_m"] for r in records if r["class"] == name])
        out[name] = {"count": int(errors.size),
                     "median_m": float(np.median(errors)) if errors.size else None,
                     "p90_m": float(np.percentile(errors, 90)) if errors.size else None,
                     "wrong_rate": float(np.mean(errors > WRONG_M)) if errors.size else None}
    return out


def solve_record(kind, obs, pixels, landmark_ids, bank, truth):
    usable = obs.valid_mask & bank.mask(obs.camera_id)
    solution = pnp.solve_pnp(pixels, bank.points(landmark_ids), obs.K, obs.T_BC, NO_CELL_GATE,
                             valid_mask=usable)
    if solution.reason != "accepted" or truth is None:
        return None  # failed another gate (or no GT): outside the comparison
    inlier_pixels = np.asarray(pixels, np.float64)[solution.inliers]
    full = pnp.occupied_cells(inlier_pixels)
    usable_cells = pnp.occupied_cells(inlier_pixels, usable)
    return {"kind": kind, "exposure_ns": obs.exposure_ns, "cells_full": full,
            "cells_usable": usable_cells, "class": classify(full, usable_cells),
            "inliers": int(solution.inliers.size),
            "error_m": float(np.linalg.norm(solution.T_WB[:2, 3] - truth[:2, 3]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--bag", type=Path, default=ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--episode", choices=("fit", "tune"), default="tune")
    parser.add_argument("--stretch", type=float, nargs=2, action="append", required=True,
                        metavar=("START_S", "END_S"))
    parser.add_argument("--camera", default="fr")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; use a fresh directory")
    episode = c.load_split()[args.episode]
    if args.stride < 1:
        parser.error("--stride must be positive")
    windows = []
    for start_s, end_s in args.stretch:
        if not 0 <= start_s < end_s <= (episode.end_ns - episode.start_ns) / c.NS_PER_S:
            parser.error("window outside the episode")
        start = episode.start_ns + int(start_s * c.NS_PER_S)
        end = episode.start_ns + int(end_s * c.NS_PER_S)
        if not episode.start_ns <= start < end <= episode.end_ns:
            parser.error("window outside the episode")
        windows.append((start_s, end_s, start, end))
    calibrations = cio.read_calibrations(args.bag)
    canonicalizer = geo.make_canonicalizer(calibrations)
    bank = lm.load_landmark_map(args.artifacts)
    index = rt.RetrievalIndex.load(args.artifacts, bank, rt.load_pinned_eigenplaces())
    worker = vision.load_anchor_worker(bank, retrieval=index)
    records, jobs = [], 0
    for start_s, end_s, start, end in windows:
        gt = build_map.read_gt_track(args.bag, start, end)
        images = []
        for image in build_map.iter_images(args.bag, args.camera, start, end,
                                           calibrations[args.camera]):
            try:
                images.append(canonicalizer.canonicalize(image))
            except c.InputError:
                continue
        for i in range(0, len(images), args.stride):
            source = images[i]
            target = next((k for k in range(i + 1, len(images)) if images[k].exposure_ns
                           >= source.exposure_ns + c.DELIVERY_MIN_DELAY_NS), None)
            snapshot = c.SearchSnapshot(
                t_ns=source.exposure_ns, state=np.zeros(6), P=np.eye(6) * 1e4,
                status=c.Status.RECOVERING, branch=c.Branch.UNKNOWN,
                auxiliary=c.AuxiliaryGeometry.unavailable("not_estimated"))
            request = c.AnchorRequest(
                job_id=jobs, purpose=c.AnchorPurpose.RECOVERY, camera_id=source.camera_id,
                source_group_id=jobs, source_exposure_ns=source.exposure_ns,
                delivery_not_before_ns=source.exposure_ns + c.DELIVERY_MIN_DELAY_NS,
                observation=source, snapshot=snapshot, reference_ids=(), map_id=bank.map_id,
                preprocessing_id=source.preprocessing_id)
            jobs += 1
            result = worker.run(request)
            if result.correspondences is None:
                continue
            corr = result.correspondences
            record = solve_record("source", source, corr.pixels, corr.landmark_ids, bank,
                                  gt.pose(source.exposure_ns))
            if record:
                records.append(record)
            if target is None:
                continue
            tracker = vision.VisionTracker(bank, calibrations)
            for k in range(i, target + 1):
                tracker.observe(images[k])
            handoff = tracker.handoff(result, images[target])
            carried = handoff.correspondences
            if carried is not None and carried.landmark_ids.size >= 4:
                record = solve_record("handoff", images[target], carried.pixels,
                                      carried.landmark_ids, bank,
                                      gt.pose(images[target].exposure_ns))
                if record:
                    records.append(record)
            tracker.settle(handoff, c.HandoffDisposition.DISCARD)
        print(f"stretch {start_s}-{end_s}: {jobs} jobs, {len(records)} records", flush=True)
    report = {
        "scope": "B coverage-grid check; offline, GT only for scoring; not a localization result",
        "map": bank.summary(), "retrieval_id": index.retrieval_id, "episode": args.episode,
        "stretches": args.stretch, "jobs": jobs, "wrong_threshold_m": WRONG_M,
        "pooled": table(records),
        "source": table([r for r in records if r["kind"] == "source"]),
        "handoff": table([r for r in records if r["kind"] == "handoff"]),
    }
    args.out.mkdir(parents=True)
    (args.out / "grid.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

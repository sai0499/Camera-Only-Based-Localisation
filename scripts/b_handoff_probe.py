"""B: where do handoff points get lost, and does a different LK carry keep more of them?

Offline measurement (owner: B). On episode stretches, each sampled image is an anchor
source (retrieval-ranked references, as in recovery). Its result is handed off to the first
image >= source + 90 ms through a fresh ``VisionTracker`` without priors (unseeded, like a
recovery handoff), once per LK carry variant:

    base      production: window 21, 4 levels, forward/backward <= 1.0 px
    fb2       forward/backward <= 2.0 px
    win31     window 31, 5 levels
    win31fb2  window 31, 5 levels, forward/backward <= 2.0 px

For each variant it reports source points, carried points, acceptance, rejection reasons and
planar error of accepted poses (wrong > 2 m). For ``base`` it splits lost rows into LK
status failures, forward/backward failures and mask/border losses. GT is read only for
scoring. Variants patch module values inside this process only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import build_map  # noqa: E402
from camloc import contracts as c  # noqa: E402
from camloc import geometry as geo  # noqa: E402
from camloc import io as cio  # noqa: E402
from camloc import landmark_map as lm  # noqa: E402
from camloc import vision  # noqa: E402
from camloc import retrieval as rt  # noqa: E402

WRONG_M = 2.0
VARIANTS = {
    "base": {"window": (21, 21), "levels": 4, "fb": 1.0},
    "fb2": {"window": (21, 21), "levels": 4, "fb": 2.0},
    "win31": {"window": (31, 31), "levels": 5, "fb": 1.0},
    "win31fb2": {"window": (31, 31), "levels": 5, "fb": 2.0},
}


@contextlib.contextmanager
def lk_variant(window, levels, fb):
    """Temporarily change the tracker's LK window/levels and its handoff carry FB bound."""
    original = (vision.LK_WINDOW, vision.LK_LEVELS, vision.lk_track)
    base_lk = original[2]

    def patched(previous, current, points, guess=None, max_fb_error_px=fb):
        return base_lk(previous, current, points, guess, max_fb_error_px)

    vision.LK_WINDOW, vision.LK_LEVELS, vision.lk_track = window, levels, patched
    try:
        yield
    finally:
        vision.LK_WINDOW, vision.LK_LEVELS, vision.lk_track = original


def loss_breakdown(previous_gray, current_gray, usable, pixels) -> dict:
    """Production LK (21 px, 4 levels, FB 1 px): why rows are lost on one transition."""
    pts = np.ascontiguousarray(pixels, np.float32).reshape(-1, 1, 2)
    fwd, st, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, pts, None,
                                          winSize=(21, 21), maxLevel=4,
                                          criteria=vision.LK_CRITERIA)
    bwd, st2, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, fwd, pts.copy(),
                                           winSize=(21, 21), maxLevel=4,
                                           criteria=vision.LK_CRITERIA,
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    fb = np.linalg.norm((bwd - pts).reshape(-1, 2), axis=1)
    status = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
    new = fwd.reshape(-1, 2)
    in_mask = vision.mask_lookup(usable, new)
    return {"status_fail": int((~status).sum()),
            "fb_over_1px": int((status & (fb > 1.0)).sum()),
            "fb_1_to_2px": int((status & (fb > 1.0) & (fb <= 2.0)).sum()),
            "mask_or_border": int((status & (fb <= 1.0) & ~in_mask).sum()),
            "kept": int((status & (fb <= 1.0) & in_mask).sum())}


def summarize(rows: list[dict]) -> dict:
    accepted = [r for r in rows if r["accepted"]]
    errors = np.array([r["error_m"] for r in accepted if r.get("error_m") is not None])
    carried = np.array([r["carried"] for r in rows])
    return {"handoffs": len(rows), "accepted": len(accepted),
            "wrong_over_2m": int((errors > WRONG_M).sum()),
            "error_median_m": float(np.median(errors)) if errors.size else None,
            "error_p90_m": float(np.percentile(errors, 90)) if errors.size else None,
            "carried_median": float(np.median(carried)) if carried.size else None,
            "carried_ge_30": int((carried >= 30).sum()),
            "reasons": dict(Counter(r["reason"] for r in rows).most_common())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--bag", type=Path, default=ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--episode", choices=("fit", "tune"), default="tune")
    parser.add_argument("--stretch", type=float, nargs=2, action="append", required=True,
                        metavar=("START_S", "END_S"))
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; use a fresh directory")
    variants = [v for v in args.variants.split(",") if v]
    episode = c.load_split()[args.episode]
    calibrations = cio.read_calibrations(args.bag)
    canonicalizer = geo.make_canonicalizer(calibrations)
    bank = lm.load_landmark_map(args.artifacts)
    index = rt.RetrievalIndex.load(args.artifacts, bank, rt.load_pinned_eigenplaces())
    worker = vision.load_anchor_worker(bank, retrieval=index)
    rows: dict[str, list[dict]] = {v: [] for v in variants}
    losses: list[dict] = []
    for start_s, end_s in args.stretch:
        start = episode.start_ns + int(start_s * c.NS_PER_S)
        end = episode.start_ns + int(end_s * c.NS_PER_S)
        gt = build_map.read_gt_track(args.bag, start, end)
        images = []
        for image in build_map.iter_images(args.bag, "fr", start, end, calibrations["fr"]):
            try:
                images.append(canonicalizer.canonicalize(image))
            except c.InputError:
                continue
        for i in range(0, len(images), args.stride):
            source = images[i]
            target = next((k for k in range(i + 1, len(images)) if images[k].exposure_ns
                           >= source.exposure_ns + c.DELIVERY_MIN_DELAY_NS), None)
            if target is None:
                continue
            snapshot = c.SearchSnapshot(
                t_ns=source.exposure_ns, state=np.zeros(6), P=np.eye(6) * 1e4,
                status=c.Status.RECOVERING, branch=c.Branch.UNKNOWN,
                auxiliary=c.AuxiliaryGeometry.unavailable("not_estimated"))
            result = worker.run(c.AnchorRequest(
                job_id=i, purpose=c.AnchorPurpose.RECOVERY, camera_id="fr", source_group_id=i,
                source_exposure_ns=source.exposure_ns,
                delivery_not_before_ns=source.exposure_ns + c.DELIVERY_MIN_DELAY_NS,
                observation=source, snapshot=snapshot, reference_ids=(), map_id=bank.map_id,
                preprocessing_id=source.preprocessing_id))
            if result.correspondences is None:
                continue
            src_points = int(result.correspondences.landmark_ids.size)
            truth = gt.pose(images[target].exposure_ns)
            if target == i + 1:
                gray0 = cv2.cvtColor(np.asarray(source.image), cv2.COLOR_RGB2GRAY)
                gray1 = cv2.cvtColor(np.asarray(images[target].image), cv2.COLOR_RGB2GRAY)
                usable = images[target].valid_mask & bank.mask("fr")
                losses.append({"source_points": src_points, **loss_breakdown(
                    gray0, gray1, usable, result.correspondences.pixels)})
            for name in variants:
                spec = VARIANTS[name]
                with lk_variant(spec["window"], spec["levels"], spec["fb"]):
                    tracker = vision.VisionTracker(bank, calibrations)
                    for k in range(i, target + 1):
                        tracker.observe(images[k])
                    handoff = tracker.handoff(result, images[target])
                    tracker.settle(handoff, c.HandoffDisposition.DISCARD)
                carried = 0 if handoff.correspondences is None else \
                    int(handoff.correspondences.landmark_ids.size)
                error = None
                if handoff.pose is not None and handoff.pose.T_WB is not None and truth is not None:
                    error = float(np.linalg.norm(handoff.pose.T_WB[:2, 3] - truth[:2, 3]))
                rows[name].append({"source_exposure_ns": source.exposure_ns, "source": src_points,
                                   "carried": carried, "accepted": handoff.accepted,
                                   "reason": handoff.reason, "error_m": error})
        print(f"stretch {start_s}-{end_s}: {len(rows[variants[0]])} handoffs", flush=True)
    loss_total = {k: int(sum(r[k] for r in losses)) for k in losses[0]} if losses else {}
    report = {"scope": "B handoff carry probe; offline, GT only for scoring; unseeded "
                       "single-delivery handoffs, not the integrated localizer",
              "map": bank.summary(), "episode": args.episode, "stretches": args.stretch,
              "variants": {v: VARIANTS[v] for v in variants},
              "summary": {v: summarize(rows[v]) for v in variants},
              "base_loss_breakdown_one_transition": loss_total}
    args.out.mkdir(parents=True)
    (args.out / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for name in variants:
            for row in rows[name]:
                handle.write(json.dumps({"variant": name, **row}) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

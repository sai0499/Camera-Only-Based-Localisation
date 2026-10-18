"""B's offline vision diagnostic on fitting/tuning images (not an estimator, not a score).

Two checks of a frozen map built by ``build_map.py``:

``anchors``   Independent single-image anchor queries: the worker matches a query image to
              <= 2 references selected from a deliberately perturbed prior (GT + noise),
              then RANSAC PnP is solved on its correspondences. Reports acceptance and pose
              error. Self-matches are excluded: query images are not map keyframes.
``sequence``  A causal B-only replay of consecutive images: LK tracking + PnP every
              image, anchor jobs at <= 2 Hz delivered >= 90 ms later through ``handoff``,
              and a constant-velocity prior extrapolated from B's own accepted poses only.
              GT supplies the one initial pose and the scoring. C's filter, scheduler and
              innovation policy are emulated crudely (5 m / 10 deg gate); this is evidence
              diagnostics, not the integrated localizer.

GT is read only here (offline) through ``build_map.read_gt_track``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import build_map  # noqa: E402
from camloc import contracts as c  # noqa: E402
from camloc import geometry as geo  # noqa: E402
from camloc import io as cio  # noqa: E402
from camloc import landmark_map as lm  # noqa: E402
from camloc import pnp, vision  # noqa: E402


def planar(T: np.ndarray) -> tuple[float, float, float]:
    return float(T[0, 3]), float(T[1, 3]), math.atan2(T[1, 0], T[0, 0])


def pose_error(T: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    """Position error, yaw error (deg), longitudinal and lateral error in the GT car frame."""
    dx, dy = T[0, 3] - gt[0, 3], T[1, 3] - gt[1, 3]
    yaw_gt = math.atan2(gt[1, 0], gt[0, 0])
    lon = math.cos(yaw_gt) * dx + math.sin(yaw_gt) * dy
    lat = -math.sin(yaw_gt) * dx + math.cos(yaw_gt) * dy
    dyaw = math.degrees(c.wrap_angle(planar(T)[2] - yaw_gt))
    return math.hypot(dx, dy), dyaw, lon, lat


def aux_from_pose(T: np.ndarray | None, source_ns: int | None) -> c.AuxiliaryGeometry:
    if T is None:
        return c.AuxiliaryGeometry.unavailable("not_estimated")
    R = T[:3, :3]
    return c.AuxiliaryGeometry(
        source=c.AuxSource.PNP, source_ns=source_ns, z_m=float(T[2, 3]),
        roll_rad=math.atan2(R[2, 1], R[2, 2]),
        pitch_rad=math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])),
        road_patch_id=None, valid=True, reason="valid")


def snapshot(t_ns: int, state, sigma_xy: float, sigma_yaw: float,
             aux: c.AuxiliaryGeometry) -> c.SearchSnapshot:
    P = np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2, 25.0, 25.0, 0.25])
    return c.SearchSnapshot(t_ns=t_ns, state=np.asarray(state, float), P=P,
                            status=c.Status.TRACKING, branch=c.Branch.UNKNOWN, auxiliary=aux)


NEAR = {"gt": None, "centers": None, "radius": 0.0}


def images(bag: Path, camera: str, start: int, end: int, calibrations, canonicalizer):
    """Canonical images; with --near-map-m only those whose GT position lies near a map
    keyframe (GT selects diagnostic images only, it is not an input of the method)."""
    for image in build_map.iter_images(bag, camera, start, end, calibrations[camera]):
        if NEAR["radius"] > 0:
            truth = NEAR["gt"].pose(image.exposure_ns)
            if truth is None or np.min(np.hypot(*(NEAR["centers"] - truth[:2, 3]).T)) > \
                    NEAR["radius"]:
                continue
        try:
            yield canonicalizer.canonicalize(image)
        except c.InputError:
            continue


def planar_T(x: float, y: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:2, 3] = [x, y]
    return T


def summarize(errors: list[tuple[float, float, float, float]]) -> dict:
    if not errors:
        return {"count": 0}
    e = np.array(errors)
    return {
        "count": len(e),
        "position_median_m": float(np.median(e[:, 0])),
        "position_p95_m": float(np.percentile(e[:, 0], 95)),
        "position_rmse_m": float(np.sqrt(np.mean(e[:, 0] ** 2))),
        "lateral_rmse_m": float(np.sqrt(np.mean(e[:, 3] ** 2))),
        "longitudinal_rmse_m": float(np.sqrt(np.mean(e[:, 2] ** 2))),
        "yaw_rmse_deg": float(np.sqrt(np.mean(e[:, 1] ** 2))),
        "within_0_5_m": float(np.mean(e[:, 0] <= 0.5)),
        "within_1_0_m": float(np.mean(e[:, 0] <= 1.0)),
    }


def request_for(job_id: int, group_id: int, obs: c.Observation, snap: c.SearchSnapshot,
                map_id: str) -> c.AnchorRequest:
    return c.AnchorRequest(
        job_id=job_id, purpose=c.AnchorPurpose.FRONT_ANCHOR if obs.camera_id == "fr"
        else c.AnchorPurpose.RECOVERY, camera_id=obs.camera_id, source_group_id=group_id,
        source_exposure_ns=obs.exposure_ns,
        delivery_not_before_ns=obs.exposure_ns + c.DELIVERY_MIN_DELAY_NS, observation=obs,
        snapshot=snap, reference_ids=(), map_id=map_id, preprocessing_id=obs.preprocessing_id)


def run_anchors(args, landmark_map, worker, calibrations, canonicalizer, gt, start, end, out):
    rng = np.random.default_rng(0)
    keyframe_times = set(int(t) for t in landmark_map.keyframe_exposure_ns)
    rows, errors, statuses = [], [], {}
    for index, obs in enumerate(images(args.bag, args.camera, start, end, calibrations,
                                       canonicalizer)):
        if index % args.stride or obs.exposure_ns in keyframe_times:
            continue
        truth = gt.pose(obs.exposure_ns)
        if truth is None:
            continue
        x, y, yaw = planar(truth)
        noisy = [x + rng.normal(0, args.prior_noise_m), y + rng.normal(0, args.prior_noise_m),
                 yaw + math.radians(rng.normal(0, 2.0)), 0, 0, 0]
        snap = snapshot(obs.exposure_ns, noisy, args.prior_noise_m, math.radians(2.0),
                        c.AuxiliaryGeometry.unavailable("not_estimated"))
        tic = time.perf_counter()
        result = worker.run(request_for(index, index, obs, snap, landmark_map.map_id))
        worker_ms = (time.perf_counter() - tic) * 1e3
        statuses[result.reason] = statuses.get(result.reason, 0) + 1
        row = {"exposure_ns": obs.exposure_ns, "status": result.reason, "worker_ms": worker_ms,
               "references": " ".join(map(str, result.matched_reference_ids))}
        if result.correspondences is not None:
            corr = result.correspondences
            sol = pnp.solve_pnp(corr.pixels, landmark_map.points(corr.landmark_ids),
                                obs.K, obs.T_BC, valid_mask=(
                                    obs.valid_mask & landmark_map.mask(obs.camera_id)))
            row.update(correspondences=int(corr.landmark_ids.size), pnp=sol.reason,
                       inliers=int(sol.inliers.size), cells=sol.occupied_cells,
                       usable_cells=pnp.occupied_cells(
                           corr.pixels[sol.inliers], landmark_map.mask(obs.camera_id))
                       if sol.inliers.size else 0)
            if sol.T_WB is not None:
                err = pose_error(sol.T_WB, truth)
                row.update(error_m=err[0], yaw_error_deg=err[1], lon_m=err[2], lat_m=err[3])
                if sol.accepted:
                    errors.append(err)
        rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break
    write_rows(out / "anchors.csv", rows)
    times = [r["worker_ms"] for r in rows[1:]] or [0.0]
    summary = {
        "queries": len(rows),
        "status_counts": statuses,
        "pnp_accepted": len(errors),
        "accepted_fraction": len(errors) / max(len(rows), 1),
        "accepted_errors": summarize(errors),
        "worker_ms_median_p95_excluding_first": [float(np.median(times)),
                                                 float(np.percentile(times, 95))],
    }
    return summary


def run_sequence(args, landmark_map, worker, calibrations, canonicalizer, gt, start, end, out):
    tracker = vision.VisionTracker(landmark_map, calibrations)
    rows, errors_all, kinds = [], [], {}
    state = None  # planar x, y, yaw, vx, vy, wz from B's own accepted poses
    last_T, last_ns, initial = None, None, None
    pending = None
    last_anchor_ns = None
    job = 0
    accepted_count = 0
    anchor_reasons: dict[str, int] = {}
    oracle_rng = np.random.default_rng(1)
    if NEAR["radius"] > 0:
        # Contiguous frames spanning the map's stretch on this pass (GT picks the window).
        times = [t for t in NEAR["gt"].t_ns[::10].tolist() if start <= t < end
                 and np.min(np.hypot(*(NEAR["centers"] - NEAR["gt"].pose(t)[:2, 3]).T))
                 <= NEAR["radius"]]
        if not times:
            raise SystemExit("no pass near the map in this window")
        start, end = times[0], times[-1]
        NEAR["radius"] = 0.0
    for group, obs in enumerate(images(args.bag, args.camera, start, end, calibrations,
                                       canonicalizer)):
        t = obs.exposure_ns
        if initial is None:
            truth0 = gt.pose(t)
            if truth0 is None:
                continue
            initial = planar(truth0)  # the one allowed initial pose (x, y, yaw only)
            state = [*initial, 0.0, 0.0, 0.0]
            last_ns = t
        dt = (t - last_ns) / c.NS_PER_S
        prior = [state[0] + state[3] * dt, state[1] + state[4] * dt,
                 c.wrap_angle(state[2] + state[5] * dt), *state[3:]]
        since = dt
        # Velocity is unknown (sigma 30 m/s, as in C's filter) until two accepted poses.
        sigma_v = 30.0 if accepted_count < 2 else 3.0
        sigma = min(math.hypot(0.5, sigma_v * since), 60.0)
        aux = aux_from_pose(last_T, last_ns if last_T is not None else None)
        if args.oracle_prior:
            # Component check only: a GT prior with noise isolates tracking/handoff mechanics
            # from bootstrapping. Never a localization result.
            oracle = gt.pose(t)
            if oracle is not None:
                ox, oy, oyaw = planar(oracle)
                prior = [ox + oracle_rng.normal(0, 0.3), oy + oracle_rng.normal(0, 0.3),
                         oyaw + math.radians(oracle_rng.normal(0, 0.3)), *state[3:]]
                sigma = 0.5
                aux = aux_from_pose(oracle, t)
        snap = snapshot(t, prior, sigma, math.radians(2.0 + 10 * since), aux)
        tracker.observe(obs)
        tic = time.perf_counter()
        tracked = tracker.track(obs, snap, aux)
        track_ms = (time.perf_counter() - tic) * 1e3
        chosen, kind = None, "prediction_only"

        def in_gate(T):
            px, py, pyaw = planar(T)
            gate = max(5.0, 3 * sigma)
            return math.hypot(px - prior[0], py - prior[1]) <= gate and \
                abs(c.wrap_angle(pyaw - prior[2])) <= max(math.radians(10), 3 * snap.P[2, 2] ** 0.5)

        if tracked.accepted and in_gate(tracked.T_WB):
            chosen, kind = tracked.T_WB, "tracking"
        handoff = None
        handoff_info = {}
        if pending is not None and t >= pending.delivery_not_before_ns_value:
            handoff = tracker.handoff(pending.result, obs)
            handoff_info = {
                "handoff_source_points": int(pending.result.correspondences.landmark_ids.size),
                "handoff_carried": 0 if handoff.correspondences is None
                else int(handoff.correspondences.landmark_ids.size),
                "handoff_inliers": 0 if handoff.pose is None else handoff.pose.inlier_count,
                "handoff_cells": 0 if handoff.pose is None else handoff.pose.occupied_cells,
            }
            pending = None
        if chosen is None and handoff is not None and handoff.accepted and \
                in_gate(handoff.pose.T_WB):
            chosen, kind = handoff.pose.T_WB, "handoff"
        if handoff is not None:
            adopt = handoff.accepted and in_gate(handoff.pose.T_WB)
            tracker.settle(handoff, c.HandoffDisposition.ADOPT if adopt
                           else c.HandoffDisposition.DISCARD)
        kinds[kind] = kinds.get(kind, 0) + 1
        if chosen is not None:
            accepted_count += 1
            x, y, yaw = planar(chosen)
            if last_T is not None and 0 < (t - last_ns) <= 500_000_000:
                gap = (t - last_ns) / c.NS_PER_S
                alpha = 1 - math.exp(-gap / 0.2)
                velocity = [(x - state[0]) / gap, (y - state[1]) / gap,
                            c.wrap_angle(yaw - state[2]) / gap]
                state[3:] = [(1 - alpha) * v0 + alpha * v1 for v0, v1 in zip(state[3:], velocity)]
            state[:3] = [x, y, yaw]
            last_T, last_ns = chosen, t
        estimate = chosen if chosen is not None else None
        truth = gt.pose(t)
        row = {"exposure_ns": t, "kind": kind, "track_reason": tracked.reason,
               "track_inliers": tracked.inlier_count, "active": tracker.active_count(obs.camera_id),
               "track_ms": track_ms,
               "handoff": "" if handoff is None else handoff.reason, **handoff_info}
        if truth is not None:
            T_est = estimate if estimate is not None else planar_T(*prior[:3])
            err = pose_error(T_est, truth)
            row.update(error_m=err[0], yaw_error_deg=err[1], lon_m=err[2], lat_m=err[3])
            errors_all.append(err)
        rows.append(row)
        # Anchor proposals: 2 Hz or when tracks run low; one job in flight; delivered >= 90 ms.
        if pending is None and (last_anchor_ns is None or t - last_anchor_ns >= 500_000_000
                                or tracker.active_count(obs.camera_id) < 40):
            if last_anchor_ns is None or t - last_anchor_ns >= 250_000_000:
                result = worker.run(request_for(job, group, obs, snap, landmark_map.map_id))
                anchor_reasons[result.reason] = anchor_reasons.get(result.reason, 0) + 1
                rows[-1]["anchor"] = result.reason
                job += 1
                last_anchor_ns = t
                if result.status is c.AnchorStatus.OK:
                    pending = _Pending(result, t + c.DELIVERY_MIN_DELAY_NS)
        if args.limit and len(rows) >= args.limit:
            break
    write_rows(out / "sequence.csv", rows)
    accepted = [r for r in rows if r["kind"] != "prediction_only"]
    return {
        "images": len(rows),
        "evidence_kinds": kinds,
        "evidence_fraction": len(accepted) / max(len(rows), 1),
        "all_outputs_errors": summarize(errors_all),
        "anchor_jobs": job,
        "anchor_reasons": anchor_reasons,
        "handoff_reasons": {k: sum(r["handoff"] == k for r in rows)
                            for k in sorted({r["handoff"] for r in rows}) if k},
        "track_ms_median_p95": [float(np.median([r["track_ms"] for r in rows])),
                                float(np.percentile([r["track_ms"] for r in rows], 95))],
        "track_reasons": {k: sum(r["track_reason"] == k for r in rows)
                          for k in sorted({r["track_reason"] for r in rows})},
    }


class _Pending:
    def __init__(self, result, delivery):
        self.result = result
        self.delivery_not_before_ns_value = delivery


def write_rows(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--bag", type=Path, default=ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--mode", choices=("anchors", "sequence", "both"), default="both")
    parser.add_argument("--episode", choices=("fit", "tune"), default="fit")
    parser.add_argument("--start-s", type=float, required=True, help="seconds after episode start")
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--camera", default="fr")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--prior-noise-m", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--oracle-prior", action="store_true",
                        help="component check: noisy GT prior and GT aux (not a result)")
    parser.add_argument("--near-map-m", type=float, default=0.0,
                        help="only images within this GT distance of a map keyframe")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; use a fresh results directory")
    episode = c.load_split()[args.episode]
    start = episode.start_ns + int(args.start_s * c.NS_PER_S)
    end = episode.start_ns + int(args.end_s * c.NS_PER_S)
    if not episode.start_ns <= start < end <= episode.end_ns:
        raise SystemExit("window outside the episode")
    landmark_map = lm.load_landmark_map(args.map)
    calibrations = cio.read_calibrations(args.bag)
    canonicalizer = geo.make_canonicalizer(calibrations)
    gt = build_map.read_gt_track(args.bag, start, end)
    NEAR.update(gt=gt, centers=landmark_map.keyframe_T_WC[:, :2, 3], radius=args.near_map_m)
    worker = vision.load_anchor_worker(landmark_map)
    args.out.mkdir(parents=True)
    report = {"map": landmark_map.summary(), "episode": args.episode,
              "window_s": [args.start_s, args.end_s], "camera": args.camera,
              "scope": "B evidence diagnostic; GT only for the initial pose, image choice and "
                       "scoring; not the integrated localizer"}
    if args.mode in ("anchors", "both"):
        report["anchors"] = run_anchors(args, landmark_map, worker, calibrations, canonicalizer,
                                        gt, start, end, args.out)
        print(json.dumps(report["anchors"], indent=2), flush=True)
    if args.mode in ("sequence", "both"):
        report["sequence"] = run_sequence(args, landmark_map, worker, calibrations,
                                          canonicalizer, gt, start, end, args.out)
        print(json.dumps(report["sequence"], indent=2), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

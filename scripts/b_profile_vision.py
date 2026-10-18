"""B Step 6: profile vision components and run the fixed mixed-precision comparison.

Offline measurement tool (owner: B), not the localizer. It times, on real canonical
images of one episode stretch:

* startup: map load, DISK/LightGlue construction, EigenPlaces + retrieval index load;
* per anchor job: retrieval query, DISK extraction, each LightGlue match, source PnP and
  the whole ``DiskAnchorWorker.run``;
* camera-rate path: ``observe``, ``track`` and current-image ``handoff`` on consecutive
  images, with the anchor result delivered at the first image >= source + 90 ms;
* peak GPU memory (torch allocator) and peak process working set (Windows).

``--fp16`` adds the plan's fixed comparison (FINAL_BUILD_PLAN section 3): the same images
through an otherwise identical backend with DISK under fp16 autocast and LightGlue
``mp=True``. The variant may only replace the fp32 baseline when the accepted-pose rate
does not decrease, the pose difference stays below 0.02 m / 0.05 deg on images both accept
and the p95 anchor latency improves. Ground truth is read only to report pose errors.
Run alone on the GPU: concurrent jobs distort timings.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import build_map  # noqa: E402
from camloc import contracts as c  # noqa: E402
from camloc import features  # noqa: E402
from camloc import geometry as geo  # noqa: E402
from camloc import io as cio  # noqa: E402
from camloc import landmark_map as lm  # noqa: E402
from camloc import pnp, vision  # noqa: E402
from camloc import retrieval as rt  # noqa: E402

POSE_DIFF_M = 0.02
POSE_DIFF_DEG = 0.05


def peak_working_set_mb() -> float | None:
    """Peak working set of this process in MiB (Windows only, else None)."""
    if sys.platform != "win32":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # pseudo-handle (HANDLE)-1
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters),
                                           ctypes.c_ulong]
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize / 2**20


def stats_ms(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    v = np.asarray(values, float)
    return {"count": int(v.size), "mean": float(v.mean()), "p50": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max())}


class Fp16Backend(features.FeatureBackend):
    """Candidate variant: DISK forward under fp16 autocast, LightGlue mixed precision."""

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device=device)
        self._matcher.conf.mp = True

    def extract(self, image: np.ndarray, mask: np.ndarray) -> features.Features:
        torch = self._torch
        tensor = torch.from_numpy(np.array(image, copy=True)).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)[None]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = self._extractor({"image": tensor})
        return features.select_keypoints(
            out["keypoints"][0].float().cpu().numpy(),
            out["keypoint_scores"][0].float().cpu().numpy(),
            out["descriptors"][0].float().cpu().numpy(), mask)


class TimedBackend:
    """Wraps a backend and records extraction/match wall times (GPU synchronised)."""

    def __init__(self, backend) -> None:
        self.inner = backend
        self.model_ids = backend.model_ids
        self.extract_ms: list[float] = []
        self.match_ms: list[float] = []

    def extract(self, image, mask):
        tic = time.perf_counter()
        out = self.inner.extract(image, mask)
        self.inner.synchronize()
        self.extract_ms.append((time.perf_counter() - tic) * 1e3)
        return out

    def match(self, a, b):
        tic = time.perf_counter()
        out = self.inner.match(a, b)
        self.inner.synchronize()
        self.match_ms.append((time.perf_counter() - tic) * 1e3)
        return out


class TimedRetrieval:
    def __init__(self, index) -> None:
        self.inner = index
        self.query_ms: list[float] = []

    def query(self, observation):
        tic = time.perf_counter()
        out = self.inner.query(observation)
        self.query_ms.append((time.perf_counter() - tic) * 1e3)
        return out


def load_images(args, calibrations, canonicalizer):
    episode = c.load_split()[args.episode]
    start = episode.start_ns + int(args.start_s * c.NS_PER_S)
    end = episode.start_ns + int(args.end_s * c.NS_PER_S)
    if not episode.start_ns <= start < end <= episode.end_ns:
        raise SystemExit("window outside the episode")
    images = []
    tic = time.perf_counter()
    for image in build_map.iter_images(args.bag, args.camera, start, end,
                                       calibrations[args.camera]):
        try:
            images.append(canonicalizer.canonicalize(image))
        except c.InputError:
            continue
    image_ms = (time.perf_counter() - tic) * 1e3
    return images, build_map.read_gt_track(args.bag, start, end), image_ms


def request(job_id: int, obs: c.Observation, map_id: str) -> c.AnchorRequest:
    snapshot = c.SearchSnapshot(
        t_ns=obs.exposure_ns, state=np.zeros(6), P=np.eye(6) * 1e4, status=c.Status.RECOVERING,
        branch=c.Branch.UNKNOWN, auxiliary=c.AuxiliaryGeometry.unavailable("not_estimated"))
    purpose = (c.AnchorPurpose.FRONT_ANCHOR if obs.camera_id == c.PRIMARY_CAMERA
               else c.AnchorPurpose.RECOVERY)
    return c.AnchorRequest(
        job_id=job_id, purpose=purpose, camera_id=obs.camera_id, source_group_id=job_id,
        source_exposure_ns=obs.exposure_ns,
        delivery_not_before_ns=obs.exposure_ns + c.DELIVERY_MIN_DELAY_NS, observation=obs,
        snapshot=snapshot, reference_ids=(), map_id=map_id,
        preprocessing_id=obs.preprocessing_id)


def anchor_pass(worker, images, bank, gt, stride: int) -> tuple[list[dict], dict]:
    """Worker job every ``stride`` images; source PnP under production gates."""
    rows, run_ms, pnp_ms = [], [], []
    for job, obs in enumerate(images[::stride]):
        tic = time.perf_counter()
        result = worker.run(request(job, obs, bank.map_id))
        run_ms.append((time.perf_counter() - tic) * 1e3)
        row = {"exposure_ns": obs.exposure_ns, "status": result.reason, "accepted": False}
        if result.correspondences is not None:
            corr = result.correspondences
            usable = obs.valid_mask & bank.mask(obs.camera_id)
            tic = time.perf_counter()
            solution = pnp.solve_pnp(corr.pixels, bank.points(corr.landmark_ids), obs.K,
                                     obs.T_BC, valid_mask=usable)
            pnp_ms.append((time.perf_counter() - tic) * 1e3)
            row.update(points=int(corr.landmark_ids.size), pnp=solution.reason,
                       accepted=solution.accepted)
            truth = gt.pose(obs.exposure_ns)
            if solution.T_WB is not None:
                row["T_WB"] = solution.T_WB
                if truth is not None:
                    row["error_m"] = float(np.linalg.norm(solution.T_WB[:2, 3] - truth[:2, 3]))
        rows.append(row)
    return rows, {"worker_run_ms": stats_ms(run_ms[1:]), "source_pnp_ms": stats_ms(pnp_ms)}


def camera_rate_pass(tracker_factory, worker, images, bank) -> dict:
    """observe/track every image; an anchor at 2 Hz delivered >= 90 ms later via handoff."""
    tracker = tracker_factory()
    observe_ms, track_ms, handoff_ms = [], [], []
    reasons: dict[str, int] = {}
    pending, last_anchor, job = None, None, 0
    aux = c.AuxiliaryGeometry.unavailable("not_estimated")
    for obs in images:
        tic = time.perf_counter()
        tracker.observe(obs)
        observe_ms.append((time.perf_counter() - tic) * 1e3)
        snapshot = c.SearchSnapshot(
            t_ns=obs.exposure_ns, state=np.zeros(6), P=np.eye(6) * 1e4, status=c.Status.TRACKING,
            branch=c.Branch.UNKNOWN, auxiliary=aux)
        tic = time.perf_counter()
        evidence = tracker.track(obs, snapshot, aux)
        if evidence.reason != "no_active_tracks":
            track_ms.append((time.perf_counter() - tic) * 1e3)
        if pending is not None and obs.exposure_ns >= pending.source_exposure_ns + \
                c.DELIVERY_MIN_DELAY_NS:
            tic = time.perf_counter()
            handoff = tracker.handoff(pending, obs)
            handoff_ms.append((time.perf_counter() - tic) * 1e3)
            reasons[handoff.reason] = reasons.get(handoff.reason, 0) + 1
            tracker.settle(handoff, c.HandoffDisposition.ADOPT if handoff.accepted
                           else c.HandoffDisposition.DISCARD)
            pending = None
        if pending is None and (last_anchor is None or obs.exposure_ns - last_anchor >= 5e8):
            result = worker.run(request(job, obs, bank.map_id))
            job += 1
            last_anchor = obs.exposure_ns
            if result.status is c.AnchorStatus.OK:
                pending = result
    return {"observe_ms": stats_ms(observe_ms), "track_ms": stats_ms(track_ms),
            "handoff_ms": stats_ms(handoff_ms), "handoff_reasons": reasons}


def compare(base_rows: list[dict], cand_rows: list[dict], base_p95: float,
            cand_p95: float) -> dict:
    """The plan's fixed optimisation rule, applied to two anchor passes on identical images."""
    by_t = {r["exposure_ns"]: r for r in cand_rows}
    diffs_m, diffs_deg = [], []
    for row in base_rows:
        other = by_t.get(row["exposure_ns"])
        if other is None or not (row["accepted"] and other["accepted"]):
            continue
        a, b = row["T_WB"], other["T_WB"]
        diffs_m.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
        cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
        diffs_deg.append(math.degrees(math.acos(float(np.clip(cos, -1, 1)))))
    rate_base = sum(r["accepted"] for r in base_rows) / max(len(base_rows), 1)
    rate_cand = sum(r["accepted"] for r in cand_rows) / max(len(cand_rows), 1)
    max_m = max(diffs_m, default=None)
    max_deg = max(diffs_deg, default=None)
    checks = {
        "same_query_images": [r["exposure_ns"] for r in base_rows] ==
                             [r["exposure_ns"] for r in cand_rows],
        "paired_pose_evidence_available": bool(diffs_m),
        "accepted_rate_not_lower": rate_cand >= rate_base,
        "pose_difference_below_0_02_m": max_m is not None and max_m < POSE_DIFF_M,
        "pose_difference_below_0_05_deg": max_deg is not None and max_deg < POSE_DIFF_DEG,
        "p95_latency_improves": cand_p95 < base_p95,
    }
    return {"accepted_rate_fp32": rate_base, "accepted_rate_fp16": rate_cand,
            "both_accepted": len(diffs_m), "pose_diff_max_m": max_m,
            "pose_diff_median_m": float(np.median(diffs_m)) if diffs_m else None,
            "pose_diff_max_deg": max_deg, "anchor_p95_ms_fp32": base_p95,
            "anchor_p95_ms_fp16": cand_p95, "checks": checks,
            "adopt_fp16": all(checks.values())}


def error_summary(rows: list[dict]) -> dict:
    errors = [r["error_m"] for r in rows if r["accepted"] and "error_m" in r]
    return {"accepted": len(errors), "jobs": len(rows),
            "error_median_m": float(np.median(errors)) if errors else None,
            "error_p95_m": float(np.percentile(errors, 95)) if errors else None,
            "status": {k: sum(r["status"] == k for r in rows)
                       for k in sorted({r["status"] for r in rows})}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--bag", type=Path, default=ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--episode", choices=("fit", "tune"), default="tune")
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--camera", default="fr")
    parser.add_argument("--stride", type=int, default=5, help="anchor job every N images")
    parser.add_argument("--fp16", action="store_true", help="add the fp16 comparison")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; use a fresh directory")
    if args.stride < 1:
        parser.error("--stride must be positive")
    import torch

    report: dict = {"scope": "B component profile and fixed fp16 comparison; offline, GT only "
                             "for pose errors; not paced full-system qualification",
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    startup = {}
    tic = time.perf_counter()
    calibrations = cio.read_calibrations(args.bag)
    startup["read_calibrations_s"] = time.perf_counter() - tic
    canonicalizer = geo.make_canonicalizer(calibrations)
    tic = time.perf_counter()
    bank = lm.load_landmark_map(args.artifacts)
    startup["map_load_s"] = time.perf_counter() - tic
    tic = time.perf_counter()
    backend = features.FeatureBackend()
    startup["disk_lightglue_init_s"] = time.perf_counter() - tic
    tic = time.perf_counter()
    model = rt.load_pinned_eigenplaces()
    index = rt.RetrievalIndex.load(args.artifacts, bank, model)
    startup["eigenplaces_index_load_s"] = time.perf_counter() - tic
    report["startup"] = startup
    report["map"] = bank.summary()

    images, gt, image_ms = load_images(args, calibrations, canonicalizer)
    report["images"] = {"count": len(images), "decode_canonicalize_ms_mean":
                        image_ms / max(len(images), 1)}
    if len(images) < 10:
        raise SystemExit("too few images in the window")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timed, timed_index = TimedBackend(backend), TimedRetrieval(index)
    worker = vision.DiskAnchorWorker(bank, timed, retrieval=timed_index)
    worker.run(request(10**6, images[0], bank.map_id))  # warm-up, excluded
    timed.extract_ms.clear()
    timed.match_ms.clear()
    timed_index.query_ms.clear()
    base_rows, anchor_times = anchor_pass(worker, images, bank, gt, args.stride)
    report["anchor_fp32"] = {**anchor_times, "retrieval_query_ms": stats_ms(timed_index.query_ms),
                             "disk_extract_ms": stats_ms(timed.extract_ms),
                             "lightglue_match_ms": stats_ms(timed.match_ms),
                             "outcome": error_summary(base_rows)}
    report["camera_rate_fp32"] = camera_rate_pass(
        lambda: vision.VisionTracker(bank, calibrations), worker, images, bank)
    if torch.cuda.is_available():
        report["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    report["process_peak_working_set_mb"] = peak_working_set_mb()

    if args.fp16:
        if not torch.cuda.is_available():
            raise SystemExit("--fp16 needs CUDA")
        fast = TimedBackend(Fp16Backend())
        fast_worker = vision.DiskAnchorWorker(bank, fast, retrieval=index)
        fast_worker.run(request(10**6, images[0], bank.map_id))
        fast.extract_ms.clear()
        fast.match_ms.clear()
        cand_rows, cand_times = anchor_pass(fast_worker, images, bank, gt, args.stride)
        report["anchor_fp16"] = {**cand_times, "disk_extract_ms": stats_ms(fast.extract_ms),
                                 "lightglue_match_ms": stats_ms(fast.match_ms),
                                 "outcome": error_summary(cand_rows)}
        report["fp16_decision"] = compare(base_rows, cand_rows,
                                          anchor_times["worker_run_ms"]["p95"],
                                          cand_times["worker_run_ms"]["p95"])

    args.out.mkdir(parents=True)
    (args.out / "profile.json").write_text(json.dumps(report, indent=2, default=float) + "\n",
                                           encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

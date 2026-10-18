"""Build the frozen fitting-only landmark map (owner: B). Offline training tool.

This is the only B code that reads ground truth: ``/tf`` map -> constructor0/base_link,
restricted to the fixed fitting interval of ``task_docs/build_split.json`` and interpolated
at each keyframe's camera exposure. The resulting map stores reference geometry and
features only; inference (``camloc/``) never imports this module or reads GT.

Pipeline (FINAL_BUILD_PLAN section 3):

1. Read calibration through A's ``camloc.io`` and canonicalise images with A's remap.
2. Select keyframes per camera on >= 4 m travel or >= 5 deg heading change.
3. Extract DISK features on usable pixels (remap-valid, bodywork masked).
4. Match each keyframe with at most 8 nearby, viewing-compatible keyframes (LightGlue).
5. Triangulate with fixed known poses and intrinsics through hloc/pycolmap, then verify the
   poses stayed fixed and the pixel-centre convention round-trips.
6. Keep landmarks with >= 3 observations, positive depths, >= 1 deg triangulation angle and
   median reprojection error <= 2 canonical px; save under a fresh output directory.

Example (small first map, ~100 fr keyframes inside the fitting interval)::

    .venv/Scripts/python.exe -B build_map.py --bag c0_2026-10-13_22-53-42_camloc_0.mcap \
        --map-mcap 23-00-38_cut_120m_update/map.mcap --split task_docs/build_split.json \
        --output artifacts/fit-small-v1 --start-s 483 --end-s 603 --max-keyframes 100
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from camloc import contracts as c
from camloc import geometry as geo
from camloc import io as cio
from camloc import landmark_map as lm
from camloc import masks

GT_TOPIC = "/tf"
GT_PARENT, GT_CHILD = "map", "constructor0/base_link"
MAX_GT_GAP_NS = 50_000_000


@dataclasses.dataclass(frozen=True)
class BuildConfig:
    travel_m: float = 4.0
    heading_deg: float = 5.0
    neighbors: int = 8
    pair_max_distance_m: float = 40.0
    pair_min_baseline_m: float = 1.0
    pair_max_axis_deg: float = 30.0
    triangulation_min_angle_deg: float = 1.0
    min_observations: int = 3
    max_median_error_px: float = 2.0
    min_angle_deg: float = 1.0
    min_landmarks_per_keyframe_warn: int = 30


# ---------------------------------------------------------------------------------------
# Ground truth (offline only)
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GtTrack:
    t_ns: np.ndarray  # int64 (N,)
    xyz: np.ndarray  # float64 (N, 3)
    quat_xyzw: np.ndarray  # float64 (N, 4)

    def pose(self, t_ns: int) -> np.ndarray | None:
        """Interpolated T_WB at ``t_ns`` (linear position, slerp), None without support."""
        times = self.t_ns
        i = int(np.searchsorted(times, t_ns, side="right"))
        if i == 0:
            return None
        if int(times[i - 1]) == t_ns:
            return _pose(self.xyz[i - 1], self.quat_xyzw[i - 1])
        if i >= len(times):
            return None
        t0, t1 = int(times[i - 1]), int(times[i])
        if t1 - t0 > MAX_GT_GAP_NS:
            return None
        w = (t_ns - t0) / (t1 - t0)
        position = (1 - w) * self.xyz[i - 1] + w * self.xyz[i]
        return _pose(position, slerp(self.quat_xyzw[i - 1], self.quat_xyzw[i], w))


def slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + w * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - w) * theta) * q0 + math.sin(w * theta) * q1) / math.sin(theta)


def _pose(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    R = geo.quaternion_xyzw_to_matrix(*(float(v) for v in quat_xyzw))
    return geo.rigid_from_rotation_translation(R, position)


def read_gt_track(bag: Path, start_ns: int, end_ns: int, margin_ns: int = 1_000_000_000) -> GtTrack:
    """``/tf`` map -> base_link samples whose stamp lies in [start - margin, end + margin)."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    rows: dict[int, tuple[float, ...]] = {}
    with open(bag, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(
                topics=[GT_TOPIC], start_time=start_ns - 2 * margin_ns,
                end_time=end_ns + 2 * margin_ns):
            for tr in msg.transforms:
                if tr.header.frame_id != GT_PARENT or tr.child_frame_id != GT_CHILD:
                    continue
                stamp = tr.header.stamp.sec * c.NS_PER_S + tr.header.stamp.nanosec
                if not start_ns - margin_ns <= stamp < end_ns + margin_ns:
                    continue
                p, q = tr.transform.translation, tr.transform.rotation
                rows[stamp] = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if len(rows) < 2:
        raise RuntimeError("no /tf ground truth in the requested window")
    stamps = np.array(sorted(rows), np.int64)
    values = np.array([rows[int(s)] for s in stamps], np.float64)
    return GtTrack(stamps, values[:, :3], values[:, 3:])


# ---------------------------------------------------------------------------------------
# Images and keyframes
# ---------------------------------------------------------------------------------------


def iter_images(bag: Path, camera_id: str, start_ns: int, end_ns: int,
                calibration: c.CameraCalibration):
    """ImageMessages of one camera with exposure in [start_ns, end_ns), in log order."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with open(bag, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, _, message, msg in reader.iter_decoded_messages(
                topics=[cio.image_topic(camera_id)], start_time=start_ns - 500_000_000,
                end_time=end_ns + 500_000_000):
            stamp = msg.header.stamp.sec * c.NS_PER_S + msg.header.stamp.nanosec
            if not start_ns <= stamp < end_ns:
                continue
            yield c.ImageMessage(
                camera_id=camera_id, exposure_ns=stamp, arrival_ns=message.log_time,
                source_id=f"{camera_id}:{stamp}", frame_id=msg.header.frame_id,
                image_format=str(msg.format), calibration_id=calibration.calibration_id,
                data=bytes(msg.data))


def heading(T_WB: np.ndarray) -> float:
    return math.atan2(T_WB[1, 0], T_WB[0, 0])


def select_keyframe(last: np.ndarray | None, pose: np.ndarray, config: BuildConfig) -> bool:
    if last is None:
        return True
    travel = float(np.linalg.norm(pose[:2, 3] - last[:2, 3]))
    turn = abs(c.wrap_angle(heading(pose) - heading(last)))
    return travel >= config.travel_m or turn >= math.radians(config.heading_deg)


def select_pairs(cameras: list[str], T_WC: np.ndarray, config: BuildConfig) -> list[tuple[int, int]]:
    """Unordered keyframe pairs: same camera, baseline in range, compatible optical axes,
    at most ``neighbors`` nearest partners per keyframe."""
    centers = T_WC[:, :3, 3]
    axes = T_WC[:, :3, 2]
    cams = np.asarray(cameras)
    pairs: set[tuple[int, int]] = set()
    cos_max = math.cos(math.radians(config.pair_max_axis_deg))
    for i in range(len(T_WC)):
        distance = np.linalg.norm(centers - centers[i], axis=1)
        ok = (cams == cams[i]) & (distance >= config.pair_min_baseline_m) \
            & (distance <= config.pair_max_distance_m) & (axes @ axes[i] >= cos_max)
        ok[i] = False
        rows = np.flatnonzero(ok)
        for j in rows[np.lexsort((rows, distance[rows]))][: config.neighbors]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


# ---------------------------------------------------------------------------------------
# Known-pose triangulation through hloc / pycolmap
# ---------------------------------------------------------------------------------------


def _image_name(camera_id: str, exposure_ns: int) -> str:
    return f"{camera_id}/{exposure_ns}.jpg"


def triangulate_known_poses(work: Path, cameras: list[str], exposures: list[int],
                            T_WC: np.ndarray, K: dict[str, np.ndarray],
                            keypoints: list[np.ndarray], pairs: list[tuple[int, int]],
                            matches: list[np.ndarray], config: BuildConfig):
    """Fixed-pose, fixed-intrinsics triangulation. Returns (reconstruction, image ids).

    hloc's importer adds the single +0.5 COLMAP pixel-origin shift to the stored OpenCV
    keypoints, so the COLMAP cameras use cx + 0.5, cy + 0.5. pycolmap 3.13's ``rigs``/
    ``frames`` map views crash on Windows, so the database is written via the getters
    instead of ``hloc.triangulation.create_db_from_model``.
    """
    import h5py
    import pycolmap
    from hloc import triangulation
    from hloc.utils.parsers import names_to_pair

    work.mkdir(parents=True, exist_ok=False)
    rec = pycolmap.Reconstruction()
    camera_ids = {cam: i + 1 for i, cam in enumerate(sorted(set(cameras)))}
    for cam, cid in camera_ids.items():
        k = K[cam]
        camera = pycolmap.Camera(model="PINHOLE", width=c.CANONICAL_WIDTH,
                                 height=c.CANONICAL_HEIGHT,
                                 params=[k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5],
                                 camera_id=cid)
        rec.add_camera(camera)
        rig = pycolmap.Rig(rig_id=cid)
        rig.add_ref_sensor(camera.sensor_id)
        rec.add_rig(rig)
    names = [_image_name(cam, t) for cam, t in zip(cameras, exposures, strict=True)]
    for row, (cam, name) in enumerate(zip(cameras, names, strict=True)):
        image_id = row + 1
        T_CW = geo.invert_rigid(T_WC[row])
        frame = pycolmap.Frame(frame_id=image_id, rig_id=camera_ids[cam])
        frame.add_data_id(pycolmap.data_t(sensor_id=rec.camera(camera_ids[cam]).sensor_id,
                                          id=image_id))
        frame.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(T_CW[:3, :3]), T_CW[:3, 3])
        rec.add_frame(frame)
        image = pycolmap.Image(name=name, camera_id=camera_ids[cam], image_id=image_id)
        image.frame_id = image_id
        rec.add_image(image)
        rec.register_frame(image_id)

    features_path, matches_path, pairs_path = work / "features.h5", work / "matches.h5", \
        work / "pairs.txt"
    with h5py.File(features_path, "w") as handle:
        for name, kps in zip(names, keypoints, strict=True):
            handle.create_group(name).create_dataset("keypoints", data=kps.astype(np.float32))
    with h5py.File(matches_path, "w") as handle:
        for (i, j), pair_matches in zip(pairs, matches, strict=True):
            matches0 = np.full(len(keypoints[i]), -1, np.int32)
            scores0 = np.zeros(len(keypoints[i]), np.float32)
            matches0[pair_matches[:, 0]] = pair_matches[:, 1]
            scores0[pair_matches[:, 0]] = 1.0
            group = handle.create_group(names_to_pair(names[i], names[j]))
            group.create_dataset("matches0", data=matches0)
            group.create_dataset("matching_scores0", data=scores0)
    pairs_path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pairs), encoding="utf-8")

    sfm = work / "sfm"
    sfm.mkdir()
    database = sfm / "database.db"
    with pycolmap.Database.open(str(database)) as db:
        for cid in camera_ids.values():
            db.write_camera(rec.camera(cid), use_camera_id=True)
            db.write_rig(rec.rig(cid), use_rig_id=True)
        for row in range(len(names)):
            db.write_frame(rec.frame(row + 1), use_frame_id=True)
            db.write_image(rec.image(row + 1), use_image_id=True)
    image_ids = {name: row + 1 for row, name in enumerate(names)}
    with pycolmap.Database.open(str(database)) as db:
        triangulation.import_features(image_ids, db, features_path)
        triangulation.import_matches(image_ids, db, pairs_path, matches_path)
        triangulation.geometric_verification(image_ids, rec, db, features_path, pairs_path,
                                             matches_path)
    (work / "images").mkdir()
    out = triangulation.run_triangulation(
        sfm, database, work / "images", rec, verbose=False,
        options={"triangulation": {"min_angle": config.triangulation_min_angle_deg}})
    return out, image_ids


def max_pose_deviation(out, T_WC: np.ndarray) -> float:
    worst = 0.0
    for row in range(len(T_WC)):
        expected = geo.invert_rigid(T_WC[row])[:3]
        actual = np.asarray(out.image(row + 1).cam_from_world().matrix())
        worst = max(worst, float(np.abs(actual - expected).max()))
    return worst


def filter_landmarks(out, T_WC: np.ndarray, K_rows: list[np.ndarray],
                     keypoints: list[np.ndarray], config: BuildConfig):
    """Apply the landmark gates with OpenCV-convention projection (independent of COLMAP)."""
    kept_points, kept_obs, stats = [], [], []
    signed = []
    rejected = {"few_observations": 0, "negative_depth": 0, "small_angle": 0, "high_error": 0}
    T_CW = np.array([geo.invert_rigid(T) for T in T_WC])
    centers = T_WC[:, :3, 3]
    for point_id in sorted(out.point3D_ids()):
        point = out.point3D(point_id)
        xyz = np.asarray(point.xyz, np.float64)
        observations: dict[int, int] = {}
        for element in point.track.elements:
            row = int(element.image_id) - 1
            observations.setdefault(row, int(element.point2D_idx))
        if len(observations) < config.min_observations:
            rejected["few_observations"] += 1
            continue
        rows = np.array(sorted(observations))
        idx = np.array([observations[r] for r in rows])
        Pc = np.einsum("nij,j->ni", T_CW[rows, :3, :3], xyz) + T_CW[rows, :3, 3]
        if np.any(Pc[:, 2] <= 0):
            rejected["negative_depth"] += 1
            continue
        proj = np.array([(K_rows[r] @ p)[:2] / p[2] for r, p in zip(rows, Pc, strict=True)])
        obs = np.array([keypoints[r][i] for r, i in zip(rows, idx, strict=True)], np.float64)
        residual = proj - obs
        errors = np.linalg.norm(residual, axis=1)
        rays = xyz - centers[rows]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        angle = math.degrees(math.acos(float(np.clip((rays @ rays.T).min(), -1.0, 1.0))))
        if angle < config.min_angle_deg:
            rejected["small_angle"] += 1
            continue
        if float(np.median(errors)) > config.max_median_error_px:
            rejected["high_error"] += 1
            continue
        kept_points.append(xyz)
        kept_obs.append((rows, idx))
        stats.append((len(rows), float(np.median(errors)), angle))
        signed.append(residual)
    return kept_points, kept_obs, stats, rejected, (np.concatenate(signed) if signed else
                                                    np.zeros((0, 2)))


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _versions() -> dict[str, str]:
    names = ("torch", "opencv-python", "pycolmap", "hloc", "lightglue", "numpy", "h5py", "kornia",
             "mcap", "mcap-ros2-support")
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def build(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    config = BuildConfig(travel_m=args.travel_m, heading_deg=args.heading_deg,
                         neighbors=args.neighbors)
    output: Path = args.output.resolve()
    work = output.with_name(output.name + ".work")
    if output.exists() or work.exists():
        raise SystemExit(f"Output {output} (or its .work) exists; use a fresh directory")
    if _sha256(args.split) != c.BUILD_SPLIT_SHA256:
        raise SystemExit("build_split.json changed; the fitting interval is frozen")
    fit = c.load_split(args.split)["fit"]
    spans = args.window or [[args.start_s, args.end_s]]
    windows = []
    for start_s, end_s in spans:
        start = fit.start_ns + int(round(start_s * c.NS_PER_S))
        end = fit.end_ns if end_s is None else fit.start_ns + int(round(end_s * c.NS_PER_S))
        if not fit.start_ns <= start < end <= fit.end_ns:
            raise SystemExit("every window must lie inside the fitting interval")
        windows.append((start, end))
    windows.sort()
    if any(a[1] > b[0] for a, b in zip(windows[:-1], windows[1:])):
        raise SystemExit("windows overlap")
    map_sha = _sha256(args.map_mcap)
    if map_sha != c.SUPPLIED_MAP_SHA256:
        raise SystemExit("supplied map.mcap hash mismatch")

    print("Reading calibration ...", flush=True)
    calibrations = cio.read_calibrations(args.bag)
    canonicalizer = geo.make_canonicalizer(calibrations)

    from camloc.features import FeatureBackend, backend_identity

    backend = FeatureBackend(device=args.device)
    cameras, exposures, poses_WB, keypoints, descriptors, scores = [], [], [], [], [], []
    usable_masks: dict[str, np.ndarray] = {}
    skipped = {"no_gt_support": 0, "decode_failure": 0}
    for camera_id in args.cameras:
        calibration = calibrations[camera_id]
        mask = canonicalizer.camera_map(camera_id).valid_mask & masks.bodywork_mask(camera_id)
        usable_masks[camera_id] = mask
        count = 0
        for start, end in windows:
            if args.max_keyframes and count >= args.max_keyframes:
                break
            print(f"Reading GT and extracting {camera_id} keyframes in "
                  f"[{(start - fit.start_ns) / 1e9:.1f}, {(end - fit.start_ns) / 1e9:.1f}) s ...",
                  flush=True)
            gt = read_gt_track(args.bag, start, end)
            last = None
            for image in iter_images(args.bag, camera_id, start, end, calibration):
                pose = gt.pose(image.exposure_ns)
                if pose is None:
                    skipped["no_gt_support"] += 1
                    continue
                if not select_keyframe(last, pose, config):
                    continue
                try:
                    observation = canonicalizer.canonicalize(image)
                except c.InputError:
                    skipped["decode_failure"] += 1
                    continue
                features = backend.extract(np.asarray(observation.image), mask)
                last = pose
                cameras.append(camera_id)
                exposures.append(image.exposure_ns)
                poses_WB.append(pose)
                keypoints.append(features.keypoints)
                descriptors.append(features.descriptors)
                scores.append(features.scores)
                count += 1
                if args.max_keyframes and count >= args.max_keyframes:
                    break
        print(f"  {count} {camera_id} keyframes", flush=True)
    if len(cameras) < 3:
        raise SystemExit("fewer than three keyframes selected")
    T_WC = np.array([pose @ calibrations[cam].T_BC for pose, cam in zip(poses_WB, cameras,
                                                                          strict=True)])
    pairs = select_pairs(cameras, T_WC, config)
    print(f"Matching {len(pairs)} keyframe pairs ...", flush=True)
    from camloc.features import Features

    feats = [Features(k, s, d) for k, s, d in zip(keypoints, scores, descriptors, strict=True)]
    pair_matches = []
    match_counts = []
    for i, j in pairs:
        found, _ = backend.match(feats[i], feats[j])
        pair_matches.append(found)
        match_counts.append(len(found))
    K = {cam: calibrations[cam].K for cam in set(cameras)}
    print("Triangulating with fixed poses ...", flush=True)
    out, _ = triangulate_known_poses(work, cameras, exposures, T_WC, K, keypoints, pairs,
                                     pair_matches, config)
    deviation = max_pose_deviation(out, T_WC)
    if deviation > 1e-9:
        raise SystemExit(f"triangulation changed fixed poses by {deviation}")
    K_rows = [K[cam] for cam in cameras]
    points, observations, stats, rejected, signed = filter_landmarks(out, T_WC, K_rows,
                                                                     keypoints, config)
    print(f"Triangulated {out.num_points3D()} points, kept {len(points)}", flush=True)

    # Assemble arrays.
    keypoint_landmarks = [np.full(len(k), -1, np.int64) for k in keypoints]
    for landmark_id, (rows, idx) in enumerate(observations):
        for r, i in zip(rows, idx, strict=True):
            if keypoint_landmarks[r][i] < 0:
                keypoint_landmarks[r][i] = landmark_id
    offsets = np.zeros(len(keypoints) + 1, np.int64)
    offsets[1:] = np.cumsum([len(k) for k in keypoints])
    stats_arr = np.array(stats, np.float64).reshape(-1, 3)
    arrays = {
        "landmark_ids": np.arange(len(points), dtype=np.int64),
        "points_W": np.array(points, np.float64).reshape(-1, 3),
        "landmark_observations": stats_arr[:, 0].astype(np.int32),
        "landmark_median_error_px": stats_arr[:, 1].astype(np.float32),
        "landmark_max_angle_deg": stats_arr[:, 2].astype(np.float32),
        "keyframe_ids": np.arange(len(cameras), dtype=np.int64),
        "keyframe_cameras": np.array(cameras, dtype="<U2"),
        "keyframe_exposure_ns": np.array(exposures, np.int64),
        "keyframe_T_WC": T_WC.astype(np.float64),
        "keypoint_offsets": offsets,
        "keypoints": np.concatenate(keypoints).astype(np.float32),
        "keypoint_scores": np.concatenate(scores).astype(np.float32),
        "descriptors": np.concatenate(descriptors).astype(np.float16),
        "keypoint_landmarks": np.concatenate(keypoint_landmarks),
    }
    for cam, mask in usable_masks.items():
        arrays[f"mask_{cam}"] = np.asarray(mask, bool)
    per_keyframe = [int(np.count_nonzero(k >= 0)) for k in keypoint_landmarks]
    weights = backend.weights
    any_cal = calibrations[args.cameras[0]]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": "build_map.py",
        "scope": "Fitting-only fixed-pose landmark map; reference geometry and features only.",
        "cameras": list(args.cameras),
        "calibration_id": any_cal.calibration_id,
        "preprocessing_id": any_cal.preprocessing_id,
        "canonical_K": {cam: calibrations[cam].K.tolist() for cam in args.cameras},
        "T_BC": {cam: calibrations[cam].T_BC.tolist() for cam in args.cameras},
        "mask_id": masks.MASK_ID,
        "fitting_interval_ns": [str(fit.start_ns), str(fit.end_ns)],
        "windows_ns": [[str(a), str(b)] for a, b in windows],
        "split_sha256": c.BUILD_SPLIT_SHA256,
        "source_bag": {"name": Path(args.bag).name, "bytes": Path(args.bag).stat().st_size,
                       "sha256": None if args.skip_bag_hash else _sha256(Path(args.bag))},
        "supplied_map_sha256": map_sha,
        "gt_source": "/tf map->constructor0/base_link inside the fitting interval, linear "
                     "position and slerp rotation at each keyframe exposure (gap <= 50 ms)",
        "feature_backend": backend_identity(),
        "model_ids": list(backend.model_ids),
        "model_sha256": {k: v["sha256"] for k, v in weights.items() if k != "torch_home"},
        "libraries": _versions(),
        "selection": dataclasses.asdict(config),
        "colmap_pixel_convention": "OpenCV centres stored; hloc importer adds +0.5; COLMAP "
                                   "cameras use cx+0.5, cy+0.5",
        "reference_map_inference_permission": "unconfirmed_request_organizer_clarification",
        "statistics": {
            "keyframes": len(cameras),
            "keyframes_per_camera": {cam: cameras.count(cam) for cam in args.cameras},
            "pairs": len(pairs),
            "matches_per_pair_median": float(np.median(match_counts)) if match_counts else 0.0,
            "triangulated_points": int(out.num_points3D()),
            "landmarks": len(points),
            "rejected_points": rejected,
            "landmark_observations_median": float(np.median(stats_arr[:, 0])) if len(points)
            else 0.0,
            "landmark_median_error_px_median": float(np.median(stats_arr[:, 1])) if len(points)
            else 0.0,
            "landmark_angle_deg_median": float(np.median(stats_arr[:, 2])) if len(points) else 0.0,
            "landmarks_per_keyframe_min_median_max": [int(min(per_keyframe)),
                                                      float(np.median(per_keyframe)),
                                                      int(max(per_keyframe))],
            "keyframes_below_landmark_warning": int(sum(
                v < config.min_landmarks_per_keyframe_warn for v in per_keyframe)),
            "signed_residual_mean_px": signed.mean(axis=0).tolist() if len(signed) else None,
            "fixed_pose_max_deviation": deviation,
            "skipped_images": skipped,
            "build_seconds": round(time.perf_counter() - started, 1),
        },
    }
    map_id = lm.save_landmark_map(output, arrays, manifest)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"map_id": map_id, **manifest["statistics"]}, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bag", type=Path, default=c.PROJECT_ROOT /
                        "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--map-mcap", type=Path, default=c.PROJECT_ROOT /
                        "23-00-38_cut_120m_update" / "map.mcap")
    parser.add_argument("--split", type=Path, default=c.DEFAULT_SPLIT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=[c.PRIMARY_CAMERA],
                        choices=list(c.CAMERA_IDS))
    parser.add_argument("--start-s", type=float, default=0.0,
                        help="window start, seconds after the fitting start")
    parser.add_argument("--end-s", type=float, default=None,
                        help="window end, seconds after the fitting start (default: fit end)")
    parser.add_argument("--window", type=float, nargs=2, action="append", metavar=("START_S", "END_S"),
                        help="repeatable fitting window in seconds after the fit start; "
                             "overrides --start-s/--end-s")
    parser.add_argument("--max-keyframes", type=int, default=0, help="per camera; 0 = no cap")
    parser.add_argument("--travel-m", type=float, default=4.0)
    parser.add_argument("--heading-deg", type=float, default=5.0)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--skip-bag-hash", action="store_true")
    args = parser.parse_args(argv)
    build(args)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()

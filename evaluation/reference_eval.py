"""Local evaluation only. Never import this module into the camera estimator."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile

import numpy as np

sys.dont_write_bytecode = True
import native_bridge as BRIDGE

ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / "23-00-38_cut_120m_update"
DEFAULT_REFERENCE = ROOT / "evaluation/reference_reconciled"
STATE = "/constructor0/state_estimation"
CAMERAS = {f"/constructor0/sensor/camera_{c}/compressed_image": c
           for c in ("fl", "fr", "cl", "cr", "r", "rl", "rr")}
POSE = ("timestamp_ns", "x", "y", "yaw")
NATIVE_LIMIT_M = 0.001  # Organizer's <0.1 mm claim, with a conservative numerical margin.


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


TRACK = module("organizer_track_frame", UPDATE / "tools/track_frame.py")
OLD = module("organizer_original_evaluate", ROOT / "tools/evaluate.py")
NEW = module("organizer_updated_evaluate", UPDATE / "tools/evaluate.py")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path, required=POSE):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not set(required) <= set(header):
            raise ValueError(f"{path}: missing required columns or duplicate header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    values = {k: [] for k in header}
    for line, row in enumerate(rows, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"{path}:{line}: malformed row")
        stamp = row["timestamp_ns"]
        if not re.fullmatch(r"[0-9]+", stamp) or not 0 <= int(stamp) <= np.iinfo(np.int64).max:
            raise ValueError(f"{path}:{line}: timestamp must be an int64 decimal integer")
        for k in header:
            value = int(stamp) if k == "timestamp_ns" else float(row[k])
            if not math.isfinite(value):
                raise ValueError(f"{path}:{line}: non-finite {k}")
            values[k].append(value)
    result = {k: np.asarray(v, dtype=np.int64 if k == "timestamp_ns" else float)
              for k, v in values.items()}
    if np.any(np.diff(result["timestamp_ns"]) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing; duplicates rejected")
    return result


def write_csv(path, columns):
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in zip(*columns.values(), strict=True):
            writer.writerow([str(int(row[0]))] + [format(float(v), ".17g") for v in row[1:]])


def stats(values):
    return {"min": float(np.min(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def project(columns, lines):
    out, audit = {k: columns[k] for k in POSE}, {}
    if "lane" in columns:
        out["lane"] = columns["lane"]
    steps = np.hypot(np.diff(columns["x"]), np.diff(columns["y"]))
    # shortcut: CSVs lack recovery events; restart on >100 m jumps, use explicit events when available.
    resets = np.flatnonzero((steps > 100) & (np.diff(columns["timestamp_ns"]) <= TRACK.MAX_DT_NS)) + 1
    boundaries = np.r_[0, resets, len(columns["x"])]
    for route, suffix in (("track", ""), ("pit", "_pit")):
        line = lines[route]
        parts = [line.xy_to_sn(columns["x"][a:b], columns["y"][a:b], columns["yaw"][a:b],
                              columns["timestamp_ns"][a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])]
        s, n, epsi = [np.concatenate([p[j] for p in parts]) for j in range(3)]
        out.update({"s" + suffix: s, "n" + suffix: n, "epsi" + suffix: epsi})
        x, y, _ = line.sn_to_xy(s, n, epsi)
        audit[route] = stats(np.hypot(x - columns["x"], y - columns["y"]))
        audit[route]["recovery_resets"] = len(resets)
    return out, audit


def native_compatibility(state, lines):
    audit = {}
    for route, suffix, lane in (("track", "", 1), ("pit", "_pit", 0)):
        x, y, _ = lines[route].sn_to_xy(state["s" + suffix], state["n" + suffix])
        error = np.hypot(x - state["x"], y - state["y"])
        active = state["lane"] == lane
        audit[route] = {"all_rows_m": stats(error),
                        "active_lane_m": stats(error[active]) if active.any() else None,
                        "compatible": bool(np.isfinite(error).all() and np.max(error) <= NATIVE_LIMIT_M)}
    return {"limit_m": NATIVE_LIMIT_M, "enabled": all(v["compatible"] for v in audit.values()),
            "routes": audit}


def cdr_stamp(data):
    # Only the fixed ROS 2 CDR Header prefix is read; JPEG bytes are never decoded here.
    if len(data) < 12 or data[:2] not in (b"\x00\x00", b"\x00\x01"):
        raise ValueError("Unsupported camera CDR encoding")
    sec, ns = struct.unpack_from("<iI" if data[1] else ">iI", data, 4)
    if sec < 0 or ns >= 1_000_000_000:
        raise ValueError("Invalid camera exposure timestamp")
    return sec * 1_000_000_000 + ns


def extract(bag, lines):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    tf, state = [], []
    stamps = {c: [] for c in CAMERAS.values()}
    decoders, checked_cameras = {}, set()
    factory = DecoderFactory()
    with open(bag, "rb") as f:
        reader = make_reader(f, validate_crcs=True)
        for schema, channel, message in reader.iter_messages(topics=["/tf", STATE, *CAMERAS]):
            if channel.topic in CAMERAS:
                if channel.message_encoding != "cdr" or schema.name != "sensor_msgs/msg/CompressedImage":
                    raise ValueError("Unexpected camera schema/encoding")
                t = cdr_stamp(message.data)
                if channel.id not in checked_cameras:
                    m = factory.decoder_for(channel.message_encoding, schema)(message.data)
                    if t != m.header.stamp.sec * 10**9 + m.header.stamp.nanosec:
                        raise ValueError("Camera Header prefix disagrees with ROS decoder")
                    checked_cameras.add(channel.id)
                stamps[CAMERAS[channel.topic]].append(t)
                continue
            if channel.id not in decoders:
                decoders[channel.id] = factory.decoder_for(channel.message_encoding, schema)
            m = decoders[channel.id](message.data)
            if channel.topic == "/tf":
                for tr in m.transforms:
                    if tr.header.frame_id == "map" and tr.child_frame_id == "constructor0/base_link":
                        q, p = tr.transform.rotation, tr.transform.translation
                        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
                        tf.append([tr.header.stamp.sec * 10**9 + tr.header.stamp.nanosec,
                                   p.x, p.y, yaw])
            else:
                sm = m.sn_map_state
                row = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
                       m.x_m, m.y_m, m.yaw_rad, int(sm.current_lane)]
                for route, st in (("track", sm.track_sn_state.sn_state),
                                  ("pit", sm.pitlane_sn_state.sn_state)):
                    line = lines[route]
                    if not 0 <= st.idx < len(line.p):
                        raise ValueError(f"Native {route} segment index is invalid")
                    row += [float(line.from_car(st.idx, st.ds)), st.n,
                            float(TRACK.wrap(m.yaw_rad - line.heading[st.idx]))]
                row += [int(sm.track_sn_state.sn_state.idx), sm.track_sn_state.sn_state.ds,
                        int(sm.pitlane_sn_state.sn_state.idx), sm.pitlane_sn_state.sn_state.ds]
                state.append(row)
    def columns(rows, header):
        return {k: np.asarray([r[j] for r in rows], dtype=np.int64 if j == 0 else float)
                for j, k in enumerate(header)}
    native_header = [*POSE, "lane", "s", "n", "epsi", "s_pit", "n_pit", "epsi_pit",
                     "track_idx", "track_ds", "pit_idx", "pit_ds"]
    return columns(tf, POSE), columns(state, native_header), {k: np.asarray(v, np.int64) for k, v in stamps.items()}


def capture(call, *args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        call(*args)
    return output.getvalue()


def tools_identity(map_path):
    paths = [map_path, ROOT / "tools/evaluate.py", ROOT / "tools/extract_gt.py",
             UPDATE / "tools/track_frame.py", UPDATE / "tools/evaluate.py",
             UPDATE / "tools/convert.py", UPDATE / "tools/extract_gt.py",
             ROOT / "task_docs/build_split.json"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def prepare(args):
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}; use a fresh directory")
    lines = TRACK.load_lines(args.map)
    identities = tools_identity(args.map)
    print("Reading GT and camera exposure stamps from the complete recording...", flush=True)
    tf, native, stamps = extract(args.bag, lines)
    publish_reference(args, tf, native, stamps, lines, identities)


def reconcile(args):
    """Upgrade the validated raw cache without re-decoding camera or state messages."""
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}; use a fresh directory")
    previous = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in previous["input_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Cached source dependency changed: {name}")
    for name, expected in previous["files"].items():
        if sha256(args.reference / name) != expected:
            raise ValueError(f"Cached source artifact changed: {name}")
    args.bag, args.map = ROOT / previous["source_bag"]["path"], ROOT / previous["map_path"]
    if sha256(args.bag) != previous["source_bag"]["sha256"]:
        raise ValueError("Cached source bag identity changed")
    tf = read_csv(args.reference / "gt_tf.csv")
    native = read_csv(args.reference / "gt_state_native_UNSAFE.csv")
    with np.load(args.reference / "camera_stamps.npz", allow_pickle=False) as archive:
        stamps = {k: archive[k] for k in archive.files}
    publish_reference(args, tf, native, stamps, TRACK.load_lines(args.map), tools_identity(args.map))


def publish_reference(args, tf, native, stamps, lines, identities):
    print(f"Reconciling {len(native['x'])} recorded native poses with the supplied map...", flush=True)
    split = json.loads((ROOT / "task_docs/build_split.json").read_text(encoding="utf-8"))
    fit_mask = ((native["timestamp_ns"] >= int(split["fit"]["start_ns"])) &
                (native["timestamp_ns"] < int(split["fit"]["end_ns"])))
    frames = BRIDGE.fit(native, fit_mask)
    recovered, cross_route = BRIDGE.reconstruct(native, frames)
    error = np.hypot(recovered["x"]-native["x"], recovered["y"]-native["y"])
    if np.max(error) > BRIDGE.MAX_ERROR_M:
        raise ValueError("Frozen native bridge fails independent position reconstruction")
    projected, reconstruction = project(recovered, lines)
    direct, _ = project(native, lines)
    agreement = {}
    for route, suffix in BRIDGE.ROUTES:
        ds = (projected["s"+suffix]-direct["s"+suffix]+lines[route].lap/2) % lines[route].lap-lines[route].lap/2
        agreement[route] = {"max_s_m": float(np.max(abs(ds))),
                            "max_n_m": float(np.max(abs(projected["n"+suffix]-direct["n"+suffix]))),
                            "max_epsi_rad": float(np.max(abs(TRACK.wrap(projected["epsi"+suffix]-direct["epsi"+suffix]))))}
        if max(agreement[route].values()) > 1e-6:
            raise ValueError(f"{route}: bridge disagrees with direct supplied-map coordinate conversion")
    verification = {"all_rows_xy_m": stats(error), "track_pit_agreement_m": stats(cross_route),
                    "direct_supplied_map_projection_agreement": agreement, "episodes": {}}
    for name in ("fit", "tune", "holdout"):
        mask = ((native["timestamp_ns"] >= int(split[name]["start_ns"])) &
                (native["timestamp_ns"] < int(split[name]["end_ns"])))
        verification["episodes"][name] = {"rows": int(mask.sum()), "xy_m": stats(error[mask])}
    verification["native_frames"] = {r: {"count": len(frames[r+"_origin"]),
                                           "min_fitting_observations": int(frames[r+"_fit_count"].min()),
                                           "scale_min": float(abs(frames[r+"_rotation"]).min()),
                                           "scale_max": float(abs(frames[r+"_rotation"]).max())} for r, _ in BRIDGE.ROUTES}
    native_gate = native_compatibility(native, lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=args.out.parent) as temp:
        dest = Path(temp) / "ready"
        dest.mkdir()
        raw = {k: native[k] for k in (*POSE, "lane", "track_idx", "track_ds", "n", "pit_idx", "pit_ds", "n_pit")}
        for name, data in (("gt_tf.csv", tf), ("source_state_native.csv", raw),
                           ("gt_state_reconciled.csv", projected)):
            write_csv(dest / name, data)
            read_csv(dest / name)
        np.savez_compressed(dest / "native_reference_bridge.npz", **frames)
        (dest / "reconciliation.json").write_text(json.dumps(verification, indent=2, allow_nan=False), encoding="utf-8")
        for name, ts in stamps.items():
            if len(ts) < 2 or np.any(np.diff(ts) <= 0):
                raise ValueError(f"Camera {name}: missing or nonmonotonic exposure stamps")
        np.savez_compressed(dest / "camera_stamps.npz", **stamps)
        tf_file, local_file = dest / "gt_tf.csv", dest / "gt_state_reconciled.csv"
        old = capture(OLD.main, tf_file, tf_file)
        updated = capture(NEW.main, tf_file, tf_file, lines["track"].lap, lines["pit"].lap)
        local = capture(NEW.main, local_file, local_file, lines["track"].lap, lines["pit"].lap)
        if "position RMSE  [m]     0.000" not in old or updated.count("<- SCORE") != 1 or local.count("<- SCORE") != 3:
            raise ValueError("Organizer scorer plumbing check failed")
        (dest / "PLUMBING_ONLY.txt").write_text(
            "GT compared with itself. NOT CAMERA LOCALIZATION RESULTS.\n\n" + old + "\n" + updated + "\n" + local,
            encoding="utf-8")
        manifest = {
            "profile": "recording native idx/ds/n -> frozen fitting-only segment frames -> supplied map s/n/epsi; locally reconciled GT",
            "source_bag": {"path": args.bag.relative_to(ROOT).as_posix(), "sha256": sha256(args.bag)},
            "map_path": args.map.relative_to(ROOT).as_posix(), "input_hashes": identities,
            "adapter_sha256": sha256(Path(__file__)),
            "bridge_code_sha256": sha256(ROOT / "evaluation/native_bridge.py"),
            "native_coordinate_bridge": {"enabled": True, "fitting_interval_ns": [split["fit"]["start_ns"], split["fit"]["end_ns"]],
                                          "reconstruction_limit_m": BRIDGE.MAX_ERROR_M,
                                          "source_fields": "track_idx,track_ds,n,pit_idx,pit_ds,n_pit; source x/y used only for fitting and verification",
                                          "target_map_sha256": sha256(args.map)},
            "projection": "unchanged organizer sequence helper; restart globally after >100 m recovery jumps or native >1 s gaps",
            "tf_rows": len(tf["x"]), "state_rows": len(native["x"]),
            "camera_counts": {k: len(v) for k, v in stamps.items()},
            "native_coordinate_gate": native_gate, "projected_roundtrip_m": reconstruction,
            "lap_lengths_m": {k: v.lap for k, v in lines.items()},
            "packages": {k: importlib.metadata.version(k) for k in ("numpy", "mcap", "mcap-ros2-support", "lz4", "zstandard")},
            "precision": "int64 nanoseconds; full double-precision poses; no GT alignment or sample removal",
            "plumbing_only": True,
            "files": {p.name: sha256(p) for p in dest.iterdir()},
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        if identities != tools_identity(args.map):
            raise ValueError("An organizer input changed during preparation")
        dest.rename(args.out)
    print(f"Reconciled reference prepared: {args.out}\nNative coordinate bridge enabled: True\nUnconverted index reuse enabled: {native_gate['enabled']}", flush=True)


def validate_submission(sub, gt, stamps):
    ts, tg = sub["timestamp_ns"], gt["timestamp_ns"]
    if ts[0] < tg[0] or ts[-1] > tg[-1]:
        raise ValueError("Submission lies outside the predetermined GT episode; no automatic trimming")
    if not np.isin(ts, stamps).all():
        raise ValueError("Submission contains timestamps absent from all seven camera exposure streams")
    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= tg[i] - tg[i - 1] <= NEW.MAX_GT_GAP_NS
    if not ok.any():
        raise ValueError("No valid GT interpolation brackets")
    bins = np.arange(tg[0], tg[-1], NEW.BIN_NS)
    coverage = len(np.unique(np.searchsorted(bins, ts[ok], side="right") - 1)) / max(len(bins), 1)
    if coverage < NEW.MIN_COVERAGE:
        raise ValueError(f"Coverage {coverage:.1%} is below organizer minimum 90%")
    return {"submitted": len(ts), "evaluated": int(ok.sum()), "coverage": coverage,
            "median_interval_ms": float(np.median(np.diff(ts)) / 1e6) if len(ts) > 1 else None,
            "max_interval_ms": float(np.max(np.diff(ts)) / 1e6) if len(ts) > 1 else None,
            "causality": "Exposure membership checked; inference input logs and prefix tests still required."}


def score(args):
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}; use a fresh directory")
    manifest = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("adapter_sha256") != sha256(Path(__file__)):
        raise ValueError("Changed evaluation adapter; revalidate and rebuild the reference")
    if manifest.get("bridge_code_sha256") != sha256(ROOT / "evaluation/native_bridge.py"):
        raise ValueError("Changed native coordinate bridge; revalidate and rebuild the reference")
    if not manifest.get("native_coordinate_bridge", {}).get("enabled"):
        raise ValueError("Native reference reconciliation is required")
    if {k: importlib.metadata.version(k) for k in manifest["packages"]} != manifest["packages"]:
        raise ValueError("Changed evaluation package versions; use the pinned environment")
    for name, expected in manifest["input_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Changed map, scorer or split: {name}; rebuild the reference")
    for name, expected in manifest["files"].items():
        if sha256(args.reference / name) != expected:
            raise ValueError(f"Changed reference artifact: {name}")
    tf = read_csv(args.reference / "gt_tf.csv")
    state = read_csv(args.reference / "gt_state_reconciled.csv")
    split = json.loads((ROOT / "task_docs/build_split.json").read_text(encoding="utf-8"))
    if args.split == "full":
        start, end = int(split["first_gt_timestamp_ns"]), int(split["last_gt_timestamp_ns"]) + 1
    else:
        start, end = int(split[args.split]["start_ns"]), int(split[args.split]["end_ns"])
    tf, state = [{k: v[(c["timestamp_ns"] >= start) & (c["timestamp_ns"] < end)] for k, v in c.items()}
                 for c in (tf, state)]
    if min(len(tf["x"]), len(state["x"])) < 2:
        raise ValueError("Episode has insufficient reference rows")
    sub = read_csv(args.submission)
    with np.load(args.reference / "camera_stamps.npz", allow_pickle=False) as cameras:
        stamps = np.concatenate([cameras[k] for k in cameras.files])
    validation = validate_submission(sub, tf, stamps)
    validate_submission(sub, state, stamps)
    lines = TRACK.load_lines(ROOT / manifest["map_path"])
    projected, projection_audit = project(sub, lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".score-", dir=args.out.parent) as temp:
        dest = Path(temp) / "ready"
        dest.mkdir()
        for name, data in (("episode_tf.csv", tf), ("episode_state_reconciled.csv", state),
                           ("submission_xy.csv", {k: sub[k] for k in POSE}),
                           ("submission_supplied_map.csv", projected)):
            write_csv(dest / name, data)
        outputs = {
            "original_xy.txt": capture(OLD.main, dest / "episode_tf.csv", dest / "submission_xy.csv"),
            "updated_xy.txt": capture(NEW.main, dest / "episode_tf.csv", dest / "submission_xy.csv",
                                      lines["track"].lap, lines["pit"].lap),
            "reconciled_supplied_map.txt": capture(NEW.main, dest / "episode_state_reconciled.csv",
                                           dest / "submission_supplied_map.csv",
                                           lines["track"].lap, lines["pit"].lap),
        }
        for name, report in outputs.items():
            expected = 2 if name == "reconciled_supplied_map.txt" else 1
            if report.count("<- SCORE") < expected or re.search(r"\b(?:nan|inf)\b", report.lower()):
                raise ValueError(f"Missing or non-finite metric section: {name}")
            (dest / name).write_text(report, encoding="utf-8")
        summary = {"split": args.split, "interval_ns": [str(start), str(end)], "validation": validation,
                   "reference_manifest_sha256": sha256(args.reference / "manifest.json"),
                   "submission_sha256": sha256(args.submission), "projected_roundtrip_m": projection_audit,
                   "unconverted_native_indices_used": False, "native_coordinate_bridge_used": True,
                   "raw_native_fields_used_for_GT": True,
                   "supplied_map_sha256": manifest["native_coordinate_bridge"]["target_map_sha256"],
                   "additional_profile": manifest["profile"]}
        (dest / "validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        dest.rename(args.out)
    print(outputs["original_xy.txt"])
    print(f"Saved both x/y reports and the reconciled supplied-map track-coordinate report to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Extract reference data and gate native-coordinate compatibility")
    p.add_argument("--bag", type=Path, default=ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap")
    p.add_argument("--map", type=Path, default=UPDATE / "map.mcap")
    p.add_argument("--out", type=Path, default=DEFAULT_REFERENCE)
    p.set_defaults(run=prepare)
    p = commands.add_parser("reconcile", help="Reconcile the original validated raw cache with the new map")
    p.add_argument("--reference", type=Path, default=ROOT / "evaluation/reference")
    p.add_argument("--out", type=Path, default=DEFAULT_REFERENCE)
    p.set_defaults(run=reconcile)
    p = commands.add_parser("score", help="Validate and score x/y/yaw; recompute local track coordinates")
    p.add_argument("submission", type=Path)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--split", choices=("fit", "tune", "holdout", "full"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=score)
    args = parser.parse_args()
    for field in ("bag", "map", "out", "submission", "reference"):
        if hasattr(args, field):
            setattr(args, field, getattr(args, field).resolve())
    try:
        args.run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Evaluation stopped: {exc}\n")


if __name__ == "__main__":
    main()

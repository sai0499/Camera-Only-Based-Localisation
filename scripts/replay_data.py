"""Passive, offline UI adapters. Never imported by the inference package."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


class Lines:
    """Incremental reader: an incomplete live line is retried without losing bytes."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0

    def read(self, finished=False):
        if not self.path.exists():
            return
        with self.path.open("rb") as handle:
            if self.path.stat().st_size < self.offset:
                raise ValueError(f"Output truncated: {self.path.name}")
            handle.seek(self.offset)
            while line := handle.readline(16_000_001):
                if len(line) > 16_000_000:
                    raise ValueError(f"Oversized record: {self.path.name}")
                if not line.endswith(b"\n"):
                    if finished:
                        raise ValueError(f"Incomplete final record: {self.path.name}")
                    break
                yield line.decode("utf-8").strip()
                self.offset = handle.tell()


def pose_details(record):
    if not record:
        return None
    pose = record.get("pose") if record.get("record") == "HandoffEvidence" else record
    pose = pose or {}
    keys = ("source", "camera_id", "inlier_count", "inlier_fraction", "median_error_px",
            "p90_error_px", "occupied_cells", "accepted", "reason")
    return {**{key: pose.get(key) for key in keys},
            "source_ns": str(record.get("source_exposure_ns", pose.get("exposure_ns", ""))),
            "measurement_ns": str(record.get("measurement_ns", pose.get("exposure_ns", "")))}


class Replay:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.readers = {name: Lines(self.folder / name) for name in
                        ("submission.csv", "diagnostics.jsonl", "runtime.jsonl")}
        self.frames = []
        self.estimates = {}
        self.groups = {}
        self.runtime = {}
        self.events = []
        self.gt = {}
        self.comparison = None
        self.evaluation = None
        self.meta = {}
        self.indices = {}
        self.group_stamps = {}
        self.updated = set()
        self.current_group = None

    def refresh(self, state=None):
        self.updated = set()
        self.meta = read_json(self.folder / "run.json", {})
        failure = read_json(self.folder / "failure.json")
        self.state = state or ("failed" if failure else
                               "completed" if "groups" in self.meta else "incomplete")
        self.failure = failure
        finished = self.state == "completed"
        # Diagnostics are joined by exact integer exposure/group, never row proximity.
        for line in self.readers["diagnostics.jsonl"].read(finished):
            r = json.loads(line)
            if not isinstance(r, dict) or not isinstance(r.get("record"), str):
                raise ValueError("Invalid diagnostic record")
            kind, gid = r["record"], r.get("group_id")
            if kind == "FrameSet":
                self.current_group = gid
            elif kind == "Estimate" and gid is not None:
                self.estimates[int(r["t_ns"])] = r
                self.updated.add(int(r["t_ns"]))
            elif kind == "GroupEvidence":
                fs = r["frame_set"]
                self.current_group = fs["group_id"]
                self.group_stamps[fs["group_id"]] = int(fs["t_ns"])
                self.updated.add(int(fs["t_ns"]))
                self.groups[int(fs["t_ns"])] = {
                    "group_id": fs["group_id"], "pose": pose_details(r.get("pose")),
                    "missing": fs.get("missing_camera_ids", []),
                    "images": [{"camera_id": x["camera_id"], "exposure_ns": str(x["exposure_ns"]),
                                "arrival_ns": str(x["arrival_ns"])}
                               for x in fs.get("images", [])],
                    "rejected": [pose_details(x) for x in r.get("rejected", [])
                                 if x.get("record") in ("PoseEvidence", "HandoffEvidence")],
                }
            elif kind in ("RecoveryEvent", "TrackerEvent", "CameraSelection") or kind.endswith("Error"):
                event_group = self.current_group if kind == "TrackerEvent" and gid is None else gid
                self.events.append({"group_id": event_group, "kind": r.get("kind", r.get("event", kind)),
                                    "reason": ("decoded=" + ",".join(r.get("decoded", [])) +
                                        " / proposed=" + ",".join(r.get("proposed", [])))
                                        if kind == "CameraSelection" else r.get("reason", r.get("detail", ""))})
        for line in self.readers["runtime.jsonl"].read(finished):
            r = json.loads(line)
            if r.get("record") == "RuntimeSample":
                self.runtime[r["group_id"]] = {
                    "processing_ms": (r["reader_ns"] + r["group_service_ns"]) / 1e6}
                if r["group_id"] in self.group_stamps:
                    self.updated.add(self.group_stamps[r["group_id"]])
        for line in self.readers["submission.csv"].read(finished):
            if line == "timestamp_ns,x,y,yaw":
                continue
            row = next(csv.reader([line]))
            if len(row) != 4 or not row[0].isdecimal():
                raise ValueError("Invalid submission row")
            stamp, values = int(row[0]), list(map(float, row[1:]))
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Nonfinite submitted pose")
            if self.frames and stamp <= int(self.frames[-1]["t_ns"]):
                raise ValueError("Submission timestamps must increase")
            self.indices[stamp] = len(self.frames)
            self.frames.append({"index": len(self.frames), "t_ns": str(stamp),
                                "x": values[0], "y": values[1], "yaw": values[2]})
        if finished and self.state == "completed" and len(self.frames) != self.meta["groups"]:
            raise ValueError("Completed run count differs from submission")

    def frame(self, index):
        row = self.frames[index]
        stamp = int(row["t_ns"])
        est, group = self.estimates.get(stamp, {}), self.groups.get(stamp, {})
        aux = est.get("auxiliary") or {}
        z = (aux.get("z_m") if aux.get("valid") and aux.get("source_ns") is not None
             and int(aux["source_ns"]) <= stamp else None)
        return {**row, **group, **self.runtime.get(group.get("group_id"), {}),
                "status": est.get("status", "unavailable"),
                "reason": est.get("reason", "diagnostics_pending"),
                "evidence": est.get("evidence_kinds", []),
                "z": z, "height_source": aux.get("source"),
                "height_source_ns": str(aux["source_ns"]) if aux.get("source_ns") is not None else None,
                **self.gt.get(stamp, {})}

    def payload(self, cursor=0):
        origin = self.meta.get("initial_pose", {}).get("timestamp_ns")
        if origin is None and self.frames:
            origin = self.frames[0]["t_ns"]
        indices = sorted(set(range(cursor, len(self.frames))) |
                         {self.indices[s] for s in self.updated if s in self.indices})
        return {"schema": 1, "state": self.state, "failure": self.failure,
                "meta": self.meta, "origin_ns": str(origin) if origin is not None else "0",
                "count": len(self.frames), "from": cursor,
                "frames": [self.frame(i) for i in indices],
                "events": self.events[-250:], "comparison": self.comparison,
                "evaluation": self.evaluation}

    def attach_gt(self, path):
        """Explicit offline action, after completion; no GT path reaches a child process."""
        if self.state != "completed":
            raise ValueError("Attach ground truth only after the run has completed")
        from scripts.export_initial_pose import verify_manifest
        from scripts.report_step5_errors import errors

        path = Path(path)
        digest = verify_manifest(path) or hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
                raise ValueError("Duplicate GT columns")
            rows = list(reader)
        gt = {k: np.array([int(r[k]) if k == "timestamp_ns" else float(r[k]) for r in rows],
                         dtype=np.int64 if k == "timestamp_ns" else float)
              for k in ("timestamp_ns", "x", "y", "yaw")}
        if len(rows) < 2 or np.any(np.diff(gt["timestamp_ns"]) <= 0):
            raise ValueError("GT must have at least two strictly increasing poses")
        if not all(np.isfinite(v).all() for v in gt.values()):
            raise ValueError("GT contains nonfinite poses")
        bounds = self.meta.get("episode", {})
        if bounds.get("end_ns") is not None:
            keep = ((gt["timestamp_ns"] >= int(bounds["start_ns"])) &
                    (gt["timestamp_ns"] < int(bounds["end_ns"])))
            gt = {k: v[keep] for k, v in gt.items()}
        if len(gt["timestamp_ns"]) < 2:
            raise ValueError("GT does not support the declared episode")
        sub = {k: np.array([int(r["t_ns"]) if k == "timestamp_ns" else r[k]
                           for r in self.frames], dtype=np.int64 if k == "timestamp_ns" else float)
               for k in ("timestamp_ns", "x", "y", "yaw")}
        stamps, pos, yaw = errors(gt, sub)
        if not len(stamps):
            raise ValueError("No timestamp-matched GT with <=50 ms brackets")
        i = np.searchsorted(gt["timestamp_ns"], stamps)
        a = (stamps - gt["timestamp_ns"][i - 1]) / (gt["timestamp_ns"][i] - gt["timestamp_ns"][i - 1])
        gx, gy, heading = [v[i - 1] + a * (v[i] - v[i - 1])
                           for v in (gt["x"], gt["y"], np.unwrap(gt["yaw"]))]
        indices = np.searchsorted(sub["timestamp_ns"], stamps)
        dx, dy = sub["x"][indices] - gx, sub["y"][indices] - gy
        self.gt = {int(s): {"gt_x": float(x), "gt_y": float(y), "error_m": float(e),
                            "yaw_error_deg": float(h),
                            "lateral_error_m": float(-np.sin(theta) * ex + np.cos(theta) * ey),
                            "longitudinal_error_m": float(np.cos(theta) * ex + np.sin(theta) * ey)}
                   for s, x, y, e, h, theta, ex, ey in zip(
                       stamps, gx, gy, pos, yaw, heading, dx, dy, strict=True)}
        self.comparison = {"reference": path.name, "sha256": digest, "matched": len(stamps),
                           "outputs": len(self.frames), "rmse_m": float(np.sqrt(np.mean(pos**2))),
                           "median_m": float(np.median(pos)), "p95_m": float(np.percentile(pos, 95)),
                           "scope": "offline timestamp-matched outputs; not an official score"}


def scene_geometry():
    """Only the hash-bound static border topic is decoded; no vehicle GT."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    from camloc import contracts as c

    path = ROOT / "23-00-38_cut_120m_update/map.mcap"
    if hashlib.sha256(path.read_bytes()).hexdigest() != c.SUPPLIED_MAP_SHA256:
        raise ValueError("Static map identity changed")
    with path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, _, _, decoded in reader.iter_decoded_messages(topics=["/constructor0/map_markers/rr"]):
            return [{"name": str(m.ns), "xyz": [[p.x, p.y, p.z] for p in m.points]}
                    for m in decoded.markers if len(m.points) > 1]
    raise ValueError("No static surveyed borders")


def wire(value):
    """Keep absolute nanoseconds exact across the browser JSON boundary."""
    if isinstance(value, dict):
        return {k: str(v) if (k.endswith("_ns") and isinstance(v, int)) else wire(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [wire(v) for v in value]
    return value

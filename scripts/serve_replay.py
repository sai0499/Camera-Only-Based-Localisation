"""Local-only run launcher and passive replay dashboard. No inference GT access."""

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import secrets
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.replay_data import Replay, read_json, scene_geometry, wire  # noqa: E402


DEFAULT_GT = "evaluation/reference_reconciled/gt_tf.csv"
BROWSE_KINDS = {
    "bag": ("file", "Select a recording", [("MCAP recording", "*.mcap"), ("All files", "*.*")], "."),
    "map": ("dir", "Select a frozen landmark-map directory", [], "artifacts"),
    "pose": ("file", "Select an initial-pose JSON", [("Initial pose JSON", "*.json"), ("All files", "*.*")], "."),
    "gt": ("file", "Select a ground-truth CSV", [("Ground-truth CSV", "*.csv"), ("All files", "*.*")],
           "evaluation/reference_reconciled"),
}
_DIALOG = (
    "import json,sys,tkinter as tk\nfrom tkinter import filedialog\n"
    "a=json.loads(sys.argv[1]);r=tk.Tk();r.withdraw();r.attributes('-topmost',True)\n"
    "p=(filedialog.askdirectory(parent=r,initialdir=a['dir'],title=a['title'],mustexist=True)"
    " if a['mode']=='dir' else filedialog.askopenfilename(parent=r,initialdir=a['dir'],"
    "title=a['title'],filetypes=[tuple(t) for t in a['types']]))\n"
    "print(json.dumps({'path':p or ''}))")


def browse(kind, root=ROOT):
    """Open a native local file dialog in a child process (the server is localhost-only)."""
    if kind not in BROWSE_KINDS:
        raise ValueError("Unknown browse target")
    mode, title, types, start = BROWSE_KINDS[kind]
    folder = Path(root) / start
    args = json.dumps({"mode": mode, "title": title, "types": types,
                       "dir": str(folder if folder.is_dir() else root)})
    done = subprocess.run([sys.executable, "-c", _DIALOG, args], capture_output=True, text=True,
                          timeout=900, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if done.returncode != 0 or not done.stdout.strip():
        raise ValueError("The file dialog could not be opened")
    chosen = json.loads(done.stdout.strip().splitlines()[-1])["path"]
    if not chosen:
        return {"path": ""}
    path = Path(chosen).resolve()
    shown = str(path.relative_to(Path(root).resolve())) if path.is_relative_to(Path(root).resolve()) else str(path)
    result = {"path": shown}
    if kind == "pose":
        if path.stat().st_size > 16_384:
            raise ValueError("Initial-pose JSON is too large")
        result["content"] = path.read_text(encoding="utf-8-sig")
    return result


def within(path, root):
    path = Path(path).resolve()
    roots = root if isinstance(root, tuple) else (root,)
    if not any(path.is_relative_to(Path(item).resolve()) for item in roots):
        raise ValueError("Path is outside the selected directory")
    return path


class Workspace:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.results = self.root / "results"
        self.replay_cache = self.root / "tmp" / "replay_ui"
        self.replay_roots = (self.results, self.replay_cache)
        self.lock = threading.RLock()
        self.replays = {}
        self.folders = {}
        self.process = None
        self.active = None
        self.geometry = None
        self.image_cache = {}

    def runs(self):
        found = [(path, base) for base in self.replay_roots
                 for pattern in ("*/run.json", "*/*/run.json") for path in base.glob(pattern)]
        items = []
        for path, base in sorted(found, key=lambda item: item[0].stat().st_mtime, reverse=True):
            folder = within(path.parent, base)
            key = hashlib.sha256(str(folder).encode()).hexdigest()[:16]
            self.folders[key] = folder
            try:
                meta = read_json(path)
                name = str(folder.relative_to(base))
                items.append({"id": key, "name": name if base == self.results else "preview/" + name,
                              "split": meta.get("split"), "groups": meta.get("groups"),
                              "running": key == self.active and self.process.poll() is None})
            except (ValueError, OSError):
                continue
        return items

    def replay(self, key):
        if key not in self.folders:
            raise ValueError("Unknown run")
        if key not in self.replays:
            self.replays[key] = Replay(self.folders[key])
        replay = self.replays[key]
        state = None
        if key == self.active:
            code = self.process.poll()
            state = "running" if code is None else "completed" if code == 0 else "failed"
        replay.refresh(state)
        self.auto_gt(replay)
        return replay

    def auto_gt(self, replay):
        """Development splits have reference GT: attach it offline once the run completes."""
        automatic = self.replay_cache / "inputs" / replay.folder.name / "inspection.json"
        if automatic.is_file():
            state = read_json(replay.folder / "evaluation_status.json", {"state": "pending"})
            if replay.state == "failed":
                state = {"state": "failed", "reason": "Localization failed; inspect the run diagnostics."}
            replay.evaluation = {k: v for k, v in state.items() if k != "gt"}
            if (replay.state == "completed" and state.get("state") == "completed"
                    and replay.comparison is None):
                replay.attach_gt(Path(state["gt"]))
            return
        if (replay.state != "completed" or replay.comparison is not None or getattr(replay, "gt_tried", False)
                or replay.meta.get("split") not in ("tune", "holdout")):
            return
        replay.gt_tried = True
        path = self.root / DEFAULT_GT
        if path.is_file():
            try:
                replay.attach_gt(path)
            except (ValueError, OSError):
                replay.comparison = None

    def config(self):
        runs = self.runs()
        completed = next((read_json(self.folders[r["id"]] / "run.json") for r in runs
                          if r["groups"] and r["split"] == "tune"), {})
        banks = [str(p.parent.relative_to(self.root)) for p in (self.root / "artifacts").glob("*/manifest.json")
                 if (p.parent / "map.npz").exists()]
        return {"runs": runs, "bags": [p.name for p in self.root.glob("*.mcap")],
                "artifacts": sorted(banks), "initial_pose": completed.get("initial_pose", {}),
                "default_gt": DEFAULT_GT,
                "default_artifacts": ("artifacts/fit-seven-rescue-v2"
                    if (self.root / "artifacts/fit-seven-rescue-v2/manifest.json").exists()
                    else "artifacts/fit-stage3-multilap-v5-rigproof-v2"
                    if (self.root / "artifacts/fit-stage3-multilap-v5-rigproof-v2/manifest.json").exists()
                    else "artifacts/fit-stage3-multilap-v5")}

    def validate(self, request):
        from camloc import contracts as c
        from run import resolve_episode

        bag = Path(request["bag"])
        bank = Path(request["artifacts"])
        bag = (self.root / bag).resolve() if not bag.is_absolute() else bag.resolve()
        bank = (self.root / bank).resolve() if not bank.is_absolute() else bank.resolve()
        if not bag.is_file() or bag.suffix.lower() != ".mcap":
            raise ValueError("Select a readable MCAP file")
        if not (bank / "map.npz").is_file():
            raise ValueError("Select a built landmark map directory")
        raw_initial = request["initial_pose"]
        initial = c.parse_initial_pose(raw_initial if isinstance(raw_initial, str) else json.dumps(raw_initial))
        mode = request.get("mode", "input")
        if mode not in ("input", "tune", "holdout"):
            raise ValueError("Unknown run mode")
        vehicle_motion = request.get("vehicle_motion") or None
        if vehicle_motion not in (None, "state-fields", "validation-twist"):
            raise ValueError("Unknown vehicle motion source")
        args = argparse.Namespace(split=None if mode == "input" else mode,
                                  start_ns=request.get("start_ns") or None,
                                  end_ns=request.get("end_ns") or None)
        episode = resolve_episode(args, initial)
        original = self.root / "c0_2026-10-13_22-53-42_camloc_0.mcap"
        if mode == "input" and bag == original.resolve():
            if episode.end_ns is None or episode.end_ns > c.load_split()["holdout"].start_ns:
                raise ValueError("For this development recording, select Tuning/Holdout or set an explicit end before holdout")
        manifest = read_json(bank / "manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("The artifact manifest is missing or invalid")
        if manifest.get("split_sha256") != c.BUILD_SPLIT_SHA256 or manifest.get("supplied_map_sha256") != c.SUPPLIED_MAP_SHA256:
            raise ValueError("Map provenance is incompatible")
        if request.get("retrieval", True) and not (bank / "retrieval.json").is_file():
            raise ValueError("This map has no retrieval index")
        # Lightweight compatibility check; production loader still verifies every asset hash.
        from camloc.io import read_calibrations
        import numpy as np

        from camloc.landmark_map import bind_calibrations

        adaptive = manifest.get("camera_policy") == "support-v1"
        calibrations = bind_calibrations(read_calibrations(bag), manifest, strict=not adaptive)
        compatible = []
        for camera in manifest["cameras"]:
            cal = calibrations[camera]
            if cal.preprocessing_id != manifest["preprocessing_id"] or not np.allclose(cal.K, manifest["canonical_K"][camera], atol=1e-8, rtol=0) or not np.allclose(
                    cal.T_BC, manifest["T_BC"][camera], atol=1e-8, rtol=0):
                if adaptive:
                    continue
                raise ValueError(f"Camera {camera} differs from the frozen map")
            compatible.append(camera)
        if not compatible:
            raise ValueError("No camera matches the frozen map")
        return {"bag": str(bag), "artifacts": str(bank), "mode": mode,
                "initial_pose": {"timestamp_ns": str(initial.timestamp_ns), "x": initial.x,
                                 "y": initial.y, "yaw": initial.yaw},
                "episode": wire(episode.summary()), "retrieval": bool(request.get("retrieval", True)),
                "vehicle_motion": vehicle_motion,
                "map_id": manifest["map_id"], "cameras": compatible}

    def upload(self, stream, length, name):
        if not 0 < length <= 64 * 1024**3 or length + 128*1024**2 > shutil.disk_usage(self.root).free:
            raise ValueError("Recording is too large for the available disk space")
        if Path(name).name != name or not name.lower().endswith(".mcap") or len(name) > 200:
            raise ValueError("Select an MCAP file")
        folder = self.replay_cache / "uploads" / secrets.token_hex(8)
        folder.mkdir(parents=True)
        path = folder / name
        try:
            with path.open("xb") as handle:
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024**2, remaining))
                    if not chunk:
                        raise ValueError("Recording upload was interrupted")
                    handle.write(chunk)
                    remaining -= len(chunk)
            from mcap.reader import make_reader
            with path.open("rb") as handle:
                if make_reader(handle).get_summary() is None:
                    raise ValueError("MCAP has no indexed summary")
        except Exception as error:
            path.unlink(missing_ok=True)
            folder.rmdir()
            if isinstance(error, (ValueError, OSError)):
                raise
            raise ValueError("The selected file is not a valid indexed MCAP recording") from error
        return {"bag": str(path.relative_to(self.root)), "name": name}

    def start_auto(self, request):
        from scripts.mcap_input import inspect_recording

        if self.process is not None and self.process.poll() is None:
            raise ValueError("A localisation process is already running")
        bag = Path(request["bag"])
        bag = bag if bag.is_absolute() else self.root / bag
        if not bag.is_file() or bag.suffix.lower() != ".mcap":
            raise ValueError("Select a readable MCAP recording")
        bank = self.root / self.config()["default_artifacts"]
        inspection = inspect_recording(bag, bank)
        return self.start({"bag": str(bag), "artifacts": str(bank), "retrieval": True,
                           **{k: inspection[k] for k in ("mode", "initial_pose", "vehicle_motion")}},
                          inspection=inspection)

    def start(self, request, inspection=None):
        if self.process is not None and self.process.poll() is None:
            raise ValueError("A localisation process is already running")
        settings = self.validate(request)
        slug = "ui-" + secrets.token_hex(5)
        folder = (self.replay_cache if inspection else self.results) / slug
        inputs = self.replay_cache / "inputs" / slug
        inputs.mkdir(parents=True, exist_ok=True)
        initial_path = inputs / "initial_pose.json"
        initial_path.write_text(json.dumps(settings["initial_pose"]), encoding="utf-8")
        command = [sys.executable, "-u", "-B", str(self.root / "run.py"),
                   "--bag", settings["bag"], "--artifacts", settings["artifacts"],
                   "--initial-pose", str(initial_path), "--output", str(folder), "--runtime"]
        if settings["mode"] == "input":
            command.append("--input-recording")
            for flag in ("start_ns", "end_ns"):
                if request.get(flag):
                    command += ["--" + flag.replace("_", "-"), str(request[flag])]
        else:
            command += ["--split", settings["mode"]]
        if settings["retrieval"]:
            command.append("--retrieval")
        if settings.get("vehicle_motion"):
            command += ["--vehicle-motion", settings["vehicle_motion"]]
        log = inputs / "localization.log"
        with log.open("wb") as handle:
            self.process = subprocess.Popen(command, cwd=self.root, stdout=handle, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        key = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16]
        self.folders[key] = folder
        self.active = key
        if inspection is not None:
            (inputs / "inspection.json").write_text(json.dumps(inspection), encoding="utf-8")
            process = self.process
            def finish():
                if process.wait() == 0:
                    from scripts.mcap_input import finish_recording
                    finish_recording(settings["bag"], folder, inputs, inspection, self.root)
            threading.Thread(target=finish, daemon=True).start()
        return {"id": key, "name": slug, "settings": settings}

    def image(self, key, index, camera):
        if camera not in ("fl", "fr", "cl", "cr", "r", "rl", "rr"):
            raise ValueError("Only selected cameras are available")
        replay = self.replay(key)
        if replay.state != "completed":
            raise ValueError("Image inspection is offline only; keep inference uncontended")
        if not 0 <= index < len(replay.frames):
            raise ValueError("Unknown output")
        row = replay.frame(index)
        image = next((x for x in row.get("images", []) if x["camera_id"] == camera), None)
        if image is None:
            raise ValueError("Camera is missing at this output")
        cache_key = (key, index, camera)
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        path = Path(replay.meta.get("query_bag_path") or self.root / replay.meta["query_bag"])
        with path.open("rb") as handle:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            arrival = int(image["arrival_ns"])
            topic = f"/constructor0/sensor/camera_{camera}/compressed_image"
            for _, _, _, decoded in reader.iter_decoded_messages(
                    topics=[topic], start_time=arrival, end_time=arrival + 1):
                stamp = decoded.header.stamp.sec * 1_000_000_000 + decoded.header.stamp.nanosec
                if stamp == int(image["exposure_ns"]):
                    data = bytes(decoded.data)
                    if not data.startswith(b"\xff\xd8"):
                        raise ValueError("Image is not a JPEG")
                    if len(self.image_cache) >= 8:
                        self.image_cache.pop(next(iter(self.image_cache)))
                    self.image_cache[cache_key] = data
                    return data
        raise ValueError("Exact recorded image was not found")

    def snapshot(self, request):
        folder = self.folders[request["id"]]
        prefix = "data:image/png;base64,"
        if not isinstance(request.get("image"), str) or not request["image"].startswith(prefix):
            raise ValueError("Expected a PNG snapshot")
        data = base64.b64decode(request["image"][len(prefix):], validate=True)
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > 4_000_000:
            raise ValueError("Invalid or oversized PNG")
        output = within(folder / "viewer-snapshots", self.replay_roots)
        output.mkdir(exist_ok=True)
        name = secrets.token_hex(8) + ".png"
        (output / name).write_bytes(data)
        return {"path": str((output / name).relative_to(self.root)),
                "url": f"/api/snapshot?id={request['id']}&file={name}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "TrackCodeLocal/1"

    def log_message(self, *_):
        pass

    def respond(self, value, status=200, mime="application/json"):
        body = (json.dumps(wire(value), allow_nan=False).encode("utf-8")
                if mime == "application/json" else value)
        self.send_response(status)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def check_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}",
                                             f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.check_host():
            return self.respond({"error": "Local host required"}, 403)
        route = urlsplit(self.path)
        query = parse_qs(route.query)
        workspace = self.server.workspace
        try:
            with workspace.lock:
                if route.path == "/api/config":
                    return self.respond({**workspace.config(), "token": self.server.token})
                if route.path == "/api/scene":
                    if workspace.geometry is None:
                        workspace.geometry = scene_geometry()
                    return self.respond(workspace.geometry)
                if route.path == "/api/run":
                    replay = workspace.replay(query["id"][0])
                    cursor = int(query.get("cursor", [0])[0])
                    if not 0 <= cursor <= len(replay.frames):
                        raise ValueError("Invalid replay cursor")
                    return self.respond(replay.payload(max(0, cursor - 64)))
                if route.path == "/api/download":
                    key, name = query["id"][0], query["file"][0]
                    if name not in ("submission.csv", "runtime.json", "run.json", "validation/score.txt", "validated.json"):
                        raise ValueError("Unknown export")
                    path = within(workspace.folders[key] / name, workspace.replay_roots)
                    return self.respond(path.read_bytes(), mime="text/plain")
                if route.path == "/api/image":
                    return self.respond(workspace.image(query["id"][0], int(query["index"][0]),
                                                        query["camera"][0]), mime="image/jpeg")
                if route.path == "/api/snapshot":
                    key, name = query["id"][0], query["file"][0]
                    if not re.fullmatch(r"[a-f0-9]{16}\.png", name):
                        raise ValueError("Unknown snapshot")
                    path = within(workspace.folders[key] / "viewer-snapshots" / name, workspace.replay_roots)
                    return self.respond(path.read_bytes(), mime="image/png")
            if route.path == "/plotly.js":
                import plotly

                return self.respond((Path(plotly.__file__).parent / "package_data/plotly.min.js").read_bytes(),
                                    mime="application/javascript")
            rel = "index.html" if route.path == "/" else route.path.lstrip("/")
            path = within(ROOT / "replay_ui" / rel, ROOT / "replay_ui")
            if path.suffix not in (".html", ".css", ".js", ".ttf", ".txt"):
                raise ValueError("Unknown asset")
            return self.respond(path.read_bytes(), mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            return self.respond({"error": str(exc)}, 400)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if (not self.check_host() or self.headers.get("X-Replay-Token") != self.server.token
                or origin not in (None, f"http://{self.headers.get('Host')}")):
            return self.respond({"error": "Local session required"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.path == "/api/upload":
                return self.respond(self.server.workspace.upload(self.rfile, length,
                    unquote(self.headers.get("X-File-Name", ""))))
            limit = 6_000_000 if self.path == "/api/snapshot" else 16_384
            if not 0 < length <= limit:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            if self.path == "/api/browse":  # a modal dialog must not hold the workspace lock
                return self.respond(browse(request.get("kind")))
            with self.server.workspace.lock:
                workspace = self.server.workspace
                if self.path == "/api/validate":
                    result = workspace.validate(request)
                elif self.path == "/api/start-auto":
                    result = workspace.start_auto(request)
                elif self.path == "/api/start":
                    result = workspace.start(request)
                elif self.path == "/api/compare":
                    replay = workspace.replay(request["id"])
                    path = Path(request["path"])
                    replay.attach_gt(path if path.is_absolute() else workspace.root / path)
                    result = replay.payload()
                elif self.path == "/api/snapshot":
                    result = workspace.snapshot(request)
                else:
                    raise ValueError("Unknown action")
            return self.respond(result)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            return self.respond({"error": str(exc)}, 400)


def make_server(port=8765, workspace=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.workspace = workspace or Workspace()
    server.token = secrets.token_urlsafe(24)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"TrackCode workspace: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Replay identity, live tails, offline comparison and launcher trust boundaries."""

import argparse
import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest

from camloc import contracts as c
from run import resolve_episode
from scripts.replay_data import Lines, Replay, wire
from scripts.serve_replay import Workspace, make_server, within

EPOCH = 1_791_918_859_509_503_883


@pytest.fixture
def saved(tmp_path):
    folder = tmp_path / "results" / "saved"
    folder.mkdir(parents=True)
    meta = {"groups": 3, "split": "input", "initial_pose": {"timestamp_ns": EPOCH},
            "episode": {"start_ns": EPOCH, "end_ns": EPOCH + 100_000_000}}
    (folder / "run.json").write_text(json.dumps(meta))
    (folder / "submission.csv").write_text("timestamp_ns,x,y,yaw\n" + "".join(
        f"{EPOCH + i * 10_000_000},1,2,0\n" for i in range(1, 4)))
    records = []
    for i in range(1, 4):
        stamp = EPOCH + i * 10_000_000
        records += [{"record": "Estimate", "group_id": i, "t_ns": stamp,
                     "status": "lost" if i == 2 else "tracking", "reason": "fixture",
                     "evidence_kinds": ["prediction_only"],
                     "auxiliary": {"valid": True, "source_ns": stamp + (1 if i == 2 else 0),
                                   "source": "pnp", "z_m": 9}},
                    {"record": "GroupEvidence", "frame_set": {"t_ns": stamp, "group_id": i,
                     "images": [{"camera_id": "fr", "exposure_ns": stamp, "arrival_ns": stamp + 1}]},
                     "pose": {"record": "PoseEvidence", "inlier_count": 35}, "rejected": []}]
    (folder / "diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return folder


def test_incremental_lines_retain_partial_record_and_detect_truncation(tmp_path):
    path = tmp_path / "log"
    path.write_bytes(b'one\ntw')
    reader = Lines(path)
    assert list(reader.read()) == ["one"]
    with path.open("ab") as handle:
        handle.write(b'o\n')
    assert list(reader.read()) == ["two"]
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        list(reader.read())


def test_completed_tail_rejects_incomplete_record(tmp_path):
    path = tmp_path / "log"
    path.write_bytes(b'{"unfinished":')
    with pytest.raises(ValueError, match="Incomplete final"):
        list(Lines(path).read(finished=True))


def test_replay_exact_epoch_join_missing_metrics_and_future_auxiliary(saved):
    replay = Replay(saved)
    replay.refresh()
    payload = wire(replay.payload())
    assert payload["count"] == 3 and payload["comparison"] is None
    assert payload["origin_ns"] == str(EPOCH)
    assert payload["frames"][0]["t_ns"] == str(EPOCH + 10_000_000)
    assert payload["frames"][0]["pose"]["inlier_count"] == 35
    assert payload["frames"][1]["status"] == "lost"
    assert payload["frames"][0]["z"] == 9 and payload["frames"][1]["z"] is None
    assert "processing_ms" not in payload["frames"][0]
    replay.refresh()
    assert len(replay.frames) == 3
    assert replay.payload(2)["frames"][0]["index"] == 2


def test_offline_gt_matches_integer_time_and_never_aligns(saved, tmp_path):
    replay = Replay(saved)
    replay.refresh()
    gt = tmp_path / "truth.csv"
    gt.write_text("timestamp_ns,x,y,yaw\n" + "".join(
        f"{EPOCH + i * 10_000_000},0,2,0\n" for i in range(5)))
    replay.attach_gt(gt)
    assert replay.comparison["rmse_m"] == 1
    assert replay.comparison["matched"] == 3
    assert replay.frame(0)["error_m"] == 1
    assert replay.frame(0)["lateral_error_m"] == 0
    assert replay.frame(0)["longitudinal_error_m"] == 1
    replay.state = "running"
    with pytest.raises(ValueError, match="completed"):
        replay.attach_gt(gt)


def test_completed_count_and_corrupt_evidence_are_not_hidden(saved):
    (saved / "submission.csv").write_text("timestamp_ns,x,y,yaw\n")
    with pytest.raises(ValueError, match="count"):
        Replay(saved).refresh()
    (saved / "diagnostics.jsonl").write_text("not JSON\n")
    with pytest.raises(ValueError):
        Replay(saved).refresh()


def test_corruption_stays_visible_on_next_poll(saved):
    (saved / "diagnostics.jsonl").write_text("not JSON\n")
    replay = Replay(saved)
    for _ in range(2):
        with pytest.raises(ValueError):
            replay.refresh()


def test_late_diagnostics_patch_an_older_output(saved):
    replay = Replay(saved)
    replay.refresh("running")
    record = {"record": "Estimate", "group_id": 1, "t_ns": EPOCH + 10_000_000,
              "status": "recovering", "reason": "delayed", "evidence_kinds": []}
    with (saved / "diagnostics.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    replay.refresh("running")
    patches = replay.payload(3)["frames"]
    assert len(patches) == 1 and patches[0]["index"] == 0
    assert patches[0]["status"] == "recovering"


def test_tracker_ledger_uses_actual_enclosing_group_and_event_name(saved):
    with (saved / "diagnostics.jsonl").open("a") as handle:
        handle.write(json.dumps({"record": "TrackerEvent", "event": "candidate_held", "camera_id": "fr"}) + "\n")
    replay = Replay(saved)
    replay.refresh()
    assert replay.events[-1]["group_id"] == 3
    assert replay.events[-1]["kind"] == "candidate_held"


def test_input_recording_uses_own_epoch_and_eof_without_gt(monkeypatch):
    initial = c.InitialPose(timestamp_ns=1234567890123456789, x=1, y=2, yaw=0)
    monkeypatch.setattr(c, "load_split", Mock(side_effect=AssertionError("input mode read split")))
    args = argparse.Namespace(split=None, start_ns=None, end_ns=None)
    bounds = resolve_episode(args, initial)
    assert bounds.start_ns == initial.timestamp_ns and bounds.end_ns is None
    assert bounds.contains(initial.timestamp_ns + 10**12)
    args.end_ns = str(initial.timestamp_ns + 100)
    assert not resolve_episode(args, initial).contains(initial.timestamp_ns + 100)
    args.start_ns = str(initial.timestamp_ns + 1)
    with pytest.raises(c.ContractError, match="outside"):
        resolve_episode(args, initial)


def test_preset_cannot_override_frozen_bounds():
    initial = c.InitialPose(timestamp_ns=1234567890123456789, x=1, y=2, yaw=0)
    with pytest.raises(c.ContractError, match="input-recording"):
        resolve_episode(argparse.Namespace(split="tune", start_ns="1", end_ns=None), initial)


@pytest.mark.parametrize("motion", [None, "state-fields", "validation-twist"])
def test_launcher_uses_one_production_process_and_no_gt(tmp_path, monkeypatch, motion):
    workspace = Workspace(tmp_path)
    settings = {"bag": "input.mcap", "artifacts": "bank", "mode": "input", "retrieval": True,
                "initial_pose": {"timestamp_ns": str(EPOCH), "x": 1, "y": 2, "yaw": 0},
                "vehicle_motion": motion}
    monkeypatch.setattr(workspace, "validate", lambda _: settings)
    child = Mock()
    child.poll.return_value = None
    popen = Mock(return_value=child)
    monkeypatch.setattr("scripts.serve_replay.subprocess.Popen", popen)
    result = workspace.start({})
    command = popen.call_args.args[0]
    assert "--input-recording" in command and "--split" not in command
    assert ("--vehicle-motion" in command) == (motion is not None)
    if motion:
        assert command[command.index("--vehicle-motion") + 1] == motion
    assert not any("gt" in item.lower() for item in command)
    assert Path(command[command.index("--initial-pose") + 1]).exists()
    assert result["id"] in workspace.folders
    with pytest.raises(ValueError, match="already running"):
        workspace.start({})


def test_http_session_host_and_path_boundaries(saved):
    workspace = Workspace(saved.parents[1])
    server = make_server(0, workspace)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/config", timeout=5) as response:
            config = json.load(response)
        assert config["runs"][0]["groups"] == 3
        request = urllib.request.Request(base + "/api/start", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 403
        request.add_header("X-Replay-Token", server.token)
        request.add_header("Origin", "https://external.invalid")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 403
        for route in ("/../pyproject.toml", "/api/download?id=missing&file=gt.csv"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + route, timeout=5)
        with pytest.raises(ValueError, match="outside"):
            within(saved / "../../outside", saved)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_offline_bundle_preserves_identity_and_refuses_overwrite(saved, tmp_path, monkeypatch):
    from render_replay import export_run

    monkeypatch.setattr("render_replay.scene_geometry", lambda: [])
    out = tmp_path / "viewer"
    entry = export_run(saved, out)
    assert 'src="data.js"' in entry.read_text(encoding="utf-8")
    data = json.loads((out / "data.js").read_text(encoding="utf-8").removeprefix("window.APEX_REPLAY=").strip().removesuffix(";"))
    assert data["payload"]["frames"][0]["t_ns"] == str(EPOCH + 10_000_000)
    assert (out / "submission.csv").read_bytes() == (saved / "submission.csv").read_bytes()
    assert (out / "plotly.js").is_file() and (out / "fonts/IBMPlexSans.ttf").is_file()
    with pytest.raises(ValueError, match="fresh"):
        export_run(saved, out)


def test_development_recording_cannot_default_into_reserved_holdout(tmp_path):
    name = "c0_2026-10-13_22-53-42_camloc_0.mcap"
    (tmp_path / name).touch()
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "map.npz").touch()
    request = {"bag": name, "artifacts": "bank", "mode": "input",
               "initial_pose": {"timestamp_ns": str(c.load_split()["tune"].start_ns),
                                "x": 1, "y": 2, "yaw": 0}}
    with pytest.raises(ValueError, match="holdout"):
        Workspace(tmp_path).validate(request)


def test_snapshot_is_bounded_to_generated_pngs(saved):
    workspace = Workspace(saved.parents[1])
    key = workspace.runs()[0]["id"]
    data = b"\x89PNG\r\n\x1a\n" + b"test fixture"
    result = workspace.snapshot({"id": key, "image": "data:image/png;base64," + base64.b64encode(data).decode()})
    assert (workspace.root / result["path"]).read_bytes() == data
    with pytest.raises(ValueError, match="PNG"):
        workspace.snapshot({"id": key, "image": "data:text/html;base64,abcd"})


def test_ui_keeps_duplicate_initial_fields_visible_to_contract_parser(tmp_path):
    (tmp_path / "input.mcap").touch()
    (tmp_path / "bank").mkdir()
    (tmp_path / "bank/map.npz").touch()
    request = {"bag": "input.mcap", "artifacts": "bank",
               "initial_pose": '{"timestamp_ns":"123","x":1,"x":2,"y":3,"yaw":0}'}
    with pytest.raises(c.ContractError, match="duplicate"):
        Workspace(tmp_path).validate(request)

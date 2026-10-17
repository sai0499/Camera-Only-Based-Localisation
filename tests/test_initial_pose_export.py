"""Tests for the offline initial-pose harness ``scripts/export_initial_pose.py``.

Synthetic GT/manifest files live in ``tmp_path``; the real-bundle check only reads the
local evaluation bundle and writes into ``tmp_path``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from camloc import contracts

ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = ROOT / "evaluation" / "reference_reconciled" / "manifest.json"
KEYS = {"timestamp_ns", "x", "y", "yaw"}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "export_initial_pose", ROOT / "scripts" / "export_initial_pose.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = _load_script()


def _write_gt(path: Path, rows, header=("timestamp_ns", "x", "y", "yaw")) -> Path:
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


def _write_manifest(directory: Path, files: dict[str, str]) -> None:
    (directory / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(gt: Path, out: Path, *bounds: str) -> int:
    return exporter.main([*bounds, "--gt", str(gt), "--out", str(out)])


def _keys(out: Path) -> set[str]:
    return set(json.loads(out.read_text(encoding="utf-8")))


def test_first_in_bounds_row_and_extra_column_dropped(tmp_path, capsys):
    gt = _write_gt(
        tmp_path / "gt.csv",
        [(100, 1.0, 2.0, 9.0, 0.1), (200, 3.0, 4.0, 9.0, 0.2), (300, 5.0, 6.0, 9.0, 0.3)],
        header=("timestamp_ns", "x", "y", "z", "yaw"),
    )
    _write_manifest(tmp_path, {"gt.csv": _sha(gt)})
    out = tmp_path / "pose.json"
    assert exporter.verify_manifest(gt) == _sha(gt)
    assert _run(gt, out, "--start-ns", "150", "--end-ns", "400") == 0
    assert _keys(out) == KEYS
    assert contracts.read_initial_pose(out) == contracts.InitialPose(
        timestamp_ns=200, x=3.0, y=4.0, yaw=0.2
    )
    assert "initial_pose timestamp_ns=200 ->" in capsys.readouterr().out


def test_open_end_runs_to_end_of_file(tmp_path):
    gt = _write_gt(tmp_path / "gt.csv", [(100, 1.0, 2.0, 0.1), (200, 3.0, 4.0, 0.2)])
    out = tmp_path / "pose.json"
    assert exporter.verify_manifest(gt) is None
    assert _run(gt, out, "--start-ns", "200") == 0
    assert contracts.read_initial_pose(out).timestamp_ns == 200


@pytest.mark.parametrize(
    "stamps",
    [(100, 100, 300), (100, 300, 200), (100, "1.5e18", 300)],
    ids=["duplicate", "decreasing", "float"],
)
def test_bad_stamps_exit_1(tmp_path, stamps):
    gt = _write_gt(tmp_path / "gt.csv", [(t, 1.0, 2.0, 0.1) for t in stamps])
    out = tmp_path / "pose.json"
    assert _run(gt, out, "--start-ns", "0") == 1
    assert not out.exists()


@pytest.mark.parametrize("entry", ["mismatch", "missing"])
def test_manifest_problems_exit_1_without_writing(tmp_path, entry):
    gt = _write_gt(tmp_path / "gt.csv", [(100, 1.0, 2.0, 0.1)])
    _write_manifest(tmp_path, {"gt.csv": "0" * 64} if entry == "mismatch" else {"other": _sha(gt)})
    out = tmp_path / "pose.json"
    with pytest.raises(contracts.ContractError):
        exporter.verify_manifest(gt)
    assert _run(gt, out, "--start-ns", "0") == 1
    assert not out.exists()


def test_no_row_in_bounds_exit_1(tmp_path):
    gt = _write_gt(tmp_path / "gt.csv", [(100, 1.0, 2.0, 0.1), (200, 3.0, 4.0, 0.2)])
    out = tmp_path / "pose.json"
    assert _run(gt, out, "--start-ns", "300", "--end-ns", "400") == 1
    assert not out.exists()


def test_existing_out_is_not_overwritten(tmp_path):
    gt = _write_gt(tmp_path / "gt.csv", [(100, 1.0, 2.0, 0.1)])
    out = tmp_path / "pose.json"
    out.write_bytes(b"keep")
    assert _run(gt, out, "--start-ns", "0") == 1
    assert out.read_bytes() == b"keep"


def test_scientific_start_is_rejected(tmp_path):
    # argparse accepts the text; parse_decimal_ns rejects it, a data error (exit 1).
    gt = _write_gt(tmp_path / "gt.csv", [(100, 1.0, 2.0, 0.1)])
    out = tmp_path / "pose.json"
    assert _run(gt, out, "--start-ns", "1e9") == 1
    assert not out.exists()


def test_usage_errors_exit_2(tmp_path):
    out = str(tmp_path / "pose.json")
    for argv in (["--out", out], ["--split", "tune", "--start-ns", "0", "--out", out]):
        with pytest.raises(SystemExit) as info:
            exporter.main(argv)
        assert info.value.code == 2


def test_split_tune_selects_first_in_bounds_row(tmp_path):
    tune = contracts.load_split()["tune"]
    s = tune.start_ns
    gt = _write_gt(
        tmp_path / "gt.csv",
        [(s - 10, 0.0, 0.0, 0.0), (s, 1.0, 2.0, 0.5), (s + 10, 3.0, 4.0, 0.6)],
    )
    out = tmp_path / "pose.json"
    assert _run(gt, out, "--split", "tune") == 0
    assert _keys(out) == KEYS
    assert contracts.read_initial_pose(out) == contracts.InitialPose(
        timestamp_ns=s, x=1.0, y=2.0, yaw=0.5
    )


@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="local evaluation bundle not present")
def test_real_bundle_tune(tmp_path, capsys):
    tune = contracts.load_split()["tune"]
    expected = None
    with open(exporter.DEFAULT_GT, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            t = int(row["timestamp_ns"])
            if tune.contains(t):
                expected = contracts.InitialPose(
                    timestamp_ns=t, x=float(row["x"]), y=float(row["y"]), yaw=float(row["yaw"])
                )
                break
    assert expected is not None
    manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    assert exporter.verify_manifest(exporter.DEFAULT_GT) == manifest["files"]["gt_tf.csv"]
    out = tmp_path / "initial_pose_tune.json"
    assert exporter.main(["--split", "tune", "--out", str(out)]) == 0
    assert _keys(out) == KEYS
    assert contracts.read_initial_pose(out) == expected
    assert f"timestamp_ns={expected.timestamp_ns}" in capsys.readouterr().out

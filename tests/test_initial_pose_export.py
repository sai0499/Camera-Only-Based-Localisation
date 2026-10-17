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

import numpy as np
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

def test_property_24_exporter_selects_first_in_bounds_row(tmp_path):
    # Feature: shared-contracts, Property 24: Exporter selects the first in-bounds GT row
    # Validates: Requirements 18.4, 18.5, 18.7
    tune = contracts.load_split()["tune"]
    for k in range(120):
        rng = np.random.default_rng([20261024, k])
        ctx = f"seed=[20261024, {k}] iter={k}"

        # Strictly increasing int64 stamps: random count 2..20, gaps >= 1.
        n = int(rng.integers(2, 21))
        gaps = rng.integers(1, 1000, size=n).astype(np.int64)
        base = int(rng.integers(0, 1_000_000))
        stamps = base + np.cumsum(gaps)
        xs = [float(v) for v in rng.uniform(-50.0, 50.0, size=n)]
        ys = [float(v) for v in rng.uniform(-50.0, 50.0, size=n)]
        yaws = [float(v) for v in rng.uniform(-10.0, 10.0, size=n)]

        # Some iterations carry an extra column; its value must never leak into the JSON.
        has_extra = bool(rng.integers(0, 2))
        extras = [float(v) for v in rng.uniform(1e6, 2e6, size=n)] if has_extra else None
        if has_extra:
            header = ("timestamp_ns", "x", "y", "yaw", "z")
            rows = [
                (int(stamps[i]), repr(xs[i]), repr(ys[i]), repr(yaws[i]), repr(extras[i]))
                for i in range(n)
            ]
        else:
            header = ("timestamp_ns", "x", "y", "yaw")
            rows = [
                (int(stamps[i]), repr(xs[i]), repr(ys[i]), repr(yaws[i])) for i in range(n)
            ]

        # Choose a mode that still leaves at least one stamp in [start, end).
        mode = int(rng.integers(0, 4))
        use_split = False
        if mode == 0:  # first row in bounds, closed end
            start = int(stamps[0])
            end = int(stamps[-1]) + 1
        elif mode == 1:  # several leading rows out, closed end
            lead = int(rng.integers(1, n))
            start = int(stamps[lead])
            end = int(stamps[-1]) + 1
        elif mode == 2:  # open end, some leading rows out
            lead = int(rng.integers(0, n))
            start = int(stamps[lead])
            end = None
        else:  # named split tune: place stamps inside the real tune bounds
            use_split = True
            span = tune.end_ns - tune.start_ns
            step = max(1, span // (n + 1))
            stamps = tune.start_ns + step * np.arange(1, n + 1, dtype=np.int64)
            if has_extra:
                rows = [
                    (int(stamps[i]), repr(xs[i]), repr(ys[i]), repr(yaws[i]), repr(extras[i]))
                    for i in range(n)
                ]
            else:
                rows = [
                    (int(stamps[i]), repr(xs[i]), repr(ys[i]), repr(yaws[i])) for i in range(n)
                ]
            start = int(stamps[0])
            end = None

        gt = _write_gt(tmp_path / f"gt_{k}.csv", rows, header=header)
        _write_manifest(tmp_path, {f"gt_{k}.csv": _sha(gt)})
        out = tmp_path / f"pose_{k}.json"

        # Compute the expected first in-bounds row independently of the exporter.
        if use_split:
            idx = next(i for i in range(n) if tune.contains(int(stamps[i])))
        else:
            idx = next(
                i
                for i in range(n)
                if start <= int(stamps[i]) and (end is None or int(stamps[i]) < end)
            )
        exp_t = int(stamps[idx])
        exp_x, exp_y, exp_yaw = xs[idx], ys[idx], yaws[idx]

        if use_split:
            rc = _run(gt, out, "--split", "tune")
        elif end is None:
            rc = _run(gt, out, "--start-ns", str(start))
        else:
            rc = _run(gt, out, "--start-ns", str(start), "--end-ns", str(end))
        assert rc == 0, f"{ctx}: exit {rc} for stamps={list(map(int, stamps))} bounds=({start},{end})"

        pose = contracts.read_initial_pose(out)
        assert pose == contracts.InitialPose(
            timestamp_ns=exp_t, x=float(exp_x), y=float(exp_y), yaw=float(exp_yaw)
        ), f"{ctx}: pose {pose} != expected row {idx} t={exp_t}"

        # Exactly four keys, no extra GT column leaked into the JSON.
        doc_keys = _keys(out)
        assert doc_keys == KEYS, f"{ctx}: keys {doc_keys} != {KEYS}"
        if has_extra:
            text = out.read_text(encoding="utf-8")
            assert repr(float(extras[idx])) not in text, f"{ctx}: extra column leaked into {text!r}"

        # x/y round-trip exactly; yaw stored wrapped per contracts.wrap_angle.
        assert pose.x.hex() == float(rows[idx][1]).hex(), f"{ctx}: x round-trip"
        assert pose.y.hex() == float(rows[idx][2]).hex(), f"{ctx}: y round-trip"
        assert pose.yaw.hex() == contracts.wrap_angle(float(rows[idx][3])).hex(), (
            f"{ctx}: yaw not stored wrapped"
        )

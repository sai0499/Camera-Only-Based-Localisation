"""Offline harness: export the one initial pose of an episode from a GT CSV.

This is the only step-1 code that reads GT. It reads the GT CSV and its sibling
``manifest.json`` as files, imports no evaluation module and is never imported by
``camloc``. The output holds exactly ``timestamp_ns``, ``x``, ``y`` and ``yaw``.

    .venv/Scripts/python.exe -B scripts/export_initial_pose.py --split tune --out tmp/c/initial_pose_tune.json
    .venv/Scripts/python.exe -B scripts/export_initial_pose.py --start-ns N [--end-ns M] [--gt PATH] --out PATH

Exit codes: 0 success, 1 data/integrity error (nothing written), 2 usage error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "evaluation" / "reference_reconciled" / "gt_tf.csv"
SPLITS = ("fit", "tune", "holdout", "full")
REQUIRED_COLUMNS = ("timestamp_ns", "x", "y", "yaw")


def _contracts() -> ModuleType:
    """Import ``camloc.contracts`` after putting the project root on ``sys.path``."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("camloc.contracts")


def resolve_bounds(split: str | None, start: str | None, end: str | None) -> Any:
    """Return the ``EpisodeBounds`` of a named split or of explicit decimal-ns bounds."""
    contracts = _contracts()
    if split is not None:
        if start is not None or end is not None:
            raise contracts.ContractError("--split excludes --start-ns/--end-ns")
        return contracts.load_split()[split]
    if start is None:
        raise contracts.ContractError("either --split or --start-ns is required")
    start_ns = contracts.parse_decimal_ns(start, "--start-ns")
    end_ns = None if end is None else contracts.parse_decimal_ns(end, "--end-ns")
    return contracts.EpisodeBounds(name="explicit", start_ns=start_ns, end_ns=end_ns)


def verify_manifest(gt_path: Path) -> str | None:
    """Check the GT file against a sibling ``manifest.json``; return its SHA-256, or None."""
    contracts = _contracts()
    gt_path = Path(gt_path)
    manifest_path = gt_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise contracts.ContractError(f"{manifest_path}: invalid JSON ({exc})") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or gt_path.name not in files:
        raise contracts.ContractError(f"{manifest_path}: no files[{gt_path.name!r}] entry")
    expected = files[gt_path.name]
    actual = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected.lower() != actual:
        raise contracts.ContractError(
            f"{gt_path}: SHA-256 {actual} does not match manifest entry {expected!r}"
        )
    return actual


def _float(contracts: ModuleType, row: dict[str, Any], key: str, line: int) -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError) as exc:
        raise contracts.ContractError(f"line {line}: {key}={row[key]!r} is not a number") from exc


def select_initial_pose(gt_path: Path, bounds: Any) -> Any:
    """Return the ``InitialPose`` of the first GT row with ``bounds.contains(t)``.

    Every stamp is parsed exactly and the whole file must be strictly increasing.
    Extra columns (e.g. organizer ``z``) are ignored.
    """
    contracts = _contracts()
    try:
        return _select(contracts, Path(gt_path), bounds)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise contracts.ContractError(f"{gt_path}: unreadable CSV ({exc})") from exc


def _select(contracts: ModuleType, gt_path: Path, bounds: Any) -> Any:
    pose = None
    previous: int | None = None
    with open(gt_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise contracts.ContractError(f"{gt_path}: missing columns {missing}")
        for row in reader:
            line = reader.line_num
            t = contracts.parse_decimal_ns(row["timestamp_ns"], f"{gt_path} line {line}")
            if previous is not None and t <= previous:
                raise contracts.ContractError(
                    f"{gt_path} line {line}: stamp {t} is not after {previous}"
                )
            previous = t
            if pose is None and bounds.contains(t):
                pose = contracts.InitialPose(
                    timestamp_ns=t,
                    x=_float(contracts, row, "x", line),
                    y=_float(contracts, row, "y", line),
                    yaw=_float(contracts, row, "yaw", line),
                )
    if pose is None:
        raise contracts.ContractError(f"{gt_path}: no row inside episode {bounds.name}")
    return pose


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--split", choices=SPLITS, help="named episode of build_split.json")
    which.add_argument("--start-ns", help="episode start, decimal integer nanoseconds")
    parser.add_argument("--end-ns", help="exclusive episode end (default: end of GT file)")
    parser.add_argument("--gt", default=str(DEFAULT_GT), help="GT CSV (timestamp_ns,x,y,yaw)")
    parser.add_argument("--out", required=True, help="initial-pose JSON to create")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)  # argparse exits 2 on usage errors
    if args.split is not None and args.end_ns is not None:
        parser.error("--end-ns requires --start-ns")
    contracts = _contracts()
    try:
        bounds = resolve_bounds(args.split, args.start_ns, args.end_ns)
        if Path(args.out).exists():
            raise contracts.ContractError(f"refusing to overwrite {args.out}")
        verify_manifest(Path(args.gt))
        pose = select_initial_pose(Path(args.gt), bounds)
        bounds.check_initial_pose(pose)
        contracts.write_initial_pose(args.out, pose)  # exclusive create
    except (contracts.ContractError, OSError) as exc:
        print(f"export_initial_pose: {exc}", file=sys.stderr)
        return 1
    print(f"initial_pose timestamp_ns={pose.timestamp_ns} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

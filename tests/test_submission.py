"""Example tests of the single submission CSV writer (Requirements 17.1-17.7, 20.3).

Every file is written under ``tmp_path``; committed assets are only read.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from camloc import contracts as c
from camloc.submission import HEADER, SubmissionWriter, format_row

ROOT = Path(__file__).resolve().parents[1]

T0 = 1_760_389_000_123_456_789  # a 19-digit exposure-like stamp


def _reference_eval():
    """Import the local scorer module, adding ``evaluation/`` to ``sys.path`` only here."""
    path = str(ROOT / "evaluation")
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("reference_eval")


def _estimate(t_ns, x=1.0, y=2.0, yaw=0.5, group_id=0):
    kinds = (
        (c.EvidenceKind.INITIAL_POSE,) if group_id is None else (c.EvidenceKind.PREDICTION_ONLY,)
    )
    return c.Estimate(
        t_ns=t_ns,
        group_id=group_id,
        state=np.array([x, y, yaw, 0.0, 0.0, 0.0], dtype=np.float64),
        P=np.eye(6),
        status=c.Status.TRACKING,
        branch=c.Branch.TRACK,
        evidence_kinds=kinds,
        reason="no_evidence",
        auxiliary=c.AuxiliaryGeometry.unavailable("no_geometry"),
        max_support_exposure_ns=None,
    )


def test_header_bytes_are_utf8_lf_without_bom(tmp_path):
    path = tmp_path / "submission.csv"
    with SubmissionWriter(path) as writer:
        writer.write(_estimate(T0, group_id=0))
        writer.write(_estimate(T0 + 1, group_id=1))
    data = path.read_bytes()
    assert HEADER == "timestamp_ns,x,y,yaw"
    assert data.startswith(b"timestamp_ns,x,y,yaw\n")
    assert b"\r" not in data
    assert not data.startswith(b"\xef\xbb\xbf")
    assert data.count(b"\n") == 3
    assert writer.rows == 2
    assert writer.last_t_ns == T0 + 1


def test_existing_path_raises_and_keeps_content(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_bytes(b"keep me\r\n")
    with pytest.raises(FileExistsError):
        SubmissionWriter(path)
    assert path.read_bytes() == b"keep me\r\n"


def test_rejected_writes_leave_file_unchanged(tmp_path):
    path = tmp_path / "submission.csv"
    with SubmissionWriter(path) as writer:
        assert writer.rows == 0 and writer.last_t_ns is None
        with pytest.raises(c.ContractError):
            writer.write(_estimate(T0, group_id=None))
        writer.write(_estimate(T0, group_id=0))
        writer.flush()
        before = path.read_bytes()
        for stale in (T0, T0 - 1):
            with pytest.raises(c.ContractError):
                writer.write(_estimate(stale, group_id=1))
        with pytest.raises(c.ContractError):
            writer.write(_estimate(T0 + 5, group_id=None))
        writer.flush()
        assert path.read_bytes() == before
        assert writer.rows == 1 and writer.last_t_ns == T0


def test_format_row_validation():
    assert format_row(T0, 1.5, -2.0, math.pi) == f"{T0},1.5,-2,{math.pi:.17g}\n"
    expected = "7,0.10000000000000001,0.20000000000000001,-0.29999999999999999\n"
    assert format_row(np.int64(7), 0.1, 0.2, -0.3) == expected
    bad = [
        (1.0, 0.0, 0.0, 0.0),
        (True, 0.0, 0.0, 0.0),
        ("1", 0.0, 0.0, 0.0),
        (1, math.nan, 0.0, 0.0),
        (1, 0.0, math.inf, 0.0),
        (1, 0.0, 0.0, -math.inf),
        (1, 0.0, 0.0, 3.5),
        (1, 0.0, 0.0, -math.pi),
        (1, True, 0.0, 0.0),
    ]
    for args in bad:
        with pytest.raises(c.ContractError):
            format_row(*args)


def test_round_trip_through_reference_reader(tmp_path):
    ev = _reference_eval()
    path = tmp_path / "submission.csv"
    stamps = [T0, T0 + 33_333_333, T0 + 66_666_667, c.INT64_MAX]
    poses = [
        (0.1, -1e-300, math.pi),
        (123.456789012345678, 5e-324, -2.5),
        (-987654.3210987654, 1.0 / 3.0, float(np.nextafter(-math.pi, 0.0))),
        (2.0**52 + 1.0, -0.0, 0.0),
    ]
    with SubmissionWriter(path) as writer:
        for k, (t_ns, (x, y, yaw)) in enumerate(zip(stamps, poses, strict=True)):
            writer.write(_estimate(t_ns, x, y, yaw, group_id=k))
    result = ev.read_csv(path)
    assert result["timestamp_ns"].dtype == np.int64
    assert [int(v) for v in result["timestamp_ns"]] == stamps
    expected = np.array(poses, dtype=np.float64)
    for column, name in enumerate(("x", "y", "yaw")):
        assert result[name].tobytes() == expected[:, column].tobytes(), name

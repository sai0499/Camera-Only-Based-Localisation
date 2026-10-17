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


# ---------------------------------------------------------------------------
# Property 22: Submission CSV round trip
# Feature: shared-contracts, Property 22: writing group estimates then reading
# them through the local scorer reproduces identical timestamps and bitwise
# identical floats, each line matches format_row, and rows/last_t_ns track the
# written data.
# Validates: Requirements 17.3, 17.6, 17.7
# ---------------------------------------------------------------------------

_ITERATIONS = 200


def _boundary_floats():
    """Finite float64 boundary values the round trip must preserve bit for bit."""
    return [
        0.0,
        -0.0,
        5e-324,  # smallest positive subnormal
        -5e-324,
        2.2250738585072014e-308,  # smallest positive normal
        float(np.finfo(np.float64).max),
        float(-np.finfo(np.float64).max),
        1.0 / 3.0,
        -987654.3210987654,
        2.0**52 + 1.0,
    ]


def _draw_coord(rng, pool):
    """Draw a finite x or y coordinate: a boundary value or a scaled random double."""
    if rng.random() < 0.5:
        return float(pool[int(rng.integers(len(pool)))])
    exponent = float(rng.integers(-308, 309))
    value = float(rng.standard_normal()) * (10.0**exponent)
    return value if math.isfinite(value) else 0.0


def _draw_yaw(rng):
    """Draw a canonical yaw in (-pi, pi], including the pi and just-above-(-pi) edges."""
    choice = rng.random()
    if choice < 0.2:
        return math.pi
    if choice < 0.4:
        return float(np.nextafter(-math.pi, 0.0))
    # Uniform on (-pi, pi] then rely on the canonical edges above for the extremes.
    return float(rng.uniform(-math.pi, math.pi))


def _draw_increasing_stamps(rng, n):
    """Return n strictly increasing int64 nanosecond stamps within [0, INT64_MAX]."""
    gaps = rng.integers(1, 10_000_000, size=n, dtype=np.int64)
    start = int(rng.integers(0, 1_000_000_000_000_000, dtype=np.int64))
    stamps = [start + int(gaps[0])]
    for k in range(1, n):
        stamps.append(stamps[-1] + int(gaps[k]))
    return stamps


def test_property_22_submission_csv_round_trip(tmp_path):
    ev = _reference_eval()
    pool = _boundary_floats()
    for seed in range(_ITERATIONS):
        rng = np.random.default_rng([seed, 22])
        n = int(rng.integers(2, 9))
        stamps = _draw_increasing_stamps(rng, n)
        estimates = []
        for k in range(n):
            x = _draw_coord(rng, pool)
            y = _draw_coord(rng, pool)
            yaw = _draw_yaw(rng)
            estimates.append(_estimate(stamps[k], x, y, yaw, group_id=k))

        path = tmp_path / f"roundtrip_{seed}.csv"
        lines = []
        with SubmissionWriter(path) as writer:
            for k, est in enumerate(estimates):
                writer.write(est)
                # Each written line equals format_row of the estimate's stored state.
                sx, sy, syaw = (float(v) for v in est.state[0:3])
                expected_line = format_row(est.t_ns, sx, sy, syaw)
                lines.append(expected_line)
                assert writer.rows == k + 1, (
                    f"seed={seed} iteration={k} input={stamps[k]}: "
                    f"rows {writer.rows} != {k + 1}"
                )
                assert writer.last_t_ns == stamps[k], (
                    f"seed={seed} iteration={k} input={stamps[k]}: "
                    f"last_t_ns {writer.last_t_ns} != {stamps[k]}"
                )
            final_rows = writer.rows
            final_last = writer.last_t_ns

        assert final_rows == n, f"seed={seed}: rows {final_rows} != {n}"
        assert final_last == stamps[-1], (
            f"seed={seed}: last_t_ns {final_last} != {stamps[-1]}"
        )

        # The file body (after the header line) equals the concatenated format_row lines.
        body = path.read_text(encoding="utf-8").split("\n", 1)[1]
        assert body == "".join(lines), f"seed={seed}: written lines differ from format_row"

        result = ev.read_csv(path)
        assert result["timestamp_ns"].dtype == np.int64, f"seed={seed}: stamp dtype"
        assert [int(v) for v in result["timestamp_ns"]] == stamps, (
            f"seed={seed}: timestamps differ after round trip"
        )
        expected = np.array(
            [[float(e.state[0]), float(e.state[1]), float(e.state[2])] for e in estimates],
            dtype=np.float64,
        )
        for column, name in enumerate(("x", "y", "yaw")):
            assert result[name].tobytes() == expected[:, column].tobytes(), (
                f"seed={seed}: {name} is not bitwise identical after round trip"
            )


# ---------------------------------------------------------------------------
# Property 23: Submission ordering and rejection
# Feature: shared-contracts, Property 23: a stale or initialization estimate is
# rejected by SubmissionWriter.write and leaves the file bytes unchanged, and
# format_row rejects non-integer stamps, non-finite coordinates and yaw outside
# (-pi, pi].
# Validates: Requirements 17.4, 17.5
# ---------------------------------------------------------------------------


def test_property_23_write_rejects_stale_and_initialization(tmp_path):
    for seed in range(_ITERATIONS):
        rng = np.random.default_rng([seed, 23])
        n = int(rng.integers(1, 7))
        stamps = _draw_increasing_stamps(rng, n)

        path = tmp_path / f"reject_{seed}.csv"
        writer = SubmissionWriter(path)
        try:
            for k in range(n):
                writer.write(_estimate(stamps[k], group_id=k))
            writer.flush()
            prefix_bytes = path.read_bytes()
            last = stamps[-1]

            if rng.random() < 0.5:
                # Out-of-order: equal or strictly smaller than the last written stamp.
                if rng.random() < 0.5:
                    stale_t = last
                else:
                    stale_t = last - int(rng.integers(1, last + 1)) if last > 0 else last
                bad = _estimate(stale_t, group_id=n)
            else:
                # Initialization estimate carries no camera group.
                bad = _estimate(last + int(rng.integers(1, 1_000_000)), group_id=None)

            with pytest.raises(c.ContractError):
                writer.write(bad)
            writer.flush()
            after_bytes = path.read_bytes()
            assert after_bytes == prefix_bytes, (
                f"seed={seed} iteration={n} input_t={bad.t_ns}: "
                f"rejected write changed the file bytes"
            )
            assert writer.rows == n, (
                f"seed={seed} iteration={n}: rows advanced on a rejected write"
            )
            assert writer.last_t_ns == last, (
                f"seed={seed} iteration={n}: last_t_ns advanced on a rejected write"
            )
        finally:
            writer.close()


def test_property_23_format_row_rejects_bad_inputs():
    non_integer_stamps = [1.0, True, np.True_, "1", "01", math.nan]
    non_finite = [math.nan, math.inf, -math.inf]
    for seed in range(_ITERATIONS):
        rng = np.random.default_rng([seed, 2323])
        t_ok = int(rng.integers(0, c.INT64_MAX, dtype=np.int64))
        x_ok = float(rng.standard_normal())
        y_ok = float(rng.standard_normal())
        yaw_ok = float(rng.uniform(-math.pi, math.pi))

        # Non-integer timestamp (float / bool / str) is rejected.
        bad_stamp = non_integer_stamps[int(rng.integers(len(non_integer_stamps)))]
        with pytest.raises(c.ContractError):
            format_row(bad_stamp, x_ok, y_ok, yaw_ok)

        # Non-finite x, y or yaw is rejected.
        slot = int(rng.integers(3))
        bad_value = non_finite[int(rng.integers(len(non_finite)))]
        args = [t_ok, x_ok, y_ok, yaw_ok]
        args[1 + slot] = bad_value
        with pytest.raises(c.ContractError):
            format_row(*args)

        # Yaw just outside (-pi, pi]: nextafter(pi, inf) above, -pi at the lower edge.
        if rng.random() < 0.5:
            bad_yaw = float(np.nextafter(math.pi, math.inf))
        else:
            bad_yaw = -math.pi
        with pytest.raises(c.ContractError):
            format_row(t_ok, x_ok, y_ok, bad_yaw)

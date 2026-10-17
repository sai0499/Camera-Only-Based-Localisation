"""The single ``timestamp_ns,x,y,yaw`` submission CSV writer.

Rows use integer nanoseconds written as ``str(int)`` and metres/radians written with
``format(v, ".17g")``, the exact format of the local scorer's CSV reader and writer. The file
is UTF-8 without BOM with ``\\n`` line endings, created exclusively (an existing file is never
overwritten). Only group estimates are written, with strictly increasing timestamps; the
initialization estimate, out-of-order rows, non-finite values and unwrapped yaw raise
``ContractError`` and leave the file unchanged.
"""

from __future__ import annotations

import math
import os
from types import TracebackType

import numpy as np

from camloc.contracts import ContractError, Estimate, as_ns

__all__ = ["HEADER", "SubmissionWriter", "format_row"]

HEADER = "timestamp_ns,x,y,yaw"


def _finite(value: object, name: str) -> float:
    """Return a finite real non-bool scalar as a Python ``float``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ContractError(f"{name}: expected a finite real number, got {type(value).__name__}")
    try:
        result = float(value)
    except OverflowError:
        raise ContractError(f"{name}: {value!r} does not fit a float") from None
    if not math.isfinite(result):
        raise ContractError(f"{name}: expected a finite real number, got {result!r}")
    return result


def format_row(t_ns: int, x: float, y: float, yaw: float) -> str:
    """Pure: return ``f"{t_ns},{x:.17g},{y:.17g},{yaw:.17g}\\n"`` after strict validation.

    ``t_ns`` must be integer nanoseconds (``as_ns``); ``x``, ``y``, ``yaw`` finite reals and
    ``yaw`` in (-pi, pi]. Yaw is never wrapped here. Violations raise ``ContractError``.
    """
    stamp = as_ns(t_ns, "t_ns")
    fx, fy, fyaw = _finite(x, "x"), _finite(y, "y"), _finite(yaw, "yaw")
    if not -math.pi < fyaw <= math.pi:
        raise ContractError(f"yaw: {fyaw!r} is outside (-pi, pi]")
    return f"{stamp},{fx:.17g},{fy:.17g},{fyaw:.17g}\n"


class SubmissionWriter:
    """Exclusive-create CSV writer of group estimates with strictly increasing ``t_ns``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._file = open(path, "x", encoding="utf-8", newline="")
        try:
            self._file.write(HEADER + "\n")
        except BaseException:
            self._file.close()
            raise
        self._rows = 0
        self._last_t_ns: int | None = None

    @property
    def rows(self) -> int:
        """Number of data rows written (the header excluded)."""
        return self._rows

    @property
    def last_t_ns(self) -> int | None:
        """Timestamp of the last written row, or None before the first row."""
        return self._last_t_ns

    def write(self, estimate: Estimate) -> None:
        """Append one row for a group ``Estimate``; any rejection writes nothing."""
        if self._file.closed:
            raise ValueError("SubmissionWriter.write: the writer is closed")
        if not isinstance(estimate, Estimate):
            raise ContractError(
                f"SubmissionWriter.write: expected an Estimate, got {type(estimate).__name__}"
            )
        if estimate.group_id is None:
            raise ContractError(
                "SubmissionWriter.write: the initialization estimate has no camera group "
                "and is never written"
            )
        if self._last_t_ns is not None and estimate.t_ns <= self._last_t_ns:
            raise ContractError(
                f"SubmissionWriter.write: t_ns {estimate.t_ns} does not exceed the previous "
                f"row's {self._last_t_ns}"
            )
        x, y, yaw = (float(v) for v in estimate.state[0:3])
        row = format_row(estimate.t_ns, x, y, yaw)
        self._file.write(row)
        self._rows += 1
        self._last_t_ns = estimate.t_ns

    def flush(self) -> None:
        """Flush buffered rows to the operating system."""
        self._file.flush()

    def close(self) -> None:
        """Close the file; closing twice is a no-op."""
        self._file.close()

    def __enter__(self) -> SubmissionWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

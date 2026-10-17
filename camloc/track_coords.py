"""Owner A, step 4: route coordinates (``camloc/track_coords.py``).

A thin adapter over the unchanged organizer helper
``23-00-38_cut_120m_update/tools/track_frame.py``. It derives the (s, n, epsi) route
coordinates of an Estimate's own x/y/yaw for both the racing line (``track``) and the
pit-lane line (``pit``), and implements the ``TrackProjector`` protocol of
``camloc.contracts``. It never reads native live idx/ds/n, GT lane or any GT state; the
projection is purely a report derived from the estimator's planar pose.

Conventions (from the helper and plan section 2a)
=================================================

Both reference lines are closed polylines in frame ``map``; the last pose repeats the first
(the closing segment is skipped). ``s`` is the XY arc length modulo the loop length, ``n``
is positive to the left of the segment direction, and ``epsi`` is the car yaw minus the
segment heading, wrapped. Loop lengths come from the loaded geometry, not rounded
constants. The reference line is neither a compulsory driving line nor an extra camera
observation.

Sequence projection and resets
==============================

The helper searches +-200 segments around its previous projection and resets to a global
search after a time gap greater than one second. This adapter drives that sequence
behaviour per route and clears its history on a confirmed recovery (``reset``), so a
recovery jump beyond the helper's local window performs a fresh global search. The original
x/y estimate is always preserved; a large XY -> (s, n) -> XY reconstruction residual is
reported through ``xy_reconstruction_residual_m`` and the record's validity, never by
rounding the pose through track coordinates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from camloc import contracts
from camloc.contracts import (
    ROUTE_IDS,
    SUPPLIED_MAP_SHA256,
    Estimate,
    RecoveryReset,
    TrackProjection,
)

_HELPER_PATH = (
    contracts.PROJECT_ROOT / "23-00-38_cut_120m_update" / "tools" / "track_frame.py"
)
# A large reconstruction residual means the clamped-endpoint inverse disagrees; above this
# the projection is reported invalid but the pose is still preserved by the caller.
_MAX_RECONSTRUCTION_RESIDUAL_M = 1.0
# The helper's own gap reset threshold (ns); feeding stamps farther apart than this makes
# it search globally, matching a confirmed recovery jump.
_GAP_RESET_NS = 1_000_000_000


def _load_helper() -> ModuleType:
    """Import the unchanged organizer ``track_frame.py`` helper by file path.

    The helper is imported as a module object, never modified. It is loaded lazily so this
    module stays importable (stdlib + numpy + camloc) for the inference-isolation scan.
    """
    spec = importlib.util.spec_from_file_location("camloc_track_frame_helper", _HELPER_PATH)
    if spec is None or spec.loader is None:
        raise contracts.ContractError(f"track_coords: cannot load helper at {_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TrackProjector:
    """Owner A's route-coordinate projector; see ``load_track_projector``."""

    def __init__(self, lines, reference_map_id: str) -> None:
        # lines: mapping route_id -> track_frame.Line
        self._lines = lines
        self._reference_map_id = reference_map_id
        # Per-route sequence state: the stamp of the previous projection, to drive the
        # helper's local-vs-global search. Cleared on reset.
        self._last_t_ns: dict[str, int] = {}
        self._last_segment: dict[str, int] = {}

    @property
    def reference_map_id(self) -> str:
        """Opaque identity of the reference geometry (``SUPPLIED_MAP_SHA256``)."""
        return self._reference_map_id

    def route_length(self, route_id: str) -> float:
        """Return the loop length of a route, computed from the loaded geometry."""
        return float(self._lines[contracts._check_route_id(route_id, "route_id")].lap)

    def _project_one(self, route_id: str, estimate: Estimate) -> TrackProjection:
        line = self._lines[route_id]
        x = float(estimate.state[0])
        y = float(estimate.state[1])
        yaw = float(estimate.state[2])

        last = self._last_t_ns.get(route_id)
        if last is None or estimate.t_ns - last > _GAP_RESET_NS:
            s, n, epsi, segment = line.xy_to_sn(x, y, yaw)
            s_m, n_m, epsi_rad, segment_index = float(s[0]), float(n[0]), float(epsi[0]), int(segment[0])
        else:
            # Reuse the helper's projection kernel with the actual previous segment.
            # Reprojecting the previous XY globally loses history at crossings.
            candidates = (self._last_segment[route_id] + np.arange(-200, 201)) % len(line.p)
            candidates = candidates[line.valid[candidates]]
            segment_index, along, n_m = line._project_one(np.array([x, y]), candidates)
            segment_index = int(segment_index)
            s_m = float(line._norm_s(line.s_ref[segment_index] + along))
            epsi_rad = yaw - float(line.heading[segment_index])
        s_m = float(np.mod(s_m, line.lap))
        n_m = float(n_m)
        epsi_rad = contracts.wrap_angle(epsi_rad, "epsi_rad")

        # XY -> (s, n) -> XY reconstruction residual: how well the clamped-segment inverse
        # reproduces the queried position. Large residuals flag an ambiguous projection.
        rx, ry, _ryaw = line.sn_to_xy(np.array([s_m]), np.array([n_m]), np.array([epsi_rad]))
        residual = float(np.hypot(rx[0] - x, ry[0] - y))

        self._last_t_ns[route_id] = estimate.t_ns
        self._last_segment[route_id] = segment_index

        if residual > _MAX_RECONSTRUCTION_RESIDUAL_M:
            return TrackProjection(
                t_ns=estimate.t_ns,
                reference_map_id=self._reference_map_id,
                route_id=route_id,
                s_m=None,
                n_m=None,
                epsi_rad=None,
                segment_index=None,
                xy_reconstruction_residual_m=residual,
                valid=False,
                reason="reconstruction_residual_too_large",
            )

        return TrackProjection(
            t_ns=estimate.t_ns,
            reference_map_id=self._reference_map_id,
            route_id=route_id,
            s_m=s_m,
            n_m=n_m,
            epsi_rad=epsi_rad,
            segment_index=segment_index,
            xy_reconstruction_residual_m=residual,
            valid=True,
            reason="valid",
        )

    def project(self, estimate: Estimate) -> tuple[TrackProjection, TrackProjection]:
        """Return the (track, pit) route projections of the estimate's own x/y/yaw.

        Both are derived from the same estimated pose; neither uses GT or native state. The
        two routes are reported independently so the caller can keep the branch ambiguity
        explicit rather than averaging.
        """
        if not isinstance(estimate, Estimate):
            raise contracts.ContractError(
                f"project: expected an Estimate, got {type(estimate).__name__}"
            )
        if any(estimate.t_ns <= stamp for stamp in self._last_t_ns.values()):
            raise contracts.ContractError("project: estimate timestamps must increase")
        return (self._project_one("track", estimate), self._project_one("pit", estimate))

    def reset(self, reset: RecoveryReset) -> None:
        """Clear the sequence-projection history so the next projection searches globally.

        A confirmed recovery can jump beyond the helper's +-200 segment window, so its local
        search would be wrong; clearing the history forces a fresh global projection.
        """
        if not isinstance(reset, RecoveryReset):
            raise contracts.ContractError(
                f"reset: expected a RecoveryReset, got {type(reset).__name__}"
            )
        self._last_t_ns.clear()
        self._last_segment.clear()



def load_track_projector(map_mcap: str | Path) -> _TrackProjector:
    """Load the reference racing/pit lines from the static map and return a TrackProjector.

    The map is hash-bound to ``SUPPLIED_MAP_SHA256``. The unchanged organizer
    ``track_frame.py`` helper parses the two reference Paths; loop lengths come from the
    loaded geometry. ``reference_map_id`` is ``SUPPLIED_MAP_SHA256``.
    """
    import hashlib

    map_mcap = Path(map_mcap)
    digest = hashlib.sha256(map_mcap.read_bytes()).hexdigest()
    if digest != SUPPLIED_MAP_SHA256:
        raise contracts.ContractError(
            f"track_coords {map_mcap}: SHA-256 {digest} != {SUPPLIED_MAP_SHA256}; the static "
            "map is hash-bound and must stay byte-for-byte unchanged"
        )
    helper = _load_helper()
    lines = helper.load_lines(str(map_mcap))
    missing = [route for route in ROUTE_IDS if route not in lines]
    if missing:
        raise contracts.ContractError(f"track_coords: missing reference lines {missing}")
    return _TrackProjector(lines, SUPPLIED_MAP_SHA256)

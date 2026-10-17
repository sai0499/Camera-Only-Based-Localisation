"""Shared contracts of the TrackCode Apex camera-only localizer (owner: C).

Teammates A (input, geometry, borders, track coordinates) and B (map, vision, retrieval,
ground motion) build against the records and call boundaries defined here; C's runner and
estimator consume them. ``task_docs/FINAL_BUILD_PLAN.md`` is the build specification; this
docstring is the short normative form of the shared conventions.

Conventions
===========

Time          Python ``int`` nanoseconds in [0, 2**63 - 1], checked by ``as_ns``. NumPy
              integers are converted to ``int``; floats (even ``1.0``), bools, strings and
              NumPy timedelta/datetime scalars are rejected. Decimal strings (JSON, split
              file) go through ``parse_decimal_ns``, which accepts ASCII ``[0-9]+`` only.
Exposure      ``exposure_ns`` is the image ``header.stamp``; ``arrival_ns`` is the MCAP log
              time in replay. Group deadlines use arrival time.
Three times   ``source_exposure_ns``: the anchor job's source image. ``measurement_ns``: the
              current same-camera image on which a pose is solved. ``t_ns``: group time =
              max exposure of the group's images = CSV timestamp. Invariant:
              source_exposure_ns <= every support stamp <= measurement_ns <= t_ns.
Frames        W = ``map``; B = ``constructor0/base_link`` (x forward, y left, z up);
              C = ``constructor0/camera_<id>`` optical (z forward, x right, y down).
Transforms    ``T_AB`` maps coordinates from frame B into frame A: p_A = T_AB @ p_B, a
              4 x 4 float64 rigid matrix. ``T_BC`` comes from /tf_static (parent base_link,
              child camera). Projection: P_C = inv(T_WB @ T_BC) @ P_W. OpenCV PnP returns
              T_CW, so T_WB = inv(T_CW) @ inv(T_BC). Invert rigid transforms in closed form
              (R.T, -R.T @ t).
Planar pose   x = T_WB[0, 3], y = T_WB[1, 3], yaw = atan2(T_WB[1, 0], T_WB[0, 0]). A full
              pose from state plus auxiliary geometry is R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
              t = (x, y, z).
Quaternions   xyzw (scalar last); convert explicitly at wxyz library boundaries.
Angles        Radians. Every angle field is normalized by ``wrap_angle`` to (-pi, pi]; -pi
              is stored as pi.
Images        Canonical RGB ``uint8`` (464, 960, 3), i.e. W x H = 960 x 464, C-contiguous;
              ``valid_mask`` ``bool`` (464, 960), True = usable. Neither is copied; both
              are marked read-only.
Canonical K   float64 3 x 3, zero skew, zero distortion, OpenCV pixel centres. A derives it
              from raw K (never P): fx' = sx*fx, fy' = sy*fy, cx' = sx*(cx + 0.5) - 0.5,
              cy' = sy*(cy + 0.5) - 0.5, sx = 960/1506, sy = 464/728. Pixels lie in
              [-0.5, 959.5] x [-0.5, 463.5].
Precision     Keypoints float32 (N, 2); world geometry, calibration, state and covariances
              float64; identity arrays int64; float16 descriptors only inside B's caches.
State         [x_m, y_m, yaw_rad, vx_mps, vy_mps, yaw_rate_radps] in map axes
              (STATE_FIELDS); P is 6 x 6 in the same order. z/roll/pitch are auxiliary
              geometry, not states.
Identity      calibration_id, preprocessing_id, map_id, model ids and reference_map_id are
              opaque non-empty owner-defined strings; contracts never compute them.
              reference_map_id is SUPPLIED_MAP_SHA256.
Cameras       CAMERA_IDS = fl, fr, cl, cr, r, rl, rr; primary fr, secondary r.
Routes        ROUTE_IDS = track, pit (the route keys of the organizer track_frame helper).
Reasons       Evidence: ``accepted`` iff reason == "accepted". Geometry: ``valid`` iff
              reason == "valid". AnchorResult: status ok iff reason == "ok". Every reason
              matches ``[a-z][a-z0-9_]*`` in full.

Errors and records
==================

``ContractError`` (a ``ValueError``) marks a structural violation, raised at construction
or at a boundary check. Weak or failed evidence is a valid record with accepted=False (or
valid=False) and a snake_case reason; presenting it for consumption raises
``RejectedEvidenceError``. ``InputError`` marks a data failure on one input, such as an
undecodable JPEG; C records it as an input event.

Records are frozen, slotted, keyword-only dataclasses. Stored small arrays are read-only
copies; images and masks are checked and made read-only without copying. Every record has
a JSON-safe ``summary()`` so each rejected or low-trust output can be logged. Contracts
enforce structure only; tuning thresholds (inlier gates, innovation limits, noise floors)
stay in the owners' modules.

Inference isolation: this module imports only the standard library and numpy. Inference
never decodes dynamic /tf, the live state topic or evaluation GT files.

Call order and flows between owners
===================================

The Protocols at the end of this module are the call boundaries (proposals for A/B
agreement); owners implement them in their own modules. They define interfaces only and
never return a plausible pose on their own. Per FrameSet k, C's runner calls, in order:
``Estimator.snapshot(k.t_ns)``; ``Canonicalizer.canonicalize`` for each camera in the
decode set; ``MappedTracker.observe`` exactly once per decoded Observation; ``track``;
``handoff`` for anchor deliveries due at k; ``verify_candidate`` while a recovery
candidate is held; road patch, ground motion and borders only when no selected in-gate
full pose exists; ``Estimator.step(GroupEvidence)``; ``MappedTracker.settle`` per
HandoffEvidence; then the CSV row and the diagnostics.

Decode set       C decides, A executes. ``fr`` every group. ``r`` while a rear anchor job
                 is in flight (from its source group through delivery or rejection), when
                 ``fr`` is missing, or when ``fr`` tracking evidence is rejected in that
                 group (decided after processing ``fr``). A's grouping never decodes;
                 header pairing of ``r`` keeps output timestamps identical across
                 frontend modes.
Road patches     C calls ``road_patch(x, y, yaw, branch)`` with the snapshot pose at most
                 once per group when plane-dependent evidence is needed and passes that
                 same RoadPatch to B's ground motion and to C's auxiliary geometry. An
                 invalid patch disables ground motion and border updates for that group.
Auxiliary        z/roll/pitch. Source ``pnp``: from the latest accepted current-image PnP
geometry         T_WB (``source_ns`` = its measurement exposure). Source ``road_patch``:
                 plane height plus a fitting-only body-to-road height offset (frozen
                 artifact, A/C) and the plane roll/pitch. Otherwise ``none``. C carries it
                 in SearchSnapshot/Estimate; A/B use it to compose T_WB. Never from GT.
Ground motion    C passes ``previous_snapshot`` and ``current_snapshot`` at the two image
                 exposures, with valid auxiliary geometry, to ``estimate``. Cache the
                 previous exposure's causal state; predict the current prior before this
                 group's updates. Do not relabel a group-time posterior as an earlier
                 exposure. The auxiliary z includes the frozen body-to-road height offset;
                 B needs no hidden height default or separate artifact lookup.
Anchor delivery  C's scheduler freezes the source Observation and SearchSnapshot in an
                 AnchorRequest; the AnchorResult holds source-image correspondences only,
                 never a pose. At the first group with t_ns >= delivery_not_before_ns
                 (>= source exposure + DELIVERY_MIN_DELAY_NS = 90 ms), C calls
                 ``handoff(result, same-camera observation of that group)``: B re-solves
                 PnP on the current image, so the measurement is the current image.
Recovery reset   C holds one candidate. Confirmation needs an accepted pose at least
                 RECOVERY_CONFIRM_MIN_NS later; C then sends a RecoveryReset through
                 ``GroupEvidence.reset`` and, after ``step()``, calls
                 ``MappedTracker.reset`` and ``TrackProjector.reset``.
Input events     A attaches anomalies to the next FrameSet (or ``trailing_events()``); C
                 creates ``decode_failure`` events when ``canonicalize`` raises
                 InputError.
Diagnostics      C writes ``record.summary()`` for every FrameSet, evidence record,
                 GroupEvidence, Estimate and anchor record to JSONL, never to the CSV.

Proposed owner entry points (finalized at integration; not implemented here):

A  ``camloc.io.open_frame_sets(bag, *, initial_pose, episode, cameras=(PRIMARY_CAMERA,
   SECONDARY_CAMERA)) -> FrameSetSource``; ``camloc.geometry.make_canonicalizer(
   calibrations) -> Canonicalizer``; ``camloc.geometry.load_road_geometry(map_mcap) ->
   RoadGeometry``; ``camloc.borders.make_border_source(road) -> BorderEvidenceSource``;
   ``camloc.track_coords.load_track_projector(map_mcap) -> TrackProjector``.
B  ``camloc.vision.load_tracker(artifacts, calibrations) -> MappedTracker``;
   ``load_anchor_worker(artifacts) -> AnchorWorker``; ``make_ground_motion() ->
   GroundMotionEstimator``.
C  ``camloc.filter`` estimator constructed from exactly one InitialPose;
   ``camloc.submission.SubmissionWriter``.

Copyable usage examples for A, B and C: ``tests/test_contract_examples.py``.
"""

import dataclasses
import enum
import hashlib
import json
import math
import os
import re
import types
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------------------
# Constants (structural only; tuning thresholds stay in the owners' modules)
# ---------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "task_docs" / "build_split.json"

CAMERA_IDS = ("fl", "fr", "cl", "cr", "r", "rl", "rr")
PRIMARY_CAMERA = "fr"
SECONDARY_CAMERA = "r"
ROUTE_IDS = ("track", "pit")  # keys of the organizer track_frame.TOPICS

MAP_FRAME = "map"
BODY_FRAME = "constructor0/base_link"
CAMERA_FRAME_PREFIX = "constructor0/camera_"

CANONICAL_WIDTH = 960
CANONICAL_HEIGHT = 464
STATE_FIELDS = ("x_m", "y_m", "yaw_rad", "vx_mps", "vy_mps", "yaw_rate_radps")

INT64_MAX = 2**63 - 1
NS_PER_S = 1_000_000_000

MAX_EXPOSURE_SPREAD_NS = 5_000_000  # absolute cap of A's min(5 ms, 0.2 T) tolerance
MAX_ARRIVAL_SPREAD_NS = 40_000_000  # absolute cap of A's min(40 ms, 1.2 T) deadline
MAX_PENDING_GROUPS = 4
DELIVERY_MIN_DELAY_NS = 90_000_000  # anchor delivery: first group with t_ns >= source + 90 ms
MAX_KEYPOINTS = 1024  # DISK retained keypoints per image
MAX_ACTIVE_TRACKS = 512
MAX_HANDOFF_POINTS = 256
MAX_HANDOFF_TRANSITIONS = 4
HISTORY_MAX_AGE_NS = 250_000_000
HISTORY_MAX_FRAMES = 32
MAX_REFERENCES_PER_JOB = 2
RETRIEVAL_CANDIDATES = 10
NORMAL_JOBS_PER_S = 3
RECOVERY_JOBS_PER_S = 4
GRID_CELLS = 16  # 4 x 4 image grid for PnP support
RECOVERY_CONFIRM_MIN_NS = 50_000_000
BORDER_MAX_RATE_HZ = 5

ROTATION_TOL = 1e-6
LAST_ROW_TOL = 1e-9
UNIT_TOL = 1e-9
SYMMETRY_TOL = 1e-9  # relative to max(1, max |P|)
PSD_TOL = 1e-9  # relative to max(1, max |P|)
SUMMARY_MAX_ARRAY_ITEMS = 64

BUILD_SPLIT_SHA256 = "894b41790af50e2f555a34001ef00821164ebc4f2f2589eb2e24280088c09222"
SUPPLIED_MAP_SHA256 = "648e8cf954d8cd39823ca31c0b62d89c8fd76b170df999ac2e3d4327439972ef"
SPLIT_TIME_ENCODING = "decimal_string_nanoseconds_convert_to_integer"
SPLIT_INTERVAL_CONVENTION = "start_inclusive_end_exclusive"
SPLIT_GUARD_NS = 2_000_000_000


# ---------------------------------------------------------------------------------------
# Enums (values are the persisted strings)
# ---------------------------------------------------------------------------------------


class Status(enum.StrEnum):
    """Estimator status: an engineering state, not proof of accuracy."""

    TRACKING = "tracking"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    LOST = "lost"


class Branch(enum.StrEnum):
    """Track branch of a pose, road patch or border association."""

    TRACK = "track"
    PIT = "pit"
    UNKNOWN = "unknown"


class FrameSetClosure(enum.StrEnum):
    """Why a FrameSet closed: all selected cameras, deadline, pending capacity or input end."""

    COMPLETE = "complete"
    DEADLINE = "deadline"
    CAPACITY = "capacity"
    EOF = "eof"


class PoseSource(enum.StrEnum):
    """Track set a PoseEvidence was solved from."""

    TRACKING = "tracking"  # the active mapped track set
    HANDOFF = "handoff"  # current-image PnP after an anchor job
    CANDIDATE = "candidate"  # the uncommitted recovery candidate's own tracks


class AnchorPurpose(enum.StrEnum):
    """Purpose of a bounded anchor job."""

    FRONT_ANCHOR = "front_anchor"
    REAR_ANCHOR = "rear_anchor"
    RECOVERY = "recovery"
    WARM_RETRIEVAL = "warm_retrieval"


class AnchorStatus(enum.StrEnum):
    """Outcome of an anchor job; only ``ok`` carries correspondences."""

    OK = "ok"
    FAILED = "failed"
    BACKEND_FAULT = "backend_fault"
    CANCELLED = "cancelled"


class AuxSource(enum.StrEnum):
    """Source of the auxiliary z/roll/pitch geometry."""

    PNP = "pnp"
    ROAD_PATCH = "road_patch"
    NONE = "none"


class EvidenceKind(enum.StrEnum):
    """Evidence applied by one estimator step."""

    INITIAL_POSE = "initial_pose"
    PNP = "pnp"
    HANDOFF_PNP = "handoff_pnp"
    GROUND_MOTION = "ground_motion"
    BORDER = "border"
    RECOVERY_RESET = "recovery_reset"
    PREDICTION_ONLY = "prediction_only"


class HandoffDisposition(enum.StrEnum):
    """C's decision on a HandoffEvidence, made after the estimator step of its group."""

    ADOPT = "adopt"
    CANDIDATE = "candidate"
    DISCARD = "discard"


class InputEventKind(enum.StrEnum):
    """Kind of an input anomaly.

    ``InputEvent.reference_ns`` means: the previous exposure of that camera for
    ``duplicate_stamp`` and ``decreasing_stamp``; the closed group's time for
    ``late_support``; the initial-pose stamp for ``before_initial_pose``; the episode end
    for ``after_episode_end``; the previous group time for ``coverage_gap``, whose
    ``exposure_ns`` is the next group's time.
    """

    NO_CALIBRATION = "no_calibration"
    CALIBRATION_MISMATCH = "calibration_mismatch"
    BEFORE_INITIAL_POSE = "before_initial_pose"
    AFTER_EPISODE_END = "after_episode_end"
    LATE_SUPPORT = "late_support"
    DUPLICATE_STAMP = "duplicate_stamp"
    DECREASING_STAMP = "decreasing_stamp"
    DECODE_FAILURE = "decode_failure"
    COVERAGE_GAP = "coverage_gap"
    END_OF_INPUT = "end_of_input"


# ---------------------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------------------


class ContractError(ValueError):
    """Structural contract violation, raised at construction or a boundary check."""


class RejectedEvidenceError(ContractError):
    """A rejected or invalid record was presented for consumption."""


class InputError(Exception):
    """A data failure on one input, e.g. an undecodable JPEG. Recorded by C, not a contract bug."""


# ---------------------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------------------

# NumPy scalars that are not plain numbers. np.timedelta64 subclasses np.integer, so it is
# rejected explicitly: int(np.timedelta64(5, "ms")) == 5 would silently mix units.
_NON_NUMERIC_SCALARS = (bool, np.bool_, np.timedelta64, np.datetime64)
_INTEGER_SCALARS = (int, np.integer)
_REAL_SCALARS = (int, float, np.integer, np.floating)

_DECIMAL_NS_PATTERN = re.compile(r"[0-9]+")
_INT64_MAX_DIGITS = len(str(INT64_MAX))
_REASON_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_POSITIVE_REASONS = frozenset({"accepted", "valid"})
_TWO_PI = 2.0 * math.pi

_EYE3 = np.eye(3)
_EYE3.setflags(write=False)
_RIGID_LAST_ROW = np.array([0.0, 0.0, 0.0, 1.0])
_RIGID_LAST_ROW.setflags(write=False)


def _describe(value: object) -> str:
    """Short text for error messages; safe for huge integers and large arrays."""
    if isinstance(value, int) and value.bit_length() > 128:
        return f"<int of {value.bit_length()} bits>"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _plain_str(value: str) -> str:
    """Exact ``str`` content of a ``str`` instance (a StrEnum member gives its value)."""
    return str.__str__(value)


def _check_int(
    value: object, name: str, *, lo: int = 0, hi: int = INT64_MAX, what: str = "an integer"
) -> int:
    """Return an ``int`` (not ``bool``) or NumPy integer as an exact ``int`` in [lo, hi]."""
    if isinstance(value, _NON_NUMERIC_SCALARS) or not isinstance(value, _INTEGER_SCALARS):
        raise ContractError(
            f"{name}: expected {what} (int or numpy integer), "
            f"got {type(value).__name__} {_describe(value)}"
        )
    result = int(value)
    if not lo <= result <= hi:
        raise ContractError(f"{name}: {_describe(result)} is outside [{lo}, {hi}]")
    return result


def as_ns(value: object, name: str = "value") -> int:
    """Return integer nanoseconds as an exact Python ``int`` in [0, 2**63 - 1].

    Accepts ``int`` (not ``bool``) and NumPy integers. Rejects ``bool``, ``np.bool_``, every
    float type (even integral values such as ``1.0``), ``str``, NumPy timedelta/datetime
    scalars and every other type, as well as out-of-range values, with a ``ContractError``
    naming ``name``.
    """
    return _check_int(value, name, what="integer nanoseconds")


def parse_decimal_ns(text: object, name: str = "value") -> int:
    """Return the nanoseconds encoded by an unsigned ASCII decimal string.

    Requires ``type(text) is str`` and ``re.fullmatch(r"[0-9]+", text)``, so signs, spaces,
    a trailing newline, underscores, decimal points, exponents and non-ASCII digits are
    rejected (``int()`` alone accepts several of them, and ``$`` would accept a trailing
    newline). The value must lie in [0, 2**63 - 1]. Violations raise ``ContractError``.
    """
    if type(text) is not str:
        raise ContractError(
            f"{name}: expected a decimal string of nanoseconds, got {type(text).__name__}"
        )
    if _DECIMAL_NS_PATTERN.fullmatch(text) is None:
        raise ContractError(f"{name}: {_describe(text)} is not an unsigned ASCII decimal integer")
    significant = text.lstrip("0")
    if len(significant) > _INT64_MAX_DIGITS:
        raise ContractError(f"{name}: {_describe(text)} is outside [0, {INT64_MAX}]")
    return as_ns(int(significant or "0"), name)


def _check_float(
    value: object, name: str, *, lo: float | None = None, hi: float | None = None
) -> float:
    """Return a finite real non-bool scalar as a Python ``float``, within [lo, hi] if given."""
    if isinstance(value, _NON_NUMERIC_SCALARS) or not isinstance(value, _REAL_SCALARS):
        raise ContractError(
            f"{name}: expected a finite real number, got {type(value).__name__} {_describe(value)}"
        )
    try:
        result = float(value)
    except OverflowError:
        raise ContractError(f"{name}: {_describe(value)} does not fit a float") from None
    if not math.isfinite(result):
        raise ContractError(f"{name}: expected a finite real number, got {result!r}")
    if lo is not None and result < lo:
        raise ContractError(f"{name}: {result!r} is below the minimum {lo!r}")
    if hi is not None and result > hi:
        raise ContractError(f"{name}: {result!r} is above the maximum {hi!r}")
    return result


def wrap_angle(angle: object, name: str = "angle") -> float:
    """Return ``angle`` in radians normalized to (-pi, pi].

    ``math.remainder(angle, 2 pi)`` is exact, so an angle already in (-pi, pi] is returned
    unchanged (including -0.0) and the result is idempotent; -pi maps to pi. Non-finite,
    ``bool`` or non-real input raises ``ContractError``. The organizer ``track_frame.wrap``
    helper agrees modulo 2 pi (it returns -pi for -pi); both scorers wrap yaw differences,
    so the +/-pi representation cannot change a score.
    """
    wrapped = math.remainder(_check_float(angle, name), _TWO_PI)
    return wrapped if wrapped > -math.pi else math.pi


def _shape_text(shape: tuple[int | None, ...]) -> str:
    dims = ["N" if dim is None else str(dim) for dim in shape]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def _frozen_array(
    value: object, name: str, *, dtype: type[np.generic], shape: tuple[int | None, ...]
) -> np.ndarray:
    """Return a read-only, C-contiguous copy of ``value`` converted to ``dtype``.

    Integer targets accept integer input only (an empty input of any real kind is allowed);
    float targets accept integer or float input and require finite values after conversion.
    ``shape`` entries of None match any length. Bool, complex, object and string arrays are
    rejected. The caller's array is never modified.
    """
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name}: not a numeric array ({exc})") from None
    target = np.dtype(dtype)
    integer_target = target.kind in "iu"
    kind = source.dtype.kind
    if kind not in ("iu" if integer_target else "iuf") and not (source.size == 0 and kind in "iuf"):
        raise ContractError(f"{name}: expected a real {target} array, got dtype {source.dtype}")
    if source.ndim != len(shape) or any(
        want is not None and got != want for got, want in zip(source.shape, shape)
    ):
        raise ContractError(f"{name}: expected shape {_shape_text(shape)}, got {source.shape}")
    if integer_target and source.size and not np.can_cast(source.dtype, target):
        info = np.iinfo(target)
        if int(source.min()) < info.min or int(source.max()) > info.max:
            raise ContractError(f"{name}: values outside the {target} range")
    with np.errstate(all="ignore"):
        result = np.array(source, dtype=target, order="C", copy=True)
    if not integer_target and not np.isfinite(result).all():
        raise ContractError(f"{name}: entries must be finite")
    result.setflags(write=False)
    return result


def check_rigid(T: object, name: str = "T") -> np.ndarray:
    """Return a read-only float64 (4, 4) copy of a rigid transform ``T_AB`` (maps B into A).

    Requires a real (4, 4) array with finite entries, ``max |R.T @ R - I| <= ROTATION_TOL``,
    ``|det(R) - 1| <= ROTATION_TOL`` (no reflection or scale) and
    ``max |T[3] - [0, 0, 0, 1]| <= LAST_ROW_TOL``; otherwise raises ``ContractError``. The
    input is not modified.
    """
    matrix = _frozen_array(T, name, dtype=np.float64, shape=(4, 4))
    rotation = matrix[:3, :3]
    with np.errstate(all="ignore"):
        orthogonality = float(np.max(np.abs(rotation.T @ rotation - _EYE3)))
        determinant = float(np.linalg.det(rotation))
        last_row = float(np.max(np.abs(matrix[3] - _RIGID_LAST_ROW)))
    if not orthogonality <= ROTATION_TOL:
        raise ContractError(
            f"{name}: rotation block is not orthonormal (max |R^T R - I| = {orthogonality:.3g})"
        )
    if not abs(determinant - 1.0) <= ROTATION_TOL:
        raise ContractError(
            f"{name}: rotation determinant {determinant:.12g} is not +1 (reflection or scale)"
        )
    if not last_row <= LAST_ROW_TOL:
        raise ContractError(f"{name}: last row must be [0, 0, 0, 1] (max deviation {last_row:.3g})")
    return matrix


def check_covariance(
    P: object, n: int | None = None, name: str = "covariance", definite: bool = False
) -> np.ndarray:
    """Return a read-only float64 (n, n) copy of a symmetric positive semidefinite matrix.

    With ``s = max(1, max |P|)``: finite entries, ``max |P - P.T| <= SYMMETRY_TOL * s`` and
    ``min(eigvalsh((P + P.T) / 2)) >= -PSD_TOL * s``; ``definite=True`` instead requires the
    minimum eigenvalue to be > 0. ``n=None`` accepts any non-empty square matrix. Violations
    raise ``ContractError``. The stored copy is the input, not its symmetrized form.
    """
    if n is None:
        cov = _frozen_array(P, name, dtype=np.float64, shape=(None, None))
        if cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
            raise ContractError(f"{name}: expected a non-empty square matrix, got {cov.shape}")
    else:
        size = _check_int(n, f"{name} size", lo=1)
        cov = _frozen_array(P, name, dtype=np.float64, shape=(size, size))
    with np.errstate(all="ignore"):
        scale = max(1.0, float(np.max(np.abs(cov))))
        asymmetry = float(np.max(np.abs(cov - cov.T)))
        symmetric = 0.5 * cov + 0.5 * cov.T
    if not asymmetry <= SYMMETRY_TOL * scale:
        raise ContractError(f"{name}: not symmetric (max |P - P.T| = {asymmetry:.3g})")
    try:
        smallest = float(np.linalg.eigvalsh(symmetric)[0])
    except np.linalg.LinAlgError as exc:
        raise ContractError(f"{name}: eigenvalue computation failed ({exc})") from None
    if definite:
        if not smallest > 0.0:
            raise ContractError(f"{name}: not positive definite (min eigenvalue {smallest:.3g})")
    elif not smallest >= -PSD_TOL * scale:
        raise ContractError(f"{name}: not positive semidefinite (min eigenvalue {smallest:.3g})")
    return cov


def check_camera_id(camera_id: object, name: str = "camera_id") -> str:
    """Return ``camera_id`` as a plain ``str`` if it is one of ``CAMERA_IDS``."""
    if not isinstance(camera_id, str) or camera_id not in CAMERA_IDS:
        raise ContractError(f"{name}: {_describe(camera_id)} is not one of {CAMERA_IDS}")
    return _plain_str(camera_id)


def check_reason(reason: object, name: str = "reason") -> str:
    """Return ``reason`` if it is a snake_case reason code (``[a-z][a-z0-9_]*`` in full)."""
    if not isinstance(reason, str) or _REASON_PATTERN.fullmatch(reason) is None:
        raise ContractError(f"{name}: {_describe(reason)} is not a snake_case reason code")
    return _plain_str(reason)


def _check_bool(value: object, name: str) -> bool:
    """Return a ``bool`` or ``np.bool_`` as ``bool``; integers are not flags."""
    if not isinstance(value, (bool, np.bool_)):
        raise ContractError(
            f"{name}: expected a bool, got {type(value).__name__} {_describe(value)}"
        )
    return bool(value)


def check_flag_reason(
    flag: object,
    reason: object,
    *,
    ok_reason: str,
    flag_name: str = "accepted",
    reason_name: str = "reason",
) -> tuple[bool, str]:
    """Validate an accepted/valid flag together with its reason; return ``(flag, reason)``.

    True requires ``reason == ok_reason`` ("accepted" for evidence, "valid" for geometry,
    "ok" for anchor results). False requires another snake_case reason that is neither
    ``ok_reason`` nor "accepted" or "valid".
    """
    ok = _check_bool(flag, flag_name)
    text = check_reason(reason, reason_name)
    if ok and text != ok_reason:
        raise ContractError(f"{flag_name}=True requires {reason_name} {ok_reason!r}, got {text!r}")
    if not ok and (text == ok_reason or text in _POSITIVE_REASONS):
        raise ContractError(f"{flag_name}=False requires a rejection {reason_name}, got {text!r}")
    return ok, text


def _check_identity(value: object, name: str) -> str:
    """Return an opaque, non-empty owner-defined identity string (never computed here)."""
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}: expected a non-empty identity string, got {_describe(value)}")
    return _plain_str(value)


def _check_enum[E: enum.Enum](value: object, enum_cls: type[E], name: str) -> E:
    """Return a member of ``enum_cls``; an exact ``str`` equal to a member value is converted.

    Members of other enums are rejected even when their values match (for example
    ``PoseSource.TRACKING`` is not a ``Status``).
    """
    if isinstance(value, enum_cls):
        return value
    if type(value) is str:
        try:
            return enum_cls(value)
        except ValueError:
            pass
    allowed = ", ".join(str(member.value) for member in enum_cls)
    raise ContractError(f"{name}: {_describe(value)} is not a {enum_cls.__name__} ({allowed})")


def _check_ids(value: object, name: str, *, max_len: int | None = None) -> np.ndarray:
    """Return a read-only int64 (N,) copy of unique non-negative identities."""
    ids = _frozen_array(value, name, dtype=np.int64, shape=(None,))
    if max_len is not None and ids.size > max_len:
        raise ContractError(f"{name}: {ids.size} ids exceed the limit of {max_len}")
    if ids.size and int(ids.min()) < 0:
        raise ContractError(f"{name}: ids must be non-negative, got {int(ids.min())}")
    unique, counts = np.unique(ids, return_counts=True)
    if unique.size != ids.size:
        raise ContractError(f"{name}: id {int(unique[counts > 1][0])} repeats; ids must be unique")
    return ids


def _check_id_tuple(values: object, name: str, *, max_len: int | None = None) -> tuple[int, ...]:
    """Return a tuple of unique non-negative integer ids from a tuple or list."""
    if not isinstance(values, (tuple, list)):
        raise ContractError(f"{name}: expected a tuple of integer ids, got {type(values).__name__}")
    ids = tuple(_check_int(value, f"{name}[{index}]") for index, value in enumerate(values))
    if max_len is not None and len(ids) > max_len:
        raise ContractError(f"{name}: {len(ids)} ids exceed the limit of {max_len}")
    if len(set(ids)) != len(ids):
        raise ContractError(f"{name}: ids must be unique, got {ids}")
    return ids


def _optional[T](check: Callable[..., T], value: object, *args: Any, **kwargs: Any) -> T | None:
    """Return None for None, otherwise ``check(value, *args, **kwargs)``."""
    return None if value is None else check(value, *args, **kwargs)


# ---------------------------------------------------------------------------------------
# Record base, summaries and consumption guard
# ---------------------------------------------------------------------------------------


class _Record:
    """Base class of every contract record.

    Subclasses are ``@dataclass(frozen=True, slots=True, kw_only=True)``; ``__post_init__``
    validates and normalizes fields through ``object.__setattr__``. Records holding numpy
    arrays, directly or through nested records, use ``eq=False`` (identity equality). Because
    ``slots=True`` recreates the class, subclass methods must not call zero-argument
    ``super()``. A subclass lists array fields that ``summary()`` always reduces to shape
    and dtype (images, masks) in the ``_SUMMARY_SHAPE_ONLY`` class variable.
    """

    __slots__ = ()
    _SUMMARY_SHAPE_ONLY: ClassVar[frozenset[str]] = frozenset()

    def summary(self) -> dict[str, Any]:
        """JSON-safe diagnostic mapping of this record; see ``summarize``."""
        return summarize(self)


def summarize(record: _Record) -> dict[str, Any]:
    """Return the JSON-safe diagnostic summary of a contract record.

    The result is ``{"record": <class name>, <field>: <value>, ...}`` over the dataclass
    fields in declaration order. Ints stay exact Python ints, floats are finite floats,
    bools stay bools, enums become their string values, tuples become lists and nested
    records become nested summaries. Arrays with at most ``SUMMARY_MAX_ARRAY_ITEMS``
    elements become ``tolist()`` lists; larger arrays and the record's shape-only fields
    (images, masks) become ``{"shape": [...], "dtype": "..."}``; bytes become
    ``{"bytes": len}``. ``json.dumps(result, allow_nan=False)`` succeeds for every valid
    record; read it back with Python ``json``, which keeps integers exact. A value without a
    JSON-safe form (including a non-finite float) raises ``ContractError``.
    """
    if not isinstance(record, _Record) or not dataclasses.is_dataclass(record):
        raise ContractError(f"summary: expected a contract record, got {type(record).__name__}")
    cls = type(record)
    result: dict[str, Any] = {"record": cls.__name__}
    for field in dataclasses.fields(record):
        result[field.name] = _summary_value(
            getattr(record, field.name),
            f"{cls.__name__}.{field.name}",
            shape_only=field.name in cls._SUMMARY_SHAPE_ONLY,
        )
    return result


def _summary_value(value: object, name: str, *, shape_only: bool = False) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if shape_only or value.size > SUMMARY_MAX_ARRAY_ITEMS:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.dtype.kind not in "biuf":
            raise ContractError(f"{name}: dtype {value.dtype} has no JSON-safe summary")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ContractError(f"{name}: non-finite array entry")
        return value.tolist()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, _INTEGER_SCALARS) and not isinstance(value, _NON_NUMERIC_SCALARS):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ContractError(f"{name}: non-finite value {number!r}")
        return number
    if isinstance(value, str):
        return _plain_str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, _Record):
        return summarize(value)
    if isinstance(value, (tuple, list)):
        return [_summary_value(item, f"{name}[{index}]") for index, item in enumerate(value)]
    raise ContractError(f"{name}: {type(value).__name__} has no JSON-safe summary")


def require_accepted[R](record: R) -> R:
    """Return ``record`` unchanged if its ``accepted`` (evidence) or ``valid`` (geometry) flag is True.

    A False flag raises ``RejectedEvidenceError("<RecordType>: <reason>")``. An object without
    exactly one of the two flags, or whose flag is not a bool, raises ``ContractError``.
    """
    kind = type(record).__name__
    flags = [flag for flag in ("accepted", "valid") if hasattr(record, flag)]
    if not flags:
        raise ContractError(f"{kind}: has no accepted/valid flag (not evidence or geometry)")
    if len(flags) > 1:
        raise ContractError(f"{kind}: has both accepted and valid flags")
    flag = getattr(record, flags[0])
    if not isinstance(flag, (bool, np.bool_)):
        raise ContractError(f"{kind}.{flags[0]}: expected a bool, got {type(flag).__name__}")
    if not flag:
        raise RejectedEvidenceError(f"{kind}: {getattr(record, 'reason', 'no reason given')}")
    return record


# ---------------------------------------------------------------------------------------
# Record field helpers
# ---------------------------------------------------------------------------------------

# Canonical image geometry. Pixel centres follow OpenCV, so canonical pixel coordinates lie
# in [-0.5, CANONICAL_WIDTH - 0.5] x [-0.5, CANONICAL_HEIGHT - 0.5].
_CANONICAL_IMAGE_SHAPE = (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3)
_CANONICAL_MASK_SHAPE = (CANONICAL_HEIGHT, CANONICAL_WIDTH)
_PIXEL_MIN = -0.5
_PIXEL_U_MAX = CANONICAL_WIDTH - 0.5
_PIXEL_V_MAX = CANONICAL_HEIGHT - 0.5


def _normalize(
    record: _Record,
    field: str,
    check: Callable[..., Any],
    *args: Any,
    optional: bool = False,
    **kwargs: Any,
) -> Any:
    """Validate one field of a record in ``__post_init__`` and store the normalized value.

    Calls ``check(value, *args, name="<Record>.<field>", **kwargs)`` and writes the result
    with ``object.__setattr__`` (records are frozen). With ``optional=True`` a None value is
    kept without calling ``check``. Returns the stored value.
    """
    value = getattr(record, field)
    if not (optional and value is None):
        value = check(value, *args, name=f"{type(record).__name__}.{field}", **kwargs)
    object.__setattr__(record, field, value)
    return value


def _check_text(value: object, name: str) -> str:
    """Return free diagnostic text (possibly empty) as a plain ``str``."""
    if not isinstance(value, str):
        raise ContractError(f"{name}: expected a str, got {type(value).__name__}")
    return _plain_str(value)


def _check_payload(value: object, name: str) -> bytes:
    """Return a non-empty ``bytes`` payload; ``bytearray`` and ``memoryview`` are rejected."""
    if not isinstance(value, bytes):
        raise ContractError(f"{name}: expected immutable bytes, got {type(value).__name__}")
    if not value:
        raise ContractError(f"{name}: the payload is empty")
    return value


def _check_camera_tuple(values: object, name: str) -> tuple[str, ...]:
    """Return a tuple or list of ids from ``CAMERA_IDS`` as a tuple of plain strings."""
    if not isinstance(values, (tuple, list)):
        raise ContractError(
            f"{name}: expected a tuple of camera ids, got {type(values).__name__}"
        )
    return tuple(
        check_camera_id(camera, f"{name}[{index}]") for index, camera in enumerate(values)
    )


def _check_record_tuple(values: object, item_type: Any, name: str) -> tuple[Any, ...]:
    """Return a tuple or list whose items are all ``item_type`` instances as a tuple."""
    label = getattr(item_type, "__name__", None) or str(item_type)
    if not isinstance(values, (tuple, list)):
        raise ContractError(f"{name}: expected a tuple of {label}, got {type(values).__name__}")
    for index, item in enumerate(values):
        if not isinstance(item, item_type):
            raise ContractError(f"{name}[{index}]: expected {label}, got {type(item).__name__}")
    return tuple(values)


def _check_intrinsics(value: object, name: str) -> np.ndarray:
    """Return a read-only float64 (3, 3) copy of a pinhole camera matrix K.

    Requires finite entries, positive focal lengths ``K[0, 0]`` and ``K[1, 1]``, zero skew
    (``K[0, 1] == K[1, 0] == 0``) and the last row exactly [0, 0, 1]. The principal point is
    bounded separately where the image size is fixed (``Observation``).
    """
    K = _frozen_array(value, name, dtype=np.float64, shape=(3, 3))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if not (fx > 0.0 and fy > 0.0):
        raise ContractError(f"{name}: focal lengths must be positive, got fx={fx!r}, fy={fy!r}")
    if K[0, 1] != 0.0 or K[1, 0] != 0.0:
        skew = (float(K[0, 1]), float(K[1, 0]))
        raise ContractError(f"{name}: skew entries K[0, 1], K[1, 0] must be zero, got {skew}")
    if K[2, 0] != 0.0 or K[2, 1] != 0.0 or K[2, 2] != 1.0:
        raise ContractError(f"{name}: last row must be [0, 0, 1], got {K[2].tolist()}")
    return K


def _check_image_buffer(
    value: object, name: str, *, dtype: type[np.generic], shape: tuple[int, ...]
) -> np.ndarray:
    """Check an image-sized numpy array by type, dtype, shape and C-contiguity only.

    Returns ``value`` itself: no element is read, nothing is copied and no flag is changed
    here. The record marks the array read-only once all of its fields are valid.
    """
    if type(value) is not np.ndarray:
        raise ContractError(f"{name}: expected a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != dtype:
        raise ContractError(f"{name}: expected dtype {np.dtype(dtype)}, got {value.dtype}")
    if value.shape != shape:
        raise ContractError(f"{name}: expected shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ContractError(
            f"{name}: expected a C-contiguous array, got strides {value.strides}"
        )
    return value


def _check_record(value: object, record_type: Any, name: str) -> Any:
    """Return ``value`` if it is an instance of ``record_type`` (a record class or a union).

    A nested record was validated when it was constructed, so nothing else is checked.
    """
    if not isinstance(value, record_type):
        label = getattr(record_type, "__name__", None) or str(record_type)
        raise ContractError(f"{name}: expected {label}, got {type(value).__name__}")
    return value


def _check_unit_vector(value: object, name: str, *, size: int) -> np.ndarray:
    """Return a read-only float64 (size,) copy of a unit vector: ``|norm - 1| <= UNIT_TOL``."""
    vector = _frozen_array(value, name, dtype=np.float64, shape=(size,))
    with np.errstate(all="ignore"):
        norm = float(np.linalg.norm(vector))
    if not abs(norm - 1.0) <= UNIT_TOL:
        raise ContractError(f"{name}: expected a unit vector, got norm {norm!r}")
    return vector


def _normalize_flag(record: _Record, flag_field: str, *, ok_reason: str) -> bool:
    """Validate a record's ``accepted``/``valid`` flag together with its ``reason`` field.

    Applies ``check_flag_reason`` with ``ok_reason`` ("accepted" for evidence, "valid" for
    geometry), stores the normalized flag and reason and returns the flag.
    """
    kind = type(record).__name__
    flag, reason = check_flag_reason(
        getattr(record, flag_field),
        getattr(record, "reason"),
        ok_reason=ok_reason,
        flag_name=f"{kind}.{flag_field}",
        reason_name=f"{kind}.reason",
    )
    object.__setattr__(record, flag_field, flag)
    object.__setattr__(record, "reason", reason)
    return flag


def _require_present(record: _Record, fields: tuple[str, ...], condition: str) -> None:
    """Raise ``ContractError`` naming every field of ``fields`` that is None under ``condition``."""
    missing = [field for field in fields if getattr(record, field) is None]
    if missing:
        raise ContractError(
            f"{type(record).__name__}: {condition} requires {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------------------
# Input records (produced by A, grouped and decoded under C's runner)
# ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ImageMessage(_Record):
    """One undecoded camera image from A's reader; grouping never decodes it.

    ``exposure_ns`` is the image header stamp; ``arrival_ns`` is the MCAP log time in replay
    (the actual arrival time live). ``source_id`` is opaque and unique per message within a
    run (A-defined), ``frame_id`` is the header frame, ``image_format`` the CompressedImage
    format string (for example "jpeg") and ``calibration_id`` the frozen calibration that
    applies to this camera. ``data`` is the non-empty compressed payload as immutable
    ``bytes`` (``bytearray`` and ``memoryview`` are rejected); it is left out of ``repr``
    and summarized by its length only.
    """

    camera_id: str
    exposure_ns: int
    arrival_ns: int
    source_id: str
    frame_id: str
    image_format: str
    calibration_id: str
    data: bytes = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "exposure_ns", as_ns)
        _normalize(self, "arrival_ns", as_ns)
        for field_name in ("source_id", "frame_id", "image_format", "calibration_id"):
            _normalize(self, field_name, _check_identity)
        _normalize(self, "data", _check_payload)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InputEvent(_Record):
    """One input anomaly, attached by A to the next FrameSet or to ``trailing_events()``.

    ``log_ns`` is the event's log time. The optional fields name the camera, image exposure,
    reference time and source message involved; ``InputEventKind`` documents what
    ``reference_ns`` means for each kind. C creates ``decode_failure`` events itself when
    ``Canonicalizer.canonicalize`` raises ``InputError``. ``detail`` is free text.
    """

    kind: InputEventKind
    log_ns: int
    camera_id: str | None = None
    exposure_ns: int | None = None
    reference_ns: int | None = None
    source_id: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        _normalize(self, "kind", _check_enum, InputEventKind)
        _normalize(self, "log_ns", as_ns)
        _normalize(self, "camera_id", check_camera_id, optional=True)
        _normalize(self, "exposure_ns", as_ns, optional=True)
        _normalize(self, "reference_ns", as_ns, optional=True)
        _normalize(self, "source_id", _check_identity, optional=True)
        _normalize(self, "detail", _check_text)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CameraCalibration(_Record):
    """Frozen first-available calibration of one camera (camera_info and /tf_static).

    ``K_raw`` (float64 3 x 3) and ``D_raw`` (float64 (5,), plumb_bob k1, k2, p1, p2, k3)
    describe the raw ``raw_width`` x ``raw_height`` image (both positive). ``K`` is the
    canonical K that A derives from ``K_raw`` (never from P) as in the module conventions.
    Both K have positive focal lengths, zero skew and last row [0, 0, 1]. ``T_BC`` is the
    rigid camera-to-body transform. Every array is stored as a read-only float64 copy.
    """

    camera_id: str
    calibration_id: str
    preprocessing_id: str
    frame_id: str
    raw_width: int
    raw_height: int
    K_raw: np.ndarray
    D_raw: np.ndarray
    K: np.ndarray
    T_BC: np.ndarray

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        for field_name in ("calibration_id", "preprocessing_id", "frame_id"):
            _normalize(self, field_name, _check_identity)
        _normalize(self, "raw_width", _check_int, lo=1, what="a positive pixel count")
        _normalize(self, "raw_height", _check_int, lo=1, what="a positive pixel count")
        _normalize(self, "K_raw", _check_intrinsics)
        _normalize(self, "D_raw", _frozen_array, dtype=np.float64, shape=(5,))
        _normalize(self, "K", _check_intrinsics)
        _normalize(self, "T_BC", check_rigid)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Observation(_Record):
    """One canonical image, produced by A's Canonicalizer from one ImageMessage.

    ``image`` is RGB ``uint8`` (464, 960, 3) and ``valid_mask`` ``bool`` (464, 960), True =
    usable; both must be C-contiguous ``numpy.ndarray`` objects. They are checked by type,
    dtype, shape and flags only, stored as the caller's own objects (never copied) and
    marked read-only in place once the whole record is valid, so the producer must not reuse
    their buffers afterwards. ``K`` is the canonical K: positive focal lengths, zero skew,
    last row [0, 0, 1] and principal point inside [-0.5, 959.5] x [-0.5, 463.5]. ``T_BC``
    is the rigid camera-to-body transform. Summaries give the image and mask as shape and
    dtype only.
    """

    camera_id: str
    exposure_ns: int
    arrival_ns: int
    source_id: str
    image: np.ndarray
    valid_mask: np.ndarray
    calibration_id: str
    preprocessing_id: str
    K: np.ndarray
    T_BC: np.ndarray

    _SUMMARY_SHAPE_ONLY: ClassVar[frozenset[str]] = frozenset({"image", "valid_mask"})

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "exposure_ns", as_ns)
        _normalize(self, "arrival_ns", as_ns)
        _normalize(self, "source_id", _check_identity)
        image = _normalize(
            self, "image", _check_image_buffer, dtype=np.uint8, shape=_CANONICAL_IMAGE_SHAPE
        )
        valid_mask = _normalize(
            self, "valid_mask", _check_image_buffer, dtype=np.bool_, shape=_CANONICAL_MASK_SHAPE
        )
        _normalize(self, "calibration_id", _check_identity)
        _normalize(self, "preprocessing_id", _check_identity)
        K = _normalize(self, "K", _check_intrinsics)
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if not (_PIXEL_MIN <= cx <= _PIXEL_U_MAX and _PIXEL_MIN <= cy <= _PIXEL_V_MAX):
            raise ContractError(
                f"Observation.K: principal point ({cx!r}, {cy!r}) lies outside the canonical "
                f"pixel range [{_PIXEL_MIN}, {_PIXEL_U_MAX}] x [{_PIXEL_MIN}, {_PIXEL_V_MAX}]"
            )
        _normalize(self, "T_BC", check_rigid)
        # Only a valid record marks the caller's buffers read-only; a rejected one leaves them.
        image.setflags(write=False)
        valid_mask.setflags(write=False)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FrameSet(_Record):
    """One header-only camera group from A's FrameSetSource; its images stay undecoded.

    ``group_id`` is >= 0. ``selected_camera_ids`` (non-empty, unique) are the cameras the
    group waits for, in order. ``images`` holds at least one ImageMessage, each camera at
    most once, ordered as selected; ``missing_camera_ids`` are the selected cameras without
    an image, in selected order. ``t_ns`` is the group time: the maximum image exposure and
    the CSV timestamp. Image exposures span at most ``MAX_EXPOSURE_SPREAD_NS`` and arrivals
    at most ``MAX_ARRIVAL_SPREAD_NS``. ``closure`` is ``complete`` exactly when no selected
    camera is missing; ``terminal_partial`` is True exactly when ``closure`` is ``eof``.
    ``input_events`` are the anomalies since the previous FrameSet, in non-decreasing
    ``log_ns`` order. Cross-group guarantees (``group_id`` 0, 1, 2, ..., strictly
    increasing ``t_ns``, no image reused) belong to the FrameSetSource and are re-checked by
    C's runner.
    """

    group_id: int
    t_ns: int
    selected_camera_ids: tuple[str, ...]
    images: tuple[ImageMessage, ...]
    missing_camera_ids: tuple[str, ...]
    closure: FrameSetClosure
    terminal_partial: bool
    input_events: tuple[InputEvent, ...]

    def __post_init__(self) -> None:
        _normalize(self, "group_id", _check_int)
        t_ns = _normalize(self, "t_ns", as_ns)
        selected = _normalize(self, "selected_camera_ids", _check_camera_tuple)
        if not selected:
            raise ContractError("FrameSet.selected_camera_ids: at least one camera is required")
        if len(set(selected)) != len(selected):
            raise ContractError(f"FrameSet.selected_camera_ids: cameras repeat in {selected}")

        images = _normalize(self, "images", _check_record_tuple, ImageMessage)
        if not images:
            raise ContractError("FrameSet.images: at least one image is required")
        rank = {camera: index for index, camera in enumerate(selected)}
        cameras = tuple(image.camera_id for image in images)
        for index, camera in enumerate(cameras):
            if camera not in rank:
                raise ContractError(
                    f"FrameSet.images[{index}]: camera {camera!r} is not selected {selected}"
                )
        if len(set(cameras)) != len(cameras):
            raise ContractError(f"FrameSet.images: a camera appears more than once in {cameras}")
        if any(rank[earlier] > rank[later] for earlier, later in zip(cameras, cameras[1:])):
            raise ContractError(
                f"FrameSet.images: cameras {cameras} are not in the selected order {selected}"
            )

        missing = _normalize(self, "missing_camera_ids", _check_camera_tuple)
        expected = tuple(camera for camera in selected if camera not in cameras)
        if missing != expected:
            raise ContractError(
                f"FrameSet.missing_camera_ids: expected {expected} (the selected cameras "
                f"without an image, in selected order), got {missing}"
            )
        closure = _normalize(self, "closure", _check_enum, FrameSetClosure)
        terminal_partial = _normalize(self, "terminal_partial", _check_bool)
        if (closure is FrameSetClosure.COMPLETE) != (not missing):
            raise ContractError(
                f"FrameSet.closure: {closure.value!r} with missing cameras {missing}; closure "
                "is 'complete' exactly when no selected camera is missing"
            )
        if terminal_partial != (closure is FrameSetClosure.EOF):
            raise ContractError(
                f"FrameSet.terminal_partial: {terminal_partial} with closure "
                f"{closure.value!r}; it is True exactly when closure is 'eof'"
            )

        exposures = [image.exposure_ns for image in images]
        if t_ns != max(exposures):
            raise ContractError(
                f"FrameSet.t_ns: {t_ns} must equal the maximum image exposure {max(exposures)}"
            )
        exposure_spread = max(exposures) - min(exposures)
        if exposure_spread > MAX_EXPOSURE_SPREAD_NS:
            raise ContractError(
                f"FrameSet.images: exposures span {exposure_spread} ns, more than "
                f"MAX_EXPOSURE_SPREAD_NS = {MAX_EXPOSURE_SPREAD_NS}"
            )
        arrivals = [image.arrival_ns for image in images]
        arrival_spread = max(arrivals) - min(arrivals)
        if arrival_spread > MAX_ARRIVAL_SPREAD_NS:
            raise ContractError(
                f"FrameSet.images: arrivals span {arrival_spread} ns, more than "
                f"MAX_ARRIVAL_SPREAD_NS = {MAX_ARRIVAL_SPREAD_NS}"
            )

        events = _normalize(self, "input_events", _check_record_tuple, InputEvent)
        for index in range(1, len(events)):
            if events[index].log_ns < events[index - 1].log_ns:
                raise ContractError(
                    f"FrameSet.input_events[{index}]: log_ns {events[index].log_ns} precedes "
                    f"{events[index - 1].log_ns}; events must be in non-decreasing log_ns order"
                )


# ---------------------------------------------------------------------------------------
# State and geometry records (A's road geometry and track coordinates, C's estimator)
# ---------------------------------------------------------------------------------------

_YAW_INDEX = STATE_FIELDS.index("yaw_rad")

# Auxiliary-geometry values: the pose values a valid record needs, plus the patch identity.
_AUX_POSE_FIELDS = ("source_ns", "z_m", "roll_rad", "pitch_rad")
_AUX_VALUE_FIELDS = (*_AUX_POSE_FIELDS, "road_patch_id")
_ROAD_PATCH_VALID_FIELDS = ("point_W", "normal_W", "residual_m")
_PROJECTION_VALUE_FIELDS = (
    "s_m",
    "n_m",
    "epsi_rad",
    "segment_index",
    "xy_reconstruction_residual_m",
)

# The evidence combinations one estimator step may report (Estimate.evidence_kinds).
_INITIALIZATION_KINDS = (EvidenceKind.INITIAL_POSE,)
_PREDICTION_ONLY_KINDS = (EvidenceKind.PREDICTION_ONLY,)
_ESTIMATE_EVIDENCE_KINDS = (
    _INITIALIZATION_KINDS,
    (EvidenceKind.PNP,),
    (EvidenceKind.HANDOFF_PNP,),
    (EvidenceKind.RECOVERY_RESET,),
    (EvidenceKind.GROUND_MOTION,),
    (EvidenceKind.BORDER,),
    (EvidenceKind.GROUND_MOTION, EvidenceKind.BORDER),
    _PREDICTION_ONLY_KINDS,
)


def _kinds_text(kinds: tuple[EvidenceKind, ...]) -> str:
    """Return evidence kinds in tuple notation, e.g. "(ground_motion, border)"."""
    values = [kind.value for kind in kinds]
    return "(" + ", ".join(values) + ("," if len(values) == 1 else "") + ")"


def _check_state(value: object, name: str) -> np.ndarray:
    """Return a read-only float64 (6,) copy of an estimator state with yaw in (-pi, pi].

    The layout is ``STATE_FIELDS`` and every entry must be finite. Only ``yaw_rad`` is an
    angle and is stored as ``wrap_angle`` returns it; ``yaw_rate_radps`` is a rate.
    """
    frozen = _frozen_array(value, name, dtype=np.float64, shape=(len(STATE_FIELDS),))
    state = np.array(frozen)  # writable copy that receives the wrapped yaw
    state[_YAW_INDEX] = wrap_angle(frozen[_YAW_INDEX], f"{name}[{_YAW_INDEX}]")
    state.setflags(write=False)
    return state


def _check_route_id(value: object, name: str) -> str:
    """Return ``value`` as a plain ``str`` if it is one of ``ROUTE_IDS``."""
    if not isinstance(value, str) or value not in ROUTE_IDS:
        raise ContractError(f"{name}: {_describe(value)} is not one of {ROUTE_IDS}")
    return _plain_str(value)


def _check_evidence_kinds(values: object, name: str) -> tuple[EvidenceKind, ...]:
    """Return a tuple or list of evidence kinds as a member tuple if Estimate allows it."""
    if not isinstance(values, (tuple, list)):
        raise ContractError(
            f"{name}: expected a tuple of EvidenceKind, got {type(values).__name__}"
        )
    kinds = tuple(
        _check_enum(value, EvidenceKind, f"{name}[{index}]") for index, value in enumerate(values)
    )
    if kinds not in _ESTIMATE_EVIDENCE_KINDS:
        allowed = ", ".join(_kinds_text(combination) for combination in _ESTIMATE_EVIDENCE_KINDS)
        raise ContractError(
            f"{name}: {_kinds_text(kinds)} is not an allowed combination "
            f"(expected one of {allowed})"
        )
    return kinds


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AuxiliaryGeometry(_Record):
    """Auxiliary z/roll/pitch that compose a full body pose with the planar state.

    These are not estimator states and are never filled from GT. Source ``pnp``: taken from
    the latest accepted current-image PnP ``T_WB``; ``source_ns`` is its measurement
    exposure. Source ``road_patch``: the plane height plus a fitting-only body-to-road height
    offset, and the plane roll/pitch, of the RoadPatch named by ``road_patch_id``. Source
    ``none``: nothing is available. A valid record has a source other than ``none`` and
    ``source_ns``, ``z_m``, ``roll_rad`` and ``pitch_rad``; a valid ``road_patch`` record
    also names ``road_patch_id``. A ``none`` record is invalid and every value is None;
    ``unavailable(reason)`` builds it. Roll and pitch are stored wrapped to (-pi, pi].
    """

    source: AuxSource
    source_ns: int | None
    z_m: float | None
    roll_rad: float | None
    pitch_rad: float | None
    road_patch_id: str | None
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        source = _normalize(self, "source", _check_enum, AuxSource)
        _normalize(self, "source_ns", as_ns, optional=True)
        _normalize(self, "z_m", _check_float, optional=True)
        _normalize(self, "roll_rad", wrap_angle, optional=True)
        _normalize(self, "pitch_rad", wrap_angle, optional=True)
        _normalize(self, "road_patch_id", _check_identity, optional=True)
        valid = _normalize_flag(self, "valid", ok_reason="valid")
        if source is AuxSource.NONE:
            if valid:
                raise ContractError("AuxiliaryGeometry: source 'none' cannot be valid")
            present = [field for field in _AUX_VALUE_FIELDS if getattr(self, field) is not None]
            if present:
                raise ContractError(
                    f"AuxiliaryGeometry: source 'none' carries no values, got {', '.join(present)}"
                )
        elif valid:
            patch_field = ("road_patch_id",) if source is AuxSource.ROAD_PATCH else ()
            _require_present(
                self, _AUX_POSE_FIELDS + patch_field, f"valid=True with source {source.value!r}"
            )

    @classmethod
    def unavailable(cls, reason: str) -> Self:
        """Return the invalid source ``none`` record: every value None, with ``reason``."""
        return cls(
            source=AuxSource.NONE,
            source_ns=None,
            z_m=None,
            roll_rad=None,
            pitch_rad=None,
            road_patch_id=None,
            valid=False,
            reason=reason,
        )


def _check_auxiliary(value: object, name: str, *, t_ns: int) -> AuxiliaryGeometry:
    auxiliary = _check_record(value, AuxiliaryGeometry, name)
    if auxiliary.source_ns is not None and auxiliary.source_ns > t_ns:
        raise ContractError(
            f"{name}.source_ns: {auxiliary.source_ns} is after t_ns {t_ns}; "
            "auxiliary geometry cannot come from later evidence"
        )
    return auxiliary


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class RoadPatch(_Record):
    """Local road plane from A's RoadGeometry for one queried pose.

    C requests at most one patch per group and passes the same record to B's ground motion
    and to C's auxiliary geometry. ``query_xy`` (float64 (2,)) is the queried map position.
    When present, ``point_W`` (float64 (3,)) is a point on the plane, ``normal_W`` is a
    float64 (3,) unit normal (within UNIT_TOL) pointing up (z > 0) and ``residual_m`` is
    >= 0. ``support`` is A's non-negative support count. A valid patch has ``point_W``,
    ``normal_W``, ``residual_m`` and branch ``track`` or ``pit``. An invalid patch keeps
    its reason and disables ground motion and border updates for that group. Arrays are
    stored as read-only float64 copies.
    """

    patch_id: str
    branch: Branch
    query_xy: np.ndarray
    point_W: np.ndarray | None
    normal_W: np.ndarray | None
    support: int
    residual_m: float | None
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "patch_id", _check_identity)
        branch = _normalize(self, "branch", _check_enum, Branch)
        _normalize(self, "query_xy", _frozen_array, dtype=np.float64, shape=(2,))
        _normalize(self, "point_W", _frozen_array, dtype=np.float64, shape=(3,), optional=True)
        normal = _normalize(self, "normal_W", _check_unit_vector, size=3, optional=True)
        if normal is not None and not normal[2] > 0.0:
            raise ContractError(
                f"RoadPatch.normal_W: the plane normal must point up (z > 0), got {normal.tolist()}"
            )
        _normalize(self, "support", _check_int, what="a support count")
        _normalize(self, "residual_m", _check_float, lo=0.0, optional=True)
        if _normalize_flag(self, "valid", ok_reason="valid"):
            _require_present(self, _ROAD_PATCH_VALID_FIELDS, "valid=True")
            if branch is Branch.UNKNOWN:
                raise ContractError(
                    "RoadPatch.branch: a valid patch needs branch 'track' or 'pit', got 'unknown'"
                )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class SearchSnapshot(_Record):
    """Frozen estimator prior at ``t_ns`` for LK seeding, reference search and anchor jobs.

    ``Estimator.snapshot(t_ns)`` returns the constant-velocity prediction without changing
    the estimator. C also caches exposure-aligned snapshots for ground motion. ``state``
    is a read-only float64 (6,) copy in
    ``STATE_FIELDS`` order with yaw wrapped to (-pi, pi]; ``P`` is its symmetric positive
    semidefinite 6 x 6 covariance, stored as a read-only float64 copy. ``auxiliary`` is the
    auxiliary geometry carried with the prior; its source time cannot exceed ``t_ns``.
    """

    t_ns: int
    state: np.ndarray
    P: np.ndarray
    status: Status
    branch: Branch
    auxiliary: AuxiliaryGeometry

    def __post_init__(self) -> None:
        t_ns = _normalize(self, "t_ns", as_ns)
        _normalize(self, "state", _check_state)
        _normalize(self, "P", check_covariance, len(STATE_FIELDS))
        _normalize(self, "status", _check_enum, Status)
        _normalize(self, "branch", _check_enum, Branch)
        _normalize(self, "auxiliary", _check_auxiliary, t_ns=t_ns)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Estimate(_Record):
    """Immutable output of one ``Estimator.step``, or the diagnostic initialization estimate.

    ``t_ns`` is the group time (the CSV timestamp) and ``group_id`` the FrameSet's id. The
    initialization estimate built from the initial pose has ``group_id`` None and is never
    written to the CSV. ``state`` and ``P`` follow SearchSnapshot. ``evidence_kinds`` is
    what this step applied, one of (initial_pose,), (pnp,), (handoff_pnp,),
    (recovery_reset,), (ground_motion,), (border,), (ground_motion, border) or
    (prediction_only,); ``group_id`` is None exactly for (initial_pose,). ``reason`` is a
    snake_case code such as "accepted", "no_evidence" or "innovation_gate".
    ``max_support_exposure_ns`` is the latest exposure of the images whose evidence was
    applied: None exactly for (initial_pose,) and (prediction_only,), otherwise at most
    ``t_ns``. Auxiliary geometry likewise cannot have a source time after ``t_ns``.
    """

    t_ns: int
    group_id: int | None
    state: np.ndarray
    P: np.ndarray
    status: Status
    branch: Branch
    evidence_kinds: tuple[EvidenceKind, ...]
    reason: str
    auxiliary: AuxiliaryGeometry
    max_support_exposure_ns: int | None

    def __post_init__(self) -> None:
        t_ns = _normalize(self, "t_ns", as_ns)
        group_id = _normalize(self, "group_id", _check_int, what="a group id", optional=True)
        _normalize(self, "state", _check_state)
        _normalize(self, "P", check_covariance, len(STATE_FIELDS))
        _normalize(self, "status", _check_enum, Status)
        _normalize(self, "branch", _check_enum, Branch)
        kinds = _normalize(self, "evidence_kinds", _check_evidence_kinds)
        _normalize(self, "reason", check_reason)
        _normalize(self, "auxiliary", _check_auxiliary, t_ns=t_ns)
        support_ns = _normalize(self, "max_support_exposure_ns", as_ns, optional=True)

        kinds_text = _kinds_text(kinds)
        if (group_id is None) != (kinds == _INITIALIZATION_KINDS):
            raise ContractError(
                f"Estimate.group_id: {group_id} with evidence_kinds {kinds_text}; group_id is "
                "None exactly for the initialization estimate (initial_pose,)"
            )
        if kinds in (_INITIALIZATION_KINDS, _PREDICTION_ONLY_KINDS):
            if support_ns is not None:
                raise ContractError(
                    "Estimate.max_support_exposure_ns: must be None for evidence_kinds "
                    f"{kinds_text}, got {support_ns}"
                )
        elif support_ns is None:
            raise ContractError(
                f"Estimate.max_support_exposure_ns: required for evidence_kinds {kinds_text}"
            )
        elif support_ns > t_ns:
            raise ContractError(
                f"Estimate.max_support_exposure_ns: {support_ns} is after t_ns {t_ns}; applied "
                "evidence cannot come from a later image"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TrackProjection(_Record):
    """Route coordinates of an Estimate's own x/y/yaw from A's TrackProjector; never from GT.

    ``route_id`` is one of ``ROUTE_IDS`` and ``reference_map_id`` the opaque identity of the
    reference geometry. ``s_m`` is the XY arc length modulo the route length, ``n_m`` is
    positive to the left of the segment direction, ``epsi_rad`` is the car yaw minus the
    segment heading (stored wrapped to (-pi, pi]), ``segment_index`` is the matched segment
    and ``xy_reconstruction_residual_m`` is the XY distance between the estimate and its
    XY -> (s, n) -> XY reconstruction. Present values satisfy ``s_m >= 0``,
    ``segment_index >= 0`` and a residual >= 0; a valid projection has every value.
    """

    t_ns: int
    reference_map_id: str
    route_id: str
    s_m: float | None
    n_m: float | None
    epsi_rad: float | None
    segment_index: int | None
    xy_reconstruction_residual_m: float | None
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "t_ns", as_ns)
        _normalize(self, "reference_map_id", _check_identity)
        _normalize(self, "route_id", _check_route_id)
        _normalize(self, "s_m", _check_float, lo=0.0, optional=True)
        _normalize(self, "n_m", _check_float, optional=True)
        _normalize(self, "epsi_rad", wrap_angle, optional=True)
        _normalize(self, "segment_index", _check_int, what="a segment index", optional=True)
        _normalize(self, "xy_reconstruction_residual_m", _check_float, lo=0.0, optional=True)
        if _normalize_flag(self, "valid", ok_reason="valid"):
            _require_present(self, _PROJECTION_VALUE_FIELDS, "valid=True")


# ---------------------------------------------------------------------------------------
# Evidence records (B's tracking, handoff and ground motion; A's borders). Weak evidence
# stays a record with accepted=False and a reason; C decides what each group consumes.
# ---------------------------------------------------------------------------------------

_UNIT_Z = np.array([0.0, 0.0, 1.0])
_UNIT_Z.setflags(write=False)

# Optional values that an accepted record must carry.
_POSE_ACCEPTED_FIELDS = ("T_WB", "median_error_px", "p90_error_px")
_HANDOFF_ACCEPTED_FIELDS = ("correspondences", "pose")
_MOTION_ACCEPTED_FIELDS = ("T_Bprev_Bcurr", "covariance", "ground_residual_m", "road_patch_id")
_BORDER_ACCEPTED_FIELDS = ("lateral_coordinate_m", "yaw_rad", "R")
# Identities an accepted HandoffEvidence shares with the pose it carries.
_HANDOFF_POSE_IDENTITIES = ("map_id", "preprocessing_id", "calibration_id")


def _check_pixels(value: object, name: str) -> np.ndarray:
    """Return a read-only float32 (N, 2) copy of canonical (u, v) pixel coordinates.

    Integer or float input is converted to float32, rounding to nearest (keypoints are never
    stored as float16). Every stored coordinate must lie in [-0.5, 959.5] x [-0.5, 463.5],
    the OpenCV pixel-centre range of the 960 x 464 canonical image. The bounds are exact
    float32 values, so rounding never moves an inside coordinate outside.
    """
    pixels = _frozen_array(value, name, dtype=np.float32, shape=(None, 2))
    u, v = pixels[:, 0], pixels[:, 1]
    inside = (u >= _PIXEL_MIN) & (u <= _PIXEL_U_MAX) & (v >= _PIXEL_MIN) & (v <= _PIXEL_V_MAX)
    if not inside.all():
        row = int(np.flatnonzero(~inside)[0])
        raise ContractError(
            f"{name}: row {row} {pixels[row].tolist()} lies outside the canonical pixel range "
            f"[{_PIXEL_MIN}, {_PIXEL_U_MAX}] x [{_PIXEL_MIN}, {_PIXEL_V_MAX}]"
        )
    return pixels


def _check_ns_tuple(values: object, name: str, *, max_len: int | None = None) -> tuple[int, ...]:
    """Return a tuple or list of integer nanosecond stamps as a tuple of exact ``int``."""
    if not isinstance(values, (tuple, list)):
        raise ContractError(
            f"{name}: expected a tuple of integer nanoseconds, got {type(values).__name__}"
        )
    if max_len is not None and len(values) > max_len:
        raise ContractError(f"{name}: {len(values)} stamps exceed the limit of {max_len}")
    return tuple(as_ns(value, f"{name}[{index}]") for index, value in enumerate(values))


def _check_planar_rigid(value: object, name: str) -> np.ndarray:
    """Return a read-only float64 (4, 4) copy of a planar rigid transform.

    On top of ``check_rigid``, the rotation is about z only (every entry of
    ``R[2, :] - [0, 0, 1]`` and of ``R[:, 2] - [0, 0, 1]`` is within ROTATION_TOL in
    magnitude) and the z translation satisfies ``|t_z| <= LAST_ROW_TOL``.
    """
    transform = check_rigid(value, name)
    rotation = transform[:3, :3]
    tilt = max(
        float(np.max(np.abs(rotation[2, :] - _UNIT_Z))),
        float(np.max(np.abs(rotation[:, 2] - _UNIT_Z))),
    )
    if not tilt <= ROTATION_TOL:
        raise ContractError(
            f"{name}: the rotation is not about z only (row and column 2 deviate from "
            f"[0, 0, 1] by up to {tilt:.3g})"
        )
    t_z = float(transform[2, 3])
    if not abs(t_z) <= LAST_ROW_TOL:
        raise ContractError(f"{name}: a planar transform has no z translation, got {t_z!r}")
    return transform


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Correspondences(_Record):
    """Unique keypoint-to-landmark correspondences on one canonical image.

    ``camera_id`` and ``exposure_ns`` name the image the pixels belong to: the source image
    in an AnchorResult, the current image in a HandoffEvidence. Row i pairs keypoint
    ``keypoint_ids[i]`` at canonical pixel ``pixels[i] = (u, v)`` with landmark
    ``landmark_ids[i]`` of map ``map_id``. Both id arrays are int64 (N,), non-negative and
    unique within the record, so a duplicated match cannot inflate PnP support. ``pixels``
    is float32 (N, 2) inside [-0.5, 959.5] x [-0.5, 463.5]. N is the same for all three
    arrays, at most MAX_KEYPOINTS, and may be zero. Arrays are stored as read-only copies.
    """

    camera_id: str
    exposure_ns: int
    keypoint_ids: np.ndarray
    pixels: np.ndarray
    landmark_ids: np.ndarray
    map_id: str

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "exposure_ns", as_ns)
        keypoint_ids = _normalize(self, "keypoint_ids", _check_ids, max_len=MAX_KEYPOINTS)
        pixels = _normalize(self, "pixels", _check_pixels)
        landmark_ids = _normalize(self, "landmark_ids", _check_ids, max_len=MAX_KEYPOINTS)
        _normalize(self, "map_id", _check_identity)
        if not keypoint_ids.size == pixels.shape[0] == landmark_ids.size:
            raise ContractError(
                f"Correspondences: keypoint_ids ({keypoint_ids.size}), pixels "
                f"({pixels.shape[0]} rows) and landmark_ids ({landmark_ids.size}) must have "
                "one entry per correspondence"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PoseEvidence(_Record):
    """B's geometric verdict on one RANSAC PnP solve; C applies the innovation gate.

    ``camera_id`` and ``exposure_ns`` name the image the pose was solved on and ``source``
    the track set it came from: ``tracking`` (the active mapped tracks), ``handoff`` (points
    carried from an anchor job's source image to this image) or ``candidate`` (the held
    recovery candidate's own tracks). ``T_WB`` is the rigid body pose, inv(T_CW) @ inv(T_BC)
    of the solve. ``track_ids`` and ``landmark_ids`` (int64 (N,), each unique and
    non-negative) identify the inliers, so ``inlier_count`` equals N; it is at most
    MAX_ACTIVE_TRACKS for ``tracking`` and MAX_HANDOFF_POINTS for ``handoff`` and
    ``candidate``. ``inlier_fraction`` is in [0, 1], present reprojection errors are >= 0
    with median <= p90, and ``occupied_cells`` counts the occupied cells of the 4 x 4 image
    grid (0..GRID_CELLS). An accepted record has ``T_WB``, both error statistics, at least
    one inlier and at least one occupied cell; inlier and error gates are B's thresholds,
    not contract rules. A rejected record keeps its reason and whatever was measured.
    """

    camera_id: str
    exposure_ns: int
    source: PoseSource
    T_WB: np.ndarray | None
    inlier_count: int
    inlier_fraction: float
    median_error_px: float | None
    p90_error_px: float | None
    occupied_cells: int
    track_ids: np.ndarray
    landmark_ids: np.ndarray
    map_id: str
    calibration_id: str
    preprocessing_id: str
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "exposure_ns", as_ns)
        source = _normalize(self, "source", _check_enum, PoseSource)
        _normalize(self, "T_WB", check_rigid, optional=True)
        limit = MAX_ACTIVE_TRACKS if source is PoseSource.TRACKING else MAX_HANDOFF_POINTS
        inlier_count = _normalize(self, "inlier_count", _check_int, what="an inlier count")
        if inlier_count > limit:
            raise ContractError(
                f"PoseEvidence.inlier_count: {inlier_count} exceeds the limit of {limit} for "
                f"source {source.value!r}"
            )
        _normalize(self, "inlier_fraction", _check_float, lo=0.0, hi=1.0)
        median = _normalize(self, "median_error_px", _check_float, lo=0.0, optional=True)
        p90 = _normalize(self, "p90_error_px", _check_float, lo=0.0, optional=True)
        if median is not None and p90 is not None and median > p90:
            raise ContractError(
                f"PoseEvidence.median_error_px: {median!r} is above p90_error_px {p90!r}"
            )
        cells = _normalize(self, "occupied_cells", _check_int, hi=GRID_CELLS, what="a cell count")
        track_ids = _normalize(self, "track_ids", _check_ids, max_len=limit)
        landmark_ids = _normalize(self, "landmark_ids", _check_ids, max_len=limit)
        if not inlier_count == track_ids.size == landmark_ids.size:
            raise ContractError(
                f"PoseEvidence.inlier_count: {inlier_count} must equal the lengths of "
                f"track_ids ({track_ids.size}) and landmark_ids ({landmark_ids.size})"
            )
        for field_name in ("map_id", "calibration_id", "preprocessing_id"):
            _normalize(self, field_name, _check_identity)
        if _normalize_flag(self, "accepted", ok_reason="accepted"):
            _require_present(self, _POSE_ACCEPTED_FIELDS, "accepted=True")
            if inlier_count < 1 or cells < 1:
                raise ContractError(
                    "PoseEvidence: accepted=True requires at least one inlier and one occupied "
                    f"cell, got inlier_count={inlier_count}, occupied_cells={cells}"
                )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class HandoffEvidence(_Record):
    """B's current-image PnP for an anchor job: how an anchor result reaches the estimator.

    The job ``job_id`` matched its source image at ``source_exposure_ns``. B carries those
    pixels through chronological same-camera LK transitions to the current image at
    ``measurement_ns`` (strictly later) and solves fresh PnP there, so no stale worker pose
    is ever fused. ``support_ns`` lists the chain's image exposures, strictly increasing
    within [source_exposure_ns, measurement_ns], at most MAX_HANDOFF_TRANSITIONS + 1 of
    them; ``transitions`` is ``len(support_ns) - 1`` (0 when empty). ``correspondences``,
    when present, holds the current pixels with the original landmark ids: same camera,
    ``measurement_ns``, at most MAX_HANDOFF_POINTS rows. ``pose``, when present, is the
    PoseEvidence solved from them: same camera, ``measurement_ns``, source ``handoff`` and
    landmark ids contained in the carried correspondences (so a pose needs them). An
    accepted record has a chain from ``source_exposure_ns`` to ``measurement_ns`` with 1 to
    MAX_HANDOFF_TRANSITIONS transitions, an age ``measurement_ns - source_exposure_ns`` of
    at most HISTORY_MAX_AGE_NS, correspondences, an accepted pose and ``map_id``,
    ``preprocessing_id`` and ``calibration_id`` equal to the pose's (``map_id`` also equal
    to the correspondences'). A rejected record keeps its reason and partial results.
    """

    job_id: int
    camera_id: str
    source_exposure_ns: int
    measurement_ns: int
    support_ns: tuple[int, ...]
    correspondences: Correspondences | None
    pose: PoseEvidence | None
    map_id: str
    preprocessing_id: str
    calibration_id: str
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "job_id", _check_int, what="a job id")
        camera_id = _normalize(self, "camera_id", check_camera_id)
        source_ns = _normalize(self, "source_exposure_ns", as_ns)
        measurement_ns = _normalize(self, "measurement_ns", as_ns)
        if not source_ns < measurement_ns:
            raise ContractError(
                f"HandoffEvidence.measurement_ns: {measurement_ns} must be after the source "
                f"exposure {source_ns}; the pose is solved on a later, current image"
            )
        support = _normalize(
            self, "support_ns", _check_ns_tuple, max_len=MAX_HANDOFF_TRANSITIONS + 1
        )
        for index, stamp in enumerate(support):
            if not source_ns <= stamp <= measurement_ns:
                raise ContractError(
                    f"HandoffEvidence.support_ns[{index}]: {stamp} lies outside "
                    f"[source_exposure_ns, measurement_ns] = [{source_ns}, {measurement_ns}]"
                )
            if index and stamp <= support[index - 1]:
                raise ContractError(
                    f"HandoffEvidence.support_ns[{index}]: {stamp} does not follow "
                    f"{support[index - 1]}; support stamps must be strictly increasing"
                )

        # Correspondences and pose belong to the current image of the same camera.
        current = (camera_id, measurement_ns)
        correspondences = _normalize(
            self, "correspondences", _check_record, Correspondences, optional=True
        )
        if correspondences is not None:
            image = (correspondences.camera_id, correspondences.exposure_ns)
            if image != current:
                raise ContractError(
                    f"HandoffEvidence.correspondences: lie on image {image}, not on the "
                    f"current image {current}"
                )
            rows = int(correspondences.landmark_ids.size)
            if rows > MAX_HANDOFF_POINTS:
                raise ContractError(
                    f"HandoffEvidence.correspondences: {rows} rows exceed "
                    f"MAX_HANDOFF_POINTS = {MAX_HANDOFF_POINTS}"
                )
        pose = _normalize(self, "pose", _check_record, PoseEvidence, optional=True)
        if pose is not None:
            image = (pose.camera_id, pose.exposure_ns)
            if image != current:
                raise ContractError(
                    f"HandoffEvidence.pose: solved on image {image}, not on the current "
                    f"image {current}"
                )
            if pose.source is not PoseSource.HANDOFF:
                raise ContractError(
                    f"HandoffEvidence.pose: source must be 'handoff', got {pose.source.value!r}"
                )
            if correspondences is None:
                raise ContractError(
                    "HandoffEvidence.pose: a carried pose needs the correspondences it was "
                    "solved from"
                )
            contained = np.isin(pose.landmark_ids, correspondences.landmark_ids, assume_unique=True)
            if not contained.all():
                outside = pose.landmark_ids[~contained].tolist()
                raise ContractError(
                    f"HandoffEvidence.pose: landmark ids {_describe(outside)} are not in the "
                    "carried correspondences"
                )
        for field_name in ("map_id", "preprocessing_id", "calibration_id"):
            _normalize(self, field_name, _check_identity)
        if not _normalize_flag(self, "accepted", ok_reason="accepted"):
            return

        # Accepted: a complete, young chain ending in an accepted pose with matching identities.
        _require_present(self, _HANDOFF_ACCEPTED_FIELDS, "accepted=True")
        transitions = self.transitions
        if not 1 <= transitions <= MAX_HANDOFF_TRANSITIONS:
            raise ContractError(
                f"HandoffEvidence.support_ns: accepted=True requires 1 to "
                f"{MAX_HANDOFF_TRANSITIONS} transitions, got {transitions}"
            )
        if support[0] != source_ns or support[-1] != measurement_ns:
            raise ContractError(
                "HandoffEvidence.support_ns: accepted=True requires the chain to run from the "
                f"source exposure {source_ns} to the measurement {measurement_ns}, got "
                f"{support[0]} to {support[-1]}"
            )
        age = measurement_ns - source_ns
        if age > HISTORY_MAX_AGE_NS:
            raise ContractError(
                f"HandoffEvidence: accepted=True requires an age of at most HISTORY_MAX_AGE_NS "
                f"= {HISTORY_MAX_AGE_NS} ns, got {age}"
            )
        if not pose.accepted:
            raise ContractError(
                f"HandoffEvidence.pose: accepted=True requires an accepted pose, got reason "
                f"{pose.reason!r}"
            )
        for field_name in _HANDOFF_POSE_IDENTITIES:
            own, solved = getattr(self, field_name), getattr(pose, field_name)
            if own != solved:
                raise ContractError(
                    f"HandoffEvidence.{field_name}: {own!r} differs from the pose's {solved!r}"
                )
        if correspondences.map_id != self.map_id:
            raise ContractError(
                f"HandoffEvidence.map_id: {self.map_id!r} differs from the correspondences' "
                f"{correspondences.map_id!r}"
            )

    @property
    def transitions(self) -> int:
        """Number of LK transitions in the support chain: ``len(support_ns) - 1``, 0 if empty."""
        return max(len(self.support_ns) - 1, 0)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MotionEvidence(_Record):
    """B's planar ground motion of one camera between two of its images.

    Static road corners tracked from the image at ``previous_ns`` to the image at
    ``current_ns`` (strictly later, same camera) are intersected with the RoadPatch named
    by ``road_patch_id`` and fitted with a robust planar rigid transform. ``T_Bprev_Bcurr``
    maps current-body coordinates into the previous body: rotation about z only and no z
    translation, within tolerance. ``covariance`` is the float64 3 x 3 symmetric positive
    definite covariance of (dx_m, dy_m, dyaw_rad) in the previous-body tangent plane.
    ``inlier_count`` (>= 0), ``inlier_fraction`` (in [0, 1]) and a present
    ``ground_residual_m`` (>= 0) describe the fit. An accepted record has the transform,
    the covariance, the residual, the road patch id and at least one inlier; depth, inlier
    and residual gates are B's thresholds. Arrays are stored as read-only float64 copies.
    """

    camera_id: str
    previous_ns: int
    current_ns: int
    T_Bprev_Bcurr: np.ndarray | None
    covariance: np.ndarray | None
    inlier_count: int
    inlier_fraction: float
    ground_residual_m: float | None
    road_patch_id: str | None
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        previous_ns = _normalize(self, "previous_ns", as_ns)
        current_ns = _normalize(self, "current_ns", as_ns)
        if not previous_ns < current_ns:
            raise ContractError(
                f"MotionEvidence.current_ns: {current_ns} must be after previous_ns {previous_ns}"
            )
        _normalize(self, "T_Bprev_Bcurr", _check_planar_rigid, optional=True)
        _normalize(self, "covariance", check_covariance, 3, definite=True, optional=True)
        inlier_count = _normalize(self, "inlier_count", _check_int, what="an inlier count")
        _normalize(self, "inlier_fraction", _check_float, lo=0.0, hi=1.0)
        _normalize(self, "ground_residual_m", _check_float, lo=0.0, optional=True)
        _normalize(self, "road_patch_id", _check_identity, optional=True)
        if _normalize_flag(self, "accepted", ok_reason="accepted"):
            _require_present(self, _MOTION_ACCEPTED_FIELDS, "accepted=True")
            if inlier_count < 1:
                raise ContractError(
                    "MotionEvidence: accepted=True requires at least one inlier, got "
                    f"inlier_count={inlier_count}"
                )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BorderEvidence(_Record):
    """A's lateral and yaw evidence from surveyed track borders seen in one image.

    Borders constrain only the lateral position and the yaw, never the along-track
    position, so the record has no along-track field. ``exposure_ns`` is the image exposure
    (not a group time) and ``branch`` the associated route branch. ``normal_xy`` is the
    float64 (2,) unit lateral direction [-sin(prior_yaw), cos(prior_yaw)];
    ``lateral_coordinate_m`` is the measured ``normal_xy . [x, y]`` and ``yaw_rad`` the
    measured yaw, stored wrapped to (-pi, pi]. ``R`` is the float64 2 x 2 symmetric
    positive definite covariance of (lateral m, yaw rad). ``support_samples`` and
    ``support_contours`` are non-negative counts; ``conditioning`` is the reciprocal
    condition number of the yaw-scaled two-parameter Jacobian, in [0, 1]. An accepted
    record has the lateral coordinate, the yaw, ``R`` and branch ``track`` or ``pit``;
    support and conditioning gates are A's thresholds. Arrays are stored as read-only
    float64 copies.
    """

    camera_id: str
    exposure_ns: int
    branch: Branch
    normal_xy: np.ndarray
    lateral_coordinate_m: float | None
    yaw_rad: float | None
    R: np.ndarray | None
    support_samples: int
    support_contours: int
    conditioning: float
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "exposure_ns", as_ns)
        branch = _normalize(self, "branch", _check_enum, Branch)
        _normalize(self, "normal_xy", _check_unit_vector, size=2)
        _normalize(self, "lateral_coordinate_m", _check_float, optional=True)
        _normalize(self, "yaw_rad", wrap_angle, optional=True)
        _normalize(self, "R", check_covariance, 2, definite=True, optional=True)
        _normalize(self, "support_samples", _check_int, what="a sample count")
        _normalize(self, "support_contours", _check_int, what="a contour count")
        _normalize(self, "conditioning", _check_float, lo=0.0, hi=1.0)
        if _normalize_flag(self, "accepted", ok_reason="accepted"):
            _require_present(self, _BORDER_ACCEPTED_FIELDS, "accepted=True")
            if branch is Branch.UNKNOWN:
                raise ContractError(
                    "BorderEvidence.branch: accepted=True requires branch 'track' or 'pit', "
                    "got 'unknown'"
                )


# Producer evidence that a GroupEvidence consumes, lists as rejected or keeps as unused.
EvidenceRecord = PoseEvidence | HandoffEvidence | MotionEvidence | BorderEvidence


# ---------------------------------------------------------------------------------------
# Anchor, recovery and group records (C's scheduler and runner, B's anchor worker). An
# anchor result never carries a pose; GroupEvidence is the single input of Estimator.step.
# ---------------------------------------------------------------------------------------

# Cameras that the camera-specific anchor purposes run on.
_ANCHOR_PURPOSE_CAMERAS = {
    AnchorPurpose.FRONT_ANCHOR: PRIMARY_CAMERA,
    AnchorPurpose.REAR_ANCHOR: SECONDARY_CAMERA,
}
# Identity fields an AnchorResult repeats from its AnchorRequest (see check_against).
_ANCHOR_IDENTITY_FIELDS = (
    "job_id",
    "purpose",
    "camera_id",
    "source_group_id",
    "source_exposure_ns",
    "map_id",
    "preprocessing_id",
)
# How the source image of an AnchorRequest's observation must match the request.
_ANCHOR_OBSERVATION_FIELDS = (
    ("camera_id", "camera_id"),
    ("exposure_ns", "source_exposure_ns"),
    ("preprocessing_id", "preprocessing_id"),
)
_ANCHOR_OK_REASON = "ok"
# The GroupEvidence pose slot holds a tracking PoseEvidence or a current-image handoff.
_POSE_SLOT_TYPES = PoseEvidence | HandoffEvidence


def _value_text(value: object) -> str:
    """Short text of a field value for error messages; enum members show their value."""
    return _describe(value.value if isinstance(value, enum.Enum) else value)


def _check_identity_tuple(values: object, name: str) -> tuple[str, ...]:
    """Return a tuple or list of opaque non-empty identity strings as a tuple of str."""
    if not isinstance(values, (tuple, list)):
        raise ContractError(
            f"{name}: expected a tuple of identity strings, got {type(values).__name__}"
        )
    return tuple(_check_identity(value, f"{name}[{index}]") for index, value in enumerate(values))


def _require_accepted_as(record: Any, name: str) -> Any:
    """Apply ``require_accepted`` and name the field ``name`` in a RejectedEvidenceError."""
    try:
        return require_accepted(record)
    except RejectedEvidenceError as exc:
        raise RejectedEvidenceError(f"{name}: {exc}") from None


def _check_accepted_pose(value: object, name: str) -> PoseEvidence:
    """Return ``value`` if it is an accepted PoseEvidence.

    Another type raises ``ContractError``; a rejected pose raises ``RejectedEvidenceError``.
    """
    return _require_accepted_as(_check_record(value, PoseEvidence, name), name)


def _measured_image(record: object, name: str) -> tuple[str, int]:
    """Return ``(camera_id, exposure_ns)`` of the image an evidence record was measured on.

    That is ``exposure_ns`` for PoseEvidence and BorderEvidence, ``measurement_ns`` (the
    current image, not the anchor source) for HandoffEvidence and ``current_ns`` for
    MotionEvidence. Other types raise ``ContractError``.
    """
    match record:
        case PoseEvidence() | BorderEvidence():
            return record.camera_id, record.exposure_ns
        case HandoffEvidence():
            return record.camera_id, record.measurement_ns
        case MotionEvidence():
            return record.camera_id, record.current_ns
        case _:
            raise ContractError(f"{name}: unsupported evidence type {type(record).__name__}")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AnchorRequest(_Record):
    """One bounded anchor job, frozen by C's scheduler at a logical group boundary.

    Job ``job_id`` serves ``purpose`` on the source image ``observation``: camera
    ``camera_id`` of group ``source_group_id``, exposed at ``source_exposure_ns``. The
    observation's camera, exposure and ``preprocessing_id`` equal the request's.
    ``snapshot`` is the estimator prior frozen with the source group, 0 to
    MAX_EXPOSURE_SPREAD_NS after the source exposure. ``delivery_not_before_ns`` is at
    least ``source_exposure_ns + DELIVERY_MIN_DELAY_NS``: C hands the result off at the
    first group with ``t_ns >= delivery_not_before_ns``, however early the worker finished.
    ``reference_ids`` names at most MAX_REFERENCES_PER_JOB unique non-negative reference
    keyframes to match; empty means retrieval inside the job. ``map_id`` and
    ``preprocessing_id`` name the frozen assets the job uses. ``front_anchor`` runs only on
    PRIMARY_CAMERA and ``rear_anchor`` only on SECONDARY_CAMERA; ``recovery`` and
    ``warm_retrieval`` may use any camera. The worker reads nothing but the request and its
    frozen assets.
    """

    job_id: int
    purpose: AnchorPurpose
    camera_id: str
    source_group_id: int
    source_exposure_ns: int
    delivery_not_before_ns: int
    observation: Observation
    snapshot: SearchSnapshot
    reference_ids: tuple[int, ...]
    map_id: str
    preprocessing_id: str

    def __post_init__(self) -> None:
        _normalize(self, "job_id", _check_int, what="a job id")
        purpose = _normalize(self, "purpose", _check_enum, AnchorPurpose)
        camera_id = _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "source_group_id", _check_int, what="a group id")
        source_ns = _normalize(self, "source_exposure_ns", as_ns)
        delivery_ns = _normalize(self, "delivery_not_before_ns", as_ns)
        observation = _normalize(self, "observation", _check_record, Observation)
        snapshot = _normalize(self, "snapshot", _check_record, SearchSnapshot)
        _normalize(self, "reference_ids", _check_id_tuple, max_len=MAX_REFERENCES_PER_JOB)
        _normalize(self, "map_id", _check_identity)
        _normalize(self, "preprocessing_id", _check_identity)

        required_camera = _ANCHOR_PURPOSE_CAMERAS.get(purpose)
        if required_camera is not None and camera_id != required_camera:
            raise ContractError(
                f"AnchorRequest.camera_id: purpose {purpose.value!r} runs on camera "
                f"{required_camera!r}, got {camera_id!r}"
            )
        earliest = source_ns + DELIVERY_MIN_DELAY_NS
        if delivery_ns < earliest:
            raise ContractError(
                f"AnchorRequest.delivery_not_before_ns: {delivery_ns} is before "
                f"source_exposure_ns + DELIVERY_MIN_DELAY_NS = {earliest}"
            )
        for observation_field, request_field in _ANCHOR_OBSERVATION_FIELDS:
            seen, wanted = getattr(observation, observation_field), getattr(self, request_field)
            if seen != wanted:
                raise ContractError(
                    f"AnchorRequest.observation: {observation_field} {_value_text(seen)} "
                    f"differs from {request_field} {_value_text(wanted)}; the job runs on its "
                    "own source image"
                )
        lag = snapshot.t_ns - source_ns
        if not 0 <= lag <= MAX_EXPOSURE_SPREAD_NS:
            raise ContractError(
                f"AnchorRequest.snapshot: t_ns {snapshot.t_ns} must lie 0 to "
                f"MAX_EXPOSURE_SPREAD_NS = {MAX_EXPOSURE_SPREAD_NS} ns after the source "
                f"exposure {source_ns} (the prior is frozen with the source group), got {lag}"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AnchorResult(_Record):
    """B's AnchorWorker answer to one AnchorRequest: source-image correspondences only.

    The record deliberately has no pose, transform, position or yaw field, so a stale
    worker pose cannot be fused: its correspondences reach the estimator only through B's
    handoff, which re-solves PnP on a current image. ``job_id``, ``purpose``,
    ``camera_id``, ``source_group_id``, ``source_exposure_ns``, ``map_id`` and
    ``preprocessing_id`` repeat the request's; ``check_against`` verifies them. ``reason``
    is "ok" exactly when ``status`` is ``ok``; every other status carries a failure reason
    (not "ok", "accepted" or "valid"). Only an ``ok`` result carries ``correspondences``:
    at least one row on the source image (``camera_id`` at ``source_exposure_ns``) with
    landmarks of ``map_id``; any other status has None. ``matched_reference_ids`` (at most
    MAX_REFERENCES_PER_JOB) and ``remaining_candidate_ids`` (at most RETRIEVAL_CANDIDATES,
    disjoint from the matched ids) are unique non-negative ids. ``model_ids`` are the opaque
    identities of the models used. ``worker_duration_ns`` is diagnostic only and never
    selects ranked evidence.
    """

    job_id: int
    purpose: AnchorPurpose
    camera_id: str
    source_group_id: int
    source_exposure_ns: int
    status: AnchorStatus
    reason: str
    correspondences: Correspondences | None
    matched_reference_ids: tuple[int, ...]
    remaining_candidate_ids: tuple[int, ...]
    map_id: str
    preprocessing_id: str
    model_ids: tuple[str, ...]
    worker_duration_ns: int | None

    def __post_init__(self) -> None:
        _normalize(self, "job_id", _check_int, what="a job id")
        _normalize(self, "purpose", _check_enum, AnchorPurpose)
        camera_id = _normalize(self, "camera_id", check_camera_id)
        _normalize(self, "source_group_id", _check_int, what="a group id")
        source_ns = _normalize(self, "source_exposure_ns", as_ns)
        status = _normalize(self, "status", _check_enum, AnchorStatus)
        reason = _normalize(self, "reason", check_reason)
        correspondences = _normalize(
            self, "correspondences", _check_record, Correspondences, optional=True
        )
        matched = _normalize(
            self, "matched_reference_ids", _check_id_tuple, max_len=MAX_REFERENCES_PER_JOB
        )
        remaining = _normalize(
            self, "remaining_candidate_ids", _check_id_tuple, max_len=RETRIEVAL_CANDIDATES
        )
        map_id = _normalize(self, "map_id", _check_identity)
        _normalize(self, "preprocessing_id", _check_identity)
        _normalize(self, "model_ids", _check_identity_tuple)
        _normalize(self, "worker_duration_ns", as_ns, optional=True)

        ok = status is AnchorStatus.OK
        if ok and reason != _ANCHOR_OK_REASON:
            raise ContractError(
                f"AnchorResult.reason: status 'ok' requires reason 'ok', got {reason!r}"
            )
        if not ok and (reason == _ANCHOR_OK_REASON or reason in _POSITIVE_REASONS):
            raise ContractError(
                f"AnchorResult.reason: status {status.value!r} requires a failure reason, "
                f"got {reason!r}"
            )
        shared = sorted(set(matched).intersection(remaining))
        if shared:
            raise ContractError(
                f"AnchorResult.remaining_candidate_ids: ids {shared} are also matched; "
                "remaining candidates are disjoint from matched_reference_ids"
            )
        if correspondences is None:
            if ok:
                raise ContractError(
                    "AnchorResult.correspondences: status 'ok' requires source-image "
                    "correspondences with at least one row"
                )
            return
        if not ok:
            raise ContractError(
                f"AnchorResult.correspondences: only status 'ok' carries correspondences, got "
                f"status {status.value!r}"
            )
        image = (correspondences.camera_id, correspondences.exposure_ns)
        source = (camera_id, source_ns)
        if image != source:
            raise ContractError(
                f"AnchorResult.correspondences: lie on image {image}, not on the source image "
                f"{source}"
            )
        if correspondences.landmark_ids.size == 0:
            raise ContractError(
                "AnchorResult.correspondences: status 'ok' requires at least one row"
            )
        if correspondences.map_id != map_id:
            raise ContractError(
                f"AnchorResult.correspondences: landmarks of map {correspondences.map_id!r}, "
                f"not of map_id {map_id!r}"
            )

    def check_against(self, request: AnchorRequest) -> None:
        """Raise ``ContractError`` unless this result answers ``request``.

        The job id, purpose, camera, source group, source exposure, map id and preprocessing
        id must equal the request's. A nonempty request.reference_ids also bounds the
        matched references; an empty tuple permits retrieval inside the job. C calls this
        before handing the result off.
        """
        if not isinstance(request, AnchorRequest):
            raise ContractError(
                "AnchorResult.check_against: expected an AnchorRequest, got "
                f"{type(request).__name__}"
            )
        differences = [
            f"{field_name} {_value_text(getattr(self, field_name))} != "
            f"{_value_text(getattr(request, field_name))}"
            for field_name in _ANCHOR_IDENTITY_FIELDS
            if getattr(self, field_name) != getattr(request, field_name)
        ]
        if differences:
            raise ContractError(
                f"AnchorResult (job {self.job_id}) does not answer the AnchorRequest (job "
                f"{request.job_id}): " + "; ".join(differences)
            )
        if request.reference_ids:
            unexpected = sorted(set(self.matched_reference_ids).difference(request.reference_ids))
            if unexpected:
                raise ContractError(
                    f"AnchorResult.matched_reference_ids: {unexpected} are outside the "
                    f"request's reference_ids {request.reference_ids}"
                )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class RecoveryReset(_Record):
    """C's confirmed recovery; the new hypothesis enters through the single update path.

    C holds at most one recovery candidate. ``candidate`` is its first accepted pose and
    ``confirmation`` an accepted pose exposed at least RECOVERY_CONFIRM_MIN_NS later and no
    later than ``t_ns``, the time of group ``group_id`` that applies the reset. ``jump_m``
    (>= 0) is the distance from the pre-reset estimate and ``reason`` a snake_case code
    such as "confirmed_recovery". C emits the record only after confirmation, as
    ``GroupEvidence.reset``, and after that step passes it to ``MappedTracker.reset`` and
    ``TrackProjector.reset``. ``reset_id`` (>= 0) identifies the reset within a run.
    Both poses must use the same map. Same-camera poses must also share calibration and
    preprocessing identities; different cameras may have camera-specific geometry IDs.
    """

    reset_id: int
    group_id: int
    t_ns: int
    candidate: PoseEvidence
    confirmation: PoseEvidence
    jump_m: float
    reason: str

    def __post_init__(self) -> None:
        _normalize(self, "reset_id", _check_int, what="a reset id")
        _normalize(self, "group_id", _check_int, what="a group id")
        t_ns = _normalize(self, "t_ns", as_ns)
        candidate = _normalize(self, "candidate", _check_accepted_pose)
        confirmation = _normalize(self, "confirmation", _check_accepted_pose)
        _normalize(self, "jump_m", _check_float, lo=0.0)
        _normalize(self, "reason", check_reason)

        identities = ("map_id",)
        if candidate.camera_id == confirmation.camera_id:
            identities += ("calibration_id", "preprocessing_id")
        for field_name in identities:
            first, later = getattr(candidate, field_name), getattr(confirmation, field_name)
            if first != later:
                raise ContractError(
                    f"RecoveryReset.confirmation.{field_name}: {later!r} differs from the "
                    f"candidate's {first!r}"
                )

        gap = confirmation.exposure_ns - candidate.exposure_ns
        if gap < RECOVERY_CONFIRM_MIN_NS:
            raise ContractError(
                f"RecoveryReset.confirmation: exposure {confirmation.exposure_ns} minus the "
                f"candidate exposure {candidate.exposure_ns} is {gap} ns, below "
                f"RECOVERY_CONFIRM_MIN_NS = {RECOVERY_CONFIRM_MIN_NS}"
            )
        if confirmation.exposure_ns > t_ns:
            raise ContractError(
                f"RecoveryReset.confirmation: exposure {confirmation.exposure_ns} is after "
                f"t_ns {t_ns}; a reset cannot use a later image"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class UnusedEvidence(_Record):
    """Accepted evidence that C did not consume in its group, with C's reason.

    ``evidence`` is an EvidenceRecord that its producer accepted; a rejected record raises
    ``ContractError`` here because it belongs in ``GroupEvidence.rejected``. ``reason`` is
    C's snake_case disposition, for example "innovation_gate", "superseded_by_pose",
    "camera_not_selected" or "recovery_candidate".
    """

    reason: str
    evidence: EvidenceRecord

    def __post_init__(self) -> None:
        _normalize(self, "reason", check_reason)
        evidence = _normalize(self, "evidence", _check_record, EvidenceRecord)
        if not evidence.accepted:
            raise ContractError(
                f"UnusedEvidence.evidence: the {type(evidence).__name__} was rejected by its "
                f"producer ({evidence.reason!r}); rejected records go to GroupEvidence.rejected"
            )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class GroupEvidence(_Record):
    """The only input of ``Estimator.step``: what C decided for one FrameSet.

    ``frame_set`` is the group; ``t_ns`` and ``group_id`` are properties taken from it.
    The consumed slots hold accepted records only: ``pose`` is the one selected, in-gate
    full pose (a PoseEvidence with source ``tracking``, or a HandoffEvidence so that its
    source and support stamps stay attached), ``motion`` and ``border`` are the fallback
    evidence and ``reset`` is a confirmed RecoveryReset. PnP comes first: a pose excludes
    motion and border, and a reset excludes every other consumed slot. A rejected record in
    a consumed slot raises ``RejectedEvidenceError``. ``rejected`` lists producer-rejected
    records and ``unused`` accepted records that C did not consume, each with C's reason.
    Every record, and a reset's confirmation, must be measured on this group's own image of
    its camera, so neither stale nor future evidence enters the group, and no record object
    may appear twice, including nested handoff poses and a consumed reset confirmation.
    A consumed pose or reset confirmation must match its image's calibration; incompatible
    evidence can still be logged in ``rejected`` or ``unused``. A reset's group id/time must
    match this group. ``consumed_kinds()`` names the consumed hierarchy level.

    ``auxiliary=None`` preserves the filter's existing geometry. C may supply A's checked
    road geometry here, or an invalid ``none`` record to disable it. Its source time is
    <= t_ns and its patch must match consumed motion. PnP geometry enters only through a
    consumed pose, whose full geometry takes precedence over this road context.
    """

    frame_set: FrameSet
    pose: PoseEvidence | HandoffEvidence | None
    motion: MotionEvidence | None
    border: BorderEvidence | None
    reset: RecoveryReset | None
    rejected: tuple[EvidenceRecord, ...]
    unused: tuple[UnusedEvidence, ...]
    auxiliary: AuxiliaryGeometry | None = None

    def __post_init__(self) -> None:
        _normalize(self, "frame_set", _check_record, FrameSet)
        _normalize(self, "pose", _check_record, _POSE_SLOT_TYPES, optional=True)
        _normalize(self, "motion", _check_record, MotionEvidence, optional=True)
        _normalize(self, "border", _check_record, BorderEvidence, optional=True)
        _normalize(self, "reset", _check_record, RecoveryReset, optional=True)
        _normalize(self, "rejected", _check_record_tuple, EvidenceRecord)
        _normalize(self, "unused", _check_record_tuple, UnusedEvidence)
        auxiliary = _normalize(self, "auxiliary", _check_auxiliary, t_ns=self.t_ns, optional=True)
        if auxiliary is not None:
            if auxiliary.source is AuxSource.PNP:
                raise ContractError("GroupEvidence.auxiliary: PnP geometry enters with a consumed pose")
            if not auxiliary.valid and (self.motion is not None or self.border is not None):
                raise ContractError("GroupEvidence.auxiliary: invalid geometry disables motion/borders")
            if self.motion is not None and auxiliary.road_patch_id != self.motion.road_patch_id:
                raise ContractError("GroupEvidence.auxiliary: road patch differs from motion's patch")
        _validate_group(self)

    @property
    def t_ns(self) -> int:
        """Group time of ``frame_set``: its maximum image exposure and the CSV timestamp."""
        return self.frame_set.t_ns

    @property
    def group_id(self) -> int:
        """Id of ``frame_set``."""
        return self.frame_set.group_id

    def consumed_kinds(self) -> tuple[EvidenceKind, ...]:
        """Evidence kinds of the consumed slots.

        ``(recovery_reset,)`` for a reset, ``(pnp,)`` for a PoseEvidence pose,
        ``(handoff_pnp,)`` for a HandoffEvidence pose, otherwise ``(ground_motion,)``,
        ``(border,)`` or ``(ground_motion, border)`` for the fallback slots, and
        ``(prediction_only,)`` when nothing is consumed. The step's Estimate reports what
        the filter actually applied, which may be less when it declines a slot, with a
        reason.
        """
        if self.reset is not None:
            return (EvidenceKind.RECOVERY_RESET,)
        if isinstance(self.pose, PoseEvidence):
            return (EvidenceKind.PNP,)
        if isinstance(self.pose, HandoffEvidence):
            return (EvidenceKind.HANDOFF_PNP,)
        kinds: list[EvidenceKind] = []
        if self.motion is not None:
            kinds.append(EvidenceKind.GROUND_MOTION)
        if self.border is not None:
            kinds.append(EvidenceKind.BORDER)
        return tuple(kinds) or _PREDICTION_ONLY_KINDS


def _validate_group(group: GroupEvidence) -> None:
    """Check a GroupEvidence with the design's validation algorithm, in its order.

    Consumed slots hold accepted records and a pose-slot PoseEvidence has source
    ``tracking``; a pose excludes motion and border (PnP first); a reset excludes the other
    consumed slots; every record, and a reset's confirmation, is measured on this group's
    image of its camera; ``rejected`` holds only rejected and ``unused`` only accepted
    records; no record object appears twice.
    """
    frame_set = group.frame_set
    exposure = {image.camera_id: image.exposure_ns for image in frame_set.images}
    calibration = {image.camera_id: image.calibration_id for image in frame_set.images}

    def current(camera_id: str, measured_ns: int, what: str) -> None:
        # Current-image rule; measured_ns <= t_ns follows because t_ns == max(exposure.values()).
        group_ns = exposure.get(camera_id)
        if group_ns != measured_ns:
            found = (
                f"group {frame_set.group_id} has no {camera_id} image"
                if group_ns is None
                else f"group {frame_set.group_id}'s {camera_id} image was exposed at {group_ns}"
            )
            raise ContractError(
                f"GroupEvidence.{what}: measured on the {camera_id} image at {measured_ns}, "
                f"but {found}; evidence must come from this group's own image of its camera"
            )

    def check(record: EvidenceRecord, what: str) -> None:
        camera_id, measured_ns = _measured_image(record, f"GroupEvidence.{what}")
        current(camera_id, measured_ns, what)

    def check_pose(record: PoseEvidence | HandoffEvidence, what: str) -> None:
        check(record, what)
        expected = calibration[record.camera_id]
        if record.calibration_id != expected:
            raise ContractError(
                f"GroupEvidence.{what}.calibration_id: {record.calibration_id!r} differs "
                f"from the current image's {expected!r}"
            )

    if group.pose is not None:
        _require_accepted_as(group.pose, "GroupEvidence.pose")
        if isinstance(group.pose, PoseEvidence) and group.pose.source is not PoseSource.TRACKING:
            raise ContractError(
                "GroupEvidence.pose: a PoseEvidence in the pose slot must have source "
                f"'tracking', got {group.pose.source.value!r}; handoff poses enter as "
                "HandoffEvidence and candidate poses through a RecoveryReset"
            )
        check_pose(group.pose, "pose")
    for slot, record in (("motion", group.motion), ("border", group.border)):
        if record is not None:
            _require_accepted_as(record, f"GroupEvidence.{slot}")
            check(record, slot)
    if group.pose is not None and (group.motion is not None or group.border is not None):
        raise ContractError(
            "GroupEvidence: PnP-first hierarchy; a consumed pose excludes the motion and "
            "border slots (list such evidence as unused)"
        )
    if group.reset is not None:
        if group.pose is not None or group.motion is not None or group.border is not None:
            raise ContractError(
                "GroupEvidence.reset: a recovery reset is the only consumed evidence of its "
                "group; the pose, motion and border slots must be empty"
            )
        if (group.reset.group_id, group.reset.t_ns) != (group.group_id, group.t_ns):
            raise ContractError(
                f"GroupEvidence.reset: group_id/t_ns ({group.reset.group_id}, "
                f"{group.reset.t_ns}) must match ({group.group_id}, {group.t_ns})"
            )
        check_pose(group.reset.confirmation, "reset.confirmation")
    for index, record in enumerate(group.rejected):  # earlier entries: rejected and current
        what = f"rejected[{index}]"
        if record.accepted:
            raise ContractError(
                f"GroupEvidence.{what}: holds an accepted {type(record).__name__}; accepted "
                "evidence that is not consumed belongs in unused"
            )
        check(record, what)
    for index, item in enumerate(group.unused):  # earlier entries: accepted and current
        what = f"unused[{index}].evidence"
        if not item.evidence.accepted:
            raise ContractError(
                f"GroupEvidence.{what}: holds a rejected {type(item.evidence).__name__}; "
                "rejected evidence belongs in rejected"
            )
        check(item.evidence, what)

    consumed = (("pose", group.pose), ("motion", group.motion), ("border", group.border))
    placed = [(slot, record) for slot, record in consumed if record is not None]
    if group.reset is not None:
        placed.append(("reset.confirmation", group.reset.confirmation))
    placed += [(f"rejected[{index}]", record) for index, record in enumerate(group.rejected)]
    placed += [
        (f"unused[{index}].evidence", item.evidence) for index, item in enumerate(group.unused)
    ]
    placed += [
        (f"{place}.pose", record.pose)
        for place, record in placed
        if isinstance(record, HandoffEvidence) and record.pose is not None
    ]
    first_place: dict[int, str] = {}
    for place, record in placed:
        earlier = first_place.setdefault(id(record), place)
        if earlier != place:
            raise ContractError(
                f"GroupEvidence.{place}: this {type(record).__name__} object already appears "
                f"as {earlier}; no evidence record may appear twice"
            )


# ---------------------------------------------------------------------------------------
# Initial pose and episodes (C's runner; the offline exporter writes the pose file)
# ---------------------------------------------------------------------------------------

_INITIAL_POSE_KEYS = frozenset({"timestamp_ns", "x", "y", "yaw"})


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class InitialPose(_Record):
    """The one initial planar pose an inference run receives; yaw is stored wrapped."""

    timestamp_ns: int
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        _normalize(self, "timestamp_ns", as_ns)
        _normalize(self, "x", _check_float)
        _normalize(self, "y", _check_float)
        _normalize(self, "yaw", wrap_angle)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeBounds(_Record):
    """Episode interval [start_ns, end_ns); ``end_ns=None`` means until the end of input."""

    name: str
    start_ns: int
    end_ns: int | None

    def __post_init__(self) -> None:
        _normalize(self, "name", _check_identity)
        start_ns = _normalize(self, "start_ns", as_ns)
        end_ns = _normalize(self, "end_ns", as_ns, optional=True)
        if end_ns is not None and end_ns <= start_ns:
            raise ContractError(
                f"EpisodeBounds({self.name}): end_ns {end_ns} must be greater than "
                f"start_ns {start_ns}"
            )

    def contains(self, t_ns: object) -> bool:
        """Return ``start_ns <= t_ns`` and (``end_ns`` is None or ``t_ns < end_ns``)."""
        t_ns = as_ns(t_ns, f"EpisodeBounds({self.name}).contains")
        return self.start_ns <= t_ns and (self.end_ns is None or t_ns < self.end_ns)

    def check_initial_pose(self, pose: InitialPose) -> None:
        """Raise ``ContractError`` unless the pose's stamp lies in [start_ns, end_ns)."""
        if not isinstance(pose, InitialPose):
            raise ContractError(f"EpisodeBounds({self.name}): expected an InitialPose")
        if not self.contains(pose.timestamp_ns):
            raise ContractError(
                f"EpisodeBounds({self.name}): initial pose stamp {pose.timestamp_ns} is outside "
                f"[{self.start_ns}, {self.end_ns if self.end_ns is not None else 'end of input'})"
            )


def _reject_json_constant(token: str) -> None:
    raise ContractError(f"initial pose: non-finite JSON constant {token}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ContractError("initial pose: duplicate JSON key")
    return dict(pairs)


def parse_initial_pose(text: str) -> InitialPose:
    """Parse the initial-pose JSON object: exactly ``timestamp_ns``, ``x``, ``y``, ``yaw``.

    ``timestamp_ns`` is a JSON integer or an unsigned decimal string; floats, bools and signed
    or padded strings are rejected. x/y/yaw must be JSON numbers (not bool) and finite after
    parsing (``1e400`` overflows to inf). NaN/Infinity constants, duplicate keys and a
    non-object top level are rejected. Every violation raises ``ContractError``.
    """
    if not isinstance(text, str):
        raise ContractError(f"initial pose: expected JSON text, got {type(text).__name__}")
    try:
        doc = json.loads(
            text, parse_constant=_reject_json_constant, object_pairs_hook=_unique_json_object
        )
    except ContractError:
        raise
    except ValueError as exc:  # JSONDecodeError, or an integer beyond the digit limit
        raise ContractError(f"initial pose: invalid JSON: {exc}") from None
    if type(doc) is not dict or set(doc) != _INITIAL_POSE_KEYS:
        raise ContractError("initial pose: exactly timestamp_ns, x, y, yaw are required")
    stamp = doc["timestamp_ns"]
    if type(stamp) is str:
        t_ns = parse_decimal_ns(stamp, "initial pose timestamp_ns")
    elif type(stamp) is int:
        t_ns = as_ns(stamp, "initial pose timestamp_ns")
    else:
        raise ContractError(
            "initial pose: timestamp_ns must be a JSON integer or decimal string, "
            f"got {type(stamp).__name__}"
        )
    for key in ("x", "y", "yaw"):
        if type(doc[key]) not in (int, float):
            raise ContractError(
                f"initial pose: {key} must be a JSON number, got {type(doc[key]).__name__}"
            )
    return InitialPose(timestamp_ns=t_ns, x=doc["x"], y=doc["y"], yaw=doc["yaw"])


def format_initial_pose(pose: InitialPose) -> str:
    """Canonical JSON line: decimal-string stamp, repr-precision floats, trailing newline."""
    if not isinstance(pose, InitialPose):
        raise ContractError(f"initial pose: expected an InitialPose, got {type(pose).__name__}")
    doc = {"timestamp_ns": str(pose.timestamp_ns), "x": pose.x, "y": pose.y, "yaw": pose.yaw}
    return json.dumps(doc, allow_nan=False) + "\n"


def read_initial_pose(path: str | os.PathLike[str]) -> InitialPose:
    """Read and parse a UTF-8 initial-pose file (a BOM is rejected as invalid JSON)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"initial pose {os.fspath(path)}: not UTF-8: {exc}") from None
    return parse_initial_pose(text)


def write_initial_pose(path: str | os.PathLike[str], pose: InitialPose) -> None:
    """Write the canonical JSON by exclusive create; an existing path raises FileExistsError."""
    text = format_initial_pose(pose)
    with open(path, "x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _split_item(doc: object, key: str, where: str) -> object:
    if not isinstance(doc, Mapping):
        raise ContractError(f"{where}: expected a JSON object, got {type(doc).__name__}")
    if key not in doc:
        raise ContractError(f"{where}: missing {key!r}")
    return doc[key]


def _split_ns(doc: object, key: str, where: str) -> int:
    return parse_decimal_ns(_split_item(doc, key, where), f"{where}.{key}")


def _split_interval(doc: object, key: str) -> tuple[int, int]:
    episode = _split_item(doc, key, "build split")
    where = f"build split.{key}"
    return _split_ns(episode, "start_ns", where), _split_ns(episode, "end_ns", where)


def episodes_from_split(doc: Mapping[str, object]) -> Mapping[str, EpisodeBounds]:
    """Return read-only fit/tune/holdout/full bounds from a parsed build-split document.

    Checks the time encoding and interval convention strings, parses every stamp with
    ``parse_decimal_ns``, requires exact 2 s guards and
    ``first == fit.start < fit.end < tune.start < tune.end < holdout.start < holdout.end
    == last + 1``. ``full`` is [first, last + 1). Bad structure raises ``ContractError``.
    """
    for key, expected in (
        ("time_encoding", SPLIT_TIME_ENCODING),
        ("interval_convention", SPLIT_INTERVAL_CONVENTION),
    ):
        value = _split_item(doc, key, "build split")
        if type(value) is not str or value != expected:
            raise ContractError(f"build split.{key}: expected {expected!r}, got {_describe(value)}")
    first = _split_ns(doc, "first_gt_timestamp_ns", "build split")
    last = _split_ns(doc, "last_gt_timestamp_ns", "build split")
    guard = _split_ns(doc, "guard_duration_ns", "build split")
    fit, tune, holdout = (_split_interval(doc, key) for key in ("fit", "tune", "holdout"))
    if guard != SPLIT_GUARD_NS or tune[0] - fit[1] != guard or holdout[0] - tune[1] != guard:
        raise ContractError(f"build split: guards must be exactly {SPLIT_GUARD_NS} ns")
    if not first == fit[0] < fit[1] < tune[0] < tune[1] < holdout[0] < holdout[1] == last + 1:
        raise ContractError(
            "build split: episodes must be ordered, disjoint and cover first .. last + 1"
        )
    intervals = {"fit": fit, "tune": tune, "holdout": holdout, "full": (first, last + 1)}
    return types.MappingProxyType(
        {
            name: EpisodeBounds(name=name, start_ns=start, end_ns=end)
            for name, (start, end) in intervals.items()
        }
    )


def load_split(path: str | os.PathLike[str] = DEFAULT_SPLIT_PATH) -> Mapping[str, EpisodeBounds]:
    """Read the hash-bound build split (bytes only, never rewritten) and return its episodes."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BUILD_SPLIT_SHA256:
        raise ContractError(
            f"build split {os.fspath(path)}: SHA-256 {digest} != {BUILD_SPLIT_SHA256}; the file "
            "is hash-bound and must stay byte-for-byte unchanged"
        )
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ContractError(f"build split: invalid JSON: {exc}") from None
    return episodes_from_split(doc)


# ---------------------------------------------------------------------------------------
# Protocols: call boundaries between owners (proposals for A/B agreement). They define
# interfaces only; owners implement them in their own modules. See "Call order and flows
# between owners" in the module docstring.
# ---------------------------------------------------------------------------------------


@runtime_checkable
class FrameSetSource(Protocol):
    """Owner A (``camloc/io.py``): header-only FrameSets from the driving recording."""

    def calibrations(self) -> Mapping[str, CameraCalibration]:
        """Frozen first-available calibration of all seven cameras, available before iteration."""
        ...

    def __iter__(self) -> Iterator[FrameSet]:
        """Header-only grouping in recorded arrival order; group_id 0, 1, 2, ...; strictly
        increasing t_ns. Groups only images with initial_pose.timestamp_ns <= exposure_ns <
        episode end; others become InputEvents. Never decodes JPEG, dynamic /tf or the live
        state topic."""
        ...

    def trailing_events(self) -> tuple[InputEvent, ...]:
        """Events after the last FrameSet (always ends with end_of_input); valid after
        exhaustion."""
        ...


@runtime_checkable
class Canonicalizer(Protocol):
    """Owner A (``camloc/geometry.py``): raw image to canonical Observation.

    C calls it for each camera in the group's decode set; a raised InputError becomes C's
    ``decode_failure`` InputEvent.
    """

    def canonicalize(self, image: ImageMessage) -> Observation:
        """One JPEG decode plus one cached raw-to-canonical remap. Raises InputError on
        failure."""
        ...


@runtime_checkable
class RoadGeometry(Protocol):
    """Owner A (``camloc/geometry.py``): local road plane from the surveyed borders.

    C calls it with the snapshot pose at most once per group when plane-dependent evidence
    is needed and passes the same RoadPatch to ground motion and auxiliary geometry.
    """

    def road_patch(self, x: float, y: float, yaw: float, branch: Branch) -> RoadPatch:
        """Deterministic local plane from paired surveyed borders; invalid patches carry a
        reason."""
        ...


@runtime_checkable
class BorderEvidenceSource(Protocol):
    """Owner A (``camloc/borders.py``): lateral/yaw border evidence.

    C calls it only when no selected in-gate full pose exists, at most BORDER_MAX_RATE_HZ
    in source time; an invalid road patch disables it for that group.
    """

    def border_evidence(
        self, observation: Observation, snapshot: SearchSnapshot, aux: AuxiliaryGeometry
    ) -> BorderEvidence:
        """Lateral/yaw-only evidence on this image; at most 200 samples and 85 candidate
        evaluations."""
        ...


@runtime_checkable
class TrackProjector(Protocol):
    """Owner A (``camloc/track_coords.py``): route coordinates of C's estimates."""

    def project(self, estimate: Estimate) -> tuple[TrackProjection, TrackProjection]:
        """(track, pit) projections of the estimate's own x/y/yaw; never GT or native state
        fields."""
        ...

    def reset(self, reset: RecoveryReset) -> None:
        """Clear sequence-projection history so the next projection searches globally."""
        ...


@runtime_checkable
class MappedTracker(Protocol):
    """Owner B (``camloc/vision.py``): camera-rate mapped tracking, handoff and recovery.

    Per group C calls ``observe`` first, then ``track``, ``handoff`` for due deliveries and
    ``verify_candidate`` while a candidate is held; after ``Estimator.step`` it calls
    ``settle`` once per HandoffEvidence and, after a confirmed recovery, ``reset``. Weak
    outcomes are returned as rejected evidence with a reason, never as an accepted pose.
    """

    def observe(self, observation: Observation) -> None:
        """Called exactly once per decoded Observation, in group order, before any other use
        of it. B keeps gray history of <= HISTORY_MAX_AGE_NS and <= HISTORY_MAX_FRAMES per
        camera."""
        ...

    def track(
        self, observation: Observation, snapshot: SearchSnapshot, aux: AuxiliaryGeometry
    ) -> PoseEvidence:
        """Bounded LK on <= MAX_ACTIVE_TRACKS mapped tracks and RANSAC PnP on this image
        (source=tracking)."""
        ...

    def handoff(self, result: AnchorResult, current: Observation) -> HandoffEvidence:
        """Carry result's source pixels to `current` (same camera) through <= 4 chronological
        LK transitions over history, <= 256 points, and solve fresh PnP at
        current.exposure_ns. Does not modify track sets; rejected outcomes carry a reason."""
        ...

    def settle(self, handoff: HandoffEvidence, disposition: HandoffDisposition) -> None:
        """C's decision, exactly once per HandoffEvidence, after step() of that group: adopt
        into the active set, hold as the single recovery candidate (replacing any earlier
        one) or discard."""
        ...

    def verify_candidate(self, observation: Observation) -> PoseEvidence | None:
        """PnP on the held candidate track set (source=candidate); None when no candidate is
        held."""
        ...

    def reset(self, reset: RecoveryReset) -> None:
        """After confirmed recovery: promote the candidate set and drop tracks of the old
        hypothesis."""
        ...


@runtime_checkable
class AnchorWorker(Protocol):
    """Owner B (``camloc/vision.py`` / ``camloc/retrieval.py``): one bounded anchor job.

    C's scheduler keeps one job in flight and one latest pending, and delivers the result
    at the first group with t_ns >= ``delivery_not_before_ns`` through
    ``MappedTracker.handoff``. The result carries no pose.
    """

    def run(self, request: AnchorRequest) -> AnchorResult:
        """Extraction, optional retrieval and <= 2 reference matches on request.observation
        only. Deterministic in (request, frozen assets); never reads estimator state, wall
        clock, other images or pending requests. Expected failures return status failed; C
        maps exceptions and watchdog expiry to backend_fault. C's scheduler alone dispatches
        and delivers."""
        ...


@runtime_checkable
class GroundMotionEstimator(Protocol):
    """Owner B (``camloc/vision.py``): planar ground motion between two same-camera images.

    C calls it only when no selected in-gate full pose exists, with the group's RoadPatch
    and causal snapshots at each image's own exposure (not at the group time). Compose
    each T_WB from snapshot x/y/yaw and auxiliary z/roll/pitch, then T_WC = T_WB @ T_BC.
    The world-frame plane can then be intersected with rays at both exposures. Auxiliary
    z already includes the fitting-only frozen body-to-road height offset when derived
    from the road; B must not assume the body origin lies on the road.
    """

    def estimate(
        self,
        previous: Observation,
        current: Observation,
        patch: RoadPatch,
        *,
        previous_snapshot: SearchSnapshot,
        current_snapshot: SearchSnapshot,
    ) -> MotionEvidence:
        """Same camera/calibration/preprocessing, previous.exposure_ns < current.exposure_ns.

        Snapshot times must equal their respective image exposures, and both auxiliary
        geometries must be valid and compatible with the patch. Missing/invalid geometry,
        incompatible patch or mismatched times/calibration/preprocessing -> rejected
        MotionEvidence with an explicit reason, never an assumed metric scale. The current
        snapshot is a causal prediction before this group's measurements; the previous
        snapshot comes from C's preceding-exposure cache, never a later posterior.
        """
        ...


@runtime_checkable
class Estimator(Protocol):
    """Owner C (``camloc/filter.py``, step 2): the single estimator and update path.

    Constructed once from exactly one InitialPose. Stale worker poses never reach it:
    anchors enter only as current-image HandoffEvidence inside GroupEvidence.
    """

    def snapshot(self, t_ns: int) -> SearchSnapshot:
        """CV-predicted prior at t_ns >= last committed time; no state change."""
        ...

    def step(self, evidence: GroupEvidence) -> Estimate:
        """The single update path: predict to evidence.t_ns, apply consumed slots once, return
        Estimate."""
        ...

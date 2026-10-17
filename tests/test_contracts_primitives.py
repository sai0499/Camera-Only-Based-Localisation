"""Tests of the contracts foundation: time, angles, ids, constants, enums, exceptions.

Example tests come first; seeded property tests follow, each a loop over
``contract_factories.seeded_rng(seed, k)`` whose failures name the seed, iteration and input.
Committed assets are only read; these tests write nothing.
"""

import dataclasses
import decimal
import enum
import fractions
import hashlib
import json
import math
import re
import unicodedata

import numpy as np
import pytest

import contract_factories as factories
from camloc import contracts as c

ROOT = factories.ROOT

# ---------------------------------------------------------------------------------------
# Integer nanoseconds (Requirements 1.2, 1.3)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [0, 1_791_918_859_504_944_006, c.INT64_MAX, np.int64(c.INT64_MAX), np.uint64(c.INT64_MAX)],
    ids=["zero", "split_stamp", "int64_max", "np_int64_max", "np_uint64_max"],
)
def test_as_ns_returns_an_exact_python_int(value):
    result = c.as_ns(value, "exposure_ns")
    assert type(result) is int
    assert result == int(value)


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 1.0, np.float64(1.0), "1", np.timedelta64(1, "ns"), None],
    ids=["bool", "np_bool", "integral_float", "np_float64", "str", "np_timedelta64", "none"],
)
def test_as_ns_rejects_non_integer_types(value):
    with pytest.raises(c.ContractError, match=r"^exposure_ns: expected integer nanoseconds"):
        c.as_ns(value, "exposure_ns")


@pytest.mark.parametrize(
    "value",
    [-1, 2**63, np.int64(-1), np.uint64(2**63)],
    ids=["minus_one", "two_pow_63", "np_int64_minus_one", "np_uint64_two_pow_63"],
)
def test_as_ns_rejects_values_outside_the_int64_range(value):
    message = r"^exposure_ns: .+ is outside \[0, 9223372036854775807\]"
    with pytest.raises(c.ContractError, match=message):
        c.as_ns(value, "exposure_ns")


# ---------------------------------------------------------------------------------------
# Angles (Requirement 2.3 and the organizer -pi boundary)
# ---------------------------------------------------------------------------------------


def test_wrap_angle_stores_minus_pi_as_pi_where_the_organizer_wrap_keeps_minus_pi():
    organizer_wrap = factories.load_organizer_track_frame().wrap
    assert float(organizer_wrap(-math.pi)) == -math.pi  # organizer: closed range [-pi, pi]
    assert c.wrap_angle(-math.pi) == math.pi  # contracts: half-open (-pi, pi]
    assert c.wrap_angle(math.pi) == math.pi == float(organizer_wrap(math.pi))
    # Both name the same angle modulo 2 pi, so wrapped yaw errors are unaffected.
    difference = c.wrap_angle(-math.pi) - float(organizer_wrap(-math.pi))
    assert math.remainder(difference, 2.0 * math.pi) == 0.0


@pytest.mark.parametrize(
    "angle",
    [math.nan, math.inf, -math.inf, np.float64(np.nan), np.float32(np.inf), 10**400],
    ids=["nan", "inf", "minus_inf", "np_float64_nan", "np_float32_inf", "int_beyond_float"],
)
def test_wrap_angle_rejects_non_finite_angles(angle):
    with pytest.raises(c.ContractError, match=r"^yaw: "):
        c.wrap_angle(angle, "yaw")


@pytest.mark.parametrize(
    "angle",
    [True, np.bool_(False), "0.5", 0.5j, np.complex128(0.5), None],
    ids=["bool", "np_bool", "str", "complex", "np_complex128", "none"],
)
def test_wrap_angle_rejects_bool_and_non_real_angles(angle):
    with pytest.raises(c.ContractError, match=r"^yaw: expected a finite real number"):
        c.wrap_angle(angle, "yaw")


# ---------------------------------------------------------------------------------------
# Camera ids and reason codes
# ---------------------------------------------------------------------------------------


def test_camera_ids_and_reason_codes_are_checked_exactly():
    assert c.check_camera_id("fr") == "fr"
    assert c.check_reason("too_few_tracks") == "too_few_tracks"
    for camera_id in ("FR", "front", "", None):
        with pytest.raises(c.ContractError, match=r"^camera_id: "):
            c.check_camera_id(camera_id)
    # A trailing newline must fail: re.fullmatch is used, not a "$" anchor.
    for reason in ("Too_few", "1st", "_x", "too few", "too_few\n", "", None):
        with pytest.raises(c.ContractError, match=r"^reason: "):
            c.check_reason(reason)


# ---------------------------------------------------------------------------------------
# Constants: task_docs/FINAL_BUILD_PLAN.md values (section in the comment) and design values
# ---------------------------------------------------------------------------------------

EXPECTED_CONSTANTS = {
    "CAMERA_IDS": ("fl", "fr", "cl", "cr", "r", "rl", "rr"),
    "PRIMARY_CAMERA": "fr",  # section 3
    "SECONDARY_CAMERA": "r",  # section 3
    "ROUTE_IDS": ("track", "pit"),  # section 2a
    "MAP_FRAME": "map",
    "BODY_FRAME": "constructor0/base_link",
    "CAMERA_FRAME_PREFIX": "constructor0/camera_",
    "CANONICAL_WIDTH": 960,  # section 2: RGB 960 x 464
    "CANONICAL_HEIGHT": 464,
    "STATE_FIELDS": ("x_m", "y_m", "yaw_rad", "vx_mps", "vy_mps", "yaw_rate_radps"),  # section 5
    "INT64_MAX": 9_223_372_036_854_775_807,
    "NS_PER_S": 1_000_000_000,
    "MAX_EXPOSURE_SPREAD_NS": 5_000_000,  # section 1: exposure tolerance min(5 ms, 0.2 T)
    "MAX_ARRIVAL_SPREAD_NS": 40_000_000,  # section 1: arrival deadline min(40 ms, 1.2 T)
    "MAX_PENDING_GROUPS": 4,  # section 1: at most four pending groups
    "DELIVERY_MIN_DELAY_NS": 90_000_000,  # section 3a: source exposure + 90 ms
    "MAX_KEYPOINTS": 1024,  # section 3: up to 1,024 retained DISK keypoints
    "MAX_ACTIVE_TRACKS": 512,  # section 3: at most 512 active LK points
    "MAX_HANDOFF_POINTS": 256,  # section 3a: cap of 256 candidate points
    "MAX_HANDOFF_TRANSITIONS": 4,  # section 3a: at most four LK transitions
    "HISTORY_MAX_AGE_NS": 250_000_000,  # section 3a: 0.25 s image history and result age
    "HISTORY_MAX_FRAMES": 32,  # section 3a: at most 32 images per active camera
    "MAX_REFERENCES_PER_JOB": 2,  # sections 3, 3a: at most two references per job
    "RETRIEVAL_CANDIDATES": 10,  # section 5: ten camera-compatible references
    "NORMAL_JOBS_PER_S": 3,  # section 3a: front 2 Hz plus rear 1 Hz
    "RECOVERY_JOBS_PER_S": 4,  # section 3a: four total jobs/s while recovering
    "GRID_CELLS": 16,  # section 3: 4 x 4 image grid
    "RECOVERY_CONFIRM_MIN_NS": 50_000_000,  # section 5: confirmation at least 50 ms later
    "BORDER_MAX_RATE_HZ": 5,  # section 4: border corrections at most 5 Hz
    "ROTATION_TOL": 1e-6,
    "LAST_ROW_TOL": 1e-9,
    "UNIT_TOL": 1e-9,
    "SYMMETRY_TOL": 1e-9,
    "PSD_TOL": 1e-9,
    "SUMMARY_MAX_ARRAY_ITEMS": 64,
    "BUILD_SPLIT_SHA256": "894b41790af50e2f555a34001ef00821164ebc4f2f2589eb2e24280088c09222",
    "SUPPLIED_MAP_SHA256": "648e8cf954d8cd39823ca31c0b62d89c8fd76b170df999ac2e3d4327439972ef",
    "SPLIT_TIME_ENCODING": "decimal_string_nanoseconds_convert_to_integer",  # section 8
    "SPLIT_INTERVAL_CONVENTION": "start_inclusive_end_exclusive",  # section 8: half-open
    "SPLIT_GUARD_NS": 2_000_000_000,  # section 8: episodes end 2 s before each gate
}


def test_constants_equal_the_build_plan_and_design_values():
    actual = {name: getattr(c, name) for name in EXPECTED_CONSTANTS}
    assert actual == EXPECTED_CONSTANTS
    # Exact types too: nanosecond constants are ints, id lists are immutable tuples.
    assert {name: type(value) for name, value in actual.items()} == {
        name: type(value) for name, value in EXPECTED_CONSTANTS.items()
    }


def test_split_and_map_identities_match_the_committed_assets():
    assert c.PROJECT_ROOT == ROOT
    assert c.DEFAULT_SPLIT_PATH == ROOT / "task_docs" / "build_split.json"
    raw = c.DEFAULT_SPLIT_PATH.read_bytes()  # read only: the split file is hash-bound
    assert hashlib.sha256(raw).hexdigest() == c.BUILD_SPLIT_SHA256
    split = json.loads(raw)
    assert split["time_encoding"] == c.SPLIT_TIME_ENCODING
    assert split["interval_convention"] == c.SPLIT_INTERVAL_CONVENTION
    assert split["guard_duration_ns"] == str(c.SPLIT_GUARD_NS)
    static_map = split["static_map"]
    assert (static_map["sha256"], static_map["frame"]) == (c.SUPPLIED_MAP_SHA256, c.MAP_FRAME)
    assert tuple(static_map["reference_path_topics"]) == c.ROUTE_IDS
    supplied_map = (ROOT / static_map["path"]).read_bytes()
    assert hashlib.sha256(supplied_map).hexdigest() == c.SUPPLIED_MAP_SHA256


def test_camera_and_route_ids_match_the_recording_and_the_organizer_helper():
    metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    recorded = re.findall(r"/constructor0/sensor/camera_([a-z]+)/compressed_image", metadata)
    assert sorted(recorded) == sorted(c.CAMERA_IDS)
    assert tuple(factories.load_organizer_track_frame().TOPICS) == c.ROUTE_IDS


# ---------------------------------------------------------------------------------------
# Enums: values are the persisted strings
# ---------------------------------------------------------------------------------------

EXPECTED_ENUM_VALUES = {
    c.Status: ("tracking", "degraded", "recovering", "lost"),
    c.Branch: ("track", "pit", "unknown"),
    c.FrameSetClosure: ("complete", "deadline", "capacity", "eof"),
    c.PoseSource: ("tracking", "handoff", "candidate"),
    c.AnchorPurpose: ("front_anchor", "rear_anchor", "recovery", "warm_retrieval"),
    c.AnchorStatus: ("ok", "failed", "backend_fault", "cancelled"),
    c.AuxSource: ("pnp", "road_patch", "none"),
    c.EvidenceKind: (
        "initial_pose",
        "pnp",
        "handoff_pnp",
        "ground_motion",
        "border",
        "recovery_reset",
        "prediction_only",
    ),
    c.HandoffDisposition: ("adopt", "candidate", "discard"),
    c.InputEventKind: (
        "no_calibration",
        "calibration_mismatch",
        "before_initial_pose",
        "after_episode_end",
        "late_support",
        "duplicate_stamp",
        "decreasing_stamp",
        "decode_failure",
        "coverage_gap",
        "end_of_input",
    ),
}


@pytest.mark.parametrize(
    ("enum_cls", "values"),
    list(EXPECTED_ENUM_VALUES.items()),
    ids=[enum_cls.__name__ for enum_cls in EXPECTED_ENUM_VALUES],
)
def test_enum_members_persist_their_design_strings(enum_cls, values):
    assert issubclass(enum_cls, enum.StrEnum)
    assert tuple(member.value for member in enum_cls) == values
    # Member names are the upper-case values, with no aliases.
    assert list(enum_cls.__members__) == [value.upper() for value in values]
    for member in enum_cls:
        assert str(member) == member.value
        assert json.dumps(member) == json.dumps(member.value)


# ---------------------------------------------------------------------------------------
# Exceptions (Requirement 14.5)
# ---------------------------------------------------------------------------------------


def test_exception_hierarchy():
    assert issubclass(c.ContractError, ValueError)
    assert issubclass(c.RejectedEvidenceError, c.ContractError)
    assert issubclass(c.InputError, Exception)
    # A data failure on one input (e.g. an undecodable JPEG) is not a contract violation.
    assert not issubclass(c.InputError, ValueError)


# ---------------------------------------------------------------------------------------
# Factories used by the property tests
# ---------------------------------------------------------------------------------------


def test_rigid_factories_build_rotations_and_closed_form_inverses():
    np.testing.assert_array_equal(
        factories.random_rotation(factories.seeded_rng(7)),
        factories.random_rotation(factories.seeded_rng(7)),
    )
    rng = factories.seeded_rng(20261017)
    for _ in range(20):
        rotation = factories.random_rotation(rng)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12)
        assert abs(np.linalg.det(rotation) - 1.0) <= 1e-12
        T_AB = factories.rigid(rotation, rng.uniform(-500.0, 500.0, size=3))
        T_BA = factories.rigid_inverse(T_AB)
        np.testing.assert_allclose(T_AB @ T_BA, np.eye(4), rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(T_BA @ T_AB, np.eye(4), rtol=0.0, atol=1e-9)
        for transform in (T_AB, T_BA):
            np.testing.assert_array_equal(c.check_rigid(transform, "T_AB"), transform)


# ---------------------------------------------------------------------------------------
# Property 1: nanosecond type discipline (Requirements 1.1, 1.2, 1.3, 1.6)
# ---------------------------------------------------------------------------------------

NS_SEED = 20261013
NS_ITERATIONS = 200
NS_FIELD_NAMES = (
    "exposure_ns",
    "ImageMessage.arrival_ns",
    "log_ns",
    "FrameSet.t_ns",
    "reference_ns",
    "AnchorRequest.delivery_not_before_ns",
)
# The recording's GT span (split "full" bounds): 61-bit stamps that a float64 detour rounds.
NS_STAMP_RANGE = (1_791_918_157_114_146_050, 1_791_919_351_506_836_680)
NS_EDGES_INSIDE = (0, 1, c.INT64_MAX - 1, c.INT64_MAX)
NS_EDGES_OUTSIDE = (-1, -(2**63), 2**63, 2**64)
# Every distinct NumPy integer scalar type; on Windows intc/int32 and uintc/uint32 differ.
# np.timedelta64 subclasses np.signedinteger but carries a unit, so it is drawn as rejected.
NUMPY_INTEGER_TYPES = tuple(
    dict.fromkeys(np.dtype(code).type for code in np.typecodes["AllInteger"])
)


def _uniform_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Return a Python int drawn uniformly from [lo, hi], for bounds of any size."""
    span = hi - lo + 1
    return lo + int.from_bytes(rng.bytes(span.bit_length() // 8 + 9), "little") % span


def _spread_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Return an int in [lo, hi] (0 <= lo <= hi) whose bit length is drawn uniformly.

    Small and large magnitudes are equally likely; a uniform draw from [0, 2**63) would
    almost always give 19-digit values.
    """
    bits = int(rng.integers(lo.bit_length(), hi.bit_length(), endpoint=True))
    return _uniform_int(rng, max(lo, (1 << bits) >> 1), min(hi, (1 << bits) - 1))


def _type_label(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _ns_cases(rng: np.random.Generator, k: int) -> list[tuple[object, int | None]]:
    """Return iteration ``k``'s ``as_ns`` inputs as (value, expected) pairs.

    ``expected`` is the exact int an accepted input must return, or None when the input
    must raise ContractError. It follows from how the value is drawn, not from ``as_ns``.
    """
    n = _spread_int(rng, 0, c.INT64_MAX)
    stamp = _uniform_int(rng, *NS_STAMP_RANGE)
    edge = NS_EDGES_INSIDE[k % len(NS_EDGES_INSIDE)]
    cases: list[tuple[object, int | None]] = [(n, n), (stamp, stamp), (edge, edge)]
    # Python ints outside [0, 2**63 - 1]: negative, 64 to 130 bits, beyond the float range.
    for value in (
        -_spread_int(rng, 1, 2**130),
        _spread_int(rng, 2**63, 2**130),
        (-1) ** k * _spread_int(rng, 2**1024, 2**1100),
        NS_EDGES_OUTSIDE[k % len(NS_EDGES_OUTSIDE)],
    ):
        cases.append((value, None))
    # Each NumPy integer type: a drawn and an edge value inside [0, 2**63 - 1], and negative
    # or (uint64) >= 2**63 values outside it.
    for scalar_type in NUMPY_INTEGER_TYPES:
        info = np.iinfo(scalar_type)
        low, high = int(info.min), int(info.max)
        top = min(high, c.INT64_MAX)
        for value in (_spread_int(rng, 0, top), (0, 1, top - 1, top)[k % 4]):
            cases.append((scalar_type(value), value))
        outside = []
        if low < 0:
            outside += [-_spread_int(rng, 1, -low), (-1, low)[k % 2]]
        if high > c.INT64_MAX:
            outside += [_uniform_int(rng, c.INT64_MAX + 1, high), (c.INT64_MAX + 1, high)[k % 2]]
        cases += [(scalar_type(value), None) for value in outside]
    # Never integer nanoseconds, whatever the value.
    small = int(rng.integers(0, 2**53))  # exactly representable as a float64
    unit = ("ns", "us", "ms", "s")[k % 4]
    special_float = (math.nan, math.inf, -math.inf, -0.0)[k % 4]
    rejected: list[object] = [True, False, np.True_, np.False_, None]
    rejected += [float(small), float(n), float(rng.uniform(-1e19, 1e19)), special_float]
    rejected += [np.float16(small % 2049), np.float32(small), np.float64(n), np.longdouble(small)]
    rejected += [str(n), np.str_(n), str(n).encode()]
    rejected += [np.timedelta64(small, unit), np.datetime64(small, unit)]
    rejected += [np.array(n, dtype=np.int64), np.array([n], dtype=np.int64), [n], (n,)]
    rejected += [decimal.Decimal(n), fractions.Fraction(n), complex(small), np.complex128(small)]
    cases += [(value, None) for value in rejected]
    return cases


def _check_as_ns(k: int, value: object, expected: int | None, name: str) -> None:
    """Check one ``as_ns`` case; a failure names the seed, iteration and input."""
    case = f"seed {NS_SEED} iteration {k}: as_ns({_type_label(value)} {value!r}, {name!r})"
    try:
        result = c.as_ns(value, name)
    except c.ContractError as exc:
        assert expected is None, f"{case} raised {exc!r}, expected {expected}"
        assert str(exc).startswith(f"{name}: "), f"{case}: message {exc!r} does not name the field"
        return
    except Exception as exc:
        pytest.fail(f"{case} raised {exc!r}, expected ContractError or an int")
    assert expected is not None, f"{case} returned {result!r}, expected ContractError"
    assert type(result) is int, f"{case} returned {_type_label(result)} {result!r}, not int"
    assert result == expected, f"{case} returned {result}, expected {expected}"


def _numpy_stamped_records(
    rng: np.random.Generator, k: int, image: np.ndarray, mask: np.ndarray
) -> list[tuple[object, dict[str, object], dict[str, int | None]]]:
    """Return iteration ``k``'s records as (record, constructor kwargs, expected stamps).

    One FrameSet holds an ``fr`` image, on some iterations an ``r`` image, and up to three
    input events; the ``fr`` Observation follows. Every stamp is passed as np.int64;
    ``expected`` maps each ``*_ns`` field to the exact int (or None) that was passed.
    """
    built: list[tuple[object, dict[str, object], dict[str, int | None]]] = []

    def build(cls: type, expected: dict[str, int | None], **kwargs: object) -> object:
        kwargs.update({key: None if ns is None else np.int64(ns) for key, ns in expected.items()})
        record = cls(**kwargs)
        built.append((record, kwargs, expected))
        return record

    # Stamps next to 0 and next to INT64_MAX on iterations 0 and 1, in the GT span afterwards.
    span = c.MAX_ARRIVAL_SPREAD_NS
    base = (0, c.INT64_MAX - span)[k] if k < 2 else _uniform_int(rng, *NS_STAMP_RANGE)
    cameras = ("fr", "r") if rng.integers(0, 2) else ("fr",)
    exposures, images = [], []
    for camera in cameras:
        exposure = base + int(rng.integers(0, c.MAX_EXPOSURE_SPREAD_NS, endpoint=True))
        latency = int(rng.integers(0, span - c.MAX_EXPOSURE_SPREAD_NS, endpoint=True))
        exposures.append(exposure)
        message = build(
            c.ImageMessage,
            {"exposure_ns": exposure, "arrival_ns": exposure + latency},
            camera_id=camera,
            source_id=f"{camera}#{k}",
            frame_id=c.CAMERA_FRAME_PREFIX + camera,
            image_format="jpeg",
            calibration_id=f"calib-{camera}",
            data=b"\xff\xd8\xff\xd9",
        )
        images.append(message)
    fr_stamps = dict(built[0][2])

    kinds = tuple(c.InputEventKind)
    events = []
    for offset in np.sort(rng.integers(0, span, size=int(rng.integers(0, 4)), endpoint=True)):
        event = build(
            c.InputEvent,
            {
                "log_ns": base + int(offset),
                "exposure_ns": None if rng.integers(0, 2) else base + int(rng.integers(0, span)),
                "reference_ns": None if rng.integers(0, 2) else base + int(rng.integers(0, span)),
            },
            kind=kinds[int(rng.integers(len(kinds)))],
        )
        events.append(event)

    missing = () if len(cameras) == 2 else ("r",)
    closures = (c.FrameSetClosure.DEADLINE, c.FrameSetClosure.CAPACITY, c.FrameSetClosure.EOF)
    closure = closures[int(rng.integers(len(closures)))] if missing else c.FrameSetClosure.COMPLETE
    build(
        c.FrameSet,
        {"t_ns": max(exposures)},
        group_id=k,
        selected_camera_ids=("fr", "r"),
        images=tuple(images),
        missing_camera_ids=missing,
        closure=closure,
        terminal_partial=closure is c.FrameSetClosure.EOF,
        input_events=tuple(events),
    )
    build(
        c.Observation,
        fr_stamps,
        camera_id="fr",
        source_id=f"fr#{k}",
        image=image,
        valid_mask=mask,
        calibration_id="calib-fr",
        preprocessing_id="canonical-960x464-v1",
        K=np.array([[600.0, 0.0, 479.5], [0.0, 600.0, 231.5], [0.0, 0.0, 1.0]]),
        T_BC=np.eye(4),
    )
    return built


def _check_stored_stamps(k: int, record: object, expected: dict[str, int | None]) -> None:
    """Check that every ``*_ns`` field of ``record`` stores its expected exact int (or None)."""
    case = f"seed {NS_SEED} iteration {k}: {type(record).__name__}"
    stored = {
        field.name: getattr(record, field.name)
        for field in dataclasses.fields(record)
        if field.name.endswith("_ns")
    }
    assert stored.keys() == expected.keys(), f"{case} has stamp fields {sorted(stored)}"
    for field, value in stored.items():
        want = expected[field]
        ok = value is None if want is None else type(value) is int and value == want
        assert ok, f"{case}.{field} stores {_type_label(value)} {value!r}, expected {want!r}"


def _check_stamp_rejected(
    k: int, cls: type, kwargs: dict[str, object], field: str, value: object
) -> None:
    """Check that ``cls`` given ``value`` as stamp ``field`` raises a ContractError naming it."""
    case = f"seed {NS_SEED} iteration {k}: {cls.__name__}({field}={_type_label(value)} {value!r})"
    try:
        cls(**{**kwargs, field: value})
    except c.ContractError as exc:
        prefix = f"{cls.__name__}.{field}: "
        assert str(exc).startswith(prefix), f"{case}: message {exc!r} does not name the field"
        return
    except Exception as exc:
        pytest.fail(f"{case} raised {exc!r}, expected ContractError")
    pytest.fail(f"{case} was accepted, expected ContractError")


# Feature: shared-contracts, Property 1: Nanosecond type discipline
def test_property_1_nanosecond_type_discipline():
    """``as_ns`` returns an exact int exactly for in-range integers; records store ints.

    For any value v, ``as_ns(v)`` returns an ``int`` equal to v exactly when v is an ``int``
    (not ``bool``) or a NumPy integer in [0, 2**63 - 1], and otherwise raises a
    ``ContractError`` naming the field: ``bool``, ``np.bool_``, every float (integral ones
    too), ``str``, NumPy timedelta/datetime scalars, other types and out-of-range values.
    Records built with np.int64 stamps store every stamp as an ``int``, and a rejected
    stamp value raises a ``ContractError`` naming ``<Record>.<field>``.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.6**
    """
    fixed_width = {np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64}
    assert fixed_width <= set(NUMPY_INTEGER_TYPES)
    # Shared canonical buffers; the first Observation marks them read-only, which is allowed.
    image = np.zeros((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3), dtype=np.uint8)
    mask = np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), dtype=bool)
    for k in range(NS_ITERATIONS):
        rng = factories.seeded_rng(NS_SEED, k)
        name = NS_FIELD_NAMES[k % len(NS_FIELD_NAMES)]
        cases = _ns_cases(rng, k)
        for value, expected in cases:
            _check_as_ns(k, value, expected, name)
        # None is a valid value of optional stamps, so it is not used against records.
        rejected = [value for value, expected in cases if expected is None and value is not None]
        for record, kwargs, expected in _numpy_stamped_records(rng, k, image, mask):
            _check_stored_stamps(k, record, expected)
            for field in expected:
                bad = rejected[int(rng.integers(len(rejected)))]
                _check_stamp_rejected(k, type(record), kwargs, field, bad)


# ---------------------------------------------------------------------------------------
# Property 2: strict decimal stamps (Requirements 1.4, 1.5)
# ---------------------------------------------------------------------------------------

DECIMAL_SEED = 20261014
DECIMAL_ITERATIONS = 200
DECIMAL_FIELD_NAMES = ("timestamp_ns", "start_ns", "end_ns", "guard_duration_ns", "fit.start_ns")
DECIMAL_EDGES_INSIDE = (0, 1, 7, 10, c.INT64_MAX - 1, c.INT64_MAX)
DECIMAL_EDGES_ABOVE = (c.INT64_MAX + 1, 2**64, 10**19, 10**24)  # 19, 20, 20 and 25 digits
# Strings this long pass int()'s default 4,300-digit limit, which counts leading zeros:
# int("0" * 5000 + "7") raises a plain ValueError.
DECIMAL_MAX_LENGTH = 5_000
# Whitespace that int() strips around a number, ASCII and Unicode.
DECIMAL_WHITESPACE = (" ", "\t", "\n", "\r", "\v", "\f", "\u00a0", "\u2003", "\u3000")
DIGIT_NAMES = ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE")
# Non-ASCII digit families by Unicode name prefix, digit d at index d. int() reads the
# decimal ones like ASCII digits: ARABIC-INDIC DIGIT THREE is U+0663, FULLWIDTH DIGIT THREE
# U+FF13. SUPERSCRIPT TWO (U+00B2) is a digit to str.isdigit() but not a decimal digit.
NON_ASCII_DIGITS = {
    family: "".join(unicodedata.lookup(f"{family} {digit}") for digit in DIGIT_NAMES)
    for family in (
        "ARABIC-INDIC DIGIT",
        "EXTENDED ARABIC-INDIC DIGIT",
        "DEVANAGARI DIGIT",
        "FULLWIDTH DIGIT",
        "MATHEMATICAL BOLD DIGIT",
        "SUPERSCRIPT",
        "SUBSCRIPT",
    )
}


def _digit_string(rng: np.random.Generator, length: int) -> str:
    """Return ``length`` random ASCII digits, the first non-zero."""
    digits = rng.integers(0, 10, size=length, dtype=np.uint8)
    digits[0] = rng.integers(1, 10)
    return (digits + ord("0")).tobytes().decode("ascii")


def _decimal_mutations(rng: np.random.Generator, text: str) -> list[tuple[str, str]]:
    """Return labelled mutations of the ASCII digit string ``text``, at drawn positions.

    Each adds a sign, whitespace, an underscore, a decimal point, an exponent or a non-ASCII
    digit. Many still read as a number to ``int()`` or ``float()``: a sign prefix, padding
    whitespace, an underscore between digits, a same-value Arabic-Indic or fullwidth digit,
    a ``.0`` suffix or an exponent.
    """

    def anywhere() -> int:  # an insertion index: prefix, inside or suffix
        return int(rng.integers(0, len(text), endpoint=True))

    def inside() -> int:  # between two characters, where int() accepts an underscore
        return int(rng.integers(1, len(text))) if len(text) > 1 else anywhere()

    def insert(index: int, piece: str) -> str:
        return text[:index] + piece + text[index:]

    def blank() -> str:  # one to three whitespace characters
        picks = rng.integers(0, len(DECIMAL_WHITESPACE), size=int(rng.integers(1, 4)))
        return "".join(DECIMAL_WHITESPACE[i] for i in picks)

    sign = "+-"[int(rng.integers(2))]
    marker = "eE"[int(rng.integers(2))]
    exponent = marker + ("", "+", "-")[int(rng.integers(3))] + str(int(rng.integers(20)))
    mutations = [
        ("plus sign", "+" + text),
        ("minus sign", "-" + text),
        ("inserted sign", insert(anywhere(), sign)),
        ("leading whitespace", blank() + text),
        ("trailing whitespace", text + blank()),
        ("surrounding whitespace", blank() + text + blank()),
        ("inner whitespace", insert(inside(), blank())),
        ("trailing newline", text + "\n"),
        ("trailing CRLF", text + "\r\n"),
        ("underscore between digits", insert(inside(), "_")),
        ("inserted underscore", insert(anywhere(), "_")),
        ("trailing point", text + "."),
        ("trailing .0", text + ".0"),
        ("drawn fraction", f"{text}.{int(rng.integers(1000))}"),
        ("inserted point", insert(anywhere(), ".")),
        ("exponent e9", text + "e9"),
        ("exponent E3", text + "E3"),
        ("drawn exponent", text + exponent),
        ("inserted exponent", insert(inside(), marker)),
    ]
    for family, digits in NON_ASCII_DIGITS.items():
        index = int(rng.integers(len(text)))
        same_value = text[:index] + digits[int(text[index])] + text[index + 1 :]
        mutations.append((f"same-value {family}", same_value))
        mutations.append((f"inserted {family}", insert(anywhere(), digits[int(rng.integers(10))])))
    family = tuple(NON_ASCII_DIGITS)[int(rng.integers(len(NON_ASCII_DIGITS)))]
    translation = str.maketrans("0123456789", NON_ASCII_DIGITS[family])
    mutations.append((f"all {family}", text.translate(translation)))
    for label, mutated in mutations:  # each adds a character other than an ASCII digit
        assert not (mutated.isascii() and mutated.isdigit()), f"{label}: {mutated!a}"
    return mutations


def _decimal_cases(rng: np.random.Generator, k: int) -> list[tuple[str, object, int | None]]:
    """Return iteration ``k``'s ``parse_decimal_ns`` inputs as (label, value, expected).

    ``expected`` is the exact int an accepted string must return, or None when the input
    must raise ContractError. It follows from how the value is built, not from
    ``parse_decimal_ns``.
    """
    values = (
        _spread_int(rng, 0, c.INT64_MAX),
        _uniform_int(rng, *NS_STAMP_RANGE),
        DECIMAL_EDGES_INSIDE[k % len(DECIMAL_EDGES_INSIDE)],
    )
    accepted = [("str(t)", str(t), t) for t in values]
    # Leading zeros keep the value ("007" is 7), also beyond int()'s digit limit.
    accepted += [("zero-padded", "0" * _spread_int(rng, 1, 40) + str(t), t) for t in values]
    padding = "0" * _spread_int(rng, 1, DECIMAL_MAX_LENGTH)
    accepted.append(("zero-padded INT64_MAX", padding + str(c.INT64_MAX), c.INT64_MAX))
    accepted.append(("zeros", "0" * int(rng.integers(1, 40)), 0))
    # Unsigned ASCII digit strings above INT64_MAX: 19 digits, up to 31, or very long.
    above = (
        _uniform_int(rng, c.INT64_MAX + 1, 10**19 - 1),
        _spread_int(rng, c.INT64_MAX + 1, 10**30),
        DECIMAL_EDGES_ABOVE[k % len(DECIMAL_EDGES_ABOVE)],
    )
    cases: list[tuple[str, object, int | None]] = list(accepted)
    cases += [("above INT64_MAX", str(value), None) for value in above]
    cases.append(("zero-padded above INT64_MAX", padding + str(above[0]), None))
    long_digits = _digit_string(rng, _spread_int(rng, 20, DECIMAL_MAX_LENGTH))
    cases.append(("long digit string", long_digits, None))
    for _, base, _ in accepted:
        cases += [(label, mutated, None) for label, mutated in _decimal_mutations(rng, base)]
    cases.append(("empty string", "", None))
    # Inputs whose type is not exactly str; np.str_ subclasses str and is rejected too.
    text, t = str(values[0]), values[0]
    for label, value in (
        ("bytes", text.encode("ascii")),
        ("bytearray", bytearray(text, "ascii")),
        ("numpy str_", np.str_(text)),
        ("numpy bytes_", np.bytes_(text.encode("ascii"))),
        ("int", t),
        ("numpy int64", np.int64(t)),
        ("float", float(t)),
        ("None", None),
        ("list", [text]),
    ):
        cases.append((label, value, None))
    return cases


def _check_parse_decimal(
    k: int, label: str, value: object, expected: int | None, name: str
) -> None:
    """Check one ``parse_decimal_ns`` case; a failure names the seed, iteration and input."""
    shown = ascii(value)
    if len(shown) > 100:  # shortened; the seed and iteration replay the full input
        shown = f"{shown[:60]}...{shown[-30:]} (ascii length {len(shown)})"
    case = (
        f"seed {DECIMAL_SEED} iteration {k} ({label}): "
        f"parse_decimal_ns({_type_label(value)} {shown}, {name!r})"
    )
    try:
        result = c.parse_decimal_ns(value, name)
    except c.ContractError as exc:
        assert expected is None, f"{case} raised {exc!a}, expected {expected}"
        assert str(exc).startswith(f"{name}: "), f"{case}: message {exc!a} does not name the field"
        return
    except Exception as exc:
        pytest.fail(f"{case} raised {exc!a}, expected ContractError or an int")
    assert expected is not None, f"{case} returned {result!a}, expected ContractError"
    assert type(result) is int, f"{case} returned {_type_label(result)} {result!a}, not int"
    assert result == expected, f"{case} returned {result}, expected {expected}"


# Feature: shared-contracts, Property 2: Strict decimal stamps
def test_property_2_strict_decimal_stamps():
    """``parse_decimal_ns`` returns t exactly for unsigned ASCII decimal strings of int64 t.

    For any t in [0, 2**63 - 1], ``parse_decimal_ns(str(t))`` and every zero-padded form
    return the ``int`` t, and digit strings above 2**63 - 1 raise. Any mutation of such a
    string that adds a sign, whitespace (a lone trailing newline included), an underscore,
    a decimal point, an exponent or a non-ASCII digit, the empty string and every input
    whose type is not exactly ``str`` (``np.str_`` included) raise a ``ContractError``
    naming the field, never another exception.

    **Validates: Requirements 1.4, 1.5**
    """
    # Generator premises: digit d of each family sits at index d; int() reads the decimal
    # families and signed, padded or underscored text; np.str_ is a str subclass.
    for digits in NON_ASCII_DIGITS.values():
        assert [unicodedata.digit(ch) for ch in digits] == list(range(10)), ascii(digits)
        assert not digits.isdecimal() or int(digits) == 123_456_789, ascii(digits)
    assert int(" +1_0\n") == 10
    assert isinstance(np.str_("1"), str)
    for k in range(DECIMAL_ITERATIONS):
        rng = factories.seeded_rng(DECIMAL_SEED, k)
        name = DECIMAL_FIELD_NAMES[k % len(DECIMAL_FIELD_NAMES)]
        for label, value, expected in _decimal_cases(rng, k):
            _check_parse_decimal(k, label, value, expected, name)


# ---------------------------------------------------------------------------------------
# Property 3: canonical angle wrap (Requirements 2.1, 2.2, 2.4, 2.5, 7.4)
# ---------------------------------------------------------------------------------------

ANGLE_SEED = 20261015
ANGLE_ITERATIONS = 200
# The organizer helper reduces by the true 2 pi and math.remainder by the float 2 pi, which
# is 2.4e-16 smaller: they drift apart by 4e-14 rad at 1e3 but by 0.04 rad at 1e15, so
# angles beyond 1e3 (up to HUGE_ANGLE) are checked for range and idempotence only.
ORGANIZER_RANGE = 1e3
ORGANIZER_TOL = 1e-12  # rad, on the difference wrapped modulo 2 pi
HUGE_ANGLE = 1e15
PI_NEIGHBOURS = (math.nextafter(-math.pi, 0.0), math.nextafter(math.pi, math.inf))
ANGLE_EDGES = (math.pi, -math.pi, *PI_NEIGHBOURS, -0.0, 0.0)  # -pi is stored as pi


def _same_float(a: float, b: float) -> bool:
    """Exact float equality that also tells -0.0 from 0.0."""
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


def _angle_cases(rng: np.random.Generator, k: int) -> list[float]:
    """Return iteration ``k``'s finite angles; only the last two may exceed 1e3 (up to 1e15)."""
    small_k, huge_k = int(ORGANIZER_RANGE / math.pi), int(HUGE_ANGLE / math.pi)
    multiple = int(rng.integers(-small_k, small_k, endpoint=True)) * math.pi
    return [
        float(rng.uniform(-ORGANIZER_RANGE, ORGANIZER_RANGE)),
        -float(rng.uniform(-math.pi, math.pi)),  # negated [-pi, pi): in (-pi, pi]
        float(rng.uniform(-1.0, 1.0)) * 10.0 ** -float(rng.uniform(0.0, 320.0)),  # to subnormal
        multiple,
        math.nextafter(multiple, (-math.inf, math.inf)[k % 2]),
        (-1.0) ** k * ORGANIZER_RANGE,
        *ANGLE_EDGES,
        float(rng.uniform(-1.0, 1.0)) * 10.0 ** float(rng.uniform(3.0, 15.0)),
        int(rng.integers(-huge_k, huge_k, endpoint=True)) * math.pi,
    ]


def _check_wrap(k: int, angle: float) -> None:
    """Check range, idempotence, in-range identity and organizer agreement for one angle."""
    case = f"seed {ANGLE_SEED} iteration {k}: wrap_angle({angle!r})"
    try:
        wrapped = c.wrap_angle(angle)
        again = c.wrap_angle(wrapped)
    except c.ContractError as exc:
        pytest.fail(f"{case} raised {exc!r}")
    assert -math.pi < wrapped <= math.pi, f"{case} = {wrapped!r} is outside (-pi, pi]"
    assert _same_float(again, wrapped), f"{case} = {wrapped!r}, which rewraps to {again!r}"
    if -math.pi < angle <= math.pi:
        assert _same_float(wrapped, angle), f"{case} = {wrapped!r} changed an in-range angle"
    if abs(angle) <= ORGANIZER_RANGE:
        theirs = float(factories.load_organizer_track_frame().wrap(angle))
        gap = abs(math.remainder(wrapped - theirs, 2.0 * math.pi))
        assert gap <= ORGANIZER_TOL, f"{case} = {wrapped!r}, organizer {theirs!r}: gap {gap!r}"


def _stored_record_angles(k: int, angles: list[float]) -> list[tuple[str, float]]:
    """Build each record with an angle field from six ``angles``; return the stored values.

    The angles feed roll, pitch, epsi, the snapshot yaw, the estimate yaw and the border yaw.
    """
    roll, pitch, epsi, snapshot_yaw, estimate_yaw, border_yaw = angles
    t_ns = factories.EXPOSURE_NS + k
    pnp = c.AuxiliaryGeometry(
        source=c.AuxSource.PNP,
        source_ns=t_ns,
        z_m=0.35,
        roll_rad=roll,
        pitch_rad=pitch,
        road_patch_id=None,
        valid=True,
        reason="valid",
    )
    projection = c.TrackProjection(
        t_ns=t_ns,
        reference_map_id=c.SUPPLIED_MAP_SHA256,
        route_id="track",
        s_m=1234.5,
        n_m=-1.75,
        epsi_rad=epsi,
        segment_index=42,
        xy_reconstruction_residual_m=0.002,
        valid=True,
        reason="valid",
    )
    def state(yaw: float) -> np.ndarray:  # [x, y, yaw, vx, vy, yaw rate]
        return np.array([105.0, -48.0, yaw, 31.0, 0.4, 0.05])

    prior = {"t_ns": t_ns, "P": np.eye(6), "status": c.Status.TRACKING, "branch": c.Branch.TRACK}
    prior["auxiliary"] = c.AuxiliaryGeometry.unavailable("no_geometry")
    snapshot = c.SearchSnapshot(state=state(snapshot_yaw), **prior)
    estimate = c.Estimate(
        state=state(estimate_yaw),
        group_id=0,
        evidence_kinds=(c.EvidenceKind.PREDICTION_ONLY,),
        reason="no_evidence",
        max_support_exposure_ns=None,
        **prior,
    )
    border = c.BorderEvidence(
        camera_id="fr",
        exposure_ns=t_ns,
        branch=c.Branch.TRACK,
        normal_xy=np.array([0.0, 1.0]),
        lateral_coordinate_m=-1.75,
        yaw_rad=border_yaw,
        R=np.diag([0.04, 1e-4]),
        support_samples=120,
        support_contours=2,
        conditioning=0.4,
        accepted=True,
        reason="accepted",
    )
    return [
        ("AuxiliaryGeometry.roll_rad", pnp.roll_rad),
        ("AuxiliaryGeometry.pitch_rad", pnp.pitch_rad),
        ("TrackProjection.epsi_rad", projection.epsi_rad),
        ("SearchSnapshot.state[2]", float(snapshot.state[2])),
        ("Estimate.state[2]", float(estimate.state[2])),
        ("BorderEvidence.yaw_rad", border.yaw_rad),
    ]


# Feature: shared-contracts, Property 3: Canonical angle wrap
def test_property_3_canonical_angle_wrap():
    """``wrap_angle`` normalizes every finite angle to (-pi, pi]; records store its result.

    For any finite angle a, ``wrap_angle(a)`` lies in (-pi, pi] and wraps to itself again,
    and an a already in (-pi, pi] comes back as the identical float, -0.0 included. For
    abs(a) <= 1e3 it is within 1e-12 rad, modulo 2 pi, of the organizer
    ``track_frame.wrap(a)``. Every record angle field (AuxiliaryGeometry roll and pitch,
    TrackProjection epsi, SearchSnapshot and Estimate state yaw, BorderEvidence yaw) stores
    exactly ``wrap_angle`` of its input.

    **Validates: Requirements 2.1, 2.2, 2.4, 2.5, 7.4**
    """
    for k in range(ANGLE_ITERATIONS):
        rng = factories.seeded_rng(ANGLE_SEED, k)
        angles = _angle_cases(rng, k)
        for angle in angles:
            _check_wrap(k, angle)
        moderate = [angle for angle in angles if abs(angle) <= ORGANIZER_RANGE]
        picks = [moderate[i] for i in rng.integers(len(moderate), size=6)]
        case = f"seed {ANGLE_SEED} iteration {k}: records built with angles {picks!r}"
        try:
            stored = _stored_record_angles(k, picks)
        except c.ContractError as exc:
            pytest.fail(f"{case} raised {exc!r}")
        for (field, value), given in zip(stored, picks, strict=True):
            want = c.wrap_angle(given)
            message = f"{case}: {field} stores {value!r} for input {given!r}, not {want!r}"
            assert _same_float(value, want), message


# ---------------------------------------------------------------------------------------
# Property 4: rigid transform acceptance (Requirements 3.1, 3.2, 4.4)
# ---------------------------------------------------------------------------------------

RIGID_SEED = 20261016
RIGID_ITERATIONS = 150
RIGID_TRANSLATION_RANGE = 1e4  # metres; large magnitudes exercise the finite-translation path


def _check_rigid_accepted(k: int, case: str, T_in: np.ndarray) -> None:
    """Check that ``check_rigid`` returns a read-only float64 (4, 4) copy equal to ``T_in``."""
    returned = c.check_rigid(T_in, "T_AB")
    assert returned.dtype == np.float64, f"{case} returned dtype {returned.dtype}, not float64"
    assert returned.shape == (4, 4), f"{case} returned shape {returned.shape}, not (4, 4)"
    assert np.array_equal(returned, T_in), f"{case} returned a transform unequal to its input"
    assert returned.flags.writeable is False, f"{case} returned a writeable transform"
    assert returned is not T_in, f"{case} returned the input object, not a copy"


def _check_rigid_rejected(k: int, case: str, T_bad: object) -> None:
    """Check that ``check_rigid`` raises a ContractError naming the field for ``T_bad``."""
    try:
        c.check_rigid(T_bad, "T_AB")
    except c.ContractError as exc:
        assert str(exc).startswith("T_AB: "), f"{case}: message {exc!r} does not name the field"
        return
    except Exception as exc:
        pytest.fail(f"{case} raised {exc!r}, expected ContractError")
    pytest.fail(f"{case} was accepted, expected ContractError")


# Feature: shared-contracts, Property 4: Rigid transform acceptance
def test_property_4_rigid_transform_acceptance():
    """``check_rigid`` accepts rigid transforms and rejects every structural violation.

    For any random rotation R (orthonormal, det +1) and finite translation t up to large
    magnitudes, ``check_rigid([[R, t], [0, 1]])`` returns a read-only float64 (4, 4) copy
    equal to the input (a copy, not the same object). Any perturbation applied alone to a
    fresh valid T raises a ``ContractError`` naming the field: a reflection (det -1), a
    uniform or tiny scale beyond ROTATION_TOL, a shear past ROTATION_TOL, a last row off by
    more than LAST_ROW_TOL, a non-finite entry or a wrong shape. A valid Python list is
    accepted (``check_rigid`` converts array-likes), so only genuinely invalid inputs are
    used for rejection.

    **Validates: Requirements 3.1, 3.2, 4.4**
    """
    for k in range(RIGID_ITERATIONS):
        rng = factories.seeded_rng(RIGID_SEED, k)
        rotation = factories.random_rotation(rng)
        translation = rng.uniform(-RIGID_TRANSLATION_RANGE, RIGID_TRANSLATION_RANGE, 3)
        T_in = factories.rigid(rotation, translation)
        prefix = f"seed {RIGID_SEED} iteration {k}"

        # Accept: the random rigid transform, a near-identity and near-pi rotation, the
        # closed-form inverse, and an otherwise valid Python list (array-like conversion).
        _check_rigid_accepted(k, f"{prefix}: random rigid", T_in)
        near_identity = factories.rigid(
            factories.random_rotation(factories.seeded_rng(RIGID_SEED, k, 1)), translation
        )
        _check_rigid_accepted(k, f"{prefix}: near-identity/near-pi rotation A", near_identity)
        near_pi = factories.rigid(
            factories.random_rotation(factories.seeded_rng(RIGID_SEED, k, 2)), translation
        )
        _check_rigid_accepted(k, f"{prefix}: near-identity/near-pi rotation B", near_pi)
        _check_rigid_accepted(k, f"{prefix}: rigid_inverse", factories.rigid_inverse(T_in))
        _check_rigid_accepted(k, f"{prefix}: Python list", T_in.tolist())

        # Reject: a reflection (negate one column so det = -1).
        column = int(rng.integers(3))
        reflected = rotation.copy()
        reflected[:, column] = -reflected[:, column]
        _check_rigid_rejected(k, f"{prefix}: reflection column {column}", factories.rigid(reflected, translation))

        # Reject: a uniform scale by 1.01 and a tiny scale by 1 + 1e-5, both past ROTATION_TOL.
        for scale in (1.01, 1.0 + 1e-5):
            _check_rigid_rejected(
                k, f"{prefix}: uniform scale {scale}", factories.rigid(rotation * scale, translation)
            )

        # Reject: a shear that pushes max|R^T R - I| just past ROTATION_TOL. Adding
        # s * (col j) to col i makes the (i, j) Gram entry s (columns are orthonormal), so
        # s a touch above ROTATION_TOL violates the orthonormality check.
        i, j = rng.permutation(3)[:2]
        sheared = rotation.copy()
        sheared[:, i] = sheared[:, i] + 1.0001e-6 * sheared[:, j]
        _check_rigid_rejected(k, f"{prefix}: shear columns {i},{j}", factories.rigid(sheared, translation))

        # Reject: last rows outside LAST_ROW_TOL (1e-9): a 1e-6 nudge, a 1e-3 third entry,
        # and a 1e-7 first entry.
        for last_row in ([1e-6, 0.0, 0.0, 1.0], [0.0, 0.0, 1e-3, 1.0], [1e-7, 0.0, 0.0, 1.0]):
            T_bad = T_in.copy()
            T_bad[3] = last_row
            _check_rigid_rejected(k, f"{prefix}: last row {last_row}", T_bad)

        # Reject: a NaN or inf inserted at a drawn cell.
        row, col = int(rng.integers(4)), int(rng.integers(4))
        for bad_value in (np.nan, np.inf):
            T_bad = T_in.copy()
            T_bad[row, col] = bad_value
            _check_rigid_rejected(k, f"{prefix}: {bad_value} at [{row},{col}]", T_bad)

        # Reject: wrong shapes.
        for shape in ((3, 4), (4, 3), (3, 3), (4, 4, 1)):
            _check_rigid_rejected(k, f"{prefix}: shape {shape}", np.zeros(shape))

# ---------------------------------------------------------------------------------------
# Property 5: covariance acceptance (Requirements 3.3, 3.4, 3.5)
# ---------------------------------------------------------------------------------------

COV_SEED = 20261017
COV_ITERATIONS = 150


def _check_cov_accepted(case: str, P_in: np.ndarray, n: int | None) -> None:
    """Check ``check_covariance`` returns a read-only float64 (n, n) copy equal to ``P_in``."""
    returned = c.check_covariance(P_in, n, name="P")
    assert returned.dtype == np.float64, f"{case} returned dtype {returned.dtype}, not float64"
    assert returned.shape == P_in.shape, f"{case} returned shape {returned.shape}, not {P_in.shape}"
    assert np.array_equal(returned, P_in), f"{case} returned a matrix unequal to its input"
    assert returned.flags.writeable is False, f"{case} returned a writeable matrix"
    assert returned is not P_in, f"{case} returned the input object, not a copy"


def _check_cov_rejected(case: str, P_bad: object, n: int | None) -> None:
    """Check ``check_covariance`` raises a ContractError naming the field for ``P_bad``."""
    try:
        c.check_covariance(P_bad, n, name="P")
    except c.ContractError as exc:
        assert str(exc).startswith("P"), f"{case}: message {exc!r} does not name the field"
        return
    except Exception as exc:
        pytest.fail(f"{case} raised {exc!r}, expected ContractError")
    pytest.fail(f"{case} was accepted, expected ContractError")


# Feature: shared-contracts, Property 5: Covariance acceptance
def test_property_5_covariance_acceptance():
    """``check_covariance`` accepts symmetric PSD matrices and rejects every violation.

    For a random real A of various sizes, ``P = A @ A.T`` is symmetric PSD, so
    ``check_covariance(P)`` returns a read-only float64 (n, n) copy equal to the input (a
    distinct object). ``definite=True`` succeeds exactly when the minimum eigenvalue is > 0:
    a full-rank square A gives a positive-definite P, while a rank-deficient A (a zeroed
    column) gives a PSD but singular P that ``definite=True`` rejects. Both the ``n`` and
    ``n=None`` forms are exercised. Perturbing a fresh valid P alone raises a
    ``ContractError`` naming the field: asymmetry past SYMMETRY_TOL * scale, a negative
    eigenvalue past PSD_TOL * scale, a NaN or inf entry, a non-square shape, a wrong ``n``,
    and (with ``definite=True``) a singular PSD matrix.

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    for k in range(COV_ITERATIONS):
        rng = factories.seeded_rng(COV_SEED, k)
        prefix = f"seed {COV_SEED} iteration {k}"
        n = int(rng.integers(1, 7))  # various n in 1..6
        # Full-rank square A -> P is symmetric positive definite.
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        P = A @ A.T
        scale = max(1.0, float(np.max(np.abs(P))))
        smallest = float(np.linalg.eigvalsh(0.5 * P + 0.5 * P.T)[0])

        # Accept: PSD acceptance in both the sized and unsized forms.
        _check_cov_accepted(f"{prefix}: A@A.T with n", P, n)
        _check_cov_accepted(f"{prefix}: A@A.T no n", P, None)

        # definite=True succeeds exactly when the minimum eigenvalue is > 0.
        returned = c.check_covariance(P, n, name="P", definite=True)
        assert smallest > 0.0 and returned.flags.writeable is False, (
            f"{prefix}: full-rank A@A.T has min eigenvalue {smallest:.3g}, definite should pass"
        )
        assert returned is not P and np.array_equal(returned, P), f"{prefix}: definite copy differs"

        # Rank-deficient A (a zeroed column) -> PSD but singular. The matrix is rebuilt from
        # its own eigendecomposition with the smallest eigenvalue set just inside the PSD
        # tolerance band (-0.5 * PSD_TOL * scale): strictly below 0, so definite=True rejects
        # it robustly despite ~1e-16 reconstruction roundoff, yet >= -PSD_TOL * scale, so PSD
        # acceptance still succeeds.
        A_singular = A.copy()
        A_singular[:, int(rng.integers(n))] = 0.0
        raw_singular = A_singular @ A_singular.T
        singular_scale = max(1.0, float(np.max(np.abs(raw_singular))))
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * raw_singular + 0.5 * raw_singular.T)
        eigenvalues[0] = -0.5 * c.PSD_TOL * singular_scale  # eigh returns them ascending
        P_singular = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        P_singular = 0.5 * P_singular + 0.5 * P_singular.T  # symmetric to machine precision
        min_singular = float(np.linalg.eigvalsh(P_singular)[0])
        assert min_singular < 0.0, (
            f"{prefix}: rank-deficient P should be non-definite, min eigenvalue {min_singular:.3g}"
        )
        _check_cov_accepted(f"{prefix}: rank-deficient PSD", P_singular, n)
        try:
            c.check_covariance(P_singular, n, name="P", definite=True)
        except c.ContractError as exc:
            assert str(exc).startswith("P"), f"{prefix}: definite rejection {exc!r} does not name P"
        else:
            pytest.fail(f"{prefix}: singular PSD accepted with definite=True")

        # Reject: break symmetry at one off-diagonal well past SYMMETRY_TOL * scale.
        if n >= 2:
            i, j = rng.permutation(n)[:2]
            P_asym = P.copy()
            P_asym[i, j] += 1.0 + 10.0 * c.SYMMETRY_TOL * scale
            _check_cov_rejected(f"{prefix}: asymmetric [{i},{j}]", P_asym, n)

        # Reject: make an eigenvalue negative past tolerance (shift P by more than its min
        # eigenvalue), and a strongly negative diagonal entry.
        lam = smallest + 1.0 + c.PSD_TOL * scale
        _check_cov_rejected(f"{prefix}: negative eigenvalue shift", P - lam * np.eye(n), None)
        P_neg_diag = P.copy()
        d = int(rng.integers(n))
        P_neg_diag[d, d] = -scale - 1.0
        _check_cov_rejected(f"{prefix}: negative diagonal [{d},{d}]", P_neg_diag, n)

        # Reject: a NaN or inf entry at a drawn cell (symmetrically, to isolate the finite check).
        for bad_value in (np.nan, np.inf):
            P_bad = P.copy()
            r1, r2 = int(rng.integers(n)), int(rng.integers(n))
            P_bad[r1, r2] = P_bad[r2, r1] = bad_value
            _check_cov_rejected(f"{prefix}: {bad_value} at [{r1},{r2}]", P_bad, None)

        # Reject: a non-square shape, and a correct square matrix given the wrong ``n``.
        _check_cov_rejected(f"{prefix}: non-square", rng.standard_normal((n, n + 1)), None)
        _check_cov_rejected(f"{prefix}: wrong n", P, n + 1)

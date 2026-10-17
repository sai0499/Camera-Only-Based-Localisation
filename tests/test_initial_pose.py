"""Example tests of InitialPose JSON, EpisodeBounds and the hash-bound build split.

The committed split file is only read; every write goes under pytest's ``tmp_path``.
"""

import copy
import hashlib
import json
import math

import pytest

from camloc import contracts as c

EXPECTED_EPISODES = {
    "fit": (1791918157114146050, 1791918857504944006),
    "tune": (1791918859504944006, 1791919130302537034),
    "holdout": (1791919132302537034, 1791919351506836680),
    "full": (1791918157114146050, 1791919351506836680),
}

VALID_POSE = '{"timestamp_ns": "1791918859504944006", "x": 12.25, "y": -3.5, "yaw": 0.125}'


def _split_doc() -> dict:
    return json.loads(c.DEFAULT_SPLIT_PATH.read_bytes().decode("utf-8"))


# ---------------------------------------------------------------------------------------
# Build split
# ---------------------------------------------------------------------------------------


def test_load_split_returns_exactly_the_design_bounds_and_leaves_the_file_unchanged():
    before = hashlib.sha256(c.DEFAULT_SPLIT_PATH.read_bytes()).hexdigest()
    episodes = c.load_split()
    assert {name: (b.start_ns, b.end_ns) for name, b in episodes.items()} == EXPECTED_EPISODES
    assert all(type(b) is c.EpisodeBounds and b.name == name for name, b in episodes.items())
    assert all(type(b.start_ns) is int and type(b.end_ns) is int for b in episodes.values())
    assert hashlib.sha256(c.DEFAULT_SPLIT_PATH.read_bytes()).hexdigest() == before
    assert before == c.BUILD_SPLIT_SHA256


def test_load_split_mapping_is_read_only():
    episodes = c.load_split()
    with pytest.raises(TypeError):
        episodes["fit"] = episodes["tune"]
    with pytest.raises(TypeError):
        del episodes["full"]


def test_load_split_rejects_a_modified_copy(tmp_path):
    raw = c.DEFAULT_SPLIT_PATH.read_bytes()
    copy_path = tmp_path / "build_split.json"
    copy_path.write_bytes(raw.replace(b'"schema_version": 2', b'"schema_version": 3'))
    with pytest.raises(c.ContractError, match="SHA-256"):
        c.load_split(copy_path)
    unchanged = tmp_path / "unchanged.json"
    unchanged.write_bytes(raw)
    assert c.load_split(unchanged)["tune"].start_ns == EXPECTED_EPISODES["tune"][0]


def test_episodes_from_split_accepts_the_committed_document():
    episodes = c.episodes_from_split(_split_doc())
    assert {name: (b.start_ns, b.end_ns) for name, b in episodes.items()} == EXPECTED_EPISODES


def _mutated(path: tuple[str, ...], value: object) -> dict:
    doc = copy.deepcopy(_split_doc())
    target = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return doc


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("guard_duration_ns",), "1000000000", "guard"),
        (("tune", "start_ns"), "1791918859504944007", "guard"),  # 2 s + 1 ns after fit
        (("time_encoding",), "decimal_string_nanoseconds", "time_encoding"),
        (("interval_convention",), "start_inclusive_end_inclusive", "interval_convention"),
        (("holdout", "end_ns"), "1791919351506836679", "ordered"),  # must be last + 1
        (("fit", "start_ns"), "1791918157114146051", "ordered"),  # must equal first GT
        (("tune", "start_ns"), 1791918859504944006, "decimal string"),
        (("tune", "end_ns"), "+1791919130302537034", "decimal"),
    ],
)
def test_episodes_from_split_rejects_bad_structure(path, value, match):
    with pytest.raises(c.ContractError, match=match):
        c.episodes_from_split(_mutated(path, value))


def test_episodes_from_split_rejects_wrong_order():
    doc = _split_doc()
    doc["fit"], doc["tune"] = doc["tune"], doc["fit"]
    with pytest.raises(c.ContractError):
        c.episodes_from_split(doc)


def test_episodes_from_split_rejects_a_missing_episode():
    doc = _split_doc()
    del doc["holdout"]
    with pytest.raises(c.ContractError, match="holdout"):
        c.episodes_from_split(doc)


# ---------------------------------------------------------------------------------------
# Initial pose JSON
# ---------------------------------------------------------------------------------------


def test_format_parse_round_trip_with_19_digit_stamp_and_exact_floats(tmp_path):
    pose = c.InitialPose(
        timestamp_ns=9_223_372_036_854_775_807, x=0.1 + 0.2, y=-1.7976931348623157e308, yaw=-0.0
    )
    text = c.format_initial_pose(pose)
    assert text.endswith("}\n") and text.count("\n") == 1
    doc = json.loads(text)
    assert set(doc) == {"timestamp_ns", "x", "y", "yaw"}
    assert doc["timestamp_ns"] == "9223372036854775807"
    parsed = c.parse_initial_pose(text)
    assert parsed == pose and type(parsed.timestamp_ns) is int
    assert parsed.x.hex() == pose.x.hex() and parsed.y.hex() == pose.y.hex()
    assert math.copysign(1.0, parsed.yaw) == -1.0

    path = tmp_path / "initial_pose.json"
    c.write_initial_pose(path, pose)
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and raw.endswith(b"}\n") and b"\r" not in raw
    assert c.read_initial_pose(path) == pose


def test_canonical_json_line_matches_the_design():
    pose = c.parse_initial_pose(VALID_POSE)
    assert c.format_initial_pose(pose) == VALID_POSE + "\n"
    assert pose == c.InitialPose(timestamp_ns=1791918859504944006, x=12.25, y=-3.5, yaw=0.125)


def test_json_integer_timestamp_is_accepted():
    pose = c.parse_initial_pose('{"timestamp_ns": 1791918859504944006, "x": 1, "y": 2, "yaw": 3}')
    assert pose.timestamp_ns == 1791918859504944006 and type(pose.timestamp_ns) is int
    assert type(pose.x) is float and pose.yaw == 3.0


def test_yaw_is_stored_wrapped():
    pose = c.parse_initial_pose('{"timestamp_ns": "5", "x": 0, "y": 0, "yaw": -3.141592653589793}')
    assert pose.yaw == math.pi


def test_write_initial_pose_refuses_an_existing_path(tmp_path):
    path = tmp_path / "initial_pose.json"
    path.write_bytes(b"keep me\n")
    pose = c.parse_initial_pose(VALID_POSE)
    with pytest.raises(FileExistsError):
        c.write_initial_pose(path, pose)
    assert path.read_bytes() == b"keep me\n"


def test_read_initial_pose_rejects_a_bom(tmp_path):
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + VALID_POSE.encode("utf-8"))
    with pytest.raises(c.ContractError):
        c.read_initial_pose(path)


@pytest.mark.parametrize(
    "text",
    [
        '{"timestamp_ns": 1.79e18, "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": 123.0, "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": true, "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": null, "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": -1, "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "+1", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "-1", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": " 1", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "1\\n", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "1_0", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "1e9", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "9223372036854775808", "x": 1.0, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "5", "x": true, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "5", "x": 1.0, "y": "2.0", "yaw": 0.5}',
        '{"timestamp_ns": "5", "x": 1.0, "y": 2.0, "yaw": false}',
        '{"timestamp_ns": "5", "x": 1.0, "y": 2.0, "yaw": null}',
        '{"timestamp_ns": "5", "x": NaN, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "5", "x": 1.0, "y": Infinity, "yaw": 0.5}',
        '{"timestamp_ns": "5", "x": 1.0, "y": 2.0, "yaw": -Infinity}',
        '{"timestamp_ns": "5", "x": 1e400, "y": 2.0, "yaw": 0.5}',
        '{"timestamp_ns": "5", "x": 1.0, "y": 2.0}',
        '{"timestamp_ns": "5", "x": 1.0, "y": 2.0, "yaw": 0.5, "z": 0.0}',
        '{"timestamp_ns": "5", "x": 1.0, "x": 1.5, "y": 2.0, "yaw": 0.5}',
        '[{"timestamp_ns": "5", "x": 1.0, "y": 2.0, "yaw": 0.5}]',
        '"5"',
        "",
        "{",
    ],
)
def test_parse_initial_pose_rejects_malformed_documents(text):
    with pytest.raises(c.ContractError):
        c.parse_initial_pose(text)


# ---------------------------------------------------------------------------------------
# Episode bounds
# ---------------------------------------------------------------------------------------


def test_episode_bounds_contains_is_start_inclusive_end_exclusive():
    bounds = c.EpisodeBounds(name="tune", start_ns=100, end_ns=200)
    assert bounds.contains(100) and bounds.contains(199)
    assert not bounds.contains(99) and not bounds.contains(200)
    bounds.check_initial_pose(c.InitialPose(timestamp_ns=199, x=0.0, y=0.0, yaw=0.0))
    for stamp in (99, 200):
        with pytest.raises(c.ContractError, match="outside"):
            bounds.check_initial_pose(c.InitialPose(timestamp_ns=stamp, x=0.0, y=0.0, yaw=0.0))


def test_episode_bounds_without_end_contains_every_later_stamp():
    bounds = c.EpisodeBounds(name="custom", start_ns=100, end_ns=None)
    assert bounds.contains(100) and bounds.contains(c.INT64_MAX)
    assert not bounds.contains(99)


@pytest.mark.parametrize("end_ns", [100, 99, 0])
def test_episode_bounds_rejects_end_not_after_start(end_ns):
    with pytest.raises(c.ContractError, match="end_ns"):
        c.EpisodeBounds(name="bad", start_ns=100, end_ns=end_ns)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "start_ns": 0, "end_ns": None},
        {"name": "x", "start_ns": 1.0, "end_ns": None},
        {"name": "x", "start_ns": 0, "end_ns": True},
    ],
)
def test_episode_bounds_rejects_bad_fields(kwargs):
    with pytest.raises(c.ContractError):
        c.EpisodeBounds(**kwargs)


# =======================================================================================
# Property tests
#
# Each property is one seeded loop over numpy.random.default_rng([seed, k]) with at least
# 100 iterations. No hypothesis. Failures name the seed, iteration and the offending input.
# The example tests above (task 7.2) are left unchanged; these sections only append.
# =======================================================================================

import numpy as np  # noqa: E402  (imported here so the example block above stays untouched)


def _rng(seed: int, k: int) -> np.random.Generator:
    """Seeded generator for iteration ``k`` of the property with base ``seed``."""
    return np.random.default_rng([seed, k])


# ---------------------------------------------------------------------------------------
# Property 19: initial-pose JSON round trip (Requirements 15.1, 15.2, 15.3, 15.5)
# ---------------------------------------------------------------------------------------

ROUND_TRIP_SEED = 20261019
ROUND_TRIP_ITERATIONS = 200
# Boundary float64 magnitudes exercised every iteration alongside random draws. These are
# the values whose repr the formatter must round-trip bit for bit.
BOUNDARY_FLOATS = (
    0.0,
    -0.0,
    5e-324,  # the smallest positive subnormal
    -5e-324,
    2.2250738585072014e-308,  # the smallest positive normal
    1.7976931348623157e308,  # the largest finite float
    -1.7976931348623157e308,
    1.0,
    -1.0,
    0.1,
    0.1 + 0.2,  # 0.30000000000000004: a classic non-decimal repr
    math.pi,
    -math.pi,
)
# Timestamp edges hit across iterations; the body also draws uniform int64 stamps.
STAMP_EDGES = (0, 1, 7, 10, 999_999_999, c.INT64_MAX - 1, c.INT64_MAX)


def _uniform_stamp(rng: np.random.Generator) -> int:
    """Draw a Python ``int`` stamp uniformly from [0, 2**63 - 1] using full 63-bit width."""
    return int(rng.integers(0, c.INT64_MAX, endpoint=True))


def _round_trip_floats(rng: np.random.Generator, k: int) -> tuple[float, float, float]:
    """Return (x, y, yaw) for iteration ``k``: boundary values rotated in, plus random draws."""
    # Rotate three distinct boundary floats into the triple so every boundary is covered.
    base = len(BOUNDARY_FLOATS)
    x = BOUNDARY_FLOATS[k % base]
    y = BOUNDARY_FLOATS[(k + 1) % base]
    yaw = BOUNDARY_FLOATS[(k + 2) % base]
    # On most iterations replace one or two coordinates with a random finite float so the
    # loop also covers ordinary values (random uniforms and exponent-scaled magnitudes).
    which = int(rng.integers(0, 4))
    scale = 10.0 ** float(rng.integers(-300, 300))
    draw = float(rng.uniform(-1.0, 1.0)) * scale
    if which == 1:
        x = draw
    elif which == 2:
        y = draw
    elif which == 3:
        yaw = float(rng.uniform(-10.0, 10.0))  # a yaw outside (-pi, pi] to be wrapped
    return x, y, yaw


def _check_round_trip(k: int, pose: c.InitialPose) -> None:
    """Formatting then parsing ``pose`` returns an equal pose with bitwise-identical floats."""
    case = (
        f"seed {ROUND_TRIP_SEED} iteration {k}: InitialPose("
        f"timestamp_ns={pose.timestamp_ns}, x={pose.x!r}, y={pose.y!r}, yaw={pose.yaw!r})"
    )
    text = c.format_initial_pose(pose)
    # 15.2: exactly four keys, timestamp_ns a decimal string, trailing newline.
    assert text.endswith("}\n"), f"{case}: formatted text {text!r} lacks a trailing newline"
    assert text.count("\n") == 1, f"{case}: formatted text {text!r} has extra newlines"
    doc = json.loads(text)
    assert set(doc) == {"timestamp_ns", "x", "y", "yaw"}, f"{case}: keys {sorted(doc)}"
    assert doc["timestamp_ns"] == str(pose.timestamp_ns), f"{case}: stamp {doc['timestamp_ns']!r}"
    assert type(doc["timestamp_ns"]) is str, f"{case}: stamp is not a JSON string"
    # 15.3, 15.5: parsing the formatted text restores an equal pose, floats bit for bit.
    parsed = c.parse_initial_pose(text)
    assert parsed == pose, f"{case}: parsed {parsed!r} != original"
    assert type(parsed.timestamp_ns) is int, f"{case}: parsed stamp type {type(parsed.timestamp_ns)}"
    assert parsed.timestamp_ns == pose.timestamp_ns, f"{case}: parsed stamp {parsed.timestamp_ns}"
    for field in ("x", "y", "yaw"):
        got = getattr(parsed, field)
        want = getattr(pose, field)
        assert got.hex() == want.hex(), f"{case}: {field} {got.hex()} != {want.hex()} (not bitwise)"
    # 15.3: a JSON-integer timestamp encoding parses to the same pose.
    int_doc = '{"timestamp_ns": %d, "x": %s, "y": %s, "yaw": %s}' % (
        pose.timestamp_ns,
        json.dumps(pose.x),
        json.dumps(pose.y),
        json.dumps(pose.yaw),
    )
    from_int = c.parse_initial_pose(int_doc)
    assert from_int == pose, f"{case}: JSON-integer stamp parsed {from_int!r} != original"
    assert from_int.timestamp_ns == pose.timestamp_ns and type(from_int.timestamp_ns) is int


# Feature: shared-contracts, Property 19: Initial-pose JSON round trip
def test_property_19_initial_pose_json_round_trip():
    """``parse_initial_pose(format_initial_pose(p)) == p`` with bitwise-identical floats.

    For any int64 timestamp and finite x, y, yaw (boundary floats -0.0, the smallest
    subnormal, the finite extremes and random uniforms), formatting then parsing an
    ``InitialPose`` restores it exactly: the floats compare equal by ``.hex()``, the
    formatted document has exactly the four keys with a decimal-string timestamp, and the
    JSON-integer timestamp encoding parses to the same pose. Yaw is stored wrapped, so the
    pose is built through ``InitialPose(...)`` and that canonical value is round-tripped
    (an arbitrary input yaw is not expected to survive unwrapped).

    **Validates: Requirements 15.1, 15.2, 15.3, 15.5**
    """
    for k in range(ROUND_TRIP_ITERATIONS):
        rng = _rng(ROUND_TRIP_SEED, k)
        x, y, yaw = _round_trip_floats(rng, k)
        # Timestamps: an edge value plus a uniform int64 draw, both exercised each iteration.
        stamps = (STAMP_EDGES[k % len(STAMP_EDGES)], _uniform_stamp(rng))
        for stamp in stamps:
            # Build through InitialPose so yaw is already wrapped; round-trip the canonical pose.
            pose = c.InitialPose(timestamp_ns=stamp, x=x, y=y, yaw=yaw)
            _check_round_trip(k, pose)


# ---------------------------------------------------------------------------------------
# Property 20: initial-pose JSON rejection (Requirement 15.4)
# ---------------------------------------------------------------------------------------

REJECTION_SEED = 20261020
REJECTION_ITERATIONS = 200


def _valid_doc(rng: np.random.Generator) -> dict:
    """A fresh valid initial-pose document with randomly drawn values each iteration."""
    return {
        "timestamp_ns": int(rng.integers(0, c.INT64_MAX, endpoint=True)),
        "x": float(rng.uniform(-1e6, 1e6)),
        "y": float(rng.uniform(-1e6, 1e6)),
        "yaw": float(rng.uniform(-math.pi, math.pi)),
    }


def _coordinate_key(rng: np.random.Generator) -> str:
    return ("x", "y", "yaw")[int(rng.integers(0, 3))]


def _rejection_cases(rng: np.random.Generator) -> list[tuple[str, str]]:
    """Return (label, raw JSON text) pairs that ``parse_initial_pose`` must reject.

    Each case mutates a freshly drawn valid document once. JSON-encodable mutations go
    through ``json.dumps``; NaN/Infinity/1e400, duplicate keys and malformed top levels are
    emitted as raw text because ``json.dumps`` cannot produce them with ``allow_nan=False``.
    """
    cases: list[tuple[str, str]] = []

    def dumped(doc: dict, label: str) -> None:
        cases.append((label, json.dumps(doc)))

    # --- timestamp mutations -----------------------------------------------------------
    base = _valid_doc(rng)
    stamp = base["timestamp_ns"]
    float_doc = dict(base)
    float_doc["timestamp_ns"] = float(stamp) + 0.0  # a JSON float timestamp
    dumped(float_doc, "float timestamp")
    frac_doc = dict(base)
    frac_doc["timestamp_ns"] = float(stamp) + 0.5 if stamp < c.INT64_MAX else 123.0
    dumped(frac_doc, "fractional float timestamp")
    bool_doc = dict(base)
    bool_doc["timestamp_ns"] = bool(int(rng.integers(0, 2)))
    dumped(bool_doc, "bool timestamp")
    for label, raw_stamp in (
        ("signed +", "+" + str(stamp)),
        ("signed -", "-" + str(max(stamp, 1))),
        ("leading whitespace", " " + str(stamp)),
        ("trailing newline", str(stamp) + "\\n"),
        ("exponent string", "1e9"),
        ("overflow string", str(c.INT64_MAX + 1 + int(rng.integers(0, 1000)))),
    ):
        rest = _valid_doc(rng)
        cases.append(
            (
                f"string timestamp ({label})",
                '{"timestamp_ns": "%s", "x": %s, "y": %s, "yaw": %s}'
                % (raw_stamp, json.dumps(rest["x"]), json.dumps(rest["y"]), json.dumps(rest["yaw"])),
            )
        )

    # --- coordinate mutations ----------------------------------------------------------
    bool_coord = _valid_doc(rng)
    bkey = _coordinate_key(rng)
    bool_coord[bkey] = bool(int(rng.integers(0, 2)))
    dumped(bool_coord, f"bool coordinate ({bkey})")
    str_coord = _valid_doc(rng)
    skey = _coordinate_key(rng)
    str_coord[skey] = str(str_coord[skey])
    dumped(str_coord, f"string coordinate ({skey})")

    # --- non-finite numbers (raw text; json.dumps(allow_nan=False) cannot emit them) ---
    for label, token in (("NaN", "NaN"), ("Infinity", "Infinity"), ("overflow 1e400", "1e400")):
        doc = _valid_doc(rng)
        key = _coordinate_key(rng)
        parts = []
        for name in ("timestamp_ns", "x", "y", "yaw"):
            value = '"%d"' % doc["timestamp_ns"] if name == "timestamp_ns" else json.dumps(doc[name])
            if name == key and name != "timestamp_ns":
                value = ("-" if (label != "overflow 1e400" and int(rng.integers(0, 2))) else "") + token
            parts.append('"%s": %s' % (name, value))
        cases.append((f"{label} coordinate ({key})", "{" + ", ".join(parts) + "}"))

    # --- key-set mutations -------------------------------------------------------------
    missing = _valid_doc(rng)
    del missing[_coordinate_key(rng) if int(rng.integers(0, 2)) else "timestamp_ns"]
    dumped(missing, "missing key")
    extra = _valid_doc(rng)
    extra["z"] = float(rng.uniform(-1.0, 1.0))
    dumped(extra, "extra key")
    dup = _valid_doc(rng)
    dkey = _coordinate_key(rng)
    cases.append(
        (
            f"duplicate key ({dkey})",
            '{"timestamp_ns": "%d", "x": %s, "y": %s, "yaw": %s, "%s": %s}'
            % (
                dup["timestamp_ns"],
                json.dumps(dup["x"]),
                json.dumps(dup["y"]),
                json.dumps(dup["yaw"]),
                dkey,
                json.dumps(float(rng.uniform(-1.0, 1.0))),
            ),
        )
    )

    # --- non-object top levels ---------------------------------------------------------
    valid_obj = json.dumps(_valid_doc(rng) | {"timestamp_ns": str(_uniform_stamp(rng))})
    cases.append(("list top level", "[%s]" % valid_obj))
    cases.append(("number top level", str(_uniform_stamp(rng))))
    cases.append(("string top level", json.dumps(str(_uniform_stamp(rng)))))
    return cases


# Feature: shared-contracts, Property 20: Initial-pose JSON rejection
def test_property_20_initial_pose_json_rejection():
    """Every malformed single-mutation document is rejected with ``ContractError``.

    Starting from a freshly drawn valid document each iteration, exactly one malformed
    mutation is applied per case and ``parse_initial_pose`` must raise ``ContractError``:
    a float, bool or signed/whitespace/exponent-string timestamp; a bool or string
    coordinate; a NaN, Infinity or 1e400 number; a missing, extra or duplicate key; and a
    non-object top level (list, number or string).

    **Validates: Requirements 15.4**
    """
    for k in range(REJECTION_ITERATIONS):
        rng = _rng(REJECTION_SEED, k)
        for label, text in _rejection_cases(rng):
            case = f"seed {REJECTION_SEED} iteration {k}: {label}: {text!r}"
            try:
                parsed = c.parse_initial_pose(text)
            except c.ContractError:
                continue
            except Exception as exc:  # pragma: no cover - a non-ContractError is a failure
                pytest.fail(f"{case} raised {exc!r}, expected ContractError")
            pytest.fail(f"{case} was accepted as {parsed!r}, expected ContractError")


# ---------------------------------------------------------------------------------------
# Property 21: episode bounds semantics (Requirements 16.1, 16.2, 16.3)
# ---------------------------------------------------------------------------------------

BOUNDS_SEED = 20261021
BOUNDS_ITERATIONS = 200


def _bounds_pair(rng: np.random.Generator, k: int) -> tuple[int, int | None]:
    """Return (start, end-or-None) int64-range values for iteration ``k``.

    ``end`` is None on some iterations; otherwise it may be below, equal to or above
    ``start`` so both the accepted and the rejected construction paths are exercised.
    """
    start = int(rng.integers(0, c.INT64_MAX, endpoint=True))
    roll = int(rng.integers(0, 5))
    if roll == 0:
        return start, None
    if roll == 1:  # end below start: rejected
        return start, int(rng.integers(0, start)) if start > 0 else start
    if roll == 2:  # end == start: rejected
        return start, start
    # end strictly above start: accepted
    high = int(rng.integers(start + 1, c.INT64_MAX, endpoint=True)) if start < c.INT64_MAX else start
    return start, high


def _probe_stamps(start: int, end: int | None, rng: np.random.Generator) -> list[int]:
    """Timestamps around the boundaries: start-1, start, end-1, end plus random draws."""
    stamps = {start}
    for candidate in (start - 1, start + 1):
        if 0 <= candidate <= c.INT64_MAX:
            stamps.add(candidate)
    if end is not None:
        for candidate in (end - 1, end):
            if 0 <= candidate <= c.INT64_MAX:
                stamps.add(candidate)
    stamps.add(int(rng.integers(0, c.INT64_MAX, endpoint=True)))
    stamps.add(0)
    stamps.add(c.INT64_MAX)
    return sorted(stamps)


# Feature: shared-contracts, Property 21: Episode bounds semantics
def test_property_21_episode_bounds_semantics():
    """``EpisodeBounds`` construction, ``contains`` and ``check_initial_pose`` agree exactly.

    For random start, end (or None) and probe timestamps including the boundaries start-1,
    start, end-1 and end, ``EpisodeBounds`` constructs exactly when ``end is None`` or
    ``end > start``; ``contains(t)`` equals ``start <= t and (end is None or t < end)``; and
    ``check_initial_pose`` raises exactly when ``contains(pose.timestamp_ns)`` is False.

    **Validates: Requirements 16.1, 16.2, 16.3**
    """
    for k in range(BOUNDS_ITERATIONS):
        rng = _rng(BOUNDS_SEED, k)
        start, end = _bounds_pair(rng, k)
        name = f"ep{k}"
        case = f"seed {BOUNDS_SEED} iteration {k}: start_ns={start}, end_ns={end}"
        should_construct = end is None or end > start
        try:
            bounds = c.EpisodeBounds(name=name, start_ns=start, end_ns=end)
        except c.ContractError as exc:
            assert not should_construct, f"{case} raised {exc!r} but should have constructed"
            continue
        assert should_construct, f"{case} constructed but end_ns is not after start_ns"
        for t in _probe_stamps(start, end, rng):
            expected = start <= t and (end is None or t < end)
            got = bounds.contains(t)
            assert got is expected, f"{case}: contains({t}) = {got}, expected {expected}"
            pose = c.InitialPose(timestamp_ns=t, x=0.0, y=0.0, yaw=0.0)
            try:
                bounds.check_initial_pose(pose)
            except c.ContractError as exc:
                assert not expected, f"{case}: check_initial_pose({t}) raised {exc!r} but t is inside"
            else:
                assert expected, f"{case}: check_initial_pose({t}) accepted but t is outside"

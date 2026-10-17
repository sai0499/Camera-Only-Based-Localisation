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

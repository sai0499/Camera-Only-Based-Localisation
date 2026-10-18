"""B's landmark map artifact: round trip, identity, compatibility and reference search."""

import json
import math

import numpy as np
import pytest

import vision_fixtures as vf
from camloc import contracts as c
from camloc import landmark_map as lm


@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


def test_round_trip_and_identity(scene, tmp_path):
    landmark_map = scene.build_map(tmp_path / "map")
    assert landmark_map.map_id.startswith("fitmap:")
    manifest = json.loads((tmp_path / "map" / lm.MANIFEST_NAME).read_text())
    assert manifest["map_id"] == landmark_map.map_id
    np.testing.assert_array_equal(landmark_map.points(scene.landmark_ids[[3, 0]]),
                                  scene.points_W[[3, 0]])
    with pytest.raises(c.ContractError):
        landmark_map.points(np.array([1]))  # ids are 5, 8, 11, ...
    keyframe = landmark_map.keyframe(3)
    assert keyframe.camera_id == "fr" and keyframe.exposure_ns == scene.t_ns(10)
    assert keyframe.keypoints.dtype == np.float32 and keyframe.descriptors.dtype == np.float32
    assert not landmark_map.has_keyframe(2)
    with pytest.raises(FileExistsError):
        scene.build_map(tmp_path / "map")


def test_tampered_arrays_are_rejected(scene, tmp_path):
    scene.build_map(tmp_path / "map")
    path = tmp_path / "map" / lm.ARRAYS_NAME
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(c.ContractError, match="identity"):
        lm.load_landmark_map(tmp_path / "map")


def test_compatibility_reasons(scene, tmp_path):
    landmark_map = scene.build_map(tmp_path / "map")
    ok = landmark_map.compatibility("fr", scene.K, scene.T_BC, landmark_map.preprocessing_id)
    assert ok == "valid"
    assert landmark_map.compatibility("r", scene.K, scene.T_BC, "x") == "camera_not_in_map"
    assert landmark_map.compatibility("fr", scene.K, scene.T_BC, "other") == \
        "preprocessing_mismatch"
    moved = scene.T_BC.copy()
    moved[0, 3] += 0.01
    assert landmark_map.compatibility("fr", scene.K, moved, landmark_map.preprocessing_id) == \
        "map_extrinsics_mismatch"


def test_reference_search_prefers_ahead_and_diverse(scene, tmp_path):
    landmark_map = scene.build_map(tmp_path / "map", keyframes=(0, 2, 4, 10, 12, 30))
    pose = scene.pose(1)
    ids = landmark_map.select_references("fr", pose[0, 3], pose[1, 3], scene.yaw)
    frames = {1: 0, 3: 2, 5: 4, 7: 10, 9: 12, 11: 30}
    # 6 m ahead of frame 1 is near frame 13; the two matched refs are >= 6 m apart.
    first, second = (frames[i] for i in ids[:2])
    assert abs(first - second) * 0.5 >= 6.0
    assert frames[ids[0]] in (10, 12)
    back = landmark_map.select_references("fr", pose[0, 3], pose[1, 3], scene.yaw + math.pi)
    assert back == ()
    assert landmark_map.select_references("rr", pose[0, 3], pose[1, 3], scene.yaw) == ()

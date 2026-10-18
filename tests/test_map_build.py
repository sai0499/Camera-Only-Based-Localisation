"""B's offline map builder: GT interpolation, keyframes, pairs and fixed-pose triangulation."""

import math

import numpy as np
import pytest

import build_map as bm
import vision_fixtures as vf
from camloc import geometry as g


def test_slerp_and_gt_interpolation_respect_support():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, math.sin(0.5), math.cos(0.5)])  # yaw 1 rad
    mid = bm.slerp(q0, q1, 0.5)
    np.testing.assert_allclose(mid, [0, 0, math.sin(0.25), math.cos(0.25)], atol=1e-12)
    track = bm.GtTrack(np.array([0, 10_000_000, 100_000_000], np.int64),
                       np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]), np.array([q0, q1, q1]))
    pose = track.pose(5_000_000)
    assert pose[0, 3] == pytest.approx(0.5)
    assert math.atan2(pose[1, 0], pose[0, 0]) == pytest.approx(0.5)
    assert track.pose(10_000_000)[0, 3] == pytest.approx(1.0)
    assert track.pose(50_000_000) is None  # 90 ms gap > 50 ms
    assert track.pose(-1) is None and track.pose(200_000_000) is None


def test_keyframe_selection_and_pairs():
    config = bm.BuildConfig()
    a = g.rigid_from_rotation_translation(np.eye(3), [0.0, 0.0, 0.0])
    near = g.rigid_from_rotation_translation(np.eye(3), [3.0, 0.0, 0.0])
    far = g.rigid_from_rotation_translation(np.eye(3), [4.0, 0.0, 0.0])
    turned = g.rigid_from_rotation_translation(vf.rot_z(math.radians(6)), [0.5, 0.0, 0.0])
    assert bm.select_keyframe(None, a, config)
    assert not bm.select_keyframe(a, near, config)
    assert bm.select_keyframe(a, far, config) and bm.select_keyframe(a, turned, config)
    scene = vf.Scene()
    T_WC = np.array([scene.pose(k) @ scene.T_BC for k in range(0, 60, 4)])  # 2 m apart
    cameras = ["fr"] * len(T_WC)
    cameras[-1] = "r"
    pairs = bm.select_pairs(cameras, T_WC, config)
    degree = np.bincount(np.array(pairs).ravel(), minlength=len(T_WC))
    assert degree[-1] == 0  # no cross-camera pairs
    assert all(i < j for i, j in pairs) and len(set(pairs)) == len(pairs)
    assert degree.max() <= 2 * config.neighbors


def test_fixed_pose_triangulation_recovers_landmarks(tmp_path):
    scene = vf.Scene(count_u=12, count_v=8)
    frames = list(range(0, 50, 8))  # 4 m baselines
    cameras = ["fr"] * len(frames)
    exposures = [scene.t_ns(k) for k in frames]
    T_WC = np.array([scene.pose(k) @ scene.T_BC for k in frames])
    keypoints, index = [], []
    for k in frames:
        pixels, visible = scene.project(k)
        rows = np.flatnonzero(visible)
        keypoints.append(pixels[rows].astype(np.float32))
        index.append(rows)
    pairs = bm.select_pairs(cameras, T_WC, bm.BuildConfig())
    matches = []
    for i, j in pairs:
        common, a, b = np.intersect1d(index[i], index[j], return_indices=True)
        matches.append(np.column_stack([a, b]).astype(np.int64))
    config = bm.BuildConfig()
    out, _ = bm.triangulate_known_poses(tmp_path / "work", cameras, exposures, T_WC,
                                        {"fr": scene.K}, keypoints, pairs, matches, config)
    assert bm.max_pose_deviation(out, T_WC) < 1e-9
    points, observations, stats, rejected, signed = bm.filter_landmarks(
        out, T_WC, [scene.K] * len(frames), keypoints, config)
    # Forward motion: points near the focus of expansion lack the 1 degree parallax.
    assert len(points) >= 40
    assert rejected["high_error"] == 0 and rejected["negative_depth"] == 0
    for xyz, (rows, idx) in zip(points, observations, strict=True):
        landmark = index[rows[0]][idx[0]]
        assert np.linalg.norm(xyz - scene.points_W[landmark]) < 0.02
    assert np.abs(signed.mean(axis=0)).max() < 0.05  # no half-pixel convention bias
    assert all(s[0] >= 3 and s[2] >= 1.0 and s[1] <= 2.0 for s in stats)

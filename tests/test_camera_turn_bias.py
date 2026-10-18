"""Opt-in turn bias in the support-v1 camera policy (causal yaw rate only)."""
from types import SimpleNamespace

import numpy as np

from camloc import camera_policy as cp

AXES = {"fr": (1, 0), "cl": (0.4, 0.9), "rl": (-0.5, 0.87), "cr": (0.5, -0.85), "rr": (-0.5, -0.9)}


class Bank:
    keyframe_ids = np.array([1, 2, 3, 4, 5])
    cams = dict(zip(range(1, 6), AXES))

    def T_BC(self, camera):
        T = np.eye(4)
        T[:2, 2] = AXES[camera]
        return T

    def keyframe(self, k):
        return SimpleNamespace(camera_id=self.cams[k], landmark_ids=np.arange(100),
                               keypoints=np.zeros((100, 2)))

    def mask(self, camera):
        return None

    def select_references(self, camera, *args, **kwargs):
        return [k for k, cam in self.cams.items() if cam == camera]


def _policy(monkeypatch, turn_bias):
    monkeypatch.setattr(cp, "occupied_cells", lambda points, mask: 8)
    return cp.CameraPolicy(Bank(), tuple(AXES), turn_bias=turn_bias)


def _snap(yaw_rate):
    return SimpleNamespace(state=np.array([0, 0, 0, 0, 0, yaw_rate]), P=np.eye(6))


def test_turn_bias_prefers_inside_side_camera(monkeypatch):
    policy = _policy(monkeypatch, True)
    available = set(AXES)
    right = policy.scores(_snap(-0.3), 10, available)  # right turn: inside is the right side
    assert right["cr"] == right["rr"] == 4 * right["cl"] == 4 * right["rl"]
    assert right["fr"] == _policy(monkeypatch, False).scores(_snap(-0.3), 10, available)["fr"]
    left = policy.scores(_snap(0.3), 10, available)
    assert left["cl"] == left["rl"] == 4 * left["cr"]
    straight = policy.scores(_snap(0.05), 10, available)
    assert straight == _policy(monkeypatch, False).scores(_snap(0.05), 10, available)


def test_turn_bias_off_by_default(monkeypatch):
    monkeypatch.setattr(cp, "occupied_cells", lambda points, mask: 8)
    policy = cp.CameraPolicy(Bank(), tuple(AXES))
    scores = policy.scores(_snap(-0.5), 10, set(AXES))
    assert scores["cr"] == scores["cl"]

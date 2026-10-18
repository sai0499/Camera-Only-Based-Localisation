"""Synthetic scenes for B's vision tests: known landmarks, poses, rendered blob images.

Landmarks are rendered as isolated Gaussian blobs at their exact projections, so LK and
PnP results can be compared with the generating poses. Nothing here is real data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import contract_factories as f
from camloc import contracts as c
from camloc import geometry as g
from camloc import landmark_map as lm

STEP_NS = 50_000_000  # 20 Hz
START_NS = f.EXPOSURE_NS - 2 * c.NS_PER_S
SPEED_MPS = 10.0
CALIBRATION_ID = "calibration_fr_v1"


def rot_z(yaw: float) -> np.ndarray:
    return np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0],
                     [0.0, 0.0, 1.0]])


class Scene:
    """A forward-moving body observing ~100 static landmarks with camera ``fr``."""

    def __init__(self, seed: int = 0, yaw: float = 0.3, count_u: int = 13, count_v: int = 8):
        rng = f.seeded_rng(seed)
        self.K = f.canonical_K()
        self.T_BC = f.camera_to_body()
        self.yaw = yaw
        self.origin = np.array([100.0, -20.0, 0.4])
        us = np.linspace(40, 920, count_u)
        vs = np.linspace(30, 430, count_v)
        pixels = np.array([[u, v] for v in vs for u in us]) + rng.uniform(-6, 6, (count_u * count_v, 2))
        depths = rng.uniform(20.0, 45.0, len(pixels))
        rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(self.K).T
        points_c = rays * depths[:, None]
        T_WC0 = self.pose(0) @ self.T_BC
        self.points_W = points_c @ T_WC0[:3, :3].T + T_WC0[:3, 3]
        self.landmark_ids = np.arange(len(self.points_W), dtype=np.int64) * 3 + 5

    def t_ns(self, k: int) -> int:
        return START_NS + k * STEP_NS

    def pose(self, k: int) -> np.ndarray:
        forward = rot_z(self.yaw) @ np.array([1.0, 0.0, 0.0])
        position = self.origin + forward * SPEED_MPS * k * STEP_NS / c.NS_PER_S
        return g.rigid_from_rotation_translation(rot_z(self.yaw), position)

    def project(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        T_CW = g.invert_rigid(self.pose(k) @ self.T_BC)
        points_c = self.points_W @ T_CW[:3, :3].T + T_CW[:3, 3]
        pixels = (points_c @ self.K.T)[:, :2] / points_c[:, 2:3]
        visible = (points_c[:, 2] > 0) & (pixels[:, 0] > 3) & (pixels[:, 0] < 956) \
            & (pixels[:, 1] > 3) & (pixels[:, 1] < 460)
        return pixels, visible

    def image(self, k: int, *, blank: bool = False) -> np.ndarray:
        rng = f.seeded_rng(1000 + k)
        canvas = 20.0 + rng.normal(0.0, 1.5, (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH))
        if not blank:
            pixels, visible = self.project(k)
            yy, xx = np.mgrid[-6:7, -6:7]
            for (u, v), amp in zip(pixels[visible], 140 + 60 * (np.arange(visible.sum()) % 3)):
                cu, cv = int(round(u)), int(round(v))
                ys, xs = yy + cv, xx + cu
                ok = (ys >= 0) & (ys < c.CANONICAL_HEIGHT) & (xs >= 0) & (xs < c.CANONICAL_WIDTH)
                blob = amp * np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2 * 2.0**2))
                canvas[ys[ok], xs[ok]] += blob[ok]
        gray = np.clip(canvas, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))

    def observation(self, k: int, **overrides) -> c.Observation:
        fields = dict(camera_id="fr", exposure_ns=self.t_ns(k), image=self.image(k),
                      calibration_id=CALIBRATION_ID, K=self.K, T_BC=self.T_BC)
        fields.update(overrides)
        return f.make_observation(**fields)

    def correspondences(self, k: int, map_id: str, limit: int | None = None) -> c.Correspondences:
        pixels, visible = self.project(k)
        rows = np.flatnonzero(visible)[:limit]
        return c.Correspondences(camera_id="fr", exposure_ns=self.t_ns(k),
                                 keypoint_ids=np.arange(rows.size, dtype=np.int64) + 1000,
                                 pixels=pixels[rows].astype(np.float32),
                                 landmark_ids=self.landmark_ids[rows], map_id=map_id)

    def snapshot(self, k: int, *, aux: bool = False, sigma: float = 0.5) -> c.SearchSnapshot:
        pose = self.pose(k)
        auxiliary = c.AuxiliaryGeometry.unavailable("not_estimated")
        if aux:
            auxiliary = c.AuxiliaryGeometry(source=c.AuxSource.PNP, source_ns=self.t_ns(k),
                                            z_m=float(pose[2, 3]), roll_rad=0.0, pitch_rad=0.0,
                                            road_patch_id=None, valid=True, reason="valid")
        forward = rot_z(self.yaw)[:2, 0] * SPEED_MPS
        return f.make_search_snapshot(
            t_ns=self.t_ns(k), state=np.array([pose[0, 3], pose[1, 3], self.yaw, *forward, 0.0]),
            P=np.diag([sigma**2, sigma**2, 0.01, 1.0, 1.0, 0.1]), auxiliary=auxiliary)

    def build_map(self, root: Path, keyframes: tuple[int, ...] = (0, 10, 20),
                  descriptor_seed: int = 7) -> lm.LandmarkMap:
        """Map whose keyframe keypoints are exact projections; descriptor row 0 encodes the
        landmark id so fake matchers can associate them."""
        kps, desc, links, T_WC = [], [], [], []
        rng = f.seeded_rng(descriptor_seed)
        for k in keyframes:
            pixels, visible = self.project(k)
            rows = np.flatnonzero(visible)
            kps.append(pixels[rows].astype(np.float32))
            d = rng.normal(0, 1, (rows.size, 128)).astype(np.float32)
            d[:, 0] = self.landmark_ids[rows]
            desc.append(d)
            links.append(self.landmark_ids[rows])
            T_WC.append(self.pose(k) @ self.T_BC)
        offsets = np.concatenate([[0], np.cumsum([len(k) for k in kps])]).astype(np.int64)
        arrays = {
            "landmark_ids": self.landmark_ids.copy(),
            "points_W": self.points_W.copy(),
            "landmark_observations": np.full(len(self.landmark_ids), 3, np.int32),
            "landmark_median_error_px": np.zeros(len(self.landmark_ids), np.float32),
            "landmark_max_angle_deg": np.full(len(self.landmark_ids), 2.0, np.float32),
            "keyframe_ids": np.arange(len(keyframes), dtype=np.int64) * 2 + 1,
            "keyframe_cameras": np.array(["fr"] * len(keyframes), dtype="<U2"),
            "keyframe_exposure_ns": np.array([self.t_ns(k) for k in keyframes], np.int64),
            "keyframe_T_WC": np.array(T_WC),
            "keypoint_offsets": offsets,
            "keypoints": np.concatenate(kps),
            "keypoint_scores": np.ones(int(offsets[-1]), np.float32),
            "descriptors": np.concatenate(desc).astype(np.float16),
            "keypoint_landmarks": np.concatenate(links).astype(np.int64),
            "mask_fr": np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool),
        }
        manifest = {
            "cameras": ["fr"], "calibration_id": CALIBRATION_ID,
            "preprocessing_id": f.PREPROCESSING_ID,
            "canonical_K": {"fr": self.K.tolist()}, "T_BC": {"fr": self.T_BC.tolist()},
            "mask_id": "synthetic",
        }
        lm.save_landmark_map(root, arrays, manifest)
        return lm.load_landmark_map(root)

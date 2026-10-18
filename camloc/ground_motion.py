"""Planar ground motion from road corners and A's checked road plane (owner: B).

Implements C's ``GroundMotionEstimator``. Corners on the road region of the previous image
are tracked to the current image with forward/backward LK. Each pixel ray is intersected
with the RoadPatch plane using the camera pose composed from that image's causal snapshot
(x/y/yaw) and auxiliary z/roll/pitch, then expressed in that image's body frame. A robust
2-D rigid fit maps current-body points into the previous body: ``T_Bprev_Bcurr``. Metric
scale comes from the plane and the camera height in the auxiliary geometry, never from an
essential-matrix direction. Weak support returns rejected evidence with a reason.

Starting gates (tune on tuning data only): camera depth 3-30 m, >= 30 inliers, >= 60 %
inlier fraction, median ground residual <= 0.25 m, inlier spread >= 1 m along the minor
axis, speed <= 95 m/s and yaw rate <= 2.5 rad/s over the exposure gap.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from camloc import contracts as c
from camloc import geometry as g
from camloc import masks
from camloc.vision import body_pose, lk_track, mask_lookup


@dataclasses.dataclass(frozen=True, slots=True)
class GroundMotionGates:
    min_depth_m: float = 3.0
    max_depth_m: float = 30.0
    max_corners: int = 400
    corner_quality: float = 0.005
    corner_min_distance_px: int = 7
    min_inliers: int = 30
    min_inlier_fraction: float = 0.6
    inlier_threshold_m: float = 0.25
    max_median_residual_m: float = 0.25
    min_minor_spread_m: float = 1.0
    max_speed_mps: float = 95.0
    max_yaw_rate_radps: float = 2.5
    ransac_iterations: int = 200
    # Patch compatibility of the auxiliary body pose.
    max_body_plane_distance_m: float = 2.0
    max_body_tilt_rad: float = math.radians(10.0)
    max_patch_distance_m: float = 40.0
    # Covariance: residual-based, inflated for plane/shared-pose uncertainty.
    sigma_floor_m: float = 0.05
    covariance_inflation: float = 4.0


def _road_depth(K: np.ndarray, T_WC: np.ndarray, patch: c.RoadPatch,
                pixels: np.ndarray) -> np.ndarray:
    """Camera z-depth at which each pixel's ray meets the plane (NaN if not in front)."""
    rays_c = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T
    rays_w = rays_c @ T_WC[:3, :3].T
    denominator = rays_w @ patch.normal_W
    numerator = float((patch.point_W - T_WC[:3, 3]) @ patch.normal_W)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = numerator / denominator
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)


def _intersect(K: np.ndarray, T_WC: np.ndarray, depth: np.ndarray,
               pixels: np.ndarray) -> np.ndarray:
    rays_c = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T
    points_c = rays_c * depth[:, None]
    return points_c @ T_WC[:3, :3].T + T_WC[:3, 3]


def fit_rigid_2d(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares (theta, t) with target ~= R(theta) @ source + t."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    theta = math.atan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
                       float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return theta, mu_t - R @ mu_s


def _residuals(source, target, theta, t):
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return np.linalg.norm(source @ R.T + t - target, axis=1)


def robust_rigid_2d(source: np.ndarray, target: np.ndarray, threshold: float,
                    iterations: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Deterministic 2-point RANSAC followed by a least-squares refit on the inliers."""
    n = len(source)
    rng = np.random.default_rng(0)
    best = np.zeros(n, bool)
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(source[i] - source[j]) < 0.5:
            continue
        theta, t = fit_rigid_2d(source[[i, j]], target[[i, j]])
        inliers = _residuals(source, target, theta, t) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 2:
        return 0.0, np.zeros(2), best
    for _ in range(3):
        theta, t = fit_rigid_2d(source[best], target[best])
        best = _residuals(source, target, theta, t) <= threshold
        if best.sum() < 2:
            break
    return theta, t, best


class RoadGroundMotion:
    """``GroundMotionEstimator`` (stateless apart from configuration)."""

    def __init__(self, gates: GroundMotionGates = GroundMotionGates()) -> None:
        self.gates = gates

    def _reject(self, previous: c.Observation, current: c.Observation, reason: str,
                patch_id: str | None = None, inliers: int = 0, fraction: float = 0.0,
                residual: float | None = None) -> c.MotionEvidence:
        return c.MotionEvidence(
            camera_id=current.camera_id, previous_ns=previous.exposure_ns,
            current_ns=current.exposure_ns, T_Bprev_Bcurr=None, covariance=None,
            inlier_count=inliers, inlier_fraction=fraction, ground_residual_m=residual,
            road_patch_id=patch_id, accepted=False, reason=reason)

    def _pose(self, snapshot: c.SearchSnapshot) -> np.ndarray:
        x, y, yaw = (float(v) for v in snapshot.state[:3])
        return body_pose(x, y, yaw, snapshot.auxiliary)

    def _compatible(self, T_WB: np.ndarray, patch: c.RoadPatch) -> bool:
        gates = self.gates
        height = float((T_WB[:3, 3] - patch.point_W) @ patch.normal_W)
        tilt = math.acos(float(np.clip(T_WB[:3, 2] @ patch.normal_W, -1.0, 1.0)))
        horizontal = float(np.hypot(*(T_WB[:2, 3] - patch.query_xy)))
        return (abs(height) <= gates.max_body_plane_distance_m and tilt <= gates.max_body_tilt_rad
                and horizontal <= gates.max_patch_distance_m)

    def estimate(self, previous: c.Observation, current: c.Observation, patch: c.RoadPatch, *,
                 previous_snapshot: c.SearchSnapshot,
                 current_snapshot: c.SearchSnapshot) -> c.MotionEvidence:
        for value, kind, name in ((previous, c.Observation, "previous"),
                                  (current, c.Observation, "current"),
                                  (patch, c.RoadPatch, "patch"),
                                  (previous_snapshot, c.SearchSnapshot, "previous_snapshot"),
                                  (current_snapshot, c.SearchSnapshot, "current_snapshot")):
            if not isinstance(value, kind):
                raise c.ContractError(f"RoadGroundMotion.estimate: {name} must be {kind.__name__}")
        if not previous.exposure_ns < current.exposure_ns:
            raise c.ContractError("RoadGroundMotion.estimate: previous must precede current")
        gates = self.gates
        if previous.camera_id != current.camera_id:
            return self._reject(previous, current, "camera_mismatch")
        if previous.calibration_id != current.calibration_id:
            return self._reject(previous, current, "calibration_mismatch")
        if previous.preprocessing_id != current.preprocessing_id:
            return self._reject(previous, current, "preprocessing_mismatch")
        if current.exposure_ns - previous.exposure_ns > c.HISTORY_MAX_AGE_NS:
            return self._reject(previous, current, "stale_motion")
        if previous_snapshot.t_ns != previous.exposure_ns or \
                current_snapshot.t_ns != current.exposure_ns:
            return self._reject(previous, current, "snapshot_time_mismatch")
        if not patch.valid:
            return self._reject(previous, current, f"road_patch_{patch.reason}"[:60])
        if not (previous_snapshot.auxiliary.valid and current_snapshot.auxiliary.valid):
            return self._reject(previous, current, "auxiliary_geometry_unavailable",
                                patch.patch_id)
        T_WB0, T_WB1 = self._pose(previous_snapshot), self._pose(current_snapshot)
        if not (self._compatible(T_WB0, patch) and self._compatible(T_WB1, patch)):
            return self._reject(previous, current, "auxiliary_incompatible_with_patch",
                                patch.patch_id)
        T_WC0, T_WC1 = T_WB0 @ previous.T_BC, T_WB1 @ current.T_BC
        bodywork = masks.bodywork_mask(previous.camera_id)
        usable0 = np.asarray(previous.valid_mask) & bodywork
        usable1 = np.asarray(current.valid_mask) & bodywork

        # Road region of the previous image: plane depth within the gate.
        vv, uu = np.mgrid[0:c.CANONICAL_HEIGHT, 0:c.CANONICAL_WIDTH]
        grid = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
        depth_grid = _road_depth(previous.K, T_WC0, patch, grid).reshape(uu.shape)
        road = usable0 & (depth_grid >= gates.min_depth_m) & (depth_grid <= gates.max_depth_m)
        if road.sum() < 1000:
            return self._reject(previous, current, "no_road_region", patch.patch_id)
        gray0 = cv2.cvtColor(np.asarray(previous.image), cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(np.asarray(current.image), cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray0, maxCorners=gates.max_corners, qualityLevel=gates.corner_quality,
            minDistance=gates.corner_min_distance_px, mask=road.astype(np.uint8) * 255)
        if corners is None or len(corners) < gates.min_inliers:
            return self._reject(previous, current, "insufficient_road_texture", patch.patch_id)
        points0 = corners.reshape(-1, 2).astype(np.float32)
        points1, ok = lk_track(gray0, gray1, points0)
        ok &= mask_lookup(usable1, points1)
        points0, points1 = points0[ok].astype(np.float64), points1[ok].astype(np.float64)
        depth0 = _road_depth(previous.K, T_WC0, patch, points0)
        depth1 = _road_depth(current.K, T_WC1, patch, points1)
        valid = (np.isfinite(depth0) & np.isfinite(depth1)
                 & (depth0 >= gates.min_depth_m) & (depth0 <= gates.max_depth_m)
                 & (depth1 >= gates.min_depth_m) & (depth1 <= gates.max_depth_m))
        n = int(valid.sum())
        if n < gates.min_inliers:
            return self._reject(previous, current, "insufficient_tracked_road_points",
                                patch.patch_id, inliers=0, fraction=0.0)
        world0 = _intersect(previous.K, T_WC0, depth0[valid], points0[valid])
        world1 = _intersect(current.K, T_WC1, depth1[valid], points1[valid])
        body0 = (world0 - T_WB0[:3, 3]) @ T_WB0[:3, :3]
        body1 = (world1 - T_WB1[:3, 3]) @ T_WB1[:3, :3]
        source, target = body1[:, :2], body0[:, :2]
        theta, t, inliers = robust_rigid_2d(source, target, gates.inlier_threshold_m,
                                            gates.ransac_iterations)
        count = int(inliers.sum())
        fraction = count / n
        residuals = _residuals(source[inliers], target[inliers], theta, t) if count else \
            np.zeros(0)
        median = float(np.median(residuals)) if count else None
        dt = (current.exposure_ns - previous.exposure_ns) / c.NS_PER_S

        def reject(reason: str) -> c.MotionEvidence:
            return self._reject(previous, current, reason, patch.patch_id, count, fraction,
                                median)

        if count < gates.min_inliers:
            return reject("too_few_inliers")
        if fraction < gates.min_inlier_fraction:
            return reject("low_inlier_fraction")
        if median > gates.max_median_residual_m:
            return reject("high_ground_residual")
        spread = np.linalg.eigvalsh(np.cov(source[inliers].T))
        if math.sqrt(max(float(spread[0]), 0.0)) < gates.min_minor_spread_m / 2.0:
            return reject("poorly_spread_support")
        if float(np.hypot(*t)) > gates.max_speed_mps * dt or \
                abs(theta) > gates.max_yaw_rate_radps * dt:
            return reject("implausible_motion")

        # Covariance of (dx, dy, dyaw) from the inlier Jacobian, floored and inflated.
        rotated = source[inliers] @ np.array([[math.cos(theta), math.sin(theta)],
                                              [-math.sin(theta), math.cos(theta)]])
        J = np.zeros((2 * count, 3))
        J[0::2, 0] = 1.0
        J[1::2, 1] = 1.0
        J[0::2, 2] = -rotated[:, 1]
        J[1::2, 2] = rotated[:, 0]
        sigma = max(float(np.sqrt(np.mean(residuals**2))), gates.sigma_floor_m)
        covariance = gates.covariance_inflation * sigma**2 * np.linalg.inv(J.T @ J)
        covariance = 0.5 * (covariance + covariance.T) + np.diag([1e-4, 1e-4, 1e-6])
        transform = np.eye(4)
        transform[:2, :2] = [[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]]
        transform[:2, 3] = t
        return c.MotionEvidence(
            camera_id=current.camera_id, previous_ns=previous.exposure_ns,
            current_ns=current.exposure_ns, T_Bprev_Bcurr=g.rigid_from_rotation_translation(
                transform[:3, :3], transform[:3, 3]),
            covariance=covariance, inlier_count=count, inlier_fraction=fraction,
            ground_residual_m=median, road_patch_id=patch.patch_id, accepted=True,
            reason="accepted")

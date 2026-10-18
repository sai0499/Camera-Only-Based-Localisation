"""RANSAC PnP with B's acceptance gates, producing PoseEvidence (owner: B).

Gates (FINAL_BUILD_PLAN section 3, starting defaults; tune only on tuning data): >= 30
inliers, >= 35 % inlier fraction, median error <= 2 px, p90 <= 4 px, inlier support in
>= 6 of the 4 x 4 image-grid cells, positive depths, a non-degenerate (not nearly
collinear) point spread and a plausible body attitude. The pose is converted to T_WB before
any check that C applies (innovation gating is C's). A failed gate returns a rejected
record that keeps everything measured.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from camloc import contracts as c
from camloc import geometry as g


@dataclasses.dataclass(frozen=True, slots=True)
class PnPGates:
    min_correspondences: int = 30
    min_inliers: int = 30
    min_inlier_fraction: float = 0.35
    max_median_px: float = 2.0
    max_p90_px: float = 4.0
    min_cells: int = 6
    ransac_threshold_px: float = 4.0
    ransac_iterations: int = 300
    ransac_confidence: float = 0.999
    # Smallest/largest singular value of the centred inlier pixels; below = nearly collinear.
    min_pixel_spread_ratio: float = 0.08
    min_depth_m: float = 0.5
    max_abs_roll_pitch_rad: float = math.radians(12.0)


DEFAULT_GATES = PnPGates()


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PnPSolution:
    """Raw result of one solve; ``inliers`` indexes the input rows."""

    T_WB: np.ndarray | None
    inliers: np.ndarray  # int64 row indices
    errors_px: np.ndarray  # float64 per inlier
    occupied_cells: int
    reason: str

    @property
    def accepted(self) -> bool:
        return self.reason == "accepted"


def usable_extent(mask: np.ndarray | None) -> tuple[float, float, float, float]:
    """(u0, u1, v0, v1) pixel-edge bounds of the usable rows/columns of a mask."""
    if mask is None or not mask.any():
        return -0.5, c.CANONICAL_WIDTH - 0.5, -0.5, c.CANONICAL_HEIGHT - 0.5
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return cols[0] - 0.5, cols[-1] + 0.5, rows[0] - 0.5, rows[-1] + 0.5


def occupied_cells(pixels: np.ndarray, mask: np.ndarray | None = None) -> int:
    """Occupied cells of a 4 x 4 grid over the canonical image, or over the usable extent
    of ``mask`` (rows/columns containing usable pixels) when a mask is given."""
    if len(pixels) == 0:
        return 0
    u0, u1, v0, v1 = usable_extent(mask)
    col = np.clip((pixels[:, 0] - u0) * 4 // (u1 - u0), 0, 3).astype(int)
    row = np.clip((pixels[:, 1] - v0) * 4 // (v1 - v0), 0, 3).astype(int)
    return int(np.unique(row * 4 + col).size)


def _attitude(T_WB: np.ndarray) -> tuple[float, float]:
    R = T_WB[:3, :3]
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    return roll, pitch


def _reprojection(points_W: np.ndarray, pixels: np.ndarray, K: np.ndarray, T_CW: np.ndarray):
    points_C = points_W @ T_CW[:3, :3].T + T_CW[:3, 3]
    depth = points_C[:, 2]
    safe = np.where(depth > 1e-9, depth, np.nan)
    projected = (points_C @ K.T)[:, :2] / safe[:, None]
    errors = np.linalg.norm(projected - pixels, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf), depth


def solve_pnp(
    pixels: np.ndarray,
    points_W: np.ndarray,
    K: np.ndarray,
    T_BC: np.ndarray,
    gates: PnPGates = DEFAULT_GATES,
) -> PnPSolution:
    """RANSAC PnP (SQPnP minimal solver) + LM refinement on its inliers, then the gates."""
    pixels = np.ascontiguousarray(pixels, np.float64).reshape(-1, 2)
    points_W = np.ascontiguousarray(points_W, np.float64).reshape(-1, 3)
    empty = np.zeros(0, np.int64)
    n = len(pixels)
    if len(points_W) != n or not (np.isfinite(pixels).all() and np.isfinite(points_W).all()):
        raise c.ContractError("PnP needs equal, finite pixel/world point rows")
    if n < max(gates.min_correspondences, 4):
        return PnPSolution(None, empty, np.zeros(0), 0, "too_few_correspondences")
    # Centre the world points: map coordinates are hundreds of metres from the origin.
    centroid = points_W.mean(axis=0)
    local = points_W - centroid
    cv2.setRNGSeed(0)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        local, pixels, K, None,
        iterationsCount=gates.ransac_iterations,
        reprojectionError=gates.ransac_threshold_px,
        confidence=gates.ransac_confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return PnPSolution(None, empty, np.zeros(0), 0, "pnp_failed")
    index = np.sort(inliers.reshape(-1).astype(np.int64))
    rvec, tvec = cv2.solvePnPRefineLM(local[index], pixels[index], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    T_CL = g.rigid_from_rotation_translation(R, tvec.reshape(3))
    T_LW = np.eye(4)
    T_LW[:3, 3] = -centroid
    T_CW = T_CL @ T_LW
    errors, depth = _reprojection(points_W, pixels, K, T_CW)
    # Final inliers: refined pose within the RANSAC threshold and in front of the camera.
    index = np.flatnonzero((errors <= gates.ransac_threshold_px) & (depth > gates.min_depth_m))
    try:
        T_WB = g.body_pose_from_pnp(c.check_rigid(T_CW, "T_CW"), T_BC)
    except c.ContractError:
        return PnPSolution(None, empty, np.zeros(0), 0, "pnp_failed")
    inlier_errors = errors[index]
    cells = occupied_cells(pixels[index])
    reason = "accepted"
    if index.size < gates.min_inliers:
        reason = "too_few_inliers"
    elif index.size / n < gates.min_inlier_fraction:
        reason = "low_inlier_fraction"
    elif float(np.median(inlier_errors)) > gates.max_median_px:
        reason = "high_median_error"
    elif float(np.percentile(inlier_errors, 90)) > gates.max_p90_px:
        reason = "high_p90_error"
    elif cells < gates.min_cells:
        reason = "poor_image_coverage"
    else:
        centred = pixels[index] - pixels[index].mean(axis=0)
        singular = np.linalg.svd(centred, compute_uv=False)
        roll, pitch = _attitude(T_WB)
        if singular[0] <= 0 or singular[-1] / singular[0] < gates.min_pixel_spread_ratio:
            reason = "degenerate_geometry"
        elif max(abs(roll), abs(pitch)) > gates.max_abs_roll_pitch_rad:
            reason = "implausible_attitude"
    return PnPSolution(T_WB, index.astype(np.int64), inlier_errors, cells, reason)


def pose_evidence(
    solution: PnPSolution,
    *,
    observation: c.Observation,
    source: c.PoseSource,
    track_ids: np.ndarray,
    landmark_ids: np.ndarray,
    num_candidates: int,
    map_id: str,
    reason: str | None = None,
) -> c.PoseEvidence:
    """Wrap a solution (or an early failure ``reason``) as PoseEvidence on ``observation``.

    ``track_ids``/``landmark_ids`` are the full candidate rows; the record keeps the inlier
    rows. An override ``reason`` (e.g. ``no_active_tracks``) produces a rejected record.
    """
    final_reason = reason or solution.reason
    inliers = solution.inliers if reason is None else np.zeros(0, np.int64)
    errors = solution.errors_px if reason is None else np.zeros(0)
    median = float(np.median(errors)) if errors.size else None
    p90 = float(np.percentile(errors, 90)) if errors.size else None
    if median is not None and p90 is not None and p90 < median:
        p90 = median
    count = int(inliers.size)
    return c.PoseEvidence(
        camera_id=observation.camera_id,
        exposure_ns=observation.exposure_ns,
        source=source,
        T_WB=solution.T_WB if reason is None else None,
        inlier_count=count,
        inlier_fraction=(count / num_candidates) if num_candidates else 0.0,
        median_error_px=median,
        p90_error_px=p90,
        occupied_cells=solution.occupied_cells if reason is None else 0,
        track_ids=np.asarray(track_ids, np.int64)[inliers],
        landmark_ids=np.asarray(landmark_ids, np.int64)[inliers],
        map_id=map_id,
        calibration_id=observation.calibration_id,
        preprocessing_id=observation.preprocessing_id,
        accepted=final_reason == "accepted",
        reason=final_reason,
    )


def rejected_pose(observation: c.Observation, source: c.PoseSource, map_id: str,
                  reason: str) -> c.PoseEvidence:
    empty = PnPSolution(None, np.zeros(0, np.int64), np.zeros(0), 0, reason)
    return pose_evidence(empty, observation=observation, source=source,
                         track_ids=np.zeros(0, np.int64), landmark_ids=np.zeros(0, np.int64),
                         num_candidates=0, map_id=map_id, reason=reason)

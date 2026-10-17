"""Owner A, step 3: valid border evidence (``camloc/borders.py``).

Surveyed track borders constrain only the lateral position and the yaw of the car, never
the along-track position: a uniform straight gives no absolute s. This module projects
branch-compatible surveyed border samples into one current camera using the prior pose and
auxiliary geometry, measures how well a lateral-shift/yaw search would be observable, and
returns a ``BorderEvidence`` record. It implements the ``BorderEvidenceSource`` protocol of
``camloc.contracts`` and the limits of ``task_docs/FINAL_BUILD_PLAN.md`` section 4.

Diagnostics first
=================

Per the build plan, border updates start as diagnostics. Associating a surveyed border with
whatever bright edge is nearest is not evidence; a correction is accepted only after manual
image review (step 4) and only when both the lateral coordinate and the yaw are observable.
This module therefore computes the observability (the conditioning of the yaw-scaled
two-parameter Jacobian) and the support honestly, but rejects by default with an explicit
reason unless strict gates pass. It never returns an accepted correction merely because an
edge is bright or nearby, and it never resolves an ambiguous pit/track branch by averaging.

Observability model
===================

With the prior yaw, the lateral direction is ``normal_xy = [-sin(yaw), cos(yaw)]``. For
each visible surveyed border sample the two-parameter residual Jacobian rows are the
sensitivity of its image position to a lateral shift and to a yaw rotation about a 10 m
lever arm. The reciprocal condition number of ``J^T J`` over the retained samples is the
``conditioning`` in [0, 1]; a near-singular system (for example a straight seen head-on)
cannot constrain both parameters and is rejected.
"""

from __future__ import annotations

import numpy as np

from camloc import contracts
from camloc import geometry
from camloc.contracts import (
    AuxiliaryGeometry,
    AuxSource,
    BorderEvidence,
    Branch,
    Observation,
    SearchSnapshot,
)
from camloc.geometry import _RoadGeometry, _Border

# Workload and gate constants (plan section 4 / step 3 guide; conservative heuristics).
MAX_BORDER_SAMPLES = 200  # frozen visible branch-compatible XYZ samples per update
MIN_SUPPORT_SAMPLES = 30  # fewer retained samples cannot support a correction
MIN_SUPPORT_CONTOURS = 3  # samples must spread across at least this many separated runs
MIN_CONDITIONING = 0.05  # reciprocal condition number floor of the 2-parameter Jacobian
YAW_LEVER_ARM_M = 10.0  # yaw is scaled by this lever arm in the Jacobian
_CONTOUR_GAP_M = 5.0  # a world gap larger than this starts a new contour run
_LATERAL_NOISE_FLOOR_M = 0.5  # measurement noise floors (plan section 5)
_YAW_NOISE_FLOOR_RAD = np.deg2rad(1.0)

# Bounded lateral/yaw search (plan section 4 / step 4 guide). No along-track search.
_COARSE_LATERAL_M = np.linspace(-2.0, 2.0, 9)  # 9 lateral offsets over +-2 m
_COARSE_YAW_RAD = np.deg2rad(np.linspace(-3.0, 3.0, 7))  # 7 yaw offsets over +-3 deg
_LOCAL_GRID = 4  # 4 x 4 local grid about the best coarse cell
_MAX_CANDIDATE_EVALUATIONS = 85  # cap including finite-difference conditioning checks
_MIN_RESIDUAL_IMPROVEMENT = 0.15  # accepted search must cut the DT residual by >= 15%
_MIN_RETAINED_SUPPORT = 0.80  # and keep >= 80% of samples on-screen and unmasked
_ORIENTATION_BINS = 8  # orientation bins for the distance-transform residual
_DT_TRUNCATION_PX = 20.0  # truncated distance-transform cap, canonical pixels


def _state_pose(snapshot: SearchSnapshot) -> tuple[float, float, float]:
    """Return (x, y, yaw) from a search snapshot's state."""
    state = snapshot.state
    return float(state[0]), float(state[1]), float(state[2])


def _compose_T_WB(x: float, y: float, yaw: float, aux: AuxiliaryGeometry) -> np.ndarray:
    """Compose the body pose T_WB from the planar state and auxiliary z/roll/pitch.

    Auxiliary geometry supplies z, roll and pitch; a valid record is required because the
    permitted initial x/y/yaw has no guaranteed z/roll/pitch. R = Rz(yaw) Ry(pitch) Rx(roll).
    """
    z = float(aux.z_m)
    roll = float(aux.roll_rad)
    pitch = float(aux.pitch_rad)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return geometry.rigid_from_rotation_translation(rz @ ry @ rx, [x, y, z])


def _visible_border_samples(
    border: _Border, T_WB: np.ndarray, T_BC: np.ndarray, K: np.ndarray, query_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (world points, canonical pixels) of border samples visible in the camera.

    Only samples with positive camera-frame depth and an in-bounds canonical pixel are
    kept, at most ``MAX_BORDER_SAMPLES`` nearest the query.
    """
    distances = np.hypot(
        border.points_W[:, 0] - query_xy[0], border.points_W[:, 1] - query_xy[1]
    )
    order = np.argsort(distances)[: 4 * MAX_BORDER_SAMPLES]  # prefilter by proximity
    candidate_W = border.points_W[order]
    camera_points = geometry.project_world_to_camera(candidate_W, T_WB, T_BC)
    in_front = camera_points[:, 2] > 1e-6
    candidate_W = candidate_W[in_front]
    camera_points = camera_points[in_front]
    if len(camera_points) == 0:
        return candidate_W, np.empty((0, 2))
    projected = camera_points @ contracts._check_intrinsics(K, "K").T
    pixels = projected[:, :2] / projected[:, 2:3]
    inside = (
        (pixels[:, 0] >= contracts._PIXEL_MIN)
        & (pixels[:, 0] <= contracts._PIXEL_U_MAX)
        & (pixels[:, 1] >= contracts._PIXEL_MIN)
        & (pixels[:, 1] <= contracts._PIXEL_V_MAX)
    )
    visible_W = candidate_W[inside]
    visible_px = pixels[inside]
    if len(visible_W) > MAX_BORDER_SAMPLES:
        visible_W = visible_W[:MAX_BORDER_SAMPLES]
        visible_px = visible_px[:MAX_BORDER_SAMPLES]
    return visible_W, visible_px


def _count_contours(points_W: np.ndarray) -> int:
    """Return the number of separated world runs among sorted border samples.

    Samples are projected onto their own principal axis and a gap larger than
    ``_CONTOUR_GAP_M`` starts a new run, so support is not all one short stretch.
    """
    if len(points_W) == 0:
        return 0
    centered = points_W[:, :2] - points_W[:, :2].mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    coordinate = np.sort(centered @ axis)
    gaps = np.diff(coordinate)
    return int(1 + np.count_nonzero(gaps > _CONTOUR_GAP_M))


def _jacobian_conditioning(
    visible_W: np.ndarray, T_WB: np.ndarray, T_BC: np.ndarray, K: np.ndarray, normal_xy: np.ndarray
) -> float:
    """Return the reciprocal condition number in [0, 1] of the yaw-scaled 2-parameter Jacobian.

    Column 0 is the image sensitivity to a lateral shift along ``normal_xy``; column 1 is
    the sensitivity to a yaw rotation, scaled by a 10 m lever arm. A near-singular system
    (both columns nearly parallel, as on a head-on straight) returns near 0.
    """
    x, y = float(T_WB[0, 3]), float(T_WB[1, 3])
    yaw = float(np.arctan2(T_WB[1, 0], T_WB[0, 0]))
    delta = 1e-3

    def pixels_for(dx: float, dy: float, dyaw: float) -> np.ndarray:
        shifted = _shift_pose(T_WB, x + dx, y + dy, yaw + dyaw)
        camera_points = geometry.project_world_to_camera(visible_W, shifted, T_BC)
        projected = camera_points @ K.T
        return projected[:, :2] / projected[:, 2:3]

    base = pixels_for(0.0, 0.0, 0.0)
    lateral = pixels_for(delta * normal_xy[0], delta * normal_xy[1], 0.0)
    yawed = pixels_for(0.0, 0.0, delta / YAW_LEVER_ARM_M)
    col_lateral = ((lateral - base) / delta).reshape(-1)
    col_yaw = ((yawed - base) / (delta / YAW_LEVER_ARM_M) / YAW_LEVER_ARM_M).reshape(-1)
    jacobian = np.column_stack([col_lateral, col_yaw])
    gram = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[-1] <= 0.0:
        return 0.0
    return float(max(eigenvalues[0], 0.0) / eigenvalues[-1])


def _shift_pose(T_WB: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Return a copy of T_WB with its planar x/y/yaw replaced, keeping z/roll/pitch."""
    rotation = T_WB[:3, :3].copy()
    # Replace only the yaw component: current yaw -> target yaw about world z.
    current_yaw = float(np.arctan2(T_WB[1, 0], T_WB[0, 0]))
    delta = yaw - current_yaw
    c, s = np.cos(delta), np.sin(delta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    shifted = np.eye(4)
    shifted[:3, :3] = rz @ rotation
    shifted[:3, 3] = [x, y, float(T_WB[2, 3])]
    return shifted


def _edge_distance_transform(observation: Observation) -> np.ndarray | None:
    """Return a truncated distance transform (float32) of the image edges, or None.

    Canny edges on the grayscale image, masked to valid pixels; the distance transform of
    the non-edge pixels gives each pixel's distance to the nearest edge, truncated to
    ``_DT_TRUNCATION_PX``. None is returned when no edge survives the mask (no evidence).
    """
    import cv2

    gray = cv2.cvtColor(np.ascontiguousarray(observation.image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges[~observation.valid_mask] = 0
    if not edges.any():
        return None
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return np.minimum(distance, _DT_TRUNCATION_PX).astype(np.float32)


def _sample_distance(distance: np.ndarray, pixels: np.ndarray) -> tuple[float, int, int]:
    """Return (mean truncated residual, on-screen count, total) for projected pixels.

    Off-screen pixels are excluded from the mean (they cannot improve a score) and counted
    as lost support. The residual is the mean distance-to-edge over the retained pixels.
    """
    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)
    height, width = distance.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    total = int(len(pixels))
    if not inside.any():
        return _DT_TRUNCATION_PX, 0, total
    residuals = distance[v[inside], u[inside]]
    return float(residuals.mean()), int(inside.sum()), total


def _project_samples(
    points_W: np.ndarray, T_WB: np.ndarray, T_BC: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Project world samples to canonical pixels, keeping only positive-depth points."""
    camera_points = geometry.project_world_to_camera(points_W, T_WB, T_BC)
    in_front = camera_points[:, 2] > 1e-6
    camera_points = camera_points[in_front]
    if len(camera_points) == 0:
        return np.empty((0, 2))
    projected = camera_points @ K.T
    return projected[:, :2] / projected[:, 2:3]


def _bounded_search(
    visible_W: np.ndarray,
    distance: np.ndarray,
    T_WB: np.ndarray,
    T_BC: np.ndarray,
    K: np.ndarray,
    normal_xy: np.ndarray,
) -> dict:
    """Run the capped lateral/yaw search and return its diagnostics.

    A 9 x 7 coarse grid over +-2 m lateral and +-3 degrees yaw, then a 4 x 4 local grid
    about the best coarse cell, scored by the mean truncated distance-to-edge of the
    projected samples. Evaluations are capped at ``_MAX_CANDIDATE_EVALUATIONS``. No
    along-track offset is ever searched. Returns the base and best residuals, the retained
    support fraction and the best lateral/yaw offset.
    """
    x0, y0 = float(T_WB[0, 3]), float(T_WB[1, 3])
    yaw0 = float(np.arctan2(T_WB[1, 0], T_WB[0, 0]))

    def score(lateral: float, yaw_offset: float) -> tuple[float, int, int]:
        x = x0 + lateral * normal_xy[0]
        y = y0 + lateral * normal_xy[1]
        pose = _shift_pose(T_WB, x, y, yaw0 + yaw_offset)
        pixels = _project_samples(visible_W, pose, T_BC, K)
        if len(pixels) == 0:
            return _DT_TRUNCATION_PX, 0, int(len(visible_W))
        return _sample_distance(distance, pixels)

    evaluations = 0
    base_residual, base_inside, total = score(0.0, 0.0)
    evaluations += 1
    best = (base_residual, 0.0, 0.0, base_inside)

    for lateral in _COARSE_LATERAL_M:
        for yaw_offset in _COARSE_YAW_RAD:
            if evaluations >= _MAX_CANDIDATE_EVALUATIONS:
                break
            residual, inside, _total = score(float(lateral), float(yaw_offset))
            evaluations += 1
            if residual < best[0]:
                best = (residual, float(lateral), float(yaw_offset), inside)

    # Local 4 x 4 refinement about the best coarse cell.
    lateral_step = (_COARSE_LATERAL_M[1] - _COARSE_LATERAL_M[0]) / _LOCAL_GRID
    yaw_step = (_COARSE_YAW_RAD[1] - _COARSE_YAW_RAD[0]) / _LOCAL_GRID
    center_lateral, center_yaw = best[1], best[2]
    for i in range(_LOCAL_GRID):
        for j in range(_LOCAL_GRID):
            if evaluations >= _MAX_CANDIDATE_EVALUATIONS:
                break
            lateral = center_lateral + (i - _LOCAL_GRID / 2 + 0.5) * lateral_step
            yaw_offset = center_yaw + (j - _LOCAL_GRID / 2 + 0.5) * yaw_step
            residual, inside, _total = score(float(lateral), float(yaw_offset))
            evaluations += 1
            if residual < best[0]:
                best = (residual, float(lateral), float(yaw_offset), inside)

    improvement = 0.0 if base_residual <= 0 else (base_residual - best[0]) / base_residual
    retained = 0.0 if total == 0 else best[3] / total
    return {
        "base_residual": base_residual,
        "best_residual": best[0],
        "lateral_m": best[1],
        "yaw_rad": best[2],
        "improvement": improvement,
        "retained_support": retained,
        "evaluations": evaluations,
    }

class _BorderEvidenceSource:
    """Owner A's conservative, diagnostics-first border evidence source."""

    def __init__(self, road: _RoadGeometry, *, allow_correction: bool = False) -> None:
        self._road = road
        self._allow_correction = allow_correction

    def _reject(
        self, observation: Observation, branch: Branch, normal_xy: np.ndarray, reason: str, *,
        support_samples: int = 0, support_contours: int = 0, conditioning: float = 0.0,
    ) -> BorderEvidence:
        return BorderEvidence(
            camera_id=observation.camera_id,
            exposure_ns=observation.exposure_ns,
            branch=branch,
            normal_xy=normal_xy,
            lateral_coordinate_m=None,
            yaw_rad=None,
            R=None,
            support_samples=support_samples,
            support_contours=support_contours,
            conditioning=conditioning,
            accepted=False,
            reason=reason,
        )

    def border_evidence(
        self, observation: Observation, snapshot: SearchSnapshot, aux: AuxiliaryGeometry
    ) -> BorderEvidence:
        """Return lateral/yaw-only border evidence for one current camera image.

        Observability and support are measured honestly; the record is rejected with an
        explicit reason unless the branch is known, the auxiliary geometry is valid, at
        least ``MIN_SUPPORT_SAMPLES`` samples spread across ``MIN_SUPPORT_CONTOURS`` runs
        are visible and the yaw-scaled Jacobian conditioning clears ``MIN_CONDITIONING``.
        An accepted record carries the measured lateral coordinate, yaw and their 2x2
        covariance; it never resolves an ambiguous branch by averaging.
        """
        x, y, yaw = _state_pose(snapshot)
        normal_xy = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        branch = snapshot.branch

        if branch is Branch.UNKNOWN:
            return self._reject(observation, branch, normal_xy, "ambiguous_branch")
        if not aux.valid or aux.source is AuxSource.NONE:
            return self._reject(observation, branch, normal_xy, "invalid_auxiliary_geometry")

        patch = self._road.road_patch(x, y, yaw, branch)
        if not patch.valid:
            return self._reject(observation, branch, normal_xy, f"invalid_patch_{patch.reason}")

        borders = self._road._borders.get(branch)
        if borders is None:
            return self._reject(observation, branch, normal_xy, "branch_without_border")

        query_xy = np.array([x, y], dtype=np.float64)
        T_WB = _compose_T_WB(x, y, yaw, aux)
        K = contracts._check_intrinsics(observation.K, "K")
        visible_W = []
        for border in borders:
            points_W, _pixels = _visible_border_samples(border, T_WB, observation.T_BC, K, query_xy)
            if len(points_W):
                visible_W.append(points_W)
        if not visible_W:
            return self._reject(observation, branch, normal_xy, "no_visible_border")
        visible_W = np.concatenate(visible_W)
        if len(visible_W) > MAX_BORDER_SAMPLES:
            visible_W = visible_W[:MAX_BORDER_SAMPLES]

        support_samples = int(len(visible_W))
        support_contours = _count_contours(visible_W)
        conditioning = _jacobian_conditioning(visible_W, T_WB, observation.T_BC, K, normal_xy)

        if support_samples < MIN_SUPPORT_SAMPLES:
            return self._reject(
                observation, branch, normal_xy, "insufficient_samples",
                support_samples=support_samples, support_contours=support_contours,
                conditioning=conditioning,
            )
        if support_contours < MIN_SUPPORT_CONTOURS:
            return self._reject(
                observation, branch, normal_xy, "insufficient_contours",
                support_samples=support_samples, support_contours=support_contours,
                conditioning=conditioning,
            )
        if conditioning < MIN_CONDITIONING:
            return self._reject(
                observation, branch, normal_xy, "poorly_conditioned",
                support_samples=support_samples, support_contours=support_contours,
                conditioning=conditioning,
            )

        # Observability is proven; run the capped lateral/yaw search against image edges.
        distance = _edge_distance_transform(observation)
        if distance is None:
            return self._reject(
                observation, branch, normal_xy, "no_edge_evidence",
                support_samples=support_samples, support_contours=support_contours,
                conditioning=conditioning,
            )
        search = _bounded_search(visible_W, distance, T_WB, observation.T_BC, K, normal_xy)

        # Gate the search result: require the residual improvement and retained-support
        # thresholds. A search that fails either is a diagnostic, not a correction.
        gated = (
            search["improvement"] >= _MIN_RESIDUAL_IMPROVEMENT
            and search["retained_support"] >= _MIN_RETAINED_SUPPORT
        )
        if not (self._allow_correction and gated):
            # Stop-rule default: report the measured diagnostic without applying a fix. The
            # bounded search ran (its evaluation budget was respected) but acceptance waits
            # on the manual image-association review and C's tuning comparison.
            reason = (
                "diagnostic_only_pending_review"
                if gated
                else "search_below_improvement_or_support"
            )
            return self._reject(
                observation, branch, normal_xy, reason,
                support_samples=support_samples, support_contours=support_contours,
                conditioning=conditioning,
            )

        # Accepted correction: map the lateral/yaw offset into the measured coordinate and
        # yaw, with conservative noise floors inflated by the search residual.
        lateral_coordinate = float(np.dot(normal_xy, [x, y])) + search["lateral_m"]
        corrected_yaw = contracts.wrap_angle(yaw + search["yaw_rad"], "yaw_rad")
        lateral_var = max(_LATERAL_NOISE_FLOOR_M, search["best_residual"] * 0.05) ** 2
        yaw_var = _YAW_NOISE_FLOOR_RAD**2 / max(conditioning, MIN_CONDITIONING)
        R = np.array([[lateral_var, 0.0], [0.0, yaw_var]], dtype=np.float64)
        return BorderEvidence(
            camera_id=observation.camera_id,
            exposure_ns=observation.exposure_ns,
            branch=branch,
            normal_xy=normal_xy,
            lateral_coordinate_m=lateral_coordinate,
            yaw_rad=corrected_yaw,
            R=R,
            support_samples=support_samples,
            support_contours=support_contours,
            conditioning=conditioning,
            accepted=True,
            reason="accepted",
        )


def make_border_source(
    road: _RoadGeometry, *, allow_correction: bool = False
) -> _BorderEvidenceSource:
    """Return a border evidence source over one RoadGeometry.

    ``allow_correction`` is False by default (the plan's stop-rule): the bounded lateral/yaw
    search still runs and its diagnostics are reported, but no correction is accepted until
    the manual image-association review and C's tuning comparison enable it. With it True a
    search that clears the improvement and retained-support gates is accepted.
    """
    if not isinstance(road, _RoadGeometry):
        raise contracts.ContractError(
            f"make_border_source: expected a RoadGeometry, got {type(road).__name__}"
        )
    return _BorderEvidenceSource(road, allow_correction=allow_correction)
"""Owner A, step 2: one pixel convention and one geometry implementation (``camloc/geometry.py``).

This module turns A's raw ``CameraCalibration`` into canonical ``Observation`` records and
provides the single transform/projection algebra every teammate shares. It implements the
``Canonicalizer`` protocol of ``camloc.contracts`` and the pixel/geometry conventions of
``task_docs/FINAL_BUILD_PLAN.md`` section 2.

Canonical remap
===============

One cached ``cv2.initUndistortRectifyMap`` followed by one ``cv2.remap`` combines
undistortion and downsampling in a single interpolation. The map is built from raw K/D,
identity rectification R and the canonical K that A already derived from raw K (never from
``camera_info.P``). The output is RGB uint8 960 x 464. A boolean valid mask marks pixels
whose source sample lies inside the raw image; remap-invalid borders are False. BGR->RGB is
explicit (OpenCV decodes JPEG as BGR). Downstream distortion is zero.

Transform algebra
=================

``T_AB`` maps coordinates from frame B into frame A: ``p_A = T_AB @ p_B``. Extrinsics are
read as ``T_BC`` (camera-to-body). A world point projects with
``P_C = inv(T_WB @ T_BC) @ P_W``; OpenCV PnP returns ``T_CW`` so the body pose is
``T_WB = inv(T_CW) @ inv(T_BC)``. Quaternions are xyzw (scalar last). Rigid inverses use the
closed form ``R.T, -R.T @ t``. A planar pose reads ``x = T_WB[0, 3]``, ``y = T_WB[1, 3]``,
``yaw = atan2(T_WB[1, 0], T_WB[0, 0])``.

hloc/COLMAP pixel centres
=========================

Internally pixel centres follow OpenCV: a pixel's centre is its integer coordinate and the
valid range is [-0.5, 959.5] x [-0.5, 463.5]. hloc's importer adds 0.5 when it imports
features, so the canonical coordinates are passed through unshifted; the half-pixel offset
is applied once, at that boundary, by hloc. ``round_trip_pixel`` projects a canonical pixel
to a bearing and back to prove the K is self-consistent.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camloc import contracts
from camloc.contracts import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    Branch,
    CameraCalibration,
    ImageMessage,
    InputError,
    Observation,
    RoadPatch,
    SUPPLIED_MAP_SHA256,
)

# ---------------------------------------------------------------------------------------
# Transform and projection algebra (pure numpy; no OpenCV needed here).
# ---------------------------------------------------------------------------------------


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 rigid transform in closed form (``R.T``, ``-R.T @ t``)."""
    matrix = contracts.check_rigid(T, "T")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def quaternion_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return the 3x3 rotation of a (normalized) xyzw quaternion (scalar last)."""
    norm = math.hypot(x, y, z, w)
    if not math.isfinite(norm) or norm <= 0.0:
        raise contracts.ContractError("quaternion: expected finite nonzero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rigid_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return a 4x4 rigid transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return contracts.check_rigid(T, "T")


def planar_pose_from_T_WB(T_WB: np.ndarray) -> tuple[float, float, float]:
    """Return ``(x, y, yaw)`` of a body pose: translation xy and yaw from the rotation."""
    matrix = contracts.check_rigid(T_WB, "T_WB")
    x = float(matrix[0, 3])
    y = float(matrix[1, 3])
    yaw = contracts.wrap_angle(float(np.arctan2(matrix[1, 0], matrix[0, 0])), "yaw")
    return x, y, yaw


def body_pose_from_pnp(T_CW: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
    """Return ``T_WB = inv(T_CW) @ inv(T_BC)`` from an OpenCV PnP result and the extrinsics."""
    return invert_rigid(contracts.check_rigid(T_CW, "T_CW")) @ invert_rigid(
        contracts.check_rigid(T_BC, "T_BC")
    )


def project_world_to_camera(points_W: np.ndarray, T_WB: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
    """Return world points in the camera frame: ``P_C = inv(T_WB @ T_BC) @ P_W``.

    ``points_W`` is (N, 3); the result is (N, 3) in the optical camera frame (z forward).
    """
    points = np.asarray(points_W, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise contracts.ContractError(f"points_W: expected (N, 3), got {points.shape}")
    T_WC = contracts.check_rigid(T_WB, "T_WB") @ contracts.check_rigid(T_BC, "T_BC")
    T_CW = invert_rigid(T_WC)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (homogeneous @ T_CW.T)[:, :3]


def project_camera_to_pixels(points_C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return canonical (u, v) pixels of camera-frame points with positive depth.

    A non-positive depth raises ``ContractError``: a point behind the camera has no pixel.
    """
    points = np.asarray(points_C, dtype=np.float64)
    K = contracts._check_intrinsics(K, "K")
    depths = points[:, 2]
    if not np.all(depths > 0.0):
        raise contracts.ContractError("project: a point has non-positive depth (behind camera)")
    projected = points @ K.T
    return projected[:, :2] / projected[:, 2:3]


def pixel_to_bearing(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return unit bearing vectors (N, 3) in the camera frame for canonical (u, v) pixels."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise contracts.ContractError(f"pixels: expected (N, 2), got {pixels.shape}")
    K = contracts._check_intrinsics(K, "K")
    K_inv = np.linalg.inv(K)
    homogeneous = np.concatenate([pixels, np.ones((pixels.shape[0], 1))], axis=1)
    rays = homogeneous @ K_inv.T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def round_trip_pixel(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return pixels re-projected from their own bearings; equals the input for a valid K.

    This proves the canonical K is self-consistent and that no half-pixel shift is applied
    inside this module (hloc applies its single +0.5 at the importer boundary).
    """
    bearings = pixel_to_bearing(pixels, K)
    return project_camera_to_pixels(bearings, K)


# ---------------------------------------------------------------------------------------
# Canonical remap and the Canonicalizer.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CameraMap:
    """A camera's cached raw-to-canonical remap plus its frozen canonical calibration."""

    calibration: CameraCalibration
    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray


def _build_camera_map(calibration: CameraCalibration) -> _CameraMap:
    """Build the single cached raw-to-canonical remap for one camera.

    One ``initUndistortRectifyMap`` with raw K/D, identity rectification and the canonical K
    yields float32 sampling maps; a full-size undistortion followed by a resize is avoided.
    The valid mask is True where the sampled source coordinate lies inside the raw image.
    """
    import cv2

    k_raw = np.ascontiguousarray(calibration.K_raw, dtype=np.float64)
    d_raw = np.ascontiguousarray(calibration.D_raw, dtype=np.float64)
    k_canonical = np.ascontiguousarray(calibration.K, dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        k_raw,
        d_raw,
        np.eye(3, dtype=np.float64),
        k_canonical,
        (CANONICAL_WIDTH, CANONICAL_HEIGHT),
        cv2.CV_32FC1,
    )
    # Bilinear interpolation must not mix an outside/constant-border sample into evidence.
    inside = (
        (map_x >= 0)
        & (map_x <= calibration.raw_width - 1)
        & (map_y >= 0)
        & (map_y <= calibration.raw_height - 1)
    )
    valid_mask = np.ascontiguousarray(inside, dtype=np.bool_)
    return _CameraMap(
        calibration=calibration,
        map_x=np.ascontiguousarray(map_x, dtype=np.float32),
        map_y=np.ascontiguousarray(map_y, dtype=np.float32),
        valid_mask=valid_mask,
    )


class _Canonicalizer:
    """Owner A's canonicalizer; see ``make_canonicalizer``."""

    def __init__(self, calibrations: Mapping[str, CameraCalibration]) -> None:
        self._maps: dict[str, _CameraMap] = {
            camera: _build_camera_map(calibration)
            for camera, calibration in calibrations.items()
        }

    def camera_map(self, camera_id: str) -> _CameraMap:
        """Return the cached remap of a camera (built once per camera at construction)."""
        camera = contracts.check_camera_id(camera_id)
        if camera not in self._maps:
            raise contracts.ContractError(f"canonicalize: no calibration for camera {camera!r}")
        return self._maps[camera]

    def canonicalize(self, image: ImageMessage) -> Observation:
        """Decode one JPEG and apply the cached remap to produce a canonical Observation.

        A decode failure or a size mismatch against the frozen raw calibration raises
        ``InputError`` so C records a ``decode_failure`` input event rather than fusing a
        bad image. The output image is RGB uint8 (464, 960, 3); ``valid_mask`` is True where
        the remap sampled inside the raw image.
        """
        import cv2

        if not isinstance(image, ImageMessage):
            raise contracts.ContractError(
                f"canonicalize: expected an ImageMessage, got {type(image).__name__}"
            )
        camera_map = self.camera_map(image.camera_id)
        calibration = camera_map.calibration
        if image.calibration_id != calibration.calibration_id:
            raise InputError(
                f"canonicalize: image calibration {image.calibration_id!r} differs from the "
                f"frozen {calibration.calibration_id!r} for camera {image.camera_id!r}"
            )
        if image.frame_id != calibration.frame_id:
            raise InputError("canonicalize: image frame differs from the frozen camera frame")

        raw = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise InputError(
                f"canonicalize: undecodable {image.image_format!r} payload for "
                f"{image.camera_id!r} at {image.exposure_ns}"
            )
        if raw.shape[:2] != (calibration.raw_height, calibration.raw_width):
            raise InputError(
                f"canonicalize: decoded size {raw.shape[1]}x{raw.shape[0]} differs from the "
                f"calibrated {calibration.raw_width}x{calibration.raw_height} for "
                f"{image.camera_id!r}"
            )

        remapped = cv2.remap(
            raw,
            camera_map.map_x,
            camera_map.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # OpenCV decodes/remaps BGR; convert to RGB explicitly and make C-contiguous.
        rgb = np.ascontiguousarray(cv2.cvtColor(remapped, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        valid_mask = camera_map.valid_mask.copy()  # Observation marks the buffer read-only

        return Observation(
            camera_id=image.camera_id,
            exposure_ns=image.exposure_ns,
            arrival_ns=image.arrival_ns,
            source_id=image.source_id,
            image=rgb,
            valid_mask=valid_mask,
            calibration_id=calibration.calibration_id,
            preprocessing_id=calibration.preprocessing_id,
            K=calibration.K,
            T_BC=calibration.T_BC,
        )


def make_canonicalizer(calibrations: Mapping[str, CameraCalibration]) -> _Canonicalizer:
    """Return a Canonicalizer that decodes and remaps selected images for the given cameras.

    Each camera's raw-to-canonical map is built once from its frozen ``CameraCalibration``.
    ``canonicalize`` then performs one JPEG decode plus one remap per image.
    """
    if not calibrations:
        raise contracts.ContractError("make_canonicalizer: no calibrations supplied")
    return _Canonicalizer(calibrations)

# ---------------------------------------------------------------------------------------
# Road geometry: a local road plane from the surveyed borders (step 3).
#
# The static map carries four ordered XYZ border polylines in /constructor0/map_markers/rr:
# the main-track left/right (closed loops) and the pit-lane left/right (open). Left/right
# arrays have different lengths and are never paired by index. For a queried pose A finds
# the nearest point on each same-branch border, associates the two opposing segments only
# when their tangents are compatible, their separation is roughly perpendicular and both
# progress on the same branch, fits a small plane to the local paired samples and reports
# support, residual, validity and a reason. A crossing, an ambiguous pit/track pairing or
# too little support yields an invalid patch that disables plane-dependent updates.
# ---------------------------------------------------------------------------------------

_BORDER_TOPIC = "/constructor0/map_markers/rr"
_BORDER_NAMESPACES = {
    Branch.TRACK: ("track_left", "track_right"),
    Branch.PIT: ("pitlane_left", "pitlane_right"),
}

# Local association window and gates (geometric heuristics; tuned conservatively).
_PATCH_RADIUS_M = 8.0  # local samples within this XY radius of the query feed the plane
_MIN_PLANE_SAMPLES = 6  # fewer paired samples than this cannot define a plane
_MAX_TANGENT_ANGLE_RAD = np.deg2rad(35.0)  # opposing tangents must be near (anti)parallel
_MIN_SEPARATION_M = 2.0  # opposite borders closer than this are an ambiguous crossing
_MAX_SEPARATION_M = 30.0  # wider than any real lane -> wrong association
_MAX_PERPENDICULAR_OFFSET = np.deg2rad(45.0)  # separation must be roughly perpendicular
_MAX_PLANE_RESIDUAL_M = 0.5  # a worse plane fit is reported invalid


def _polyline_tangents(points_xy: np.ndarray, closed: bool) -> np.ndarray:
    """Return unit tangent directions (N, 2) along an ordered XY polyline.

    A closed loop wraps; an open polyline repeats its end directions. Zero-length steps
    inherit the previous direction so a duplicated closing point does not create a NaN.
    """
    n = len(points_xy)
    if closed:
        deltas = np.roll(points_xy, -1, axis=0) - points_xy
    else:
        deltas = np.empty_like(points_xy)
        deltas[:-1] = points_xy[1:] - points_xy[:-1]
        deltas[-1] = deltas[-2]
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    tangents = np.zeros_like(deltas)
    nonzero = lengths > 1e-9
    tangents[nonzero] = deltas[nonzero] / lengths[nonzero, None]
    for index in np.where(~nonzero)[0]:
        tangents[index] = tangents[(index - 1) % n]
    return tangents


@dataclass(frozen=True, slots=True)
class _Border:
    """One ordered surveyed border polyline with cached XY tangents."""

    namespace: str
    points_W: np.ndarray  # (N, 3) float64
    tangents_xy: np.ndarray  # (N, 2) float64 unit
    closed: bool

    def nearest_index(self, query_xy: np.ndarray) -> tuple[int, float]:
        """Return the index of the nearest polyline point and its XY distance."""
        distances = np.hypot(
            self.points_W[:, 0] - query_xy[0], self.points_W[:, 1] - query_xy[1]
        )
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def local_points(self, query_xy: np.ndarray, radius: float) -> np.ndarray:
        """Return the contiguous local branch, excluding nearby hairpin/crossing arms."""
        distances = np.hypot(
            self.points_W[:, 0] - query_xy[0], self.points_W[:, 1] - query_xy[1]
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] > radius:
            return self.points_W[:0]
        indices = [nearest]
        seen = {nearest}
        for direction in (-1, 1):
            side = []
            index = nearest + direction
            while self.closed or 0 <= index < len(distances):
                index %= len(distances)
                if index in seen or distances[index] > radius:
                    break
                seen.add(index)
                side.append(index)
                index += direction
            if direction < 0:
                indices = side[::-1] + indices
            else:
                indices.extend(side)
        return self.points_W[indices]


def _load_border_markers(map_mcap: Path) -> dict[str, np.ndarray]:
    """Read the four ordered XYZ border arrays from the static map's rr MarkerArray."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    arrays: dict[str, np.ndarray] = {}
    with open(map_mcap, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, decoded in reader.iter_decoded_messages(
            topics=[_BORDER_TOPIC]
        ):
            for marker in decoded.markers:
                arrays[str(marker.ns)] = np.array(
                    [[p.x, p.y, p.z] for p in marker.points], dtype=np.float64
                )
            break
    expected = {name for names in _BORDER_NAMESPACES.values() for name in names}
    missing = sorted(expected - set(arrays))
    if missing:
        raise contracts.ContractError(f"road geometry: border markers missing {missing}")
    return arrays


def _fit_plane(points_W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a plane to 3D points by PCA; return (point_on_plane, up_normal, rms_residual).

    The normal is the smallest-variance eigenvector, flipped so its z component is positive
    (the road faces up). The residual is the RMS of the signed point-to-plane distances.
    """
    centroid = points_W.mean(axis=0)
    centered = points_W - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    if _s[1] < 1e-6 or _s[1] < 1e-3 * _s[0]:
        raise contracts.ContractError("road geometry: collinear plane support")
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if not norm > 0.0:
        raise contracts.ContractError("road geometry: degenerate plane normal")
    normal = normal / norm
    residuals = centered @ normal
    rms = float(np.sqrt(np.mean(residuals**2)))
    return centroid, normal, rms


class _RoadGeometry:
    """Owner A's local road plane source; see ``load_road_geometry``."""

    def __init__(self, borders: Mapping[Branch, tuple[_Border, _Border]], map_id: str) -> None:
        self._borders = dict(borders)
        self._map_id = map_id
        self._counter = 0

    @property
    def map_id(self) -> str:
        """Opaque identity of the reference geometry (``SUPPLIED_MAP_SHA256``)."""
        return self._map_id

    def _next_patch_id(self, branch: Branch) -> str:
        self._counter += 1
        return f"patch:{self._map_id[:12]}:{branch.value}:{self._counter}"

    def _invalid(self, query_xy: np.ndarray, branch: Branch, reason: str) -> RoadPatch:
        return RoadPatch(
            patch_id=self._next_patch_id(branch),
            branch=branch,
            query_xy=query_xy,
            point_W=None,
            normal_W=None,
            support=0,
            residual_m=None,
            valid=False,
            reason=reason,
        )

    def road_patch(self, x: float, y: float, yaw: float, branch: Branch) -> RoadPatch:
        """Return a local road plane for the queried pose on the requested branch.

        ``yaw`` is accepted for interface symmetry and future tangent gating; the plane
        itself is orientation-free. An invalid patch (ambiguous association, too little
        support, poor fit) carries a reason and no plane, and disables plane-dependent
        updates for that group.
        """
        query_xy = np.array(
            [contracts._check_float(x, "x"), contracts._check_float(y, "y")], dtype=np.float64
        )
        contracts.wrap_angle(yaw, "yaw")  # validate; value not used by the planar fit
        branch = contracts._check_enum(branch, Branch, "branch")
        if branch not in self._borders:
            return self._invalid(query_xy, branch, "branch_without_border")

        left, right = self._borders[branch]
        left_index, left_distance = left.nearest_index(query_xy)
        right_index, right_distance = right.nearest_index(query_xy)

        # Association gate: opposing tangents near (anti)parallel, separation roughly
        # perpendicular and within a plausible lane width. A crossing fails here.
        left_tangent = left.tangents_xy[left_index]
        right_tangent = right.tangents_xy[right_index]
        alignment = abs(float(np.dot(left_tangent, right_tangent)))
        if alignment < np.cos(_MAX_TANGENT_ANGLE_RAD):
            return self._invalid(query_xy, branch, "incompatible_tangents")

        separation_vector = right.points_W[right_index, :2] - left.points_W[left_index, :2]
        separation = float(np.linalg.norm(separation_vector))
        if separation < _MIN_SEPARATION_M:
            return self._invalid(query_xy, branch, "ambiguous_crossing")
        if separation > _MAX_SEPARATION_M:
            return self._invalid(query_xy, branch, "separation_too_wide")
        fraction = float((query_xy - left.points_W[left_index, :2]) @ separation_vector) / separation**2
        margin = 2.0 / separation
        if (max(left_distance, right_distance) > separation + _PATCH_RADIUS_M
                or not -margin <= fraction <= 1.0 + margin):
            return self._invalid(query_xy, branch, "outside_road_corridor")
        separation_unit = separation_vector / separation
        perpendicularity = abs(float(np.dot(separation_unit, left_tangent)))
        if perpendicularity > np.sin(_MAX_PERPENDICULAR_OFFSET):
            return self._invalid(query_xy, branch, "non_perpendicular_separation")

        # A wide road can put the opposite border farther than the local along-road radius.
        left_local = left.local_points(left.points_W[left_index, :2], _PATCH_RADIUS_M)
        right_local = right.local_points(right.points_W[right_index, :2], _PATCH_RADIUS_M)
        if min(len(left_local), len(right_local)) < 3:
            return self._invalid(query_xy, branch, "insufficient_support")
        if np.dot(left_tangent, right_tangent) < 0:
            right_local = right_local[::-1]
        paired = np.argmin(np.linalg.norm(left_local[:, None, :2] - right_local[None, :, :2], axis=2), axis=1)
        if np.any(np.diff(paired) < 0) or len(np.unique(paired)) < 3:
            return self._invalid(query_xy, branch, "non_monotonic_pairing")
        offsets = right_local[paired, :2] - left_local[:, :2]
        widths = np.linalg.norm(offsets, axis=1)
        along = np.abs(offsets @ left_tangent)
        if (np.any(widths < _MIN_SEPARATION_M) or np.any(widths > _MAX_SEPARATION_M)
                or np.any(along > widths * np.sin(_MAX_PERPENDICULAR_OFFSET))):
            return self._invalid(query_xy, branch, "incompatible_local_pairing")
        local = np.unique(np.concatenate([left_local, right_local[np.unique(paired)]]), axis=0)
        try:
            point_W, normal_W, residual = _fit_plane(local)
        except contracts.ContractError:
            return self._invalid(query_xy, branch, "degenerate_plane")
        if residual > _MAX_PLANE_RESIDUAL_M:
            return self._invalid(query_xy, branch, "plane_residual_too_large")

        return RoadPatch(
            patch_id=self._next_patch_id(branch),
            branch=branch,
            query_xy=query_xy,
            point_W=point_W,
            normal_W=normal_W,
            support=int(len(local)),
            residual_m=residual,
            valid=True,
            reason="valid",
        )

def load_road_geometry(map_mcap: str | Path) -> _RoadGeometry:
    """Load the surveyed borders from the static map and return a RoadGeometry.

    The map is hash-bound: its SHA-256 must equal ``SUPPLIED_MAP_SHA256`` (the reference
    map identity). Track borders are closed loops, pit borders open; left/right arrays keep
    their own lengths and are never paired by index. Paths and markers are duplicate
    geometry, so only the rr MarkerArray is read.
    """
    map_mcap = Path(map_mcap)
    digest = hashlib.sha256(map_mcap.read_bytes()).hexdigest()
    if digest != SUPPLIED_MAP_SHA256:
        raise contracts.ContractError(
            f"road geometry {map_mcap}: SHA-256 {digest} != {SUPPLIED_MAP_SHA256}; the static "
            "map is hash-bound and must stay byte-for-byte unchanged"
        )
    arrays = _load_border_markers(map_mcap)
    borders: dict[Branch, tuple[_Border, _Border]] = {}
    for branch, (left_ns, right_ns) in _BORDER_NAMESPACES.items():
        pair = []
        for namespace in (left_ns, right_ns):
            points = arrays[namespace]
            closed = bool(np.allclose(points[0], points[-1]))
            pair.append(
                _Border(
                    namespace=namespace,
                    points_W=points,
                    tangents_xy=_polyline_tangents(points[:, :2], closed),
                    closed=closed,
                )
            )
        borders[branch] = (pair[0], pair[1])
    return _RoadGeometry(borders, SUPPLIED_MAP_SHA256)

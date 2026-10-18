"""Frozen fitting-only landmark map artifact: format, loading and reference search (owner: B).

``build_map.py`` writes a map directory; inference loads it read-only. Layout::

    <dir>/manifest.json   provenance, identities, calibration, thresholds, statistics
    <dir>/map.npz         arrays below (uncompressed .npy members, loaded fully)

Arrays (keypoints float32, world geometry and calibration float64, descriptors float16):

* ``landmark_ids`` int64 (L,) sorted unique; ``points_W`` float64 (L, 3) in ``map``;
  ``landmark_observations`` int32 (L,); ``landmark_median_error_px`` float32 (L,);
  ``landmark_max_angle_deg`` float32 (L,).
* ``keyframe_ids`` int64 (K,) sorted unique; ``keyframe_cameras`` <U2 (K,);
  ``keyframe_exposure_ns`` int64 (K,); ``keyframe_T_WC`` float64 (K, 4, 4).
* ``keypoint_offsets`` int64 (K + 1,); ``keypoints`` float32 (N, 2); ``keypoint_scores``
  float32 (N,); ``descriptors`` float16 (N, 128); ``keypoint_landmarks`` int64 (N,), -1
  when the keypoint has no retained landmark. Keyframe k owns rows
  ``keypoint_offsets[k]:keypoint_offsets[k + 1]``.
* ``mask_<camera>`` bool (464, 960) usable-pixel masks (bodywork excluded).

``map_id`` hashes the arrays digest and complete manifest (including geometry and models);
the loader re-checks both. The artifact holds reference geometry only: no query trajectory,
no timestamp-to-pose lookup. Keyframe exposure times are provenance, never search keys.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from camloc import contracts as c

FORMAT = "camloc-landmark-map-v2"
MANIFEST_NAME = "manifest.json"
ARRAYS_NAME = "map.npz"
# Canonical intrinsics/extrinsics of a run must match the map's within these tolerances.
K_TOLERANCE_PX = 0.05
T_BC_TOLERANCE = 1e-6


def raw_calibration_geometry(calibration):
    """Exact inputs of the per-camera canonical remap, excluding unrelated sensor mounts."""
    return {"frame_id": calibration.frame_id, "raw_width": calibration.raw_width,
            "raw_height": calibration.raw_height, "K_raw": calibration.K_raw.tolist(),
            "D_raw": calibration.D_raw.tolist(), "K": calibration.K.tolist(),
            "T_BC": calibration.T_BC.tolist()}


def bind_calibrations(calibrations, manifest):
    """Bind a matching mapped camera to its frozen remap ID using hash-bound raw proof.

    Legacy banks without raw proof retain their original strict identity check. A change
    to another camera cannot invalidate an exactly identical mapped camera's pixels.
    """
    from camloc.io import PREPROCESSING_RECIPE

    result = dict(calibrations)
    for camera in manifest["cameras"]:
        calibration = calibrations[camera]
        if calibration.preprocessing_id == manifest["preprocessing_id"]:
            continue
        if (manifest.get("preprocessing_recipe") != PREPROCESSING_RECIPE
                or manifest.get("raw_camera_geometry", {}).get(camera) != raw_calibration_geometry(calibration)
                or not np.array_equal(calibration.K, manifest["canonical_K"][camera])
                or not np.array_equal(calibration.T_BC, manifest["T_BC"][camera])):
            raise c.ContractError(f"camera {camera} lacks matching frozen raw preprocessing geometry")
        result[camera] = dataclasses.replace(calibration, preprocessing_id=manifest["preprocessing_id"])
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _map_id(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "map_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "fitmap:" + hashlib.sha256(encoded).hexdigest()[:16]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Keyframe:
    """One reference image of the map: its fixed camera pose and stored features."""

    keyframe_id: int
    camera_id: str
    exposure_ns: int
    T_WC: np.ndarray
    keypoints: np.ndarray  # float32 (N, 2)
    descriptors: np.ndarray  # float32 (N, 128)
    landmark_ids: np.ndarray  # int64 (N,), -1 = no landmark


def camera_heading(T_WC: np.ndarray) -> float:
    """Map-frame heading of the optical axis (camera +z) projected onto the XY plane."""
    axis = np.asarray(T_WC)[:3, 2]
    return math.atan2(float(axis[1]), float(axis[0]))


def planar_camera_center(x: float, y: float, yaw: float, T_BC: np.ndarray) -> np.ndarray:
    """Approximate XY camera centre of a planar body pose (z/roll/pitch ignored)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    offset = np.asarray(T_BC)[:2, 3]
    return np.array([x + cy * offset[0] - sy * offset[1], y + sy * offset[0] + cy * offset[1]])


def planar_camera_heading(yaw: float, T_BC: np.ndarray) -> float:
    axis = np.asarray(T_BC)[:3, 2]
    return c.wrap_angle(yaw + math.atan2(float(axis[1]), float(axis[0])))


class LandmarkMap:
    """Read-only loaded map. Construct with ``load_landmark_map``."""

    def __init__(self, root: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]):
        self.root = root
        self.manifest = manifest
        self.map_id: str = manifest["map_id"]
        self.preprocessing_id: str = manifest["preprocessing_id"]
        self.calibration_id: str = manifest["calibration_id"]
        self.mask_id: str = manifest["mask_id"]
        self.cameras: tuple[str, ...] = tuple(manifest["cameras"])
        self._K = {cam: np.array(v, np.float64) for cam, v in manifest["canonical_K"].items()}
        self._T_BC = {cam: np.array(v, np.float64) for cam, v in manifest["T_BC"].items()}
        for value in (*self._K.values(), *self._T_BC.values()):
            value.setflags(write=False)
        self.landmark_ids = arrays["landmark_ids"]
        self.points_W = arrays["points_W"]
        self.keyframe_ids = arrays["keyframe_ids"]
        self.keyframe_cameras = arrays["keyframe_cameras"]
        self.keyframe_exposure_ns = arrays["keyframe_exposure_ns"]
        self.keyframe_T_WC = arrays["keyframe_T_WC"]
        self._offsets = arrays["keypoint_offsets"]
        self._keypoints = arrays["keypoints"]
        self._descriptors = arrays["descriptors"]
        self._keypoint_landmarks = arrays["keypoint_landmarks"]
        self._masks = {cam: arrays[f"mask_{cam}"] for cam in self.cameras}
        for array in arrays.values():
            array.setflags(write=False)
        self._centers = self.keyframe_T_WC[:, :2, 3].copy()
        self._headings = np.array([camera_heading(T) for T in self.keyframe_T_WC])
        self._landmark_counts = np.array(
            [int(np.count_nonzero(self._keypoint_landmarks[a:b] >= 0))
             for a, b in zip(self._offsets[:-1], self._offsets[1:], strict=True)]
        )

    # -- identity and calibration ---------------------------------------------------------

    def K(self, camera_id: str) -> np.ndarray:
        return self._K[camera_id]

    def T_BC(self, camera_id: str) -> np.ndarray:
        return self._T_BC[camera_id]

    def mask(self, camera_id: str) -> np.ndarray:
        """Usable-pixel mask the map was built with (bodywork excluded)."""
        return self._masks[camera_id]

    def compatibility(self, camera_id: str, K: np.ndarray, T_BC: np.ndarray,
                      preprocessing_id: str) -> str:
        """``"valid"`` or a reason why images with this geometry cannot use the map.

        The run's opaque calibration id may differ from the training recording's (it hashes
        all seven cameras); the camera's canonical K and T_BC must match numerically.
        """
        if camera_id not in self.cameras:
            return "camera_not_in_map"
        if preprocessing_id != self.preprocessing_id:
            return "preprocessing_mismatch"
        if not np.allclose(K, self._K[camera_id], atol=K_TOLERANCE_PX, rtol=0.0):
            return "map_intrinsics_mismatch"
        if not np.allclose(T_BC, self._T_BC[camera_id], atol=T_BC_TOLERANCE, rtol=0.0):
            return "map_extrinsics_mismatch"
        return "valid"

    # -- landmarks and keyframes ----------------------------------------------------------

    @property
    def num_landmarks(self) -> int:
        return int(self.landmark_ids.size)

    def points(self, landmark_ids: np.ndarray) -> np.ndarray:
        """World points (N, 3) of known landmark ids; unknown ids raise ContractError."""
        ids = np.asarray(landmark_ids, np.int64)
        rows = np.searchsorted(self.landmark_ids, ids)
        rows = np.clip(rows, 0, max(self.landmark_ids.size - 1, 0))
        if ids.size and (self.landmark_ids.size == 0 or np.any(self.landmark_ids[rows] != ids)):
            raise c.ContractError(f"landmark ids not in map {self.map_id}")
        return self.points_W[rows]

    def has_keyframe(self, keyframe_id: int) -> bool:
        row = int(np.searchsorted(self.keyframe_ids, keyframe_id))
        return row < self.keyframe_ids.size and int(self.keyframe_ids[row]) == keyframe_id

    def keyframe(self, keyframe_id: int) -> Keyframe:
        row = int(np.searchsorted(self.keyframe_ids, keyframe_id))
        if row >= self.keyframe_ids.size or int(self.keyframe_ids[row]) != keyframe_id:
            raise c.ContractError(f"keyframe {keyframe_id} not in map {self.map_id}")
        a, b = int(self._offsets[row]), int(self._offsets[row + 1])
        return Keyframe(
            keyframe_id=int(keyframe_id),
            camera_id=str(self.keyframe_cameras[row]),
            exposure_ns=int(self.keyframe_exposure_ns[row]),
            T_WC=self.keyframe_T_WC[row],
            keypoints=self._keypoints[a:b],
            descriptors=self._descriptors[a:b].astype(np.float32),
            landmark_ids=self._keypoint_landmarks[a:b],
        )

    def select_references(
        self,
        camera_id: str,
        x: float,
        y: float,
        yaw: float,
        *,
        sigma_xy_m: float = 0.0,
        count: int = c.RETRIEVAL_CANDIDATES,
        base_radius_m: float = 25.0,
        max_heading_rad: float = math.radians(30.0),
        min_landmarks: int = 30,
        lead_m: float = 6.0,
        min_separation_m: float = 6.0,
    ) -> tuple[int, ...]:
        """Spatially plausible, viewing-compatible reference keyframes for a planar prior.

        Candidates share the camera, have at least ``min_landmarks`` landmark keypoints, a
        camera centre within ``base_radius_m + 3 sigma_xy_m`` of the predicted centre and an
        optical-axis heading within ``max_heading_rad``. They are ranked by the distance to a
        point ``lead_m`` ahead along the predicted optical axis (a reference slightly ahead
        sees the landmarks the query still has in view) + 10 m/rad heading difference, ties
        by keyframe id. The first MAX_REFERENCES_PER_JOB entries are made diverse: each is
        at least ``min_separation_m`` from the earlier ones when such a candidate exists,
        so the two matched references contribute different landmarks. Geometry only, never
        fitting-lap time.
        """
        if camera_id not in self.cameras or not self.keyframe_ids.size:
            return ()
        T_BC = self._T_BC[camera_id]
        center = planar_camera_center(x, y, yaw, T_BC)
        heading = planar_camera_heading(yaw, T_BC)
        distance = np.hypot(*(self._centers - center).T)
        target = center + lead_m * np.array([math.cos(heading), math.sin(heading)])
        to_target = np.hypot(*(self._centers - target).T)
        dheading = np.abs((self._headings - heading + math.pi) % (2 * math.pi) - math.pi)
        radius = base_radius_m + 3.0 * max(float(sigma_xy_m), 0.0)
        ok = (
            (self.keyframe_cameras == camera_id)
            & (distance <= radius)
            & (dheading <= max_heading_rad)
            & (self._landmark_counts >= min_landmarks)
        )
        rows = np.flatnonzero(ok)
        cost = to_target[rows] + 10.0 * dheading[rows]
        order = list(rows[np.lexsort((self.keyframe_ids[rows], cost))])
        chosen: list[int] = []
        for row in order:
            if len(chosen) >= c.MAX_REFERENCES_PER_JOB:
                break
            if all(np.hypot(*(self._centers[row] - self._centers[k])) >= min_separation_m
                   for k in chosen):
                chosen.append(row)
        ranked = chosen + [row for row in order if row not in chosen]
        return tuple(int(self.keyframe_ids[row]) for row in ranked[:count])

    def summary(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "cameras": list(self.cameras),
            "landmarks": self.num_landmarks,
            "keyframes": int(self.keyframe_ids.size),
            "keypoints": int(self._keypoints.shape[0]),
            "preprocessing_id": self.preprocessing_id,
            "mask_id": self.mask_id,
        }


_REQUIRED_ARRAYS = {
    "landmark_ids": (np.int64, 1), "points_W": (np.float64, 2),
    "landmark_observations": (np.int32, 1), "landmark_median_error_px": (np.float32, 1),
    "landmark_max_angle_deg": (np.float32, 1), "keyframe_ids": (np.int64, 1),
    "keyframe_exposure_ns": (np.int64, 1), "keyframe_T_WC": (np.float64, 3),
    "keypoint_offsets": (np.int64, 1), "keypoints": (np.float32, 2),
    "keypoint_scores": (np.float32, 1), "descriptors": (np.float16, 2),
    "keypoint_landmarks": (np.int64, 1),
}


def _validate_arrays(arrays: dict[str, np.ndarray], cameras: tuple[str, ...]) -> None:
    for name, (dtype, ndim) in _REQUIRED_ARRAYS.items():
        if name not in arrays:
            raise c.ContractError(f"map array {name} missing")
        if arrays[name].dtype != dtype or arrays[name].ndim != ndim:
            raise c.ContractError(f"map array {name}: expected {np.dtype(dtype)} {ndim}-D")
    lm, kf = arrays["landmark_ids"], arrays["keyframe_ids"]
    if lm.size and (np.any(np.diff(lm) <= 0) or lm[0] < 0):
        raise c.ContractError("landmark_ids must be sorted, unique and non-negative")
    if kf.size and (np.any(np.diff(kf) <= 0) or kf[0] < 0):
        raise c.ContractError("keyframe_ids must be sorted, unique and non-negative")
    if arrays["points_W"].shape != (lm.size, 3) or not np.isfinite(arrays["points_W"]).all():
        raise c.ContractError("points_W must be finite (L, 3)")
    offsets = arrays["keypoint_offsets"]
    total = arrays["keypoints"].shape[0]
    if offsets.shape != (kf.size + 1,) or offsets[0] != 0 or offsets[-1] != total \
            or np.any(np.diff(offsets) < 0):
        raise c.ContractError("keypoint_offsets inconsistent with keypoints")
    if arrays["descriptors"].shape[0] != total or arrays["keypoint_landmarks"].shape != (total,):
        raise c.ContractError("descriptor/landmark rows must match keypoints")
    for name, shape in (("keypoints", (total, 2)), ("descriptors", (total, 128)),
                        ("keypoint_scores", (total,)), ("landmark_observations", (lm.size,)),
                        ("landmark_median_error_px", (lm.size,)),
                        ("landmark_max_angle_deg", (lm.size,)),
                        ("keyframe_exposure_ns", (kf.size,)),
                        ("keyframe_T_WC", (kf.size, 4, 4)),
                        ("keyframe_cameras", (kf.size,))):
        if name not in arrays or arrays[name].shape != shape:
            raise c.ContractError(f"map array {name}: expected {shape}")
        if arrays[name].dtype.kind in "fiu" and not np.isfinite(arrays[name]).all():
            raise c.ContractError(f"map array {name}: non-finite values")
    if np.any(arrays["keypoint_landmarks"] < -1):
        raise c.ContractError("invalid negative landmark link")
    for pose in arrays["keyframe_T_WC"]:
        c.check_rigid(pose, "keyframe_T_WC")
    linked = arrays["keypoint_landmarks"][arrays["keypoint_landmarks"] >= 0]
    if linked.size and not np.isin(linked, lm).all():
        raise c.ContractError("keypoint_landmarks reference unknown landmarks")
    if not set(arrays["keyframe_cameras"].tolist()) <= set(cameras):
        raise c.ContractError("keyframes use cameras missing from the manifest")
    for cam in cameras:
        mask = arrays.get(f"mask_{cam}")
        if mask is None or mask.dtype != np.bool_ or mask.shape != (c.CANONICAL_HEIGHT,
                                                                  c.CANONICAL_WIDTH):
            raise c.ContractError(f"mask_{cam} must be bool (464, 960)")


def load_landmark_map(path: str | Path) -> LandmarkMap:
    """Load and verify a map directory written by ``save_landmark_map``."""
    root = Path(path)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise c.ContractError(f"{root}: not a {FORMAT} artifact")
    arrays_path = root / ARRAYS_NAME
    digest = _sha256(arrays_path)
    if digest != manifest["map_npz_sha256"] or manifest["map_id"] != _map_id(manifest):
        raise c.ContractError(f"{root}: map arrays or manifest do not match their identity")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cameras = tuple(manifest["cameras"])
    if not cameras or len(cameras) != len(set(cameras)):
        raise c.ContractError("map cameras must be nonempty and unique")
    for camera in cameras:
        c.check_camera_id(camera)
        c._check_intrinsics(manifest["canonical_K"][camera], f"map K {camera}")
        c.check_rigid(manifest["T_BC"][camera], f"map T_BC {camera}")
    _validate_arrays(arrays, cameras)
    return LandmarkMap(root, manifest, arrays)


def save_landmark_map(path: str | Path, arrays: dict[str, np.ndarray],
                      manifest: dict[str, Any]) -> str:
    """Write a new map directory (must not exist) and return its map_id."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    cameras = tuple(manifest["cameras"])
    _validate_arrays(arrays, cameras)
    arrays_path = root / ARRAYS_NAME
    with arrays_path.open("xb") as handle:
        np.savez(handle, **arrays)
    digest = _sha256(arrays_path)
    full = {**manifest, "format": FORMAT, "map_npz_sha256": digest}
    full["map_id"] = _map_id(full)
    (root / MANIFEST_NAME).write_text(json.dumps(full, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    return full["map_id"]

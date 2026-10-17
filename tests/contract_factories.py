"""Seeded helpers for the shared-contracts tests (imported by tests only, never by camloc).

``tests/`` has no ``__init__.py``; pytest's default prepend import mode puts this directory on
``sys.path``, so test modules use ``import contract_factories``. The helpers here are geometry
primitives, the read-only organizer loader and valid record builders. Each builder takes
keyword overrides of valid defaults, so a test can break exactly one field.
"""

import functools
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from camloc import contracts as c

ROOT = Path(__file__).resolve().parents[1]
ORGANIZER_TRACK_FRAME = ROOT / "23-00-38_cut_120m_update" / "tools" / "track_frame.py"


def seeded_rng(seed: int, *stream: int) -> np.random.Generator:
    """Return a reproducible generator for a fixed test seed and optional sub-stream ids.

    Each test passes its own fixed seed. A property loop can draw iteration ``k`` from
    ``seeded_rng(seed, k)``, so a failure reported as ``(seed, k)`` replays on its own.
    """
    return np.random.default_rng([seed, *stream])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed 3 x 3 rotation matrix (orthonormal, determinant +1).

    The QR decomposition of a standard normal matrix gives an orthonormal Q. Scaling its
    columns by the signs of diag(R) makes Q uniform over O(3); negating one column when
    det(Q) < 0 (the determinant fix) maps reflections onto rotations.
    """
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.where(np.diag(r) < 0.0, -1.0, 1.0)
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def rigid(rotation: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Return the float64 4 x 4 transform T_AB = [[R, t], [0, 0, 0, 1]], so p_A = R @ p_B + t."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rigid_inverse(transform: ArrayLike) -> np.ndarray:
    """Return inv(T_AB) = T_BA in closed form: [[R.T, -R.T @ t], [0, 0, 0, 1]]."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a (4, 4) transform, got shape {matrix.shape}")
    rotation_t = matrix[:3, :3].T
    return rigid(rotation_t, -rotation_t @ matrix[:3, 3])


@functools.cache
def load_organizer_track_frame() -> ModuleType:
    """Return the unchanged organizer ``track_frame`` helper, loaded read-only.

    The file runs from its own path under a private module name (not added to
    ``sys.modules``). Bytecode writing is disabled while it loads, so no ``__pycache__``
    appears in the protected organizer folder even when pytest runs without ``-B``.
    """
    spec = importlib.util.spec_from_file_location("organizer_track_frame", ORGANIZER_TRACK_FRAME)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ORGANIZER_TRACK_FRAME}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


# ---------------------------------------------------------------------------------------
# Input records: ImageMessage, InputEvent, CameraCalibration, Observation, FrameSet
# ---------------------------------------------------------------------------------------

EXPOSURE_NS = 1_791_918_859_614_207_351  # 19-digit header stamp, 109 ms into the tune episode
ARRIVAL_DELAY_NS = 42_000_000  # MCAP log time minus exposure of a default image
GROUP_EXPOSURE_STEP_NS = 500_000  # exposure step between consecutive images of a group
PREPROCESSING_ID = "canonical_960x464_v1"
JPEG_PAYLOAD = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9"  # opaque: grouping never decodes it
RAW_WIDTH = 1506
RAW_HEIGHT = 728
RAW_DISTORTION = (-0.0512, 0.0127, 0.0003, -0.0002, 0.0)  # plumb_bob k1, k2, p1, p2, k3
# R_BC of an optical frame (z forward, x right, y down) looking along body x (x forward,
# y left, z up): camera z -> body x, camera x -> body -y, camera y -> body -z; det = +1.
OPTICAL_TO_BODY = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
CAMERA_TRANSLATION_M = (1.25, -0.30, 0.95)  # camera origin in body coordinates


def canonical_K(
    fx: float = 860.0, fy: float = 860.0, cx: float = 479.5, cy: float = 231.5
) -> np.ndarray:
    """Return a writable float64 canonical K (zero skew, last row [0, 0, 1])."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def raw_K() -> np.ndarray:
    """Return the raw 1506 x 728 K that the module's resize formulas map onto ``canonical_K()``.

    Inverts fx' = sx*fx and cx' = sx*(cx + 0.5) - 0.5 (likewise for y) with
    sx = 960/1506 and sy = 464/728.
    """
    canonical = canonical_K()
    sx = c.CANONICAL_WIDTH / RAW_WIDTH
    sy = c.CANONICAL_HEIGHT / RAW_HEIGHT
    return canonical_K(
        fx=canonical[0, 0] / sx,
        fy=canonical[1, 1] / sy,
        cx=(canonical[0, 2] + 0.5) / sx - 0.5,
        cy=(canonical[1, 2] + 0.5) / sy - 0.5,
    )


def camera_to_body() -> np.ndarray:
    """Return a writable rigid T_BC of a forward-looking camera (maps camera into body)."""
    return rigid(OPTICAL_TO_BODY, CAMERA_TRANSLATION_M)


def canonical_image() -> np.ndarray:
    """Return a fresh, writable, C-contiguous RGB uint8 (464, 960, 3) image of zeros."""
    return np.zeros((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3), dtype=np.uint8)


def canonical_mask() -> np.ndarray:
    """Return a fresh, writable, C-contiguous bool (464, 960) mask, all pixels usable."""
    return np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), dtype=np.bool_)


def make_image_message(**overrides: Any) -> c.ImageMessage:
    """Return a valid ImageMessage of camera ``fr``; keyword arguments replace fields.

    Unless overridden, ``arrival_ns`` is ``exposure_ns + ARRIVAL_DELAY_NS`` and
    ``source_id``, ``frame_id`` and ``calibration_id`` are derived from the camera (and the
    exposure for ``source_id``), so several cameras or exposures give distinct messages.
    """
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "exposure_ns": EXPOSURE_NS,
        "image_format": "jpeg",
        "data": JPEG_PAYLOAD,
    }
    fields.update(overrides)
    camera, exposure = fields["camera_id"], fields["exposure_ns"]
    if "arrival_ns" not in fields:
        fields["arrival_ns"] = exposure + ARRIVAL_DELAY_NS
    if "source_id" not in fields:
        fields["source_id"] = f"{camera}@{exposure}"
    if "frame_id" not in fields:
        fields["frame_id"] = f"{c.CAMERA_FRAME_PREFIX}{camera}"
    if "calibration_id" not in fields:
        fields["calibration_id"] = f"calibration_{camera}_v1"
    return c.ImageMessage(**fields)


def make_input_event(**overrides: Any) -> c.InputEvent:
    """Return a valid ``late_support`` InputEvent logged at the default arrival time."""
    fields: dict[str, Any] = {
        "kind": c.InputEventKind.LATE_SUPPORT,
        "log_ns": EXPOSURE_NS + ARRIVAL_DELAY_NS,
    }
    fields.update(overrides)
    return c.InputEvent(**fields)


def make_calibration(**overrides: Any) -> c.CameraCalibration:
    """Return a valid CameraCalibration of camera ``fr``; keyword arguments replace fields."""
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "preprocessing_id": PREPROCESSING_ID,
        "raw_width": RAW_WIDTH,
        "raw_height": RAW_HEIGHT,
        "D_raw": np.array(RAW_DISTORTION),
    }
    fields.update(overrides)
    camera = fields["camera_id"]
    if "calibration_id" not in fields:
        fields["calibration_id"] = f"calibration_{camera}_v1"
    if "frame_id" not in fields:
        fields["frame_id"] = f"{c.CAMERA_FRAME_PREFIX}{camera}"
    if "K_raw" not in fields:
        fields["K_raw"] = raw_K()
    if "K" not in fields:
        fields["K"] = canonical_K()
    if "T_BC" not in fields:
        fields["T_BC"] = camera_to_body()
    return c.CameraCalibration(**fields)


def make_observation(**overrides: Any) -> c.Observation:
    """Return a valid Observation of camera ``fr``; keyword arguments replace fields.

    Fresh image and mask buffers are allocated only when not overridden, and become
    read-only once the record is valid. Identities follow ``make_image_message``.
    """
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "exposure_ns": EXPOSURE_NS,
        "preprocessing_id": PREPROCESSING_ID,
    }
    fields.update(overrides)
    camera, exposure = fields["camera_id"], fields["exposure_ns"]
    if "arrival_ns" not in fields:
        fields["arrival_ns"] = exposure + ARRIVAL_DELAY_NS
    if "source_id" not in fields:
        fields["source_id"] = f"{camera}@{exposure}"
    if "calibration_id" not in fields:
        fields["calibration_id"] = f"calibration_{camera}_v1"
    if "image" not in fields:
        fields["image"] = canonical_image()
    if "valid_mask" not in fields:
        fields["valid_mask"] = canonical_mask()
    if "K" not in fields:
        fields["K"] = canonical_K()
    if "T_BC" not in fields:
        fields["T_BC"] = camera_to_body()
    return c.Observation(**fields)


def make_frame_set(present: Sequence[str] | None = None, **overrides: Any) -> c.FrameSet:
    """Return a valid FrameSet (group 0 of cameras ``fr`` and ``r``); overrides replace fields.

    ``present`` lists the cameras that get an image (default: every selected camera). The
    k-th of them is exposed at ``EXPOSURE_NS + k * GROUP_EXPOSURE_STEP_NS`` and arrives
    ``ARRIVAL_DELAY_NS`` later. Unless overridden, ``t_ns`` is the maximum exposure,
    ``missing_camera_ids`` the selected cameras without an image, ``closure`` is
    ``complete`` when nothing is missing and ``deadline`` otherwise, and
    ``terminal_partial`` is True exactly for an ``eof`` closure. Only ImageMessage items
    of an overridden ``images`` feed these derivations, so a malformed item still reaches
    the record's own checks.
    """
    fields: dict[str, Any] = {
        "group_id": 0,
        "selected_camera_ids": (c.PRIMARY_CAMERA, c.SECONDARY_CAMERA),
        "input_events": (),
    }
    fields.update(overrides)
    selected = fields["selected_camera_ids"]
    if "images" not in fields:
        cameras = selected if present is None else present
        fields["images"] = tuple(
            make_image_message(
                camera_id=camera, exposure_ns=EXPOSURE_NS + k * GROUP_EXPOSURE_STEP_NS
            )
            for k, camera in enumerate(cameras)
        )
    images = [image for image in fields["images"] if isinstance(image, c.ImageMessage)]
    if "t_ns" not in fields:
        fields["t_ns"] = max((image.exposure_ns for image in images), default=EXPOSURE_NS)
    if "missing_camera_ids" not in fields:
        cameras_with_image = {image.camera_id for image in images}
        fields["missing_camera_ids"] = tuple(
            camera for camera in selected if camera not in cameras_with_image
        )
    if "closure" not in fields:
        complete = not fields["missing_camera_ids"]
        fields["closure"] = c.FrameSetClosure.COMPLETE if complete else c.FrameSetClosure.DEADLINE
    if "terminal_partial" not in fields:
        fields["terminal_partial"] = fields["closure"] == c.FrameSetClosure.EOF
    return c.FrameSet(**fields)


# ---------------------------------------------------------------------------------------
# Evidence, anchor and group records. Defaults line up with ``make_frame_set()``: camera
# ``fr`` measured at EXPOSURE_NS, the group's own ``fr`` image, so they can be consumed.
# ---------------------------------------------------------------------------------------

MAP_ID = "track_map_v1"
JOB_ID = 7
HANDOFF_AGE_NS = 100_000_000  # source-to-current age of a default handoff (two transitions)
MOTION_STEP_NS = 50_000_000  # previous-to-current step of a default ground motion


def make_correspondences(**overrides: Any) -> c.Correspondences:
    """Return valid three-row Correspondences on the ``fr`` image at EXPOSURE_NS."""
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "exposure_ns": EXPOSURE_NS,
        "keypoint_ids": np.array([0, 1, 2]),
        "pixels": np.array([[100.0, 100.0], [480.0, 240.0], [800.0, 400.0]]),
        "landmark_ids": np.array([10, 11, 12]),
        "map_id": MAP_ID,
    }
    fields.update(overrides)
    return c.Correspondences(**fields)


def make_pose_evidence(**overrides: Any) -> c.PoseEvidence:
    """Return an accepted ``tracking`` PoseEvidence of camera ``fr`` with three inliers.

    ``T_WB`` is the identity. Unless overridden, ``calibration_id`` follows the camera.
    Pass ``accepted=False`` with a failure ``reason`` for a rejected record.
    """
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "exposure_ns": EXPOSURE_NS,
        "source": c.PoseSource.TRACKING,
        "T_WB": np.eye(4),
        "inlier_count": 3,
        "inlier_fraction": 0.75,
        "median_error_px": 0.8,
        "p90_error_px": 1.6,
        "occupied_cells": 3,
        "track_ids": np.array([0, 1, 2]),
        "landmark_ids": np.array([10, 11, 12]),
        "map_id": MAP_ID,
        "preprocessing_id": PREPROCESSING_ID,
        "accepted": True,
        "reason": "accepted",
    }
    fields.update(overrides)
    if "calibration_id" not in fields:
        fields["calibration_id"] = f"calibration_{fields['camera_id']}_v1"
    return c.PoseEvidence(**fields)


def make_motion_evidence(**overrides: Any) -> c.MotionEvidence:
    """Return accepted planar ground motion of ``fr`` ending at EXPOSURE_NS (1.5 m forward).

    Unless overridden, ``previous_ns`` is ``current_ns - MOTION_STEP_NS``.
    """
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "current_ns": EXPOSURE_NS,
        "T_Bprev_Bcurr": rigid(np.eye(3), (1.5, 0.0, 0.0)),
        "covariance": np.diag([0.01, 0.01, 0.001]),
        "inlier_count": 20,
        "inlier_fraction": 0.8,
        "ground_residual_m": 0.02,
        "road_patch_id": "road_patch_0",
        "accepted": True,
        "reason": "accepted",
    }
    fields.update(overrides)
    if "previous_ns" not in fields:
        fields["previous_ns"] = fields["current_ns"] - MOTION_STEP_NS
    return c.MotionEvidence(**fields)


def make_border_evidence(**overrides: Any) -> c.BorderEvidence:
    """Return accepted ``track`` BorderEvidence of ``fr`` at EXPOSURE_NS (prior yaw 0)."""
    fields: dict[str, Any] = {
        "camera_id": c.PRIMARY_CAMERA,
        "exposure_ns": EXPOSURE_NS,
        "branch": c.Branch.TRACK,
        "normal_xy": np.array([0.0, 1.0]),
        "lateral_coordinate_m": 0.4,
        "yaw_rad": 0.01,
        "R": np.diag([0.04, 0.0004]),
        "support_samples": 120,
        "support_contours": 2,
        "conditioning": 0.3,
        "accepted": True,
        "reason": "accepted",
    }
    fields.update(overrides)
    return c.BorderEvidence(**fields)


def make_handoff_evidence(**overrides: Any) -> c.HandoffEvidence:
    """Return an accepted two-transition HandoffEvidence of ``fr`` measured at EXPOSURE_NS.

    Unless overridden, ``source_exposure_ns`` is ``measurement_ns - HANDOFF_AGE_NS``, the
    chain runs source, midpoint, measurement, and ``correspondences`` and the ``handoff``
    pose are built on the current image of the camera with matching identities.
    """
    fields: dict[str, Any] = {
        "job_id": JOB_ID,
        "camera_id": c.PRIMARY_CAMERA,
        "measurement_ns": EXPOSURE_NS,
        "map_id": MAP_ID,
        "preprocessing_id": PREPROCESSING_ID,
        "accepted": True,
        "reason": "accepted",
    }
    fields.update(overrides)
    camera, measurement = fields["camera_id"], fields["measurement_ns"]
    if "source_exposure_ns" not in fields:
        fields["source_exposure_ns"] = measurement - HANDOFF_AGE_NS
    source = fields["source_exposure_ns"]
    if "support_ns" not in fields:
        fields["support_ns"] = (source, (source + measurement) // 2, measurement)
    if "calibration_id" not in fields:
        fields["calibration_id"] = f"calibration_{camera}_v1"
    if "correspondences" not in fields:
        fields["correspondences"] = make_correspondences(
            camera_id=camera, exposure_ns=measurement, map_id=fields["map_id"]
        )
    if "pose" not in fields:
        fields["pose"] = make_pose_evidence(
            camera_id=camera,
            exposure_ns=measurement,
            source=c.PoseSource.HANDOFF,
            map_id=fields["map_id"],
            preprocessing_id=fields["preprocessing_id"],
            calibration_id=fields["calibration_id"],
        )
    return c.HandoffEvidence(**fields)


def make_search_snapshot(**overrides: Any) -> c.SearchSnapshot:
    """Return a valid ``tracking`` SearchSnapshot at EXPOSURE_NS without auxiliary geometry."""
    fields: dict[str, Any] = {
        "t_ns": EXPOSURE_NS,
        "state": np.array([120.0, -35.0, 0.25, 18.0, 0.1, 0.02]),
        "P": np.diag([0.25, 0.25, 0.01, 0.5, 0.5, 0.05]),
        "status": c.Status.TRACKING,
        "branch": c.Branch.TRACK,
        "auxiliary": c.AuxiliaryGeometry.unavailable("not_estimated"),
    }
    fields.update(overrides)
    return c.SearchSnapshot(**fields)


def make_anchor_request(**overrides: Any) -> c.AnchorRequest:
    """Return a valid ``front_anchor`` AnchorRequest (job JOB_ID) on the ``fr`` image of group 0.

    Unless overridden, ``delivery_not_before_ns`` is the earliest allowed
    (``source_exposure_ns + DELIVERY_MIN_DELAY_NS``), and ``observation`` and ``snapshot``
    are built for the request's camera at ``source_exposure_ns``.
    """
    fields: dict[str, Any] = {
        "job_id": JOB_ID,
        "purpose": c.AnchorPurpose.FRONT_ANCHOR,
        "camera_id": c.PRIMARY_CAMERA,
        "source_group_id": 0,
        "source_exposure_ns": EXPOSURE_NS,
        "reference_ids": (3, 8),
        "map_id": MAP_ID,
        "preprocessing_id": PREPROCESSING_ID,
    }
    fields.update(overrides)
    source = fields["source_exposure_ns"]
    if "delivery_not_before_ns" not in fields:
        fields["delivery_not_before_ns"] = source + c.DELIVERY_MIN_DELAY_NS
    if "observation" not in fields:
        fields["observation"] = make_observation(camera_id=fields["camera_id"], exposure_ns=source)
    if "snapshot" not in fields:
        fields["snapshot"] = make_search_snapshot(t_ns=source)
    return c.AnchorRequest(**fields)


def make_anchor_result(**overrides: Any) -> c.AnchorResult:
    """Return an ``ok`` AnchorResult answering ``make_anchor_request()``.

    Unless overridden, ``reason`` is "ok" for status ``ok`` and "insufficient_matches"
    otherwise, and ``correspondences`` lie on the source image for status ``ok`` and are
    None otherwise.
    """
    fields: dict[str, Any] = {
        "job_id": JOB_ID,
        "purpose": c.AnchorPurpose.FRONT_ANCHOR,
        "camera_id": c.PRIMARY_CAMERA,
        "source_group_id": 0,
        "source_exposure_ns": EXPOSURE_NS,
        "status": c.AnchorStatus.OK,
        "matched_reference_ids": (3,),
        "remaining_candidate_ids": (8,),
        "map_id": MAP_ID,
        "preprocessing_id": PREPROCESSING_ID,
        "model_ids": ("disk_v1", "lightglue_v1"),
        "worker_duration_ns": 35_000_000,
    }
    fields.update(overrides)
    ok = fields["status"] == c.AnchorStatus.OK
    if "reason" not in fields:
        fields["reason"] = "ok" if ok else "insufficient_matches"
    if "correspondences" not in fields:
        fields["correspondences"] = (
            make_correspondences(
                camera_id=fields["camera_id"],
                exposure_ns=fields["source_exposure_ns"],
                map_id=fields["map_id"],
            )
            if ok
            else None
        )
    return c.AnchorResult(**fields)


def make_recovery_reset(**overrides: Any) -> c.RecoveryReset:
    """Return a valid RecoveryReset confirmed on the ``fr`` image of ``make_frame_set()``.

    The ``candidate`` pose is exposed RECOVERY_CONFIRM_MIN_NS before the ``confirmation``
    (both source ``candidate``; confirmation at EXPOSURE_NS). Unless overridden, ``t_ns``
    is the confirmation exposure plus GROUP_EXPOSURE_STEP_NS, the default group time.
    """
    fields: dict[str, Any] = {
        "reset_id": 0,
        "group_id": 0,
        "jump_m": 4.5,
        "reason": "confirmed_recovery",
    }
    fields.update(overrides)
    if "confirmation" not in fields:
        fields["confirmation"] = make_pose_evidence(source=c.PoseSource.CANDIDATE)
    confirmation = fields["confirmation"]
    if "candidate" not in fields:
        fields["candidate"] = make_pose_evidence(
            source=c.PoseSource.CANDIDATE,
            exposure_ns=EXPOSURE_NS - c.RECOVERY_CONFIRM_MIN_NS,
        )
    if "t_ns" not in fields:
        fields["t_ns"] = (
            confirmation.exposure_ns + GROUP_EXPOSURE_STEP_NS
            if isinstance(confirmation, c.PoseEvidence)
            else EXPOSURE_NS + GROUP_EXPOSURE_STEP_NS
        )
    return c.RecoveryReset(**fields)


def make_unused_evidence(**overrides: Any) -> c.UnusedEvidence:
    """Return UnusedEvidence of an accepted default border ("superseded_by_pose")."""
    fields: dict[str, Any] = {"reason": "superseded_by_pose"}
    fields.update(overrides)
    if "evidence" not in fields:
        fields["evidence"] = make_border_evidence()
    return c.UnusedEvidence(**fields)


def make_group_evidence(**overrides: Any) -> c.GroupEvidence:
    """Return a GroupEvidence over ``make_frame_set()`` with every slot empty (prediction only)."""
    fields: dict[str, Any] = {
        "pose": None,
        "motion": None,
        "border": None,
        "reset": None,
        "rejected": (),
        "unused": (),
    }
    fields.update(overrides)
    if "frame_set" not in fields:
        fields["frame_set"] = make_frame_set()
    return c.GroupEvidence(**fields)

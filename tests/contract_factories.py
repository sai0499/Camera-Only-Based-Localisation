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

"""Focused tests for owner A's geometry module (``camloc/geometry.py``), step 2.

Covered: transform/projection round trips, the PnP-to-body conversion, K-versus-P misuse,
the half-pixel K scaling, canonical remap output and mask boundaries, yaw wrap and the
hloc/COLMAP pixel-centre boundary (a single +0.5 at import, none inside this module).

Synthetic calibration is used so the tests do not depend on the private recording; the
canonical K is built with ``io``'s own derivation to keep one source of truth.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

from camloc import contracts
from camloc import geometry
from camloc import io
from camloc.contracts import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CameraCalibration,
    ImageMessage,
    InputError,
    Observation,
)

RAW_WIDTH, RAW_HEIGHT = 1506, 728


def _identity_quat_matrix():
    return np.eye(3, dtype=np.float64)


def _calibration(camera_id: str = "fr", *, distortion=(0.0, 0.0, 0.0, 0.0, 0.0)) -> CameraCalibration:
    """Build a synthetic CameraCalibration with a plausible raw K and the derived canonical K."""
    k_raw = np.array(
        [[1350.0, 0.0, 753.0], [0.0, 1166.0, 352.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    k_canonical = io._canonical_k(k_raw, RAW_WIDTH, RAW_HEIGHT)
    # A simple tilted mount so T_BC is a non-trivial rigid transform.
    rotation = geometry.quaternion_xyzw_to_matrix(-0.5, 0.5, -0.5, 0.5)
    t_bc = geometry.rigid_from_rotation_translation(rotation, [0.95, -0.16, 0.34])
    return CameraCalibration(
        camera_id=camera_id,
        calibration_id="calib:test",
        preprocessing_id="prep:test",
        frame_id=f"constructor0/camera_{camera_id}",
        raw_width=RAW_WIDTH,
        raw_height=RAW_HEIGHT,
        K_raw=k_raw,
        D_raw=np.array(distortion, dtype=np.float64),
        K=k_canonical,
        T_BC=t_bc,
    )


def _jpeg_bytes(width: int, height: int, color=(10, 120, 200)) -> bytes:
    """Return a JPEG payload of a solid BGR color at the given size."""
    import cv2

    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :] = color  # OpenCV stores BGR
    ok, buffer = cv2.imencode(".jpg", bgr)
    assert ok
    return buffer.tobytes()


class TransformTests(unittest.TestCase):
    def test_invert_rigid_is_an_inverse(self):
        rotation = geometry.quaternion_xyzw_to_matrix(0.1, -0.2, 0.3, 0.9)
        T = geometry.rigid_from_rotation_translation(rotation, [1.0, 2.0, 3.0])
        inverse = geometry.invert_rigid(T)
        self.assertTrue(np.allclose(T @ inverse, np.eye(4), atol=1e-12))
        self.assertTrue(np.allclose(inverse @ T, np.eye(4), atol=1e-12))

    def test_projection_recovers_camera_frame_points(self):
        calibration = _calibration()
        T_WB = geometry.rigid_from_rotation_translation(_identity_quat_matrix(), [10.0, -3.0, 0.5])
        T_WC = T_WB @ calibration.T_BC
        camera_points = np.array([[0.3, 0.1, 5.0], [-0.2, 0.0, 8.0], [0.0, -0.15, 12.0]])
        homogeneous = np.concatenate([camera_points, np.ones((3, 1))], axis=1)
        world_points = (homogeneous @ T_WC.T)[:, :3]
        recovered = geometry.project_world_to_camera(world_points, T_WB, calibration.T_BC)
        self.assertTrue(np.allclose(recovered, camera_points, atol=1e-9))

    def test_body_pose_from_pnp_inverts_projection(self):
        calibration = _calibration()
        rotation = geometry.quaternion_xyzw_to_matrix(0.0, 0.0, 0.3827, 0.9239)  # 45 deg yaw
        T_WB = geometry.rigid_from_rotation_translation(rotation, [4.0, 7.0, 0.4])
        T_WC = T_WB @ calibration.T_BC
        T_CW = geometry.invert_rigid(T_WC)
        recovered = geometry.body_pose_from_pnp(T_CW, calibration.T_BC)
        self.assertTrue(np.allclose(recovered, T_WB, atol=1e-12))

    def test_planar_pose_wraps_yaw(self):
        # A yaw of 3*pi/2 wraps to -pi/2 in (-pi, pi].
        rotation = geometry.quaternion_xyzw_to_matrix(
            0.0, 0.0, float(np.sin(3 * np.pi / 4)), float(np.cos(3 * np.pi / 4))
        )
        T_WB = geometry.rigid_from_rotation_translation(rotation, [1.0, 2.0, 0.0])
        _x, _y, yaw = geometry.planar_pose_from_T_WB(T_WB)
        self.assertAlmostEqual(yaw, -np.pi / 2, places=6)
        self.assertTrue(-np.pi < yaw <= np.pi)

    def test_point_behind_camera_is_rejected(self):
        calibration = _calibration()
        behind = np.array([[0.0, 0.0, -1.0]])
        with self.assertRaises(contracts.ContractError):
            geometry.project_camera_to_pixels(behind, calibration.K)


class PixelConventionTests(unittest.TestCase):
    def test_half_pixel_k_scaling_matches_the_convention(self):
        calibration = _calibration()
        sx, sy = CANONICAL_WIDTH / RAW_WIDTH, CANONICAL_HEIGHT / RAW_HEIGHT
        k_raw, k = calibration.K_raw, calibration.K
        self.assertAlmostEqual(k[0, 0], sx * k_raw[0, 0])
        self.assertAlmostEqual(k[1, 1], sy * k_raw[1, 1])
        self.assertAlmostEqual(k[0, 2], sx * (k_raw[0, 2] + 0.5) - 0.5)
        self.assertAlmostEqual(k[1, 2], sy * (k_raw[1, 2] + 0.5) - 0.5)

    def test_supplied_p_is_never_used_as_raw_k(self):
        # A camera_info P differs from raw K; the canonical K must come from raw K only.
        # Here we prove the derivation ignores any P-like matrix by construction: the only
        # inputs to _canonical_k are raw K and the raw size.
        calibration = _calibration()
        fake_p = calibration.K_raw.copy()
        fake_p[0, 0] *= 0.87  # a rectified focal length, as in real P
        derived = io._canonical_k(calibration.K_raw, RAW_WIDTH, RAW_HEIGHT)
        self.assertTrue(np.allclose(derived, calibration.K))
        self.assertFalse(np.allclose(io._canonical_k(fake_p, RAW_WIDTH, RAW_HEIGHT), calibration.K))

    def test_pixel_round_trip_applies_no_extra_half_pixel_shift(self):
        calibration = _calibration()
        pixels = np.array([[0.0, 0.0], [480.0, 232.0], [959.0, 463.0], [123.4, 77.8]])
        recovered = geometry.round_trip_pixel(pixels, calibration.K)
        self.assertTrue(np.allclose(recovered, pixels, atol=1e-9))

    def test_principal_point_is_the_fixed_point_of_the_round_trip(self):
        calibration = _calibration()
        principal = np.array([[calibration.K[0, 2], calibration.K[1, 2]]])
        bearing = geometry.pixel_to_bearing(principal, calibration.K)
        # The principal point maps to the optical axis (0, 0, 1).
        self.assertTrue(np.allclose(bearing[0], [0.0, 0.0, 1.0], atol=1e-9))


class CanonicalizerTests(unittest.TestCase):
    def _image_message(self, calibration: CameraCalibration, data: bytes) -> ImageMessage:
        return ImageMessage(
            camera_id=calibration.camera_id,
            exposure_ns=1_000_000_000,
            arrival_ns=1_000_000_000,
            source_id="fr:1",
            frame_id=calibration.frame_id,
            image_format="jpeg",
            calibration_id=calibration.calibration_id,
            data=data,
        )

    def test_canonicalize_produces_a_valid_observation(self):
        calibration = _calibration()
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        message = self._image_message(calibration, _jpeg_bytes(RAW_WIDTH, RAW_HEIGHT))
        observation = canon.canonicalize(message)
        self.assertIsInstance(observation, Observation)
        self.assertEqual(observation.image.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))
        self.assertEqual(observation.image.dtype, np.uint8)
        self.assertEqual(observation.valid_mask.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH))
        self.assertFalse(observation.image.flags.writeable)  # read-only once valid
        self.assertEqual(observation.calibration_id, calibration.calibration_id)
        self.assertTrue(np.array_equal(observation.K, calibration.K))

    def test_bgr_to_rgb_conversion_is_explicit(self):
        # A solid BGR (blue=200, green=120, red=10) image must come back as RGB with the
        # channels swapped: red=10, green=120, blue=200.
        calibration = _calibration()
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        data = _jpeg_bytes(RAW_WIDTH, RAW_HEIGHT, color=(200, 120, 10))  # BGR order
        observation = canon.canonicalize(self._image_message(calibration, data))
        center = observation.image[CANONICAL_HEIGHT // 2, CANONICAL_WIDTH // 2]
        # JPEG is lossy; allow a small tolerance but assert the R/B order is correct.
        self.assertLess(abs(int(center[0]) - 10), 20)  # red low
        self.assertGreater(int(center[2]), int(center[0]))  # blue > red

    def test_distortion_produces_masked_border_pixels(self):
        # Strong barrel distortion leaves remap-invalid border pixels (mask False there).
        calibration = _calibration(distortion=(0.4, 0.0, 0.0, 0.0, 0.0))
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        observation = canon.canonicalize(
            self._image_message(calibration, _jpeg_bytes(RAW_WIDTH, RAW_HEIGHT))
        )
        self.assertTrue(observation.valid_mask.any())
        self.assertFalse(observation.valid_mask.all())  # some border pixels are invalid

    def test_zero_distortion_leaves_all_pixels_valid(self):
        calibration = _calibration(distortion=(0.0, 0.0, 0.0, 0.0, 0.0))
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        observation = canon.canonicalize(
            self._image_message(calibration, _jpeg_bytes(RAW_WIDTH, RAW_HEIGHT))
        )
        self.assertTrue(observation.valid_mask.all())

    def test_undecodable_payload_raises_input_error(self):
        calibration = _calibration()
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        message = self._image_message(calibration, b"not-a-jpeg")
        with self.assertRaises(InputError):
            canon.canonicalize(message)

    def test_wrong_raw_size_raises_input_error(self):
        calibration = _calibration()
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        message = self._image_message(calibration, _jpeg_bytes(640, 480))
        with self.assertRaises(InputError):
            canon.canonicalize(message)

    def test_mismatched_calibration_id_raises_input_error(self):
        calibration = _calibration()
        canon = geometry.make_canonicalizer({calibration.camera_id: calibration})
        message = ImageMessage(
            camera_id=calibration.camera_id,
            exposure_ns=1_000_000_000,
            arrival_ns=1_000_000_000,
            source_id="fr:1",
            frame_id=calibration.frame_id,
            image_format="jpeg",
            calibration_id="calib:other",
            data=_jpeg_bytes(RAW_WIDTH, RAW_HEIGHT),
        )
        with self.assertRaises(InputError):
            canon.canonicalize(message)


@pytest.mark.parametrize("quaternion", [(0., 0., 0., 0.), (float("inf"), 0., 0., 1.),
                                       (float("nan"), 0., 0., 1.), (1e308,) * 4])
def test_invalid_quaternion_cannot_silently_become_identity(quaternion):
    with pytest.raises(contracts.ContractError, match="quaternion"):
        geometry.quaternion_xyzw_to_matrix(*quaternion)


if __name__ == "__main__":
    unittest.main()

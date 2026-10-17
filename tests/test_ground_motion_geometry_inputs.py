"""Synthetic examples of the geometry C must pass to B; no ground-motion backend here."""

import inspect
import math

import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c


def _rotation(yaw, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _snapshot(exposure_ns, body_pose, patch):
    r = body_pose[:3, :3]
    yaw = math.atan2(r[1, 0], r[0, 0])
    return f.make_search_snapshot(
        t_ns=exposure_ns,
        state=[body_pose[0, 3], body_pose[1, 3], yaw, 0.0, 0.0, 0.0],
        auxiliary=c.AuxiliaryGeometry(
            source=c.AuxSource.ROAD_PATCH,
            source_ns=exposure_ns,
            z_m=body_pose[2, 3],
            roll_rad=math.atan2(r[2, 1], r[2, 2]),
            pitch_rad=math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])),
            road_patch_id=patch.patch_id,
            valid=True,
            reason="valid",
        ),
    )


def _body_pose(snapshot):
    aux = c.require_accepted(snapshot.auxiliary)
    return f.rigid(
        _rotation(snapshot.state[2], aux.pitch_rad, aux.roll_rad),
        [snapshot.state[0], snapshot.state[1], aux.z_m],
    )


def _intersect_road(pixels, observation, snapshot, patch):
    camera_pose = _body_pose(snapshot) @ observation.T_BC
    rays_c = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(observation.K).T
    rays_w = rays_c @ camera_pose[:3, :3].T
    origin = camera_pose[:3, 3]
    depths = ((patch.point_W - origin) @ patch.normal_W) / (rays_w @ patch.normal_W)
    assert np.all((3.0 <= depths) & (depths <= 30.0))
    return origin + depths[:, None] * rays_w


def _patch(point, normal):
    return c.RoadPatch(
        patch_id="synthetic_road",
        branch=c.Branch.TRACK,
        query_xy=np.asarray(point)[:2],
        point_W=point,
        normal_W=normal,
        support=40,
        residual_m=0.0,
        valid=True,
        reason="valid",
    )


def test_exposure_geometry_supplies_scale_on_a_sloped_road_with_translation_and_yaw():
    surface_rotation = _rotation(0.2, pitch=-0.12, roll=0.05)
    origin = np.array([100.0, -35.0, 2.0])
    normal = surface_rotation[:, 2]
    patch = _patch(origin, normal)
    body_height_m = 0.4
    previous_pose = f.rigid(surface_rotation, origin + body_height_m * normal)
    delta = f.rigid(_rotation(0.04), [1.2, 0.08, 0.0])
    current_pose = previous_pose @ delta
    previous = f.make_observation(exposure_ns=f.EXPOSURE_NS - f.MOTION_STEP_NS)
    current = f.make_observation()
    previous_snapshot = _snapshot(previous.exposure_ns, previous_pose, patch)
    current_snapshot = _snapshot(current.exposure_ns, current_pose, patch)
    inputs = inspect.signature(c.GroundMotionEstimator.estimate).bind(
        None, previous, current, patch,
        previous_snapshot=previous_snapshot,
        current_snapshot=current_snapshot,
    ).arguments

    points_b = np.array([[x, y, -body_height_m] for x in (9.0, 12.0, 15.0) for y in (-1.5, 0, 1.5)])
    points_w = points_b @ previous_pose[:3, :3].T + previous_pose[:3, 3]
    reconstructed_body = []
    for key, expected_pose in (("previous", previous_pose), ("current", current_pose)):
        observation, snapshot = inputs[key], inputs[f"{key}_snapshot"]
        assert snapshot.t_ns == observation.exposure_ns
        np.testing.assert_allclose(_body_pose(snapshot), expected_pose, atol=1e-12)
        camera_pose = expected_pose @ observation.T_BC
        points_c = (points_w - camera_pose[:3, 3]) @ camera_pose[:3, :3]
        pixels_h = points_c @ observation.K.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        assert np.all((0 <= pixels) & (pixels < [c.CANONICAL_WIDTH, c.CANONICAL_HEIGHT]))
        recovered = _intersect_road(pixels, observation, snapshot, patch)
        np.testing.assert_allclose(recovered, points_w, atol=1e-10)
        reconstructed_body.append((recovered - expected_pose[:3, 3]) @ expected_pose[:3, :3])

    # The required delta maps current-body points into the previous body, not vice versa.
    np.testing.assert_allclose(
        reconstructed_body[1] @ delta[:3, :3].T + delta[:3, 3],
        reconstructed_body[0], atol=1e-10,
    )
    motion = f.make_motion_evidence(T_Bprev_Bcurr=delta, road_patch_id=patch.patch_id)
    assert motion.accepted and motion.T_Bprev_Bcurr[0, 3] == pytest.approx(1.2)


def test_same_road_pixels_have_different_metric_scale_for_different_body_heights():
    observation = f.make_observation()
    patch = _patch([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    pixels = np.array([[480.0, 403.5], [500.0, 403.5]])
    distances = []
    heights = (0.25, 1.25)
    for body_height_m in heights:
        snapshot = _snapshot(f.EXPOSURE_NS, f.rigid(np.eye(3), [0, 0, body_height_m]), patch)
        points = _intersect_road(pixels, observation, snapshot, patch)
        distances.append(np.linalg.norm(points[1] - points[0]))
    camera_mount_height = observation.T_BC[2, 3]
    expected_ratio = (heights[1] + camera_mount_height) / (heights[0] + camera_mount_height)
    assert distances[1] / distances[0] == pytest.approx(expected_ratio)
    assert distances[1] != pytest.approx(distances[0])

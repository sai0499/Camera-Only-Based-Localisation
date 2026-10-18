"""B's ground motion on a rendered textured road plane with known body motion."""

import math

import cv2
import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c
from camloc import geometry as g
from camloc import vision
from camloc.ground_motion import RoadGroundMotion, fit_rigid_2d

BODY_HEIGHT_M = 0.4
TEXTURE_ORIGIN = np.array([60.0, -60.0])
TEXTURE_M_PER_PX = 0.04


def _rz(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0],
                     [0, 0, 1.0]])


@pytest.fixture(scope="module")
def texture():
    rng = np.random.default_rng(4)
    noise = rng.uniform(0, 255, (3000, 3000)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 1.6)


def _render(texture, T_WB, T_BC, K, *, flat=False):
    T_WC = T_WB @ T_BC
    vv, uu = np.mgrid[0:c.CANONICAL_HEIGHT, 0:c.CANONICAL_WIDTH]
    rays = np.column_stack([uu.ravel(), vv.ravel(), np.ones(uu.size)]) @ np.linalg.inv(K).T
    rays_w = rays @ T_WC[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -T_WC[2, 3] / rays_w[:, 2]
    points = T_WC[:3, 3] + s[:, None] * rays_w
    map_x = ((points[:, 0] - TEXTURE_ORIGIN[0]) / TEXTURE_M_PER_PX).reshape(uu.shape)
    map_y = ((points[:, 1] - TEXTURE_ORIGIN[1]) / TEXTURE_M_PER_PX).reshape(uu.shape)
    ground = (s > 0).reshape(uu.shape)
    map_x = np.where(ground, map_x, -1).astype(np.float32)
    map_y = np.where(ground, map_y, -1).astype(np.float32)
    source = np.full_like(texture, 90.0) if flat else texture
    gray = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=40.0)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))


def _patch(**overrides):
    fields = dict(patch_id="synthetic_road", branch=c.Branch.TRACK,
                  query_xy=np.array([100.0, 0.0]), point_W=np.array([100.0, 0.0, 0.0]),
                  normal_W=np.array([0.0, 0.0, 1.0]), support=40, residual_m=0.0, valid=True,
                  reason="valid")
    fields.update(overrides)
    return c.RoadPatch(**fields)


def _snapshot(t_ns, T_WB, aux=True):
    yaw = math.atan2(T_WB[1, 0], T_WB[0, 0])
    auxiliary = c.AuxiliaryGeometry(
        source=c.AuxSource.ROAD_PATCH, source_ns=t_ns, z_m=float(T_WB[2, 3]), roll_rad=0.0,
        pitch_rad=0.0, road_patch_id="synthetic_road", valid=True, reason="valid") if aux \
        else c.AuxiliaryGeometry.unavailable("not_estimated")
    return f.make_search_snapshot(t_ns=t_ns, state=np.array([T_WB[0, 3], T_WB[1, 3], yaw, 0, 0, 0]),
                                  auxiliary=auxiliary)


def _case(texture, *, flat=False):
    K, T_BC = f.canonical_K(), f.camera_to_body()
    previous_pose = g.rigid_from_rotation_translation(_rz(0.1), [100.0, 0.0, BODY_HEIGHT_M])
    delta = g.rigid_from_rotation_translation(_rz(0.02), [1.0, 0.05, 0.0])
    current_pose = previous_pose @ delta
    t1 = f.EXPOSURE_NS
    t0 = t1 - f.MOTION_STEP_NS
    # Camera ``r`` has no bodywork polygon, so the whole lower image is road.
    previous = f.make_observation(camera_id="r", exposure_ns=t0, K=K, T_BC=T_BC,
                                  image=_render(texture, previous_pose, T_BC, K, flat=flat))
    current = f.make_observation(camera_id="r", exposure_ns=t1, K=K, T_BC=T_BC,
                                 image=_render(texture, current_pose, T_BC, K, flat=flat))
    return previous, current, _snapshot(t0, previous_pose), _snapshot(t1, current_pose), delta


def test_recovers_metric_planar_motion(texture):
    previous, current, s0, s1, delta = _case(texture)
    motion = RoadGroundMotion().estimate(previous, current, _patch(), previous_snapshot=s0,
                                         current_snapshot=s1)
    assert motion.accepted, motion.reason
    np.testing.assert_allclose(motion.T_Bprev_Bcurr[:2, 3], delta[:2, 3], atol=0.03)
    yaw = math.atan2(motion.T_Bprev_Bcurr[1, 0], motion.T_Bprev_Bcurr[0, 0])
    assert yaw == pytest.approx(0.02, abs=math.radians(0.15))
    assert motion.T_Bprev_Bcurr[2, 3] == 0.0 and motion.road_patch_id == "synthetic_road"
    assert motion.inlier_count >= 30 and motion.ground_residual_m <= 0.25
    assert np.all(np.linalg.eigvalsh(motion.covariance) > 0)
    assert vision.make_ground_motion().gates == RoadGroundMotion().gates


@pytest.mark.parametrize(
    ("case", "reason"),
    [("invalid_patch", "road_patch_insufficient_support"),
     ("no_aux", "auxiliary_geometry_unavailable"),
     ("time", "snapshot_time_mismatch"),
     ("flat", "insufficient_road_texture"),
     ("far_patch", "auxiliary_incompatible_with_patch")],
)
def test_rejections_never_assume_scale(texture, case, reason):
    previous, current, s0, s1, _ = _case(texture, flat=case == "flat")
    patch = _patch()
    if case == "invalid_patch":
        patch = _patch(point_W=None, normal_W=None, residual_m=None, valid=False,
                       reason="insufficient_support")
    elif case == "no_aux":
        s0 = f.make_search_snapshot(t_ns=s0.t_ns, state=s0.state)
    elif case == "time":
        s1 = f.make_search_snapshot(t_ns=s1.t_ns - 1, state=s1.state)
    elif case == "far_patch":
        patch = _patch(point_W=np.array([100.0, 0.0, -5.0]))
    motion = RoadGroundMotion().estimate(previous, current, patch, previous_snapshot=s0,
                                         current_snapshot=s1)
    assert not motion.accepted and motion.reason == reason
    assert motion.T_Bprev_Bcurr is None and motion.covariance is None


def test_camera_mismatch_and_order_errors(texture):
    previous, current, s0, s1, _ = _case(texture)
    other = f.make_observation(camera_id="fr", exposure_ns=current.exposure_ns)
    motion = RoadGroundMotion().estimate(previous, other, _patch(), previous_snapshot=s0,
                                         current_snapshot=s1)
    assert motion.reason == "camera_mismatch"
    with pytest.raises(c.ContractError):
        RoadGroundMotion().estimate(current, previous, _patch(), previous_snapshot=s1,
                                    current_snapshot=s0)


def test_rigid_fit_direction_maps_current_into_previous():
    source = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 3.0], [4.0, 4.0]])
    theta, t = 0.1, np.array([1.0, -0.5])
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    fitted_theta, fitted_t = fit_rigid_2d(source, source @ R.T + t)
    assert fitted_theta == pytest.approx(theta)
    np.testing.assert_allclose(fitted_t, t)

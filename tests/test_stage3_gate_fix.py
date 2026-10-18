"""Stage 3 visual-baseline gate fix: bug-condition exploration and preservation properties.

Property 1 (bug condition) tests encode the expected behaviour of design Properties 1-4 and
fail on the unfixed code. Property 2 (preservation) tests pin the unaffected behaviour to a
reference computed in-test with plain ``transition``/``_predict``/``_update`` math or the
unchanged ``select_references`` ranking, so they hold before and after the fix.
Seeded ``numpy`` loops stand in for ``hypothesis`` (not a dependency). Synthetic only: no
localization accuracy or native high-rate claim follows from these tests.
"""

import collections
import math
from types import SimpleNamespace

import numpy as np
import pytest

import contract_factories as f
import test_filter as tf
import vision_fixtures as vf
from camloc import borders, contracts as c, geometry, landmark_map, vision
from camloc import filter as module
from camloc.filter import CameraFilter
from camloc.frontend import VisualFrontend

T = f.EXPOSURE_NS
MS = 1_000_000
TAU_S = 2.0
RATES = (10, 20, 30, 60, 120)


# -- fixtures --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


@pytest.fixture(scope="module")
def line_bank(scene, tmp_path_factory):
    """40 ``fr`` references 10 m apart along the scene heading; every 7th is ineligible
    (no landmark links), so ``eligible`` is a strict subset of the bank."""
    root = tmp_path_factory.mktemp("stage3_bank")
    base = scene.build_map(root / "base", keyframes=(0,))
    with np.load(base.root / "map.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count, spacing = 40, 10.0
    forward = np.r_[vf.rot_z(scene.yaw)[:2, 0], 0.0]
    rows = int(arrays["keypoint_offsets"][1])
    poses = []
    for i in range(count):
        T_WC = arrays["keyframe_T_WC"][0].copy()
        T_WC[:3, 3] += forward * spacing * i
        poses.append(T_WC)
    links = []
    for i in range(count):
        link = arrays["keypoint_landmarks"].copy()
        if i % 7 == 3:
            link[:] = -1
        links.append(link)
    for key in ("keypoints", "keypoint_scores", "descriptors"):
        arrays[key] = np.concatenate([arrays[key]] * count)
    arrays["keypoint_landmarks"] = np.concatenate(links).astype(np.int64)
    arrays["keyframe_T_WC"] = np.array(poses)
    arrays["keyframe_exposure_ns"] = (arrays["keyframe_exposure_ns"][0]
                                      + np.arange(count, dtype=np.int64) * c.NS_PER_S)
    arrays["keyframe_ids"] = np.arange(count, dtype=np.int64) * 2 + 1
    arrays["keyframe_cameras"] = np.array(["fr"] * count, dtype="<U2")
    arrays["keypoint_offsets"] = np.arange(count + 1, dtype=np.int64) * rows
    landmark_map.save_landmark_map(root / "line", arrays, dict(base.manifest))
    bank = landmark_map.load_landmark_map(root / "line")
    eligible = {int(k) for i, k in enumerate(bank.keyframe_ids) if i % 7 != 3}
    return SimpleNamespace(map=bank, eligible=eligible, spacing=spacing, count=count,
                           forward=forward[:2], left=np.array([-forward[1], forward[0]]))


@pytest.fixture(scope="module")
def road():
    return geometry.load_road_geometry(c.PROJECT_ROOT / "23-00-38_cut_120m_update/map.mcap")


def _initial(scene, stamp=None):
    return c.InitialPose(timestamp_ns=scene.t_ns(0) if stamp is None else stamp,
                         x=float(scene.origin[0]), y=float(scene.origin[1]), yaw=scene.yaw)


def _ref_frontend(bank, scene):
    """A VisualFrontend whose ``_references`` sees the line bank (no image path needed)."""
    tracker = SimpleNamespace(map=bank, events=collections.deque())
    return VisualFrontend(None, tracker, CameraFilter(_initial(scene)),
                          road=None, borders=None, motion=None)


def _snapshot(stamp, x, y, yaw, sigma, sigma_yaw):
    return f.make_search_snapshot(
        t_ns=stamp, state=np.array([x, y, yaw, 0.0, 0.0, 0.0]),
        P=np.diag([sigma**2, sigma**2, sigma_yaw**2, 1.0, 1.0, 0.1]))


def _query(stamp):
    return SimpleNamespace(camera_id="fr", exposure_ns=stamp)


def _pnp(stamp, x, y, yaw):
    return f.make_pose_evidence(
        exposure_ns=stamp, T_WB=tf.planar(x, y, yaw), inlier_count=64,
        track_ids=np.arange(64), landmark_ids=np.arange(64) + 100, inlier_fraction=0.8,
        occupied_cells=8, median_error_px=0.5, p90_error_px=1)


def _fuse_motion(estimator, t0, x0, y0, yaw0, velocity, yaw_rate, count, step_ns):
    """Fuse ``count`` PnP poses moving at ``velocity``/``yaw_rate``; return the last stamp."""
    stamp = t0
    for k in range(count):
        stamp = t0 + k * step_ns
        dt = (stamp - t0) / c.NS_PER_S
        pose = _pnp(stamp, x0 + velocity[0] * dt, y0 + velocity[1] * dt,
                    c.wrap_angle(yaw0 + yaw_rate * dt))
        result = estimator.step(f.make_group_evidence(frame_set=tf.fr_frame(k, stamp), pose=pose))
        assert result.reason == "accepted", (k, result.reason)
    return stamp


def _corr(scene, k, map_id, rows, offset=1000):
    pixels, visible = scene.project(k)
    pick = np.flatnonzero(visible)[rows]
    return c.Correspondences(camera_id="fr", exposure_ns=scene.t_ns(k),
                             keypoint_ids=np.arange(pick.size, dtype=np.int64) + offset,
                             pixels=pixels[pick].astype(np.float32),
                             landmark_ids=scene.landmark_ids[pick], map_id=map_id)


def _adopt_handoff(scene, tracker, obs, rows, job_id):
    result = f.make_anchor_result(job_id=job_id, source_exposure_ns=scene.t_ns(0),
                                  map_id=tracker.map.map_id,
                                  correspondences=_corr(scene, 0, tracker.map.map_id, rows),
                                  matched_reference_ids=(1,), remaining_candidate_ids=())
    handoff = tracker.handoff(result, obs)
    assert handoff.accepted, handoff.reason
    tracker.settle(handoff, c.HandoffDisposition.ADOPT)
    return handoff


def _reprojection(scene, tracker, landmark_ids, pixels, T_WB):
    projected, depth = vision.project(tracker.map.points(landmark_ids), T_WB, scene.T_BC, scene.K)
    return np.linalg.norm(projected - pixels, axis=1), depth


# ============================================================================================
# Property 1: Bug Condition (task 1) - these FAIL on the unfixed code by design.
# ============================================================================================

# Feature: stage3-visual-baseline-gate-fix, Property 1: Bug Condition
def test_bug_stranded_search_returns_eligible_references(scene, line_bank):
    """Case (a) stranded: an unsupported snapshot >= 250 m from every eligible reference.

    **Validates: Requirements 2.1, 2.3**
    """
    for iteration in range(40):
        rng = f.seeded_rng(301, iteration)
        frontend = _ref_frontend(line_bank.map, scene)
        along = rng.uniform(0, line_bank.spacing * (line_bank.count - 1))
        lateral = rng.uniform(250, 400) * rng.choice([-1.0, 1.0])
        xy = scene.origin[:2] + line_bank.forward * along + line_bank.left * lateral
        yaw = scene.yaw + rng.uniform(-0.3, 0.3)
        snapshot = _snapshot(T, *xy, yaw, rng.uniform(0.5, 60.0), rng.uniform(0.001, 0.05))
        refs = frontend._references(_query(T), snapshot, False)
        assert 1 <= len(refs) <= 2, (iteration, xy.tolist(), refs)
        assert set(refs) <= line_bank.eligible, (iteration, refs)


# Feature: stage3-visual-baseline-gate-fix, Property 1: Bug Condition
def test_bug_repeated_pair_while_untried_references_exist(scene, line_bank):
    """Case (a) repeated pair: unsupported groups 300 ms apart near the line.

    **Validates: Requirements 2.1, 2.3**
    """
    for iteration in range(40):
        rng = f.seeded_rng(302, iteration)
        frontend = _ref_frontend(line_bank.map, scene)
        along = rng.uniform(0, line_bank.spacing * (line_bank.count - 1))
        xy = scene.origin[:2] + line_bank.forward * along + line_bank.left * rng.uniform(-5, 5)
        snapshot = _snapshot(T, *xy, scene.yaw + rng.uniform(-0.1, 0.1),
                             rng.uniform(5.0, 20.0), rng.uniform(0.001, 0.05))
        first = frontend._references(_query(T), snapshot, False)
        second = frontend._references(_query(T + 300 * MS), snapshot, False)
        assert len(first) == 2 and len(line_bank.eligible - set(first)) > 0
        assert not set(first) & set(second), (iteration, first, second)


# Feature: stage3-visual-baseline-gate-fix, Property 1: Bug Condition
def test_bug_prolonged_loss_extrapolation_is_bounded():
    """Case (b): fuse ~37 m/s motion, then predict 270 s past the last full pose.

    **Validates: Requirements 2.4**
    """
    for iteration in range(20):
        rng = f.seeded_rng(303, iteration)
        heading = rng.uniform(-math.pi, math.pi)
        speed = 37.0 if iteration == 0 else rng.uniform(15.0, 40.0)
        velocity = speed * np.array([math.cos(heading), math.sin(heading)])
        yaw_rate = -0.282 if iteration == 0 else rng.uniform(-0.3, 0.3)
        x0, y0 = rng.uniform(-100, 100, 2)
        estimator = CameraFilter(c.InitialPose(timestamp_ns=T, x=x0, y=y0, yaw=heading))
        last = _fuse_motion(estimator, T, x0, y0, heading, velocity, yaw_rate, 11, 100 * MS)
        boundary = estimator.snapshot(last + 3 * c.NS_PER_S)
        late = estimator.snapshot(last + 270 * c.NS_PER_S)
        v_B = float(np.linalg.norm(boundary.state[3:5]))
        displacement = float(np.linalg.norm(late.state[:2] - boundary.state[:2]))
        yaw_change = abs(c.wrap_angle(late.state[2] - boundary.state[2]))
        assert displacement <= TAU_S * v_B + 1e-6, (iteration, v_B, displacement)
        assert yaw_change <= TAU_S * abs(boundary.state[5]) + 1e-6, (iteration, yaw_change)


# Feature: stage3-visual-baseline-gate-fix, Property 1: Bug Condition
def test_bug_adoption_retains_stale_tracks(scene, tmp_path):
    """Case (c): adopt A, make A's pooled pixels stale by 10 px, adopt B.

    **Validates: Requirements 2.2**
    """
    bank = scene.build_map(tmp_path / "bank")
    for iteration in range(4):
        rng = f.seeded_rng(304, iteration)
        tracker = vision.VisionTracker(bank)
        obs = {k: scene.observation(k) for k in (0, 1)}
        for k in (0, 1):
            tracker.observe(obs[k])
        visible = int(scene.project(0)[1].sum())
        order = rng.permutation(visible)
        rows_a, rows_b = np.sort(order[: visible // 2]), np.sort(order[visible // 2:])
        _adopt_handoff(scene, tracker, obs[1], rows_a, job_id=1)
        angle = rng.uniform(-math.pi, math.pi)
        active = tracker._active["fr"]
        active.pixels = (active.pixels + 10.0 * np.array([math.cos(angle), math.sin(angle)])
                         ).astype(np.float32)
        handoff_b = _adopt_handoff(scene, tracker, obs[1], rows_b, job_id=2)
        pool = tracker._active["fr"]
        old = ~np.isin(pool.landmark_ids, handoff_b.correspondences.landmark_ids)
        error, depth = _reprojection(scene, tracker, pool.landmark_ids[old], pool.pixels[old],
                                     handoff_b.pose.T_WB)
        stale = pool.track_ids[old][(error > 4.0) | (depth <= 0.5)]
        assert stale.size == 0, (iteration, int(old.sum()), stale.tolist()[:8],
                                 float(error.max(initial=0.0)))


# Feature: stage3-visual-baseline-gate-fix, Property 1: Bug Condition
def test_bug_unsupported_groups_propose_every_group(scene, road, tmp_path):
    """Case (d): 10 consecutive unsupported 100 ms groups with an accepting scheduler.

    **Validates: Requirements 2.2, 2.3**
    """
    bank = scene.build_map(tmp_path / "bank")
    tracker = vision.load_tracker(bank)
    estimator = CameraFilter(_initial(scene))
    observations = {k: scene.observation(2 * k, image=scene.image(2 * k, blank=True))
                    for k in range(10)}
    canonicalizer = SimpleNamespace(
        canonicalize=lambda image: observations[(image.exposure_ns - scene.t_ns(0)) // (100 * MS)])
    frontend = VisualFrontend(canonicalizer, tracker, estimator, road=road,
                              borders=borders.make_border_source(road),
                              motion=vision.make_ground_motion())
    proposals = collections.Counter()
    for k in range(10):
        image = f.make_image_message(exposure_ns=observations[k].exposure_ns)
        frame = f.make_frame_set(group_id=k, selected_camera_ids=("fr",), images=(image,))
        context = SimpleNamespace(results=(), rear_in_flight=False,
                                  propose=lambda k=k, **kwargs: proposals.update([k]))
        evidence = frontend(frame, estimator.snapshot(frame.t_ns), context)
        assert evidence.pose is None
        estimator.step(evidence)
    per_group = [proposals[k] for k in range(10)]
    assert per_group == [1] * 10, per_group


# ============================================================================================
# Property 2: Preservation (task 2) - these PASS on the unfixed code and must keep passing.
# ============================================================================================

def _ref_predict_group(old, stamp):
    """Reference for a prediction-only group: one CV prediction, symmetrized; cache unsymmetrized."""
    scale = module._NOISE_SCALE[old.status]
    state, P = module._predict(old.state, old.P, stamp - old.t_ns, scale)
    return state, (P + P.T) * 0.5, (state, P, old.status)


def _ref_pnp_group(old, pose, stamp, last_full):
    scale = module._NOISE_SCALE[old.status]
    state, P = module._predict(old.state.copy(), old.P.copy(), stamp - old.t_ns, scale)
    quality = max(1.0, 40 / pose.inlier_count, 8 / pose.occupied_cells,
                  pose.median_error_px / 2, pose.p90_error_px / 4)
    interval = 0.1 if last_full is None else (stamp - last_full) / c.NS_PER_S
    R = np.diag(module.POSE_STD**2) * quality * max(1.0, 0.1 / interval)
    state, P = module._update(state, P, module._pose_values(pose.T_WB), np.eye(6)[:3], R)
    cached = module._predict(state, P, 0, scale)
    final_state, final_P = module._predict(state, P, 0, scale)
    return final_state, (final_P + final_P.T) * 0.5, (*cached, c.Status.TRACKING)


def _expected_status(stamp, last_full):
    if last_full is None or stamp - last_full > c.NS_PER_S:
        return c.Status.LOST
    return c.Status.DEGRADED


def _assert_cached(estimator, stamp, cached):
    snap = estimator.exposure_snapshot("fr", stamp)
    np.testing.assert_array_equal(snap.state, cached[0])
    np.testing.assert_array_equal(snap.P, cached[1])
    assert snap.status is cached[2]


# Feature: stage3-visual-baseline-gate-fix, Property 2: Preservation
def test_preservation_pre_boundary_filter_is_plain_cv(scene):
    """Every prediction ends at or before B = last_full_ns + 3 s at 10-120 Hz: state, P,
    status, reason, kinds and cached exposure snapshots equal the plain CV reference.

    **Validates: Requirements 3.6, 3.7**
    """
    for iteration in range(100):
        rng = f.seeded_rng(310, iteration)
        rate = RATES[iteration % len(RATES)]
        x0, y0 = rng.uniform(-100, 100, 2)
        yaw0 = rng.uniform(-math.pi, math.pi)
        estimator = CameraFilter(c.InitialPose(timestamp_ns=T, x=x0, y=y0, yaw=yaw0))
        speed, heading = rng.uniform(0, 40), rng.uniform(-math.pi, math.pi)
        velocity = speed * np.array([math.cos(heading), math.sin(heading)])
        yaw_rate = rng.uniform(-0.3, 0.3)
        t, group_id, last_full, pending = T, 0, None, None
        for k in range(int(rng.integers(0, 5))):
            if k:
                t += int(rng.integers(50, 201)) * MS
            dt = (t - T) / c.NS_PER_S
            pose = _pnp(t, x0 + velocity[0] * dt, y0 + velocity[1] * dt,
                        c.wrap_angle(yaw0 + yaw_rate * dt))
            old = estimator.estimate
            gated = estimator.pose_in_gate(pose)
            ref = (_ref_pnp_group(old, pose, t, last_full) if gated
                   else _ref_predict_group(old, t))
            result = estimator.step(f.make_group_evidence(frame_set=tf.fr_frame(group_id, t),
                                                          pose=pose))
            if pending is not None:
                _assert_cached(estimator, *pending)
            np.testing.assert_array_equal(result.state, ref[0])
            np.testing.assert_array_equal(result.P, ref[1])
            if gated:
                last_full = t
                assert result.reason == "accepted" and result.status is c.Status.TRACKING
                assert result.evidence_kinds == (c.EvidenceKind.PNP,)
            else:
                assert result.reason == "innovation_gate"
                assert result.status is _expected_status(t, last_full)
            pending, group_id = (t, ref[2]), group_id + 1
        boundary = (last_full if last_full is not None else T) + 3 * c.NS_PER_S
        start = t
        steps = int((boundary - start) * rate // c.NS_PER_S)
        steps = int(rng.integers(0, steps + 1))
        for i in range(1, steps + 1):
            stamp = start + i * c.NS_PER_S // rate
            assert stamp <= boundary
            old = estimator.estimate
            probe = int(rng.integers(old.t_ns, boundary + 1))
            snap = estimator.snapshot(probe)
            ref_snap = module._predict(old.state, old.P, probe - old.t_ns,
                                       module._NOISE_SCALE[old.status])
            np.testing.assert_array_equal(snap.state, ref_snap[0])
            np.testing.assert_array_equal(snap.P, ref_snap[1])
            assert snap.status is old.status
            ref = _ref_predict_group(old, stamp)
            result = estimator.step(f.make_group_evidence(frame_set=tf.fr_frame(group_id, stamp)))
            if pending is not None:
                _assert_cached(estimator, *pending)
            np.testing.assert_array_equal(result.state, ref[0])
            np.testing.assert_array_equal(result.P, ref[1])
            assert result.reason == "no_evidence"
            assert result.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
            assert result.status is _expected_status(stamp, last_full)
            pending, group_id = (stamp, ref[2]), group_id + 1


# Feature: stage3-visual-baseline-gate-fix, Property 2: Preservation
def test_preservation_reset_and_following_pre_boundary_rows():
    """A confirmed RECOVERY_RESET and the prediction rows within 3 s of it.

    **Validates: Requirements 3.6, 3.7**
    """
    P0 = np.diag(np.r_[module.POSE_STD, module.INITIAL_STD[3:]] ** 2)
    for iteration in range(100):
        rng = f.seeded_rng(311, iteration)
        rate = RATES[iteration % len(RATES)]
        estimator = CameraFilter(tf.initial_at(T - 200 * MS, yaw=rng.uniform(-math.pi, math.pi)))
        camera = str(rng.choice(["fr", "r"]))
        front = T + int(rng.integers(0, 50)) * MS
        rear = front + int(rng.integers(0, 5 * MS))
        x, y, yaw = *rng.uniform(-50, 50, 2), rng.uniform(-math.pi, math.pi)
        evidence = tf.reset_evidence(tf.two_camera_frame(0, front, rear), camera,
                                     tf.planar(x, y, yaw))
        m, now = (front if camera == "fr" else rear), evidence.t_ns
        old = estimator.estimate
        scale = module._NOISE_SCALE[old.status]
        state0 = np.r_[module._pose_values(tf.planar(x, y, yaw)), 0.0, 0.0, 0.0]
        state, P = module._predict(state0, P0, now - m, scale)
        result = estimator.step(evidence)
        np.testing.assert_array_equal(result.state, state)
        np.testing.assert_array_equal(result.P, (P + P.T) * 0.5)
        assert result.status is c.Status.TRACKING
        assert result.evidence_kinds == (c.EvidenceKind.RECOVERY_RESET,)
        cached_front = module._predict(state0, P0, front - m, scale) if front >= m else None
        boundary, group_id, pending = m + 3 * c.NS_PER_S, 1, None
        steps = int(rng.integers(0, min(40, (boundary - now) * rate // c.NS_PER_S) + 1))
        for i in range(1, steps + 1):
            stamp = now + i * c.NS_PER_S // rate
            old = estimator.estimate
            ref = _ref_predict_group(old, stamp)
            result = estimator.step(f.make_group_evidence(frame_set=tf.fr_frame(group_id, stamp)))
            if i == 1 and cached_front is not None:
                _assert_cached(estimator, front, (*cached_front, c.Status.TRACKING))
            if pending is not None:
                _assert_cached(estimator, *pending)
            np.testing.assert_array_equal(result.state, ref[0])
            np.testing.assert_array_equal(result.P, ref[1])
            assert result.reason == "no_evidence"
            assert result.status is _expected_status(stamp, m)
            pending, group_id = (stamp, ref[2]), group_id + 1


# Feature: stage3-visual-baseline-gate-fix, Property 2: Preservation
def test_preservation_supported_proposals_use_capped_shortlist(scene, line_bank):
    """Supported groups propose ``select_references(sigma <= 60 m)[:2]``, unchanged.

    **Validates: Requirements 3.2, 3.5**
    """
    for iteration in range(150):
        rng = f.seeded_rng(312, iteration)
        frontend = _ref_frontend(line_bank.map, scene)
        along = rng.uniform(-50, line_bank.spacing * line_bank.count + 50)
        lateral = rng.uniform(-300, 300) if iteration % 3 == 0 else rng.uniform(-20, 20)
        xy = scene.origin[:2] + line_bank.forward * along + line_bank.left * lateral
        yaw = scene.yaw + rng.uniform(-0.6, 0.6)
        sigma, sigma_yaw = rng.uniform(0.1, 200.0), rng.uniform(0.001, 0.3)
        snapshot = _snapshot(T, *xy, yaw, sigma, sigma_yaw)
        expected = line_bank.map.select_references(
            "fr", *xy, yaw, sigma_xy_m=min(sigma, 60.0),
            max_heading_rad=min(math.pi, math.radians(30) + 3 * sigma_yaw))[:2]
        for repeat in range(2):
            stamp = T + repeat * 300 * MS
            assert frontend._references(_query(stamp), snapshot, True) == expected


# Feature: stage3-visual-baseline-gate-fix, Property 2: Preservation
def test_preservation_consistent_adoption_keeps_identical_pool(scene, tmp_path):
    """Adoptions whose old tracks all reproject <= 4 px in front of the camera, and
    adoptions with no prior active set, give the unchanged merged pool.

    **Validates: Requirements 3.2, 3.4**
    """
    bank = scene.build_map(tmp_path / "bank")
    visible = int(scene.project(0)[1].sum())
    for iteration in range(12):
        rng = f.seeded_rng(313, iteration)
        tracker = vision.VisionTracker(bank)
        obs = {k: scene.observation(k) for k in (0, 1)}
        for k in (0, 1):
            tracker.observe(obs[k])
        rows_b = np.sort(rng.choice(visible, int(visible * 0.6), replace=False))
        if iteration % 3 == 0:  # no prior active set
            handoff = _adopt_handoff(scene, tracker, obs[1], rows_b, job_id=2)
            pool = tracker._active["fr"]
            np.testing.assert_array_equal(pool.landmark_ids, handoff.correspondences.landmark_ids)
            np.testing.assert_array_equal(pool.pixels, handoff.correspondences.pixels)
            assert np.unique(pool.track_ids).size == len(pool) <= c.MAX_ACTIVE_TRACKS
            continue
        rows_a = np.sort(rng.choice(visible, int(visible * 0.6), replace=False))
        _adopt_handoff(scene, tracker, obs[1], rows_a, job_id=1)
        before = tracker._active["fr"]
        before = SimpleNamespace(track_ids=before.track_ids.copy(),
                                 landmark_ids=before.landmark_ids.copy(),
                                 pixels=before.pixels.copy())
        handoff = _adopt_handoff(scene, tracker, obs[1], rows_b, job_id=2)
        error, depth = _reprojection(scene, tracker, before.landmark_ids, before.pixels,
                                     handoff.pose.T_WB)
        assert (error <= 4.0).all() and (depth > 0.5).all()  # non-bug domain precondition
        fresh = handoff.correspondences
        keep = ~np.isin(before.landmark_ids, fresh.landmark_ids)
        pool = tracker._active["fr"]
        np.testing.assert_array_equal(
            pool.landmark_ids,
            np.concatenate([fresh.landmark_ids, before.landmark_ids[keep]])[:c.MAX_ACTIVE_TRACKS])
        np.testing.assert_array_equal(
            pool.pixels, np.concatenate([fresh.pixels, before.pixels[keep]])[:c.MAX_ACTIVE_TRACKS])
        np.testing.assert_array_equal(pool.track_ids[len(fresh.landmark_ids):],
                                      before.track_ids[keep])
        assert np.unique(pool.track_ids).size == len(pool) <= c.MAX_ACTIVE_TRACKS
        assert np.unique(pool.landmark_ids).size == len(pool)

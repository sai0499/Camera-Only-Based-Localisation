"""Stage 3 fix: the unsupported reference search cannot strand or repeat (design Properties 1, 4).

Seeded ``numpy`` loops stand in for ``hypothesis`` (not a dependency). Synthetic only: no
localization accuracy or native high-rate claim follows from these tests.
"""

import collections
import math
from types import SimpleNamespace

import numpy as np
import pytest

import contract_factories as f
import stage4_fakes as s4
import vision_fixtures as vf
from camloc import borders, contracts as c, geometry, landmark_map, vision
from camloc.filter import CameraFilter
from camloc.frontend import VisualFrontend
from camloc.scheduler import AnchorScheduler

T = f.EXPOSURE_NS
MS = 1_000_000


@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


@pytest.fixture(scope="module")
def line_bank(scene, tmp_path_factory):
    """40 ``fr`` references 10 m apart along the scene heading; every 7th has no landmark
    links (ineligible). Reference 20 sits at the initial pose."""
    root = tmp_path_factory.mktemp("stage3_search_bank")
    base = scene.build_map(root / "base", keyframes=(0,))
    with np.load(base.root / "map.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    count, spacing, zero = 40, 10.0, 20
    forward = np.r_[vf.rot_z(scene.yaw)[:2, 0], 0.0]
    rows = int(arrays["keypoint_offsets"][1])
    poses, links = [], []
    for i in range(count):
        T_WC = arrays["keyframe_T_WC"][0].copy()
        T_WC[:3, 3] += forward * spacing * (i - zero)
        poses.append(T_WC)
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
    T_BC = bank.T_BC("fr")
    center = landmark_map.planar_camera_center(*scene.origin[:2], scene.yaw, T_BC)
    heading = landmark_map.planar_camera_heading(scene.yaw, T_BC)
    axis = np.array([math.cos(heading), math.sin(heading)])
    offsets = {int(k): float((bank.keyframe_T_WC[i, :2, 3] - center) @ axis)
               for i, k in enumerate(bank.keyframe_ids)}
    return SimpleNamespace(map=bank, eligible=eligible, offsets=offsets, spacing=spacing,
                           forward=forward[:2], left=np.array([-forward[1], forward[0]]),
                           start=scene.origin[:2] - forward[:2] * spacing * zero,
                           length=spacing * (count - 1))


@pytest.fixture(scope="module")
def road():
    return geometry.load_road_geometry(c.PROJECT_ROOT / "23-00-38_cut_120m_update/map.mcap")


def _initial(scene):
    return c.InitialPose(timestamp_ns=scene.t_ns(0), x=float(scene.origin[0]),
                         y=float(scene.origin[1]), yaw=scene.yaw)


def _frontend(bank, scene, *, started=False):
    tracker = SimpleNamespace(map=bank, events=collections.deque())
    estimator = CameraFilter(_initial(scene))
    if started:  # a full pose was fused since the InitialPose: no startup ordering
        estimator._last_pnp_ns = scene.t_ns(0)
    return VisualFrontend(None, tracker, estimator, road=None, borders=None, motion=None)


def _snapshot(x, y, yaw, sigma, sigma_yaw):
    return f.make_search_snapshot(
        t_ns=T, state=np.array([x, y, yaw, 0.0, 0.0, 0.0]),
        P=np.diag([sigma**2, sigma**2, sigma_yaw**2, 1.0, 1.0, 0.1]))


def _query(stamp=T):
    return SimpleNamespace(camera_id="fr", exposure_ns=stamp)


def _random_snapshot(rng, bank):
    along = rng.uniform(-100, bank.length + 100)
    lateral = rng.uniform(-400, 400) if rng.random() < 0.4 else rng.uniform(-20, 20)
    xy = bank.start + bank.forward * along + bank.left * lateral
    sigma = rng.uniform(0.1, 200.0) if rng.random() < 0.7 else rng.uniform(0.1, 3.0)
    sigma_yaw = rng.uniform(0.001, 0.3)
    yaw = rng.uniform(-math.pi, math.pi) if rng.random() < 0.2 else rng.uniform(-0.3, 0.3) + 0.3
    return xy, yaw, sigma, sigma_yaw, _snapshot(*xy, yaw, sigma, sigma_yaw)


# Feature: stage3-visual-baseline-gate-fix, Property 1: Recovery and startup search never strands or repeats
def test_unsupported_search_sweeps_eligible_untried_references(scene, line_bank):
    """1-2 eligible references, untried unless exhausted, and the candidate set covers every
    reference inside 25 m + 3 sigma (uncapped) with the unchanged heading limit.

    **Validates: Requirements 2.1, 2.3**
    """
    bank = line_bank.map
    for iteration in range(60):
        rng = f.seeded_rng(401, iteration)
        frontend = _frontend(bank, scene, started=rng.random() < 0.5)
        xy, yaw, sigma, sigma_yaw, snapshot = _random_snapshot(rng, line_bank)
        heading = min(math.pi, math.radians(30) + 3 * sigma_yaw)
        radius = set(bank.select_references("fr", *xy, yaw, sigma_xy_m=sigma, count=10_000,
                                            max_heading_rad=heading))
        candidates = radius or line_bank.eligible
        seen, tried = [], set()
        for k in range(int(rng.integers(1, 30))):
            refs = frontend._references(_query(T + k * 100 * MS), snapshot, False)
            assert 1 <= len(refs) <= 2 and len(set(refs)) == len(refs), (iteration, k, refs)
            assert set(refs) <= candidates <= line_bank.eligible, (iteration, refs)
            if candidates - tried:
                assert not set(refs) & tried, (iteration, k, refs, tried)
            else:
                tried = set()
            tried |= set(refs)
            seen.extend(refs)
        # The sweep reaches every candidate once enough groups were proposed.
        if len(seen) >= len(candidates) + 2:
            assert set(seen) == candidates, iteration


# Feature: stage3-visual-baseline-gate-fix, Property 1: Recovery and startup search never strands or repeats
def test_empty_radius_falls_back_to_whole_eligible_bank(scene, line_bank):
    """A far prediction with a narrow heading gate still sweeps the whole eligible bank.

    **Validates: Requirements 2.1, 2.3**
    """
    for iteration in range(20):
        rng = f.seeded_rng(402, iteration)
        frontend = _frontend(line_bank.map, scene, started=True)
        xy = line_bank.start + line_bank.left * rng.uniform(800, 2000) * rng.choice([-1.0, 1.0])
        yaw = scene.yaw + math.pi  # facing away: heading gate also excludes everything
        snapshot = _snapshot(*xy, yaw, rng.uniform(0.1, 5.0), 0.001)
        assert not line_bank.map.select_references("fr", *xy, yaw, sigma_xy_m=5.0, count=10_000)
        seen = set()
        for k in range(len(line_bank.eligible) // 2 + 1):
            refs = frontend._references(_query(T + k * 100 * MS), snapshot, False)
            assert 1 <= len(refs) <= 2 and set(refs) <= line_bank.eligible
            seen |= set(refs)
        assert seen == line_bank.eligible, iteration


# Feature: stage3-visual-baseline-gate-fix, Property 1: Recovery and startup search never strands or repeats
def test_startup_orders_references_ahead_of_initial_camera_first(scene, line_bank):
    """Before any full pose, references at along-heading offset >= -5 m come first, nearest
    ahead first; after a full pose the plain ranking is used.

    **Validates: Requirements 2.1, 2.3**
    """
    for iteration in range(20):
        rng = f.seeded_rng(403, iteration)
        frontend = _frontend(line_bank.map, scene)
        xy = scene.origin[:2] + line_bank.left * rng.uniform(-3, 3)
        snapshot = _snapshot(*xy, scene.yaw + rng.uniform(-0.05, 0.05), rng.uniform(20, 80), 0.01)
        order = []
        while True:  # one full sweep: stop at the restart
            refs = frontend._references(_query(), snapshot, False)
            if set(refs) & set(order):
                break
            order.extend(refs)
        ahead = [ref for ref in order if line_bank.offsets[ref] >= -5.0]
        assert order[:len(ahead)] == ahead, (iteration, order)
        assert ahead == sorted(ahead, key=line_bank.offsets.get), (iteration, ahead)
        assert line_bank.offsets[order[0]] >= -5.0
    started = _frontend(line_bank.map, scene, started=True)
    snapshot = _snapshot(*scene.origin[:2], scene.yaw, 50.0, 0.01)
    expected = line_bank.map.select_references(
        "fr", *scene.origin[:2], scene.yaw, sigma_xy_m=50.0, count=10_000,
        max_heading_rad=math.radians(30) + 0.03)
    assert started._references(_query(), snapshot, False) == expected[:2]


def test_supported_query_clears_the_episode(scene, line_bank):
    frontend = _frontend(line_bank.map, scene, started=True)
    snapshot = _snapshot(*scene.origin[:2], scene.yaw, 10.0, 0.01)
    first = frontend._references(_query(), snapshot, False)
    assert frontend._references(_query(), snapshot, False) != first
    frontend._references(_query(), snapshot, True)
    assert frontend._references(_query(), snapshot, False) == first


# Feature: stage3-visual-baseline-gate-fix, Property 4: Recovery proposals use the scheduler ceiling
def test_unsupported_groups_propose_each_group_within_scheduler_ceilings(scene, road, tmp_path):
    """30 consecutive unsupported 100 ms groups: one proposal per group into the real
    AnchorScheduler; one in flight, one pending and <= 4 recovery dispatches per 1 s hold.

    **Validates: Requirements 2.2, 2.3, 3.3**
    """
    bank = scene.build_map(tmp_path / "bank")
    tracker = vision.load_tracker(bank)
    estimator = CameraFilter(_initial(scene))
    observations = {k: scene.observation(2 * k, image=scene.image(2 * k, blank=True))
                    for k in range(30)}
    canonicalizer = SimpleNamespace(
        canonicalize=lambda image: observations[(image.exposure_ns - scene.t_ns(0)) // (100 * MS)])
    frontend = VisualFrontend(canonicalizer, tracker, estimator, road=road,
                              borders=borders.make_border_source(road),
                              motion=vision.make_ground_motion())
    worker = s4.FakeWorker(variant=lambda request: "failed")
    scheduler = AnchorScheduler(worker, map_id=bank.map_id)
    scheduler.start(_initial(scene))
    status, per_group = c.Status.LOST, []
    for k in range(30):
        image = f.make_image_message(exposure_ns=observations[k].exposure_ns)
        frame = f.make_frame_set(group_id=k, selected_camera_ids=("fr",), images=(image,))
        context = scheduler.begin_group(frame, estimator.snapshot(frame.t_ns), status)
        evidence = frontend(frame, estimator.snapshot(frame.t_ns), context)
        assert evidence.pose is None
        per_group.append(len(context._proposals))
        estimate = estimator.step(evidence)
        status = estimate.status
        scheduler.end_group(estimate, frontend_failed=False)
    assert per_group == [1] * 30, per_group
    assert worker.max_concurrency <= 1
    stamps = [request.source_exposure_ns for request in worker.requests]
    assert stamps and stamps == sorted(stamps)
    for s in stamps:
        assert sum(s - c.NS_PER_S < t <= s for t in stamps) <= c.RECOVERY_JOBS_PER_S, stamps
    assert scheduler.counts()["anchor_rate_limited"] > 0

"""Stage 3 gate fix round 2: exploration of the retrieval gap (design Property 1, bug condition).

Both tests are designed to FAIL on the unfixed code and to PASS once task 3 lands:

- the moving-car sweep constructs ``VisualFrontend(..., retrieval_enabled=True)``; on the
  unfixed code that keyword does not exist, and the unfixed proposals stay near the stale
  snapshot while the car drives away;
- the failed-result carry reads ``GroupContext.settled``, which the unfixed scheduler lacks
  (FAILED results with ``remaining_candidate_ids`` never reach the frontend).

Synthetic only: blank images, a stub reference bank and ``stage4_fakes`` workers. No
localization accuracy, retrieval recall or native high-rate claim follows from these tests.
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

MS = 1_000_000
GROUP_NS = 250 * MS  # unsupported proposals at 4/s
SWEEP_S = 30
WARMUP_NS = 3 * c.NS_PER_S
TRUE_SPEED_MPS = 35.0
SPACING_M = 4.0
FRESH_NS = 500 * MS  # carry freshness bound (design: source <= 0.5 s old)
NEAR_M = 15.0


@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


@pytest.fixture(scope="module")
def road():
    return geometry.load_road_geometry(c.PROJECT_ROOT / "23-00-38_cut_120m_update/map.mcap")


def build_sweep_bank(scene, root):
    """``fr`` references every 4 m along the scene heading from -20 m to past the 30 s path
    of a 35 m/s car; all eligible (the base keyframe's landmark links are kept)."""
    base = scene.build_map(root / "base", keyframes=(0,))
    with np.load(base.root / "map.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    zero = 5
    count = zero + int(math.ceil(TRUE_SPEED_MPS * SWEEP_S / SPACING_M)) + 6
    forward = np.r_[vf.rot_z(scene.yaw)[:2, 0], 0.0]
    rows = int(arrays["keypoint_offsets"][1])
    poses = []
    for i in range(count):
        T_WC = arrays["keyframe_T_WC"][0].copy()
        T_WC[:3, 3] += forward * SPACING_M * (i - zero)
        poses.append(T_WC)
    for key in ("keypoints", "keypoint_scores", "descriptors", "keypoint_landmarks"):
        arrays[key] = np.concatenate([arrays[key]] * count)
    arrays["keypoint_landmarks"] = arrays["keypoint_landmarks"].astype(np.int64)
    arrays["keyframe_T_WC"] = np.array(poses)
    arrays["keyframe_exposure_ns"] = (arrays["keyframe_exposure_ns"][0]
                                      + np.arange(count, dtype=np.int64) * 100 * MS)
    arrays["keyframe_ids"] = np.arange(count, dtype=np.int64) * 2 + 1
    arrays["keyframe_cameras"] = np.array(["fr"] * count, dtype="<U2")
    arrays["keypoint_offsets"] = np.arange(count + 1, dtype=np.int64) * rows
    landmark_map.save_landmark_map(root / "line", arrays, dict(base.manifest))
    bank = landmark_map.load_landmark_map(root / "line")
    along = {int(k): SPACING_M * (i - zero) for i, k in enumerate(bank.keyframe_ids)}
    return SimpleNamespace(map=bank, along=along)


@pytest.fixture(scope="module")
def sweep_bank(scene, tmp_path_factory):
    return build_sweep_bank(scene, tmp_path_factory.mktemp("stage3_retrieval_gap_bank"))


def _initial(scene):
    return c.InitialPose(timestamp_ns=scene.t_ns(0), x=float(scene.origin[0]),
                         y=float(scene.origin[1]), yaw=scene.yaw)


class RecordingWorker(s4.FakeWorker):
    """``FakeWorker`` that also keeps every AnchorResult it returned, in call order."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results: list[c.AnchorResult] = []

    def run(self, request):
        result = super().run(request)
        with self._lock:
            self.results.append(result)
        return result


def retrieval_frontend(canonicalizer, tracker, estimator, **components):
    """The planned fixed API (design.md): ``VisualFrontend(..., retrieval_enabled=True)``."""
    return VisualFrontend(canonicalizer, tracker, estimator, retrieval_enabled=True, **components)


def moving_car_sweep(scene, road, bank, make_frontend=retrieval_frontend):
    """Drive ``SWEEP_S`` s of unsupported 4 Hz ``fr`` groups through the real frontend,
    CameraFilter and AnchorScheduler, with every anchor job FAILED with remaining ids.

    The images are blank, so no pose is ever supported and the snapshot stays frozen at the
    initial pose while the (test-only) true car position advances at 35 m/s. Returns one
    record per group: its time, the frontend's proposals and the results settled so far.
    """
    groups = SWEEP_S * c.NS_PER_S // GROUP_NS
    t0 = scene.t_ns(0)
    steps = GROUP_NS // vf.STEP_NS

    def index(exposure_ns):
        return (exposure_ns - t0) // vf.STEP_NS

    canonicalizer = SimpleNamespace(canonicalize=lambda image: scene.observation(
        index(image.exposure_ns), image=scene.image(index(image.exposure_ns), blank=True)))
    tracker = vision.load_tracker(bank.map)
    estimator = CameraFilter(_initial(scene))
    frontend = make_frontend(canonicalizer, tracker, estimator, road=road,
                             borders=borders.make_border_source(road),
                             motion=vision.make_ground_motion())
    worker = RecordingWorker(variant=lambda request: "failed")
    scheduler = AnchorScheduler(worker, map_id=bank.map.map_id)
    scheduler.start(_initial(scene))
    status, records = c.Status.LOST, []
    for k in range(groups):
        stamp = scene.t_ns(k * steps)
        image = f.make_image_message(exposure_ns=stamp)
        frame = f.make_frame_set(group_id=k, selected_camera_ids=("fr",), images=(image,))
        prior = estimator.snapshot(frame.t_ns)
        context = scheduler.begin_group(frame, prior, status)
        settled = tuple(worker.results)  # every job dispatched before this group is delivered
        evidence = frontend(frame, estimator.snapshot(frame.t_ns), context)
        assert evidence.pose is None, k
        records.append(SimpleNamespace(
            k=k, t_ns=stamp, elapsed_ns=stamp - t0, settled=settled,
            frozen_m=float(np.linalg.norm(prior.state[:2] - scene.origin[:2])),
            true_along_m=TRUE_SPEED_MPS * (stamp - t0) / c.NS_PER_S,
            proposals=[(p.purpose, p.observation.exposure_ns, p.reference_ids)
                       for p in context._proposals]))
        estimate = estimator.step(evidence)
        status = estimate.status
        scheduler.end_group(estimate, frontend_failed=False)
    assert worker.max_concurrency <= 1
    return records


def _fresh_carry(refs, exposure_ns, settled):
    """True when ``refs`` are 1-2 untried-candidate ids of a result settled <= 0.5 s ago."""
    return 1 <= len(refs) <= c.MAX_REFERENCES_PER_JOB and any(
        0 <= exposure_ns - result.source_exposure_ns <= FRESH_NS
        and set(refs) <= set(result.remaining_candidate_ids)
        for result in settled)


def _near_true(refs, record, bank):
    return sum(abs(bank.along[ref] - record.true_along_m) <= NEAR_M for ref in refs)


# Feature: stage3-gate-fix-round2, Property 1: Bug Condition
def test_bug_moving_car_unsupported_groups_request_retrieval(scene, road, sweep_bank):
    """Case (a): snapshot frozen at 0 m, the car at 35 m/s, unsupported proposals at 4/s
    for 30 s with a loaded index. After the first 3 s every proposal is one RECOVERY
    request per group that is either a retrieval request ``()`` or 1-2 carried ids of a
    result settled <= 0.5 s before the proposal's exposure. Never a stale-centre sweep.

    **Validates: Requirements 2.1, 2.3**
    """
    records = moving_car_sweep(scene, road, sweep_bank)
    late = [record for record in records if record.elapsed_ns >= WARMUP_NS]
    assert late and max(record.frozen_m for record in records) <= 1.0  # bug condition holds
    bad, kinds = [], {"retrieval": 0, "carried": 0}
    for record in late:
        assert len(record.proposals) == 1, (record.k, record.proposals)
        purpose, exposure_ns, refs = record.proposals[0]
        assert purpose is c.AnchorPurpose.RECOVERY, (record.k, purpose)
        if refs == ():
            kinds["retrieval"] += 1
        elif _fresh_carry(refs, exposure_ns, record.settled):
            kinds["carried"] += 1
        else:
            bad.append((record.k, round(record.true_along_m), refs,
                        [sweep_bank.along[ref] for ref in refs],
                        _near_true(refs, record, sweep_bank)))
    assert not bad, (f"{len(bad)}/{len(late)} late proposals neither retrieval nor fresh "
                     f"carry; first (group, true_along_m, refs, ref_along_m, near): {bad[:5]}")
    # Not vacuous: retrieval requests are issued and fresh failed-result candidates are used.
    assert kinds["retrieval"] > 0 and kinds["carried"] > 0, kinds


# Feature: stage3-gate-fix-round2, Property 1: Bug Condition
def test_bug_failed_result_remaining_ids_reach_frontend():
    """Case (c): a FAILED result with ``remaining_candidate_ids`` is visible to the frontend
    through ``GroupContext.settled`` at its Delivery_Group, beside every OK result settled
    there; ``GroupContext.results`` keeps only the OK ones (unchanged contents).

    **Validates: Requirements 2.1, 2.3**
    """
    frames = s4.synthetic_stream(16, 4.0, cameras=(c.PRIMARY_CAMERA,))
    worker = RecordingWorker(variant=lambda request: "failed" if request.job_id % 2 == 0 else "ok")
    scheduler = AnchorScheduler(worker, map_id="stage3_retrieval_gap_map")
    scheduler.start(c.InitialPose(timestamp_ns=s4.T0, x=0.0, y=0.0, yaw=0.0))
    seen_failed = seen_ok = 0
    settled_ids: list[int] = []
    for frame in frames:
        snapshot = f.make_search_snapshot(t_ns=frame.t_ns, status=c.Status.LOST)
        context = scheduler.begin_group(frame, snapshot, c.Status.LOST)
        settled = context.settled  # absent on the unfixed scheduler
        assert isinstance(settled, tuple), type(settled)
        assert [r.job_id for r in context.results] == [
            r.job_id for r in settled if r.status is c.AnchorStatus.OK]
        for result in settled:
            produced = [r for r in worker.results if r.job_id == result.job_id]
            assert len(produced) == 1 and produced[0].status is result.status, result.job_id
            assert result.remaining_candidate_ids == produced[0].remaining_candidate_ids
            settled_ids.append(result.job_id)
            if result.status is c.AnchorStatus.FAILED:
                assert result.remaining_candidate_ids, result
                seen_failed += 1
            else:
                seen_ok += 1
        context.propose(purpose=c.AnchorPurpose.RECOVERY, observation=s4.observation(frame.images[0]),
                        reference_ids=(), track_count=0)
        scheduler.end_group(c.Estimate(
            t_ns=frame.t_ns, group_id=frame.group_id, state=np.array(snapshot.state),
            P=np.array(snapshot.P), status=c.Status.LOST, branch=c.Branch.TRACK,
            evidence_kinds=(c.EvidenceKind.PREDICTION_ONLY,), reason="no_evidence",
            auxiliary=c.AuxiliaryGeometry.unavailable("not_estimated"),
            max_support_exposure_ns=None), frontend_failed=False)
    scheduler.finish()
    # Every dispatched job except the last is settled exactly once, FAILED ones included.
    assert seen_failed > 0 and seen_ok > 0, (seen_failed, seen_ok)
    assert len(settled_ids) == len(set(settled_ids)) >= len(worker.results) - 1, settled_ids


# --- Fix checking (task 3): carry freshness, exhaustion, None vs () -----------------------


class ScriptWorker:
    """Fails every job. A retrieval request (``()``) matches the first 2 of a seeded ranking
    of the bank's ids and leaves the next ``remaining`` ids; named requests match their refs
    and leave none (the real worker's split)."""

    def __init__(self, bank_ids, remaining, seed=0):
        self.bank_ids, self.remaining, self.seed = list(bank_ids), remaining, seed
        self.requests: list[c.AnchorRequest] = []

    def run(self, request):
        self.requests.append(request)
        if request.reference_ids:
            matched, rest = request.reference_ids, ()
        else:
            rng = np.random.default_rng([self.seed, request.job_id])
            ranking = tuple(int(k) for k in rng.permutation(self.bank_ids)[: 2 + self.remaining])
            matched, rest = ranking[:2], ranking[2:]
        return f.make_anchor_result(
            job_id=request.job_id, purpose=request.purpose, camera_id=request.camera_id,
            source_group_id=request.source_group_id, source_exposure_ns=request.source_exposure_ns,
            map_id=request.map_id, preprocessing_id=request.preprocessing_id,
            status=c.AnchorStatus.FAILED, matched_reference_ids=matched,
            remaining_candidate_ids=rest, worker_duration_ns=None)


def drive_unsupported(scene, road, bank, worker, *, group_ns, groups, retrieval_enabled=True):
    """``groups`` unsupported (blank) ``fr`` groups every ``group_ns`` through the real
    frontend, filter and scheduler; one record per group with the settled results and the
    frontend's proposals."""
    t0, steps = scene.t_ns(0), group_ns // vf.STEP_NS
    canonicalizer = SimpleNamespace(canonicalize=lambda image: scene.observation(
        (image.exposure_ns - t0) // vf.STEP_NS,
        image=scene.image((image.exposure_ns - t0) // vf.STEP_NS, blank=True)))
    estimator = CameraFilter(_initial(scene))
    frontend = VisualFrontend(canonicalizer, vision.load_tracker(bank), estimator, road=road,
                              borders=borders.make_border_source(road),
                              motion=vision.make_ground_motion(),
                              retrieval_enabled=retrieval_enabled)
    scheduler = AnchorScheduler(worker, map_id=bank.map_id)
    scheduler.start(_initial(scene))
    status, records = c.Status.LOST, []
    for k in range(groups):
        stamp = scene.t_ns(k * steps)
        frame = f.make_frame_set(group_id=k, selected_camera_ids=("fr",),
                                 images=(f.make_image_message(exposure_ns=stamp),))
        context = scheduler.begin_group(frame, estimator.snapshot(frame.t_ns), status)
        evidence = frontend(frame, estimator.snapshot(frame.t_ns), context)
        assert evidence.pose is None, k
        records.append(SimpleNamespace(k=k, t_ns=stamp, settled=context.settled, proposals=[
            (p.purpose, p.observation.exposure_ns, p.reference_ids) for p in context._proposals]))
        estimate = estimator.step(evidence)
        status = estimate.status
        scheduler.end_group(estimate, frontend_failed=False)
    scheduler.finish()
    return records, scheduler


def expected_proposals(records, requests):
    """Spec model: the newest settled retrieval result with remaining ids is the carry;
    ids matched by results settled since are tried; fresh (<= 0.5 s) untried ids first
    (at most 2), else ``()``."""
    by_job, carry, out = {r.job_id: r for r in requests}, None, []
    for record in records:
        for result in record.settled:
            if carry is not None:
                carry["tried"] |= set(result.matched_reference_ids)
            if (by_job[result.job_id].reference_ids == () and result.remaining_candidate_ids
                    and (carry is None or result.source_exposure_ns > carry["source"])):
                carry = {"source": result.source_exposure_ns, "tried": set(result.matched_reference_ids),
                         "ids": result.remaining_candidate_ids}
        refs = ()
        if carry is not None and 0 <= record.t_ns - carry["source"] <= FRESH_NS:
            refs = tuple([i for i in carry["ids"] if i not in carry["tried"]][:2])
        out.append([(c.AnchorPurpose.RECOVERY, record.t_ns, refs)])
    return out


def _kinds(records):
    return collections.Counter("retrieval" if p[2] == () else "carried"
                               for record in records for p in record.proposals)


# Feature: stage3-gate-fix-round2, Property 1: Unsupported groups request global retrieval
def test_carry_proposals_follow_freshness_and_tried_rules(scene, road, sweep_bank):
    """Random group periods and carry lengths: every unsupported group proposes exactly the
    spec model's refs: fresh untried carried ids (<= 2), else a retrieval request.

    **Validates: Requirements 2.1, 2.3**
    """
    ids = [int(k) for k in sweep_bank.map.keyframe_ids]
    kinds = collections.Counter()
    for iteration in range(10):
        rng = np.random.default_rng([71, iteration])
        group_ns = int(rng.choice([50, 100, 150, 200, 250, 300, 400, 600])) * MS
        worker = ScriptWorker(ids, int(rng.integers(0, 9)), seed=iteration)
        records, scheduler = drive_unsupported(scene, road, sweep_bank.map, worker,
                                               group_ns=group_ns, groups=int(rng.integers(20, 40)))
        assert [record.proposals for record in records] == expected_proposals(
            records, worker.requests), (iteration, group_ns, worker.remaining)
        stamps = [request.source_exposure_ns for request in worker.requests]
        for s in stamps:  # the unchanged recovery ceiling holds
            assert sum(s - c.NS_PER_S < t <= s for t in stamps) <= c.RECOVERY_JOBS_PER_S
        kinds.update(_kinds(records))
    assert kinds["retrieval"] and kinds["carried"], kinds


def test_fresh_carry_is_swept_then_exhausted_then_retrieval(scene, road, sweep_bank):
    ids = [int(k) for k in sweep_bank.map.keyframe_ids]
    worker = ScriptWorker(ids, 8)
    records, _ = drive_unsupported(scene, road, sweep_bank.map, worker, group_ns=100 * MS, groups=4)
    first = records[1].settled[0]
    assert records[0].proposals[0][2] == ()
    assert [r.proposals[0][2] for r in records[1:]] == [
        first.remaining_candidate_ids[0:2], first.remaining_candidate_ids[2:4],
        first.remaining_candidate_ids[4:6]]
    # Exhausted while still fresh: two remaining ids, tried at the next delivery -> ().
    short = ScriptWorker(ids, 2)
    records, _ = drive_unsupported(scene, road, sweep_bank.map, short, group_ns=100 * MS, groups=3)
    assert [r.proposals[0][2] for r in records] == [
        (), records[1].settled[0].remaining_candidate_ids, ()]
    assert records[2].t_ns - records[1].settled[0].source_exposure_ns <= FRESH_NS


def test_stale_carry_is_never_proposed(scene, road, sweep_bank):
    """Carry sources older than 0.5 s (600 ms groups) are never used: every proposal is a
    retrieval request."""
    ids = [int(k) for k in sweep_bank.map.keyframe_ids]
    records, _ = drive_unsupported(scene, road, sweep_bank.map, ScriptWorker(ids, 8),
                                   group_ns=600 * MS, groups=12)
    assert any(record.settled for record in records)
    assert all(record.proposals == [(c.AnchorPurpose.RECOVERY, record.t_ns, ())]
               for record in records)


def _plain_frontend(bank, scene, retrieval_enabled):
    tracker = SimpleNamespace(map=bank, events=collections.deque())
    estimator = CameraFilter(_initial(scene))
    estimator._last_pnp_ns = scene.t_ns(0)
    return VisualFrontend(None, tracker, estimator, road=None, borders=None, motion=None,
                          retrieval_enabled=retrieval_enabled)


def _snapshot(xy, yaw, sigma, sigma_yaw):
    return f.make_search_snapshot(t_ns=f.EXPOSURE_NS, state=np.array([*xy, yaw, 0.0, 0.0, 0.0]),
                                  P=np.diag([sigma**2, sigma**2, sigma_yaw**2, 1.0, 1.0, 0.1]))


# Feature: stage3-gate-fix-round2, Property 5: Preservation - Supported, no-index and named-reference paths
def test_supported_branch_unchanged_and_none_vs_retrieval_request(scene, sweep_bank):
    """Supported groups propose the same geometric refs (or nothing, None) with and without
    retrieval; an unsupported group with retrieval and no carry is ``()``, without it the
    geometric sweep.

    **Validates: Requirements 3.1, 3.2**
    """
    bank = sweep_bank.map
    forward = vf.rot_z(scene.yaw)[:2, 0]
    query = SimpleNamespace(camera_id="fr", exposure_ns=f.EXPOSURE_NS)
    for iteration in range(40):
        rng = np.random.default_rng([72, iteration])
        xy = scene.origin[:2] + forward * rng.uniform(-50, 200) + rng.normal(0, 30, 2)
        snapshot = _snapshot(xy, scene.yaw + rng.uniform(-0.5, 0.5), rng.uniform(0.1, 80.0),
                             rng.uniform(0.001, 0.2))
        enabled = _plain_frontend(bank, scene, True)
        disabled = _plain_frontend(bank, scene, False)
        refs = enabled._proposal_refs(query, snapshot, True)
        assert refs == (disabled._references(query, snapshot, True) or None), iteration
    far = _snapshot(scene.origin[:2] + 5000.0, scene.yaw, 0.1, 0.001)
    assert _plain_frontend(bank, scene, True)._proposal_refs(query, far, True) is None
    assert _plain_frontend(bank, scene, False)._proposal_refs(query, far, True) is None
    assert _plain_frontend(bank, scene, True)._proposal_refs(query, far, False) == ()
    geometric = _plain_frontend(bank, scene, False)._proposal_refs(query, far, False)
    assert geometric and 1 <= len(geometric) <= c.MAX_REFERENCES_PER_JOB

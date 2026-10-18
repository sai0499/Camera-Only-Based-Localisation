"""B's DiskAnchorWorker with a fake feature backend: references, uniqueness, failures."""

import math

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import vision
from camloc.features import Features


class FakeBackend:
    """Query keypoints are the scene's exact projections; descriptor[0] holds landmark ids
    (decoys use -1) and ``match`` pairs equal codes, optionally duplicating matches."""

    model_ids = ("fake_disk:000", "fake_lightglue:000")

    def __init__(self, scene, k, *, limit=None, decoys=20, duplicate=False):
        pixels, visible = scene.project(k)
        rows = np.flatnonzero(visible)[:limit]
        rng = np.random.default_rng(5)
        decoy = np.column_stack([rng.uniform(0, 959, decoys), rng.uniform(0, 463, decoys)])
        self.keypoints = np.concatenate([pixels[rows], decoy]).astype(np.float32)
        codes = np.concatenate([scene.landmark_ids[rows], -np.ones(decoys)])
        self.descriptors = np.zeros((len(codes), 128), np.float32)
        self.descriptors[:, 0] = codes
        self.duplicate = duplicate
        self.extract_calls = 0

    def extract(self, image, mask):
        self.extract_calls += 1
        n = len(self.keypoints)
        return Features(self.keypoints, np.linspace(1, 0, n, dtype=np.float32), self.descriptors)

    def match(self, query, reference):
        ref_codes = {int(v): j for j, v in enumerate(reference.descriptors[:, 0]) if v >= 0}
        pairs = [(i, ref_codes[int(v)]) for i, v in enumerate(query.descriptors[:, 0])
                 if int(v) in ref_codes]
        if self.duplicate and pairs:
            pairs += [(pairs[0][0], pairs[1][1])]  # one query keypoint matched twice
        pairs = np.array(pairs, np.int64).reshape(-1, 2)
        return pairs, np.linspace(0.9, 0.5, len(pairs)).astype(np.float32)


@pytest.fixture(scope="module")
def scene():
    return vf.Scene()


@pytest.fixture()
def landmark_map(scene, tmp_path):
    return scene.build_map(tmp_path / "map")


def _request(scene, landmark_map, k, **overrides):
    fields = dict(job_id=3, source_exposure_ns=scene.t_ns(k), observation=scene.observation(k),
                  snapshot=scene.snapshot(k), reference_ids=(), map_id=landmark_map.map_id)
    fields.update(overrides)
    return f.make_anchor_request(**fields)


def test_ok_result_has_unique_source_correspondences_and_no_pose(scene, landmark_map):
    worker = vision.DiskAnchorWorker(landmark_map, FakeBackend(scene, 4, duplicate=True))
    request = _request(scene, landmark_map, 4)
    result = worker.run(request)
    assert result.status is c.AnchorStatus.OK, result.reason
    result.check_against(request)
    corr = result.correspondences
    assert (corr.camera_id, corr.exposure_ns) == ("fr", scene.t_ns(4))
    assert np.unique(corr.keypoint_ids).size == corr.keypoint_ids.size
    assert np.unique(corr.landmark_ids).size == corr.landmark_ids.size
    assert len(result.matched_reference_ids) <= c.MAX_REFERENCES_PER_JOB
    assert not set(result.matched_reference_ids) & set(result.remaining_candidate_ids)
    assert result.model_ids == FakeBackend.model_ids
    assert not hasattr(result, "T_WB")


def test_frozen_reference_ids_are_respected(scene, landmark_map):
    worker = vision.DiskAnchorWorker(landmark_map, FakeBackend(scene, 4))
    request = _request(scene, landmark_map, 4, reference_ids=(5,))
    result = worker.run(request)
    result.check_against(request)
    assert result.matched_reference_ids == (5,)


def test_snapshot_reference_search_is_geometric(scene, landmark_map):
    worker = vision.DiskAnchorWorker(landmark_map, FakeBackend(scene, 12))
    near = worker.candidates(_request(scene, landmark_map, 12))
    assert near and near[0] in (3, 5)  # keyframes at frames 10 and 20 (ids 3 and 5)
    pose = scene.pose(12)
    far = f.make_search_snapshot(t_ns=scene.t_ns(12),
                                 state=np.array([pose[0, 3] + 500.0, pose[1, 3], 0.3, 0, 0, 0]))
    assert worker.candidates(_request(scene, landmark_map, 12, snapshot=far)) == ()
    turned = f.make_search_snapshot(t_ns=scene.t_ns(12),
                                    state=np.array([pose[0, 3], pose[1, 3], 0.3 + math.pi, 0, 0, 0]))
    assert worker.candidates(_request(scene, landmark_map, 12, snapshot=turned)) == ()


@pytest.mark.parametrize(
    ("setup", "reason"),
    [("map", "map_mismatch"), ("far", "no_reference_candidates"),
     ("few", "insufficient_matches"), ("intrinsics", "map_intrinsics_mismatch")],
)
def test_expected_failures_return_status_failed(scene, landmark_map, setup, reason):
    backend = FakeBackend(scene, 4, limit=10 if setup == "few" else None)
    worker = vision.DiskAnchorWorker(landmark_map, backend)
    overrides = {}
    if setup == "map":
        overrides["map_id"] = "another_map"
    elif setup == "far":
        overrides["snapshot"] = f.make_search_snapshot(
            t_ns=scene.t_ns(4), state=np.array([5000.0, 0.0, 0.0, 0, 0, 0]))
    elif setup == "intrinsics":
        K = scene.K.copy()
        K[1, 2] += 1.0
        overrides["observation"] = scene.observation(4, K=K)
    request = _request(scene, landmark_map, 4, **overrides)
    result = worker.run(request)
    assert result.status is c.AnchorStatus.FAILED and result.reason == reason
    assert result.correspondences is None
    result.check_against(request)


def test_unique_pairs_keeps_best_score_per_keypoint_and_landmark():
    q = np.array([0, 0, 1, 2, 3])
    lms = np.array([10, 11, 10, 12, 12])
    s = np.array([0.5, 0.9, 0.8, 0.4, 0.7])
    keep = vision.unique_pairs(q, lms, s)
    pairs = {(int(q[i]), int(lms[i])) for i in keep}
    assert pairs == {(0, 11), (1, 10), (3, 12)}


# --- Stage 3 gate fix round 2: global retrieval for empty-reference requests ----------


class StubRetrieval:
    """Test-only stand-in for ``camloc.retrieval.RetrievalIndex``: a fixed ranking per call."""

    def __init__(self, ranking):
        self.ranking = tuple(ranking)
        self.queries = []

    def query(self, observation):
        self.queries.append((observation.camera_id, observation.exposure_ns))
        return self.ranking


@pytest.fixture()
def wide_map(scene, tmp_path):
    """Twelve ``fr`` keyframes (ids 1, 3, ..., 23) so a ranking can exceed RETRIEVAL_CANDIDATES."""
    return scene.build_map(tmp_path / "wide", keyframes=tuple(range(0, 24, 2)))


# Feature: stage3-gate-fix-round2, Property 2: Worker retrieval ranking
def test_empty_reference_request_uses_retrieval_split(scene, wide_map):
    """An empty-reference request with an index takes the index ranking: matched = first 2,
    remaining = next <= 8, the same on every call; the observation is the request's.

    **Validates: Requirements 2.1, 2.2**
    """
    ids = [int(k) for k in wide_map.keyframe_ids]
    for iteration in range(12):
        rng = f.seeded_rng(611, iteration)
        ranking = tuple(int(k) for k in rng.permutation(ids)[: int(rng.integers(1, 11))])
        stub = StubRetrieval(ranking)
        worker = vision.DiskAnchorWorker(wide_map, FakeBackend(scene, 4), retrieval=stub)
        request = _request(scene, wide_map, 4)
        assert worker.candidates(request) == ranking
        first, second = worker.run(request), worker.run(request)
        first.check_against(request)
        assert first.status is c.AnchorStatus.OK, (iteration, first.reason)
        assert first.matched_reference_ids == ranking[:2]
        assert first.remaining_candidate_ids == ranking[2:10]
        assert (second.status, second.matched_reference_ids, second.remaining_candidate_ids) == (
            first.status, first.matched_reference_ids, first.remaining_candidate_ids)
        assert first.model_ids == FakeBackend.model_ids  # map identities stay unchanged
        assert set(stub.queries) == {("fr", scene.t_ns(4))}


def test_retrieval_candidates_keep_only_mapped_same_camera_ids(scene, landmark_map):
    stub = StubRetrieval((99, 5, 3, 1))  # 99 is not a keyframe of this map
    worker = vision.DiskAnchorWorker(landmark_map, FakeBackend(scene, 4), retrieval=stub)
    assert worker.candidates(_request(scene, landmark_map, 4)) == (5, 3, 1)
    stub.ranking = ()
    result = worker.run(_request(scene, landmark_map, 4))
    assert result.status is c.AnchorStatus.FAILED and result.reason == "no_reference_candidates"
    with pytest.raises(c.ContractError, match="retrieval"):
        vision.DiskAnchorWorker(landmark_map, FakeBackend(scene, 4), retrieval=object())


# Feature: stage3-gate-fix-round2, Property 5: Preservation - Supported, no-index and named-reference paths
def test_named_reference_and_no_index_paths_are_unchanged(scene, wide_map):
    """Named references never query the index and give the same candidates and result as
    the no-index worker; empty references without an index keep the geometric search.

    **Validates: Requirements 3.1, 3.2, 3.3**
    """
    ids = [int(k) for k in wide_map.keyframe_ids]
    plain = vision.DiskAnchorWorker(wide_map, FakeBackend(scene, 12))
    for iteration in range(12):
        rng = f.seeded_rng(612, iteration)
        stub = StubRetrieval(tuple(int(k) for k in rng.permutation(ids)[:10]))
        indexed = vision.DiskAnchorWorker(wide_map, FakeBackend(scene, 12), retrieval=stub)
        named = tuple(int(k) for k in rng.choice(ids + [99], int(rng.integers(1, 3)),
                                                 replace=False))
        request = _request(scene, wide_map, 12, reference_ids=named)
        assert indexed.candidates(request) == plain.candidates(request)
        with_index, without = indexed.run(request), plain.run(request)
        assert (with_index.status, with_index.reason, with_index.matched_reference_ids,
                with_index.remaining_candidate_ids) == (
            without.status, without.reason, without.matched_reference_ids,
            without.remaining_candidate_ids)
        assert stub.queries == []
    empty = _request(scene, wide_map, 12)
    x, y, yaw = (float(v) for v in empty.snapshot.state[:3])
    expected = wide_map.select_references(
        "fr", x, y, yaw,
        sigma_xy_m=min(vision.snapshot_sigma_xy(empty.snapshot), plain.config.max_sigma_xy_m),
        base_radius_m=plain.config.reference_radius_m, max_heading_rad=plain.config.max_heading_rad)
    assert plain.candidates(empty) == expected

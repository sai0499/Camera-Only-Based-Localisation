"""Frozen spatial feature selection retains weaker lower-image evidence within the same caps."""
import dataclasses

import numpy as np

import build_map
from camloc import contracts as c, features


def test_lower_features_survive_horizon_score_dominance_and_stay_sorted():
    n = 2000
    rng = np.random.default_rng(4)
    points = np.column_stack([rng.uniform(1, 958, n), np.r_[
        rng.uniform(1, 100, 1800), rng.uniform(180, 260, 200)]]).astype(np.float32)
    scores = np.linspace(1., 0., n).astype(np.float32)
    descriptors = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 128))
    mask = np.zeros((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    mask[:280] = True
    plain = features.select_keypoints(points, scores, descriptors, mask)
    spatial = features.select_keypoints(points, scores, descriptors, mask, selection="grid-reserve-32")
    assert not (plain.keypoints[:, 1] >= 180).any()
    assert (spatial.keypoints[:, 1] >= 180).sum() >= 100
    assert len(spatial) == c.MAX_KEYPOINTS
    assert (np.diff(spatial.scores) <= 0).all()
    ids = spatial.descriptors[:, 0].astype(int)
    assert len(np.unique(ids)) == len(ids)
    np.testing.assert_array_equal(spatial.keypoints, points[ids])
    assert features.backend_identity() != features.backend_identity("grid-reserve-32")


def test_empty_spatial_selection_is_valid():
    result = features.select_keypoints(np.empty((0, 2)), np.empty(0), np.empty((0, 128)),
        np.zeros((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool), selection="grid-reserve-32")
    assert len(result) == 0


def test_wider_pair_pass_keeps_degree_cap_and_adds_useful_baselines():
    poses = np.tile(np.eye(4), (60, 1, 1))
    poses[:, 0, 3] = np.arange(60) * 1.1
    cfg = build_map.BuildConfig()
    near = build_map.select_pairs(["fr"] * 60, poses, cfg)
    wider = build_map.select_pairs(["fr"] * 60, poses,
                                  dataclasses.replace(cfg, pair_target_baseline_m=12.))
    assert np.bincount(np.array(wider).ravel()).max() <= cfg.neighbors
    assert len(set(wider)) == len(wider)
    assert sum(j-i >= 8 for i,j in wider) > sum(j-i >= 8 for i,j in near)

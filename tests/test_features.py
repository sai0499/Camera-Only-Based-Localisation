"""Step 1 toolchain check: the pinned DISK/LightGlue backend runs offline on canonical images."""

import cv2
import numpy as np
import pytest

from camloc import contracts as c
from camloc import features


@pytest.fixture(scope="module")
def backend():
    return features.FeatureBackend()


def _textured_image(seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH)).astype(np.float32)
    gray = cv2.GaussianBlur(noise, (0, 0), 2.0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))


def test_weights_are_verified_and_identified():
    weights = features.verified_weights()
    ids = features.model_identities(weights)
    assert ids[0].startswith("disk-depth:") and ids[1].startswith("disk-lightglue:")


def test_extraction_respects_mask_cap_and_canonical_pixels(backend):
    image = _textured_image()
    mask = np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    mask[240:] = False
    feats = backend.extract(image, mask)
    assert 0 < len(feats) <= c.MAX_KEYPOINTS
    assert feats.keypoints.dtype == np.float32 and feats.descriptors.shape == (len(feats), 128)
    assert np.all(np.rint(feats.keypoints[:, 1]) < 240)
    assert np.all(np.diff(feats.scores) <= 0)


def test_matching_recovers_a_known_shift(backend):
    image = _textured_image(1)
    shifted = np.roll(image, (3, 7), axis=(0, 1))
    mask = np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    a, b = backend.extract(image, mask), backend.extract(shifted, mask)
    pairs, scores = backend.match(a, b)
    assert len(pairs) > 100 and scores.dtype == np.float32
    flow = b.keypoints[pairs[:, 1]] - a.keypoints[pairs[:, 0]]
    np.testing.assert_allclose(np.median(flow, axis=0), [7.0, 3.0], atol=0.5)
    empty = features.Features(np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                              np.zeros((0, 128), np.float32))
    assert backend.match(empty, a)[0].shape == (0, 2)

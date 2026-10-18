"""Camera selection changes work priority, never geometric acceptance or source budgets."""
from types import SimpleNamespace

import numpy as np

from camloc import masks
from camloc.camera_policy import CameraPolicy


class Bank:
    keyframe_ids = np.array([1, 2])

    def keyframe(self, frame_id):
        pixels = (np.array([[x, y] for x in (100, 300, 600, 850) for y in (40, 130, 230, 340)])
                  if frame_id == 2 else np.array([[x, 130] for x in range(16)]))
        return SimpleNamespace(camera_id="cl" if frame_id == 2 else "fr", keypoints=pixels,
                               landmark_ids=np.arange(16))

    def mask(self, camera):
        return np.ones((464, 960), bool)

    def select_references(self, camera, x, y, yaw, **kwargs):
        return (1 if camera == "fr" else 2,) if x < 50 else ()


def test_spatially_spread_auxiliary_wins_and_failure_penalty_expires_in_source_time():
    policy = CameraPolicy(Bank(), ("fr", "cl"))
    prior = SimpleNamespace(state=np.zeros(6), P=np.eye(6))
    assert policy.rank(policy.scores(prior, 10**9, {"fr", "cl"}))[0] == "cl"
    policy.feedback("cl", False, 10**9)
    policy.feedback("cl", False, 10**9)
    assert policy.rank(policy.scores(prior, 10**9 + 1, {"fr", "cl"}))[0] == "fr"
    assert policy.rank(policy.scores(prior, 2*10**9, {"fr", "cl"}))[0] == "cl"
    assert policy.rank(policy.scores(prior, 2*10**9, {"fr"})) == ["fr"]


def test_unmapped_sector_reserves_front_global_recovery_and_ignores_future_feedback():
    policy = CameraPolicy(Bank(), ("fr", "cl"))
    prior = SimpleNamespace(state=np.array([100., 0., 0., 0., 0., 0.]), P=np.eye(6))
    assert policy.rank(policy.scores(prior, 10**9, {"fr", "cl"})) == ["fr"]
    prior.state[0] = 0.
    before = policy.scores(prior, 10**9, {"fr", "cl"})
    policy.feedback("cl", False, 2*10**9)
    assert policy.scores(prior, 10**9, {"fr", "cl"}) == before


def test_auxiliary_masks_remove_bodywork_without_changing_the_front_mask():
    np.testing.assert_array_equal(masks.bodywork_mask("fr", rescue=True), masks.bodywork_mask("fr"))
    np.testing.assert_array_equal(masks.bodywork_mask("fl", rescue=True), masks.bodywork_mask("fr")[:, ::-1])
    for camera, cutoff in (("cl", 200), ("cr", 200), ("rl", 360), ("rr", 260)):
        usable = masks.bodywork_mask(camera, rescue=True)
        assert usable[:cutoff].all() and not usable[cutoff:].any()
        assert usable.dtype == np.bool_
    assert set(masks.SUPPORTED_MASK_IDS) == {masks.MASK_ID, masks.RESCUE_MASK_ID}

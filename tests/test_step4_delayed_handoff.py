"""Step 4 pass test: delayed anchor results at 10/20/30/60 Hz with B's real VisionTracker.

A result delivered within HISTORY_MAX_AGE_NS over a complete same-camera chain is re-solved
on the current image (never the source pose); older results, missing chains and backend
faults are explicit. Earlier evidence records never change. Synthetic rendered scenes only.
"""

import copy

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import geometry as g
from camloc import vision

RATES_HZ = (10, 20, 30, 60)


class RateScene(vf.Scene):
    """The fixture scene sampled at an arbitrary camera rate (same 10 m/s motion)."""

    def __init__(self, rate_hz: int):
        self.step_ns = round(c.NS_PER_S / rate_hz)  # before Scene.__init__ calls pose()
        super().__init__()

    def t_ns(self, k: int) -> int:
        return vf.START_NS + k * self.step_ns

    def pose(self, k: int) -> np.ndarray:
        forward = vf.rot_z(self.yaw) @ np.array([1.0, 0.0, 0.0])
        position = self.origin + forward * vf.SPEED_MPS * k * self.step_ns / c.NS_PER_S
        return g.rigid_from_rotation_translation(vf.rot_z(self.yaw), position)


def _frames_for_age(scene: RateScene, age_ns: int) -> int:
    """Number of frames after the source whose exposure is first >= source + age."""
    return -(-age_ns // scene.step_ns)


def _result(scene, tracker, k, job_id=1):
    return f.make_anchor_result(
        job_id=job_id, source_exposure_ns=scene.t_ns(k), map_id=tracker.map.map_id,
        correspondences=scene.correspondences(k, tracker.map.map_id),
        matched_reference_ids=(1,), remaining_candidate_ids=())


def _setup(rate_hz, tmp_path):
    scene = RateScene(rate_hz)
    keyframes = tuple(sorted({0, _frames_for_age(scene, 400_000_000),
                              _frames_for_age(scene, 800_000_000)}))
    tracker = vision.VisionTracker(scene.build_map(tmp_path / f"map{rate_hz}", keyframes))
    return scene, tracker


@pytest.mark.parametrize("rate_hz", RATES_HZ)
@pytest.mark.parametrize("age_ms", (90, 150, 250))
def test_delayed_result_is_resolved_on_the_current_image(rate_hz, age_ms, tmp_path):
    scene, tracker = _setup(rate_hz, tmp_path)
    delay = _frames_for_age(scene, c.DELIVERY_MIN_DELAY_NS if age_ms == 90 else age_ms * 10**6)
    if scene.t_ns(delay) - scene.t_ns(0) > c.HISTORY_MAX_AGE_NS:
        delay -= 1  # the newest frame still inside the 0.25 s history bound
    for k in range(delay + 1):
        tracker.observe(scene.observation(k))
    observation = scene.observation(delay)
    handoff = tracker.handoff(_result(scene, tracker, 0), observation)
    assert handoff.accepted, (rate_hz, age_ms, handoff.reason)
    assert handoff.measurement_ns == scene.t_ns(delay) == handoff.pose.exposure_ns
    assert handoff.source_exposure_ns == scene.t_ns(0)
    assert 1 <= handoff.transitions <= c.MAX_HANDOFF_TRANSITIONS
    assert handoff.support_ns[0] == scene.t_ns(0) and handoff.support_ns[-1] == scene.t_ns(delay)
    assert handoff.correspondences.landmark_ids.size <= c.MAX_HANDOFF_POINTS
    error = np.linalg.norm(handoff.pose.T_WB[:3, 3] - scene.pose(delay)[:3, 3])
    source_error = np.linalg.norm(handoff.pose.T_WB[:3, 3] - scene.pose(0)[:3, 3])
    assert error < 0.05 < source_error  # current pose, not the stale source pose
    tracker.settle(handoff, c.HandoffDisposition.ADOPT)


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_result_older_than_history_is_stale(rate_hz, tmp_path):
    scene, tracker = _setup(rate_hz, tmp_path)
    delay = _frames_for_age(scene, c.HISTORY_MAX_AGE_NS + 1)
    for k in range(delay + 1):
        tracker.observe(scene.observation(k))
    handoff = tracker.handoff(_result(scene, tracker, 0), scene.observation(delay))
    assert not handoff.accepted and handoff.reason == "stale_result"
    assert handoff.pose is None
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_missing_source_image_is_rejected(rate_hz, tmp_path):
    scene, tracker = _setup(rate_hz, tmp_path)
    delay = _frames_for_age(scene, c.DELIVERY_MIN_DELAY_NS)
    for k in range(1, delay + 1):  # the source image was never observed
        tracker.observe(scene.observation(k))
    handoff = tracker.handoff(_result(scene, tracker, 0), scene.observation(delay))
    assert not handoff.accepted and handoff.reason == "missing_image_chain"
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)


def _snapshot(record):
    return {name: copy.deepcopy(np.array(getattr(record, name)) if isinstance(
        getattr(record, name), np.ndarray) else getattr(record, name))
        for name in record.__dataclass_fields__}


def _same(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if isinstance(a[name], np.ndarray):
            np.testing.assert_array_equal(a[name], b[name])
        elif isinstance(a[name], (c.PoseEvidence, c.Correspondences)):
            _same(_snapshot(a[name]), _snapshot(b[name]))
        else:
            assert a[name] == b[name], name


@pytest.mark.parametrize("rate_hz", RATES_HZ)
def test_delivery_never_changes_earlier_outputs(rate_hz, tmp_path):
    scene, tracker = _setup(rate_hz, tmp_path)
    first = _frames_for_age(scene, c.DELIVERY_MIN_DELAY_NS)
    for k in range(first + 1):
        tracker.observe(scene.observation(k))
    emitted = [tracker.handoff(_result(scene, tracker, 0, job_id=1), scene.observation(first))]
    tracker.settle(emitted[0], c.HandoffDisposition.ADOPT)
    k = first
    for _ in range(3):
        k += 1
        observation = scene.observation(k)
        tracker.observe(observation)
        emitted.append(tracker.track(observation, scene.snapshot(k),
                                     c.AuxiliaryGeometry.unavailable("not_estimated")))
    assert all(record.accepted for record in emitted), [r.reason for r in emitted]
    frozen = [_snapshot(record) for record in emitted]
    # A second delayed anchor from an earlier source is delivered and adopted afterwards.
    source = k - _frames_for_age(scene, c.DELIVERY_MIN_DELAY_NS)
    late = tracker.handoff(_result(scene, tracker, source, job_id=2), scene.observation(k))
    tracker.settle(late, c.HandoffDisposition.ADOPT)
    for record, before in zip(emitted, frozen, strict=True):
        _same(_snapshot(record), before)
        measured = getattr(record, "measurement_ns", getattr(record, "exposure_ns", None))
        assert measured <= late.measurement_ns


class _FailingBackend:
    model_ids = ("fake_disk:000", "fake_lightglue:000")

    def extract(self, image, mask):
        raise RuntimeError("CUDA error: device-side fault")

    def match(self, a, b):  # pragma: no cover - never reached
        raise AssertionError


def test_backend_fault_is_raised_never_turned_into_a_result(tmp_path):
    scene, tracker = _setup(20, tmp_path)
    worker = vision.DiskAnchorWorker(tracker.map, _FailingBackend())
    request = f.make_anchor_request(
        job_id=4, source_exposure_ns=scene.t_ns(4), observation=scene.observation(4),
        snapshot=scene.snapshot(4), reference_ids=(), map_id=tracker.map.map_id)
    with pytest.raises(RuntimeError, match="device-side"):
        worker.run(request)

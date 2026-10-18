"""B+C blackout integration: a > 3 s image blackout on the rendered scene, then restored images.

Real tracker/PnP/worker (B) with the real scheduler/filter/recovery/runner (C). During the
blackout the estimate must stay bounded past B = last full pose + 3 s (stage3 OU loss model),
rows must stay Lost/prediction-only with honest reasons, the frontend must keep proposing
anchor jobs, and after restoration a full pose may only come from accepted, non-blank evidence.
Synthetic software verification only; not localization accuracy.
"""

import json

import numpy as np

import contract_factories as f
import vision_fixtures as vf
from camloc import borders, contracts as c, filter as module, geometry, vision
from camloc.features import Features
from camloc.filter import CameraFilter
from camloc.frontend import VisualFrontend
from camloc.recovery import RecoveryManager
from camloc.scheduler import AnchorScheduler
from run import run_stream
from test_anchor_worker import FakeBackend
from validate_run import validate

SLOW_MPS = 2.0  # keeps the static landmarks in view across the whole replay
NORMAL = tuple(range(10))
BLACKOUT = tuple(range(10, 86, 2))  # 10 Hz blank images, 3.8 s
RESTORED = tuple(range(86, 106))
FULL = {c.EvidenceKind.PNP, c.EvidenceKind.HANDOFF_PNP, c.EvidenceKind.RECOVERY_RESET}


class SlowScene(vf.Scene):
    def pose(self, k):
        forward = vf.rot_z(self.yaw) @ np.array([1.0, 0.0, 0.0])
        position = self.origin + forward * SLOW_MPS * k * vf.STEP_NS / c.NS_PER_S
        return geometry.rigid_from_rotation_translation(vf.rot_z(self.yaw), position)


class RecordingFilter(CameraFilter):
    def __init__(self, initial):
        super().__init__(initial)
        self.history = []  # (estimate, last full-pose ns after the step)

    def step(self, *args, **kwargs):
        estimate = super().step(*args, **kwargs)
        self.history.append((estimate, self._last_full_ns()))
        return estimate


def _blackout_run(tmp_path):
    scene = SlowScene()
    bank = scene.build_map(tmp_path / "bank")
    ks = NORMAL + BLACKOUT + RESTORED
    blank = set(BLACKOUT)
    observations = {k: scene.observation(k, image=scene.image(k, blank=k in blank)) for k in ks}
    frames = [f.make_frame_set(group_id=g, selected_camera_ids=("fr",),
                               images=(f.make_image_message(exposure_ns=scene.t_ns(k)),))
              for g, k in enumerate(ks)]
    by_ns = {scene.t_ns(k): k for k in ks}

    class Canonicalizer:
        def canonicalize(self, image):
            return observations[by_ns[image.exposure_ns]]

    class Backend:
        model_ids = FakeBackend.model_ids

        def extract(self, image, mask):
            k = next(k for k, obs in observations.items() if np.array_equal(obs.image, image))
            if k in blank:  # blank image: no features at all
                return Features(np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                                np.zeros((0, 128), np.float32))
            return FakeBackend(scene, k).extract(image, mask)

        def match(self, a, b):
            return FakeBackend(scene, 0).match(a, b)

    initial = c.InitialPose(timestamp_ns=scene.t_ns(0), x=scene.origin[0], y=scene.origin[1],
                            yaw=scene.yaw)
    estimator = RecordingFilter(initial)
    tracker = vision.load_tracker(bank)
    worker = vision.DiskAnchorWorker(bank, Backend())
    road = geometry.load_road_geometry(c.PROJECT_ROOT / "23-00-38_cut_120m_update/map.mcap")
    frontend = VisualFrontend(Canonicalizer(), tracker, estimator, road=road,
                              borders=borders.make_border_source(road),
                              motion=vision.make_ground_motion())
    episode = c.EpisodeBounds(name="tune", start_ns=scene.t_ns(0), end_ns=scene.t_ns(ks[-1] + 1))
    csv, log = tmp_path / "submission.csv", tmp_path / "diagnostics.jsonl"
    counts = run_stream(frames, estimator, frontend, output=csv, diagnostics=log, episode=episode,
                        scheduler=AnchorScheduler(worker, map_id=bank.map_id),
                        recovery=RecoveryManager(), tracker=tracker)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    return scene, ks, frames, episode, csv, log, counts, records, estimator.history


def test_blackout_is_bounded_honest_keeps_searching_and_recovers_only_by_rules(tmp_path):
    scene, ks, frames, episode, csv, log, counts, records, history = _blackout_run(tmp_path)
    assert counts["frontend_errors"] == 0
    groups = {g: k for g, k in enumerate(ks)}
    first_blank = ks.index(BLACKOUT[0])
    first_restored = ks.index(RESTORED[0])
    assert len(history) == len(ks)

    # A full pose was fused before the blackout; none from blank images.
    assert any(set(e.evidence_kinds) & FULL for e, _ in history[:first_blank])
    blackout = history[first_blank:first_restored]
    assert all(not set(e.evidence_kinds) & FULL for e, _ in blackout)

    # B and the boundary state s_B (CV from the last full-pose estimate to B).
    last_full_ns = blackout[-1][1]
    boundary = last_full_ns + module.FULL_POSE_LOSS_NS
    assert boundary < blackout[-1][0].t_ns  # the blackout really crosses B
    anchor = max((e for e, _ in history[:first_blank] if set(e.evidence_kinds) & FULL),
                 key=lambda e: e.t_ns)
    later = [e for e, _ in history[:first_blank] if e.t_ns > anchor.t_ns]
    start = later[-1] if later else anchor
    s_B, _ = module._propagate(start.state, start.P, start.t_ns, boundary,
                               module._NOISE_SCALE[start.status], last_full_ns)
    reach = module.LOSS_TAU_S * float(np.hypot(*s_B[3:5])) + 0.5
    turn = module.LOSS_TAU_S * abs(float(s_B[5])) + 1e-3
    past = [e for e, _ in blackout if e.t_ns > boundary]
    assert past
    for estimate in past:
        assert np.all(np.isfinite(estimate.state)) and np.all(np.isfinite(estimate.P))
        assert float(np.hypot(*(estimate.state[:2] - s_B[:2]))) <= reach
        assert abs(c.wrap_angle(float(estimate.state[2] - s_B[2]))) <= turn
        # Honest rows: Lost, prediction-only, with a stated reason.
        assert estimate.status is c.Status.LOST
        assert list(estimate.evidence_kinds) == [c.EvidenceKind.PREDICTION_ONLY]
        assert estimate.reason

    # The frontend keeps proposing anchor jobs in every blank group past B.
    proposed = {r["group_id"] for r in records if r["record"] == "SchedulerEvent"
                and (r["kind"] == "created"
                     or (r["kind"] == "rate_limited" and r.get("stage") == "create"))}
    past_groups = {e.group_id for e in past}
    assert past_groups <= proposed

    # After restoration, any full pose comes only from accepted evidence on restored images.
    restored_ns = {scene.t_ns(k) for k in RESTORED}
    full_after = [e for e, _ in history[first_restored:] if set(e.evidence_kinds) & FULL]
    assert full_after  # the restored scene is re-acquired ...
    # ... first through an anchor-confirmed handoff/reset, never a bare tracker PnP from Lost.
    assert set(full_after[0].evidence_kinds) <= {c.EvidenceKind.HANDOFF_PNP,
                                                 c.EvidenceKind.RECOVERY_RESET}
    for estimate in full_after:
        assert groups[estimate.group_id] in RESTORED
        assert estimate.max_support_exposure_ns in restored_ns
    report = validate(csv, log, frames, episode)
    assert report["valid"] and report["rows"] == len(ks)

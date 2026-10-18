"""Production assembly uses real B tracking/PnP and C scheduling/filter on a known scene."""

import dataclasses
import json

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import borders, contracts as c, geometry, vision
from camloc.filter import CameraFilter
from camloc.frontend import VisualFrontend
from camloc.recovery import RecoveryManager
from camloc.scheduler import AnchorScheduler
from run import run_stream
from test_anchor_worker import FakeBackend
from validate_run import check_metrics, validate


def _scene_run(tmp_path, *, black=False):
    scene = vf.Scene()
    bank = scene.build_map(tmp_path / "bank")
    observations = [scene.observation(k, image=scene.image(k, blank=black and k >= 8))
                    for k in range(16)]
    images = [f.make_image_message(exposure_ns=obs.exposure_ns) for obs in observations]
    frames = [f.make_frame_set(group_id=k, selected_camera_ids=("fr",), images=(image,))
              for k, image in enumerate(images)]

    class Canonicalizer:
        def canonicalize(self, image):
            return observations[(image.exposure_ns - scene.t_ns(0)) // vf.STEP_NS]

    class Backend:
        model_ids = FakeBackend.model_ids

        def extract(self, image, mask):
            k = next(k for k, obs in enumerate(observations) if np.array_equal(obs.image, image))
            return FakeBackend(scene, k).extract(image, mask)

        def match(self, a, b):
            return FakeBackend(scene, 0).match(a, b)

    initial = c.InitialPose(timestamp_ns=scene.t_ns(0), x=scene.origin[0], y=scene.origin[1], yaw=scene.yaw)
    estimator = CameraFilter(initial)
    tracker = vision.load_tracker(bank)
    worker = vision.DiskAnchorWorker(bank, Backend())
    road = geometry.load_road_geometry(c.PROJECT_ROOT / "23-00-38_cut_120m_update/map.mcap")
    frontend = VisualFrontend(Canonicalizer(), tracker, estimator, road=road,
                              borders=borders.make_border_source(road), motion=vision.make_ground_motion())
    episode = c.EpisodeBounds(name="tune", start_ns=scene.t_ns(0), end_ns=scene.t_ns(16))
    csv, log = tmp_path / "submission.csv", tmp_path / "diagnostics.jsonl"
    counts = run_stream(frames, estimator, frontend, output=csv, diagnostics=log, episode=episode,
        scheduler=AnchorScheduler(worker, map_id=bank.map_id), recovery=RecoveryManager(), tracker=tracker)
    return counts, frames, episode, csv, log


@pytest.mark.parametrize("black", [False, True])
def test_real_tracker_scheduler_filter_and_validator(tmp_path, black):
    counts, frames, episode, csv, log = _scene_run(tmp_path, black=black)
    assert counts["frontend_errors"] == 0
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(r["record"] == "Estimate" and r["evidence_kinds"] == ["handoff_pnp"] for r in records)
    assert any(r["record"] == "Estimate" and r["evidence_kinds"] == ["pnp"] for r in records)
    report = validate(csv, log, frames, episode)
    assert report["valid"] and report["rows"] == 16 and report["visual_updates"] >= 2
    if black:
        assert counts["prediction_only"] > 2


@pytest.mark.parametrize("corrupt", ["timestamp", "yaw", "future", "truncate", "aborted"])
def test_validator_rejects_invalid_runs(tmp_path, corrupt):
    _, frames, episode, csv, log = _scene_run(tmp_path)
    rows = csv.read_text().splitlines()
    if corrupt in ("timestamp", "yaw"):
        values = rows[3].split(",")
        values[0 if corrupt == "timestamp" else 3] = "7" if corrupt == "timestamp" else "360"
        rows[3] = ",".join(values)
        csv.write_text("\n".join(rows) + "\n")
    elif corrupt == "truncate":
        csv.write_text("\n".join(rows[:-2]) + "\n")
    else:
        records = [json.loads(line) for line in log.read_text().splitlines()]
        if corrupt == "aborted":
            records.append({"record": "RunAborted"})
        else:
            record = next(r for r in records if r["record"] == "Estimate" and r["group_id"] is not None)
            record["max_support_exposure_ns"] = episode.end_ns + 1
        log.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(c.ContractError):
        validate(csv, log, frames, episode)


def test_missing_metric_sections_are_failure(tmp_path):
    (tmp_path / "original_xy.txt").write_text("scorer returned without metrics\n")
    with pytest.raises(c.ContractError, match="metric section"):
        check_metrics(tmp_path)


def test_manifest_metadata_tampering_changes_map_identity(tmp_path):
    vf.Scene().build_map(tmp_path / "bank")
    path = tmp_path / "bank/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["canonical_K"]["fr"][0][0] += 2
    path.write_text(json.dumps(manifest))
    from camloc.landmark_map import load_landmark_map

    with pytest.raises(c.ContractError, match="identity"):
        load_landmark_map(tmp_path / "bank")


def test_rejected_handoff_settles_once_and_cannot_be_held(tmp_path):
    scene = vf.Scene()
    bank = scene.build_map(tmp_path / "bank")
    tracker = vision.load_tracker(bank)
    current = scene.observation(1)
    tracker.observe(current)
    result = f.make_anchor_result(source_exposure_ns=scene.t_ns(0), map_id=bank.map_id,
        status=c.AnchorStatus.FAILED, reason="insufficient_matches", correspondences=None)
    handoff = tracker.handoff(result, current)
    with pytest.raises(c.ContractError, match="accepted"):
        tracker.settle(handoff, c.HandoffDisposition.CANDIDATE)
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)
    with pytest.raises(c.ContractError, match="settled"):
        tracker.settle(handoff, c.HandoffDisposition.DISCARD)


def test_observed_identity_must_match(tmp_path):
    scene = vf.Scene()
    tracker = vision.load_tracker(scene.build_map(tmp_path / "bank"))
    current = scene.observation(1)
    tracker.observe(current)
    with pytest.raises(c.ContractError, match="identity"):
        tracker.track(dataclasses.replace(current, source_id="different-image"), scene.snapshot(1),
                      scene.snapshot(1).auxiliary)


def test_worker_correspondences_are_provisional_until_current_pose_passes(tmp_path, monkeypatch):
    from camloc import pnp
    from test_anchor_worker import _request

    scene = vf.Scene()
    bank = scene.build_map(tmp_path / "bank")
    real_solve = pnp.solve_pnp

    def failed_gate(*args, **kwargs):
        return dataclasses.replace(real_solve(*args, **kwargs), reason="poor_image_coverage")

    monkeypatch.setattr(pnp, "solve_pnp", failed_gate)
    result = vision.DiskAnchorWorker(bank, FakeBackend(scene, 4)).run(_request(scene, bank, 4))
    assert result.status is c.AnchorStatus.OK and result.correspondences is not None
    tracker = vision.load_tracker(bank)
    for k in (4, 5, 6):
        tracker.observe(scene.observation(k))
    handoff = tracker.handoff(result, scene.observation(6))
    assert not handoff.accepted and handoff.reason == "poor_image_coverage"
    tracker.settle(handoff, c.HandoffDisposition.DISCARD)


def test_builder_gt_support_cannot_cross_fitting_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace as N

    import mcap.reader

    import build_map

    fit = c.load_split()["fit"]
    stamps = [fit.start_ns - 1, fit.start_ns, fit.start_ns + 10_000_000,
              fit.end_ns - 10_000_000, fit.end_ns, fit.end_ns + 1]
    transforms = [N(header=N(frame_id="map", stamp=N(sec=t // c.NS_PER_S, nanosec=t % c.NS_PER_S)),
        child_frame_id="constructor0/base_link", transform=N(translation=N(x=1., y=2., z=3.),
            rotation=N(x=0., y=0., z=0., w=1.))) for t in stamps]

    class Reader:
        def iter_decoded_messages(self, **kwargs):
            yield None, None, None, N(transforms=transforms)

    monkeypatch.setattr(mcap.reader, "make_reader", lambda *args, **kwargs: Reader())
    bag = tmp_path / "offline.mcap"
    bag.write_bytes(b"")
    track = build_map.read_gt_track(bag, fit.start_ns, fit.end_ns, support_bounds=fit)
    assert track.t_ns.tolist() == stamps[1:4]
    assert track.pose(fit.start_ns - 1) is None and track.pose(fit.end_ns) is None


def test_changed_future_images_do_not_change_completed_outputs(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    first = _scene_run(before)[3].read_text().splitlines()
    altered = _scene_run(after, black=True)[3].read_text().splitlines()
    assert first[:9] == altered[:9]  # header plus all eight unaffected exposures


def test_pnp_rejects_nonfinite_or_mismatched_correspondences():
    from camloc.pnp import solve_pnp

    scene = vf.Scene()
    pixels, visible = scene.project(0)
    points = scene.points_W[visible].copy()
    points[0, 0] = np.nan
    with pytest.raises(c.ContractError, match="finite"):
        solve_pnp(pixels[visible], points, scene.K, scene.T_BC)
    with pytest.raises(c.ContractError, match="equal"):
        solve_pnp(pixels[visible], scene.points_W[visible][:-1], scene.K, scene.T_BC)

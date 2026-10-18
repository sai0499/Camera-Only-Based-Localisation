"""Reject unsupported optimization decisions and out-of-split diagnostic reads."""

from types import SimpleNamespace

import numpy as np
import pytest

from camloc import contracts as c
from scripts import b_grid_check, b_handoff_probe, b_profile_vision


def row(stamp, accepted=True):
    return {"exposure_ns": stamp, "accepted": accepted, "T_WB": np.eye(4)}


@pytest.mark.parametrize("base,candidate", [([], []), ([row(1, False)], [row(1, False)]),
    ([row(1), row(2, False)], [row(1, False), row(2)]), ([row(1), row(2)], [row(1)])])
def test_fp16_cannot_pass_without_comparable_accepted_poses(base, candidate):
    result = b_profile_vision.compare(base, candidate, 100., 50.)
    assert not result["adopt_fp16"]
    if result["both_accepted"] == 0:
        assert result["pose_diff_max_m"] is result["pose_diff_max_deg"] is None


def test_fp16_rule_can_pass_with_paired_equivalent_poses():
    result = b_profile_vision.compare([row(1)], [row(1)], 100., 50.)
    assert result["adopt_fp16"] and all(result["checks"].values())
    assert result["pose_diff_max_m"] == result["pose_diff_max_deg"] == 0.


@pytest.mark.parametrize("module", [b_grid_check, b_handoff_probe])
@pytest.mark.parametrize("window", [(-1, 1), (0, 1000), (2, 1), (float("nan"), 1),
                                  (0, float("inf")), (0, 1e-12)])
def test_all_windows_are_checked_before_any_recording_read(module, window, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid diagnostic window reached a recording/GT reader")

    monkeypatch.setattr(module.cio, "read_calibrations", forbidden)
    monkeypatch.setattr(module.build_map, "read_gt_track", forbidden)
    monkeypatch.setattr("sys.argv", ["diagnostic", "--artifacts", "unused", "--out",
        str(tmp_path / "out"), "--episode", "tune", "--stretch", "0", "1", "--stretch",
        str(window[0]), str(window[1])])
    with pytest.raises(SystemExit, match="2"):
        module.main()
    assert not (tmp_path / "out").exists()


def test_image_timing_excludes_gt_loading(monkeypatch):
    episode = c.load_split()["fit"]
    args = SimpleNamespace(episode="fit", start_s=0., end_s=1., bag="unused", camera="fr")
    clock = [10.]
    monkeypatch.setattr(b_profile_vision.time, "perf_counter", lambda: clock[0])

    def recording_images(*args):
        clock[0] += .25
        yield object()

    truth = object()

    def read_truth(bag, start, end):
        assert (start, end) == (episode.start_ns, episode.start_ns + c.NS_PER_S)
        clock[0] += 30.
        return truth

    monkeypatch.setattr(b_profile_vision.build_map, "iter_images", recording_images)
    monkeypatch.setattr(b_profile_vision.build_map, "read_gt_track", read_truth)
    canonicalizer = SimpleNamespace(canonicalize=lambda image: "canonical")
    images, gt, image_ms = b_profile_vision.load_images(args, {"fr": None}, canonicalizer)
    assert images == ["canonical"] and gt is truth and image_ms == 250.

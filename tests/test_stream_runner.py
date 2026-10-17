"""Synthetic streaming integration; test-only frontends never imply real localization."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c
from camloc.filter import CameraFilter
from run import run_stream
from test_filter import T, group, initial, pose
from test_inference_isolation import scan_source


def no_evidence(frame, prior):
    assert prior.t_ns == frame.t_ns
    return f.make_group_evidence(frame_set=frame)


def test_stream_preserves_failed_and_weak_groups_and_separates_diagnostics(tmp_path):
    frames = [group(i, i * 100_000_000).frame_set for i in range(4)]

    def frontend(frame, prior):
        if frame.group_id == 1:
            raise c.InputError("synthetic decode failure")
        if frame.group_id == 2:
            rejected = f.make_pose_evidence(
                exposure_ns=frame.t_ns, accepted=False, reason="too_few_inliers")
            return f.make_group_evidence(frame_set=frame, rejected=(rejected,))
        return no_evidence(frame, prior)

    output, diagnostics = tmp_path / "submission.csv", tmp_path / "diagnostics.jsonl"
    counts = run_stream(frames, CameraFilter(initial()), frontend, output=output, diagnostics=diagnostics)
    assert counts == {"groups": 4, "frontend_errors": 1, "prediction_only": 4}
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["timestamp_ns"]) for row in rows] == [frame.t_ns for frame in frames]
    assert np.isfinite([[float(row[key]) for key in ("x", "y", "yaw")] for row in rows]).all()
    records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert records[0]["evidence_kinds"] == ["initial_pose"]
    assert sum(item["record"] == "FrontendError" for item in records) == 1
    estimates = [r for r in records if r["record"] == "Estimate" and r["group_id"] is not None]
    assert len(estimates) == len(rows) and all(r["status"] == "lost" for r in estimates)
    assert "too_few_inliers" in diagnostics.read_text()
    assert output.read_bytes().startswith(b"timestamp_ns,x,y,yaw\n")


def test_completed_prefix_ignores_future_evidence_and_repeats_exactly(tmp_path):
    frames = [group(i, i * 100_000_000).frame_set for i in range(5)]

    def run(name, change_future):
        def frontend(frame, prior):
            measured = pose(frame.t_ns, x=2 if change_future and frame.group_id >= 3 else 0)
            return f.make_group_evidence(frame_set=frame, pose=measured)

        output = tmp_path / f"{name}.csv"
        run_stream(frames, CameraFilter(initial()), frontend, output=output,
                   diagnostics=tmp_path / f"{name}.jsonl")
        return output.read_bytes().splitlines()

    first = run("first", False)
    assert first == run("repeat", False)
    assert first[:4] == run("changed", True)[:4]  # header plus three completed groups


@pytest.mark.parametrize("bad", ["before_initial", "outside_episode", "changed_calibration"])
def test_stream_rejects_input_boundary_violations(tmp_path, bad):
    frames = [group(0).frame_set, group(1, 100_000_000).frame_set]
    if bad == "before_initial":
        frames = [f.make_frame_set(selected_camera_ids=("fr",),
                                  images=(f.make_image_message(exposure_ns=T - 1),))]
    elif bad == "changed_calibration":
        frames[1] = f.make_frame_set(
            group_id=1, selected_camera_ids=("fr",),
            images=(f.make_image_message(exposure_ns=T + 100_000_000, calibration_id="other"),))
    episode = c.EpisodeBounds(name="synthetic", start_ns=T, end_ns=T + 100_000_000)
    with pytest.raises(c.ContractError):
        run_stream(frames, CameraFilter(initial()), no_evidence,
                   episode=episode if bad == "outside_episode" else None,
                   output=tmp_path / "output.csv", diagnostics=tmp_path / "diagnostics.jsonl")


def test_existing_diagnostics_are_preserved_without_creating_submission(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    diagnostics.write_text("keep")
    output = tmp_path / "submission.csv"
    with pytest.raises(FileExistsError):
        run_stream([], CameraFilter(initial()), no_evidence, output=output, diagnostics=diagnostics)
    assert diagnostics.read_text() == "keep" and not output.exists()
    with pytest.raises(c.ContractError):
        run_stream([], CameraFilter(initial()), no_evidence, output=output, diagnostics=output)


def test_wrong_frontend_group_is_logged_and_becomes_prediction_only(tmp_path):
    counts = run_stream([group(0).frame_set], CameraFilter(initial()),
                        lambda frame, prior: group(0), output=tmp_path / "output.csv",
                        diagnostics=tmp_path / "diagnostics.jsonl")
    assert counts["frontend_errors"] == 1 and counts["prediction_only"] == 1


def test_runner_imports_remain_isolated_from_reference_code():
    root = Path(__file__).resolve().parents[1]
    assert not scan_source((root / "run.py").read_text(), "run.py")


def test_trailing_events_do_not_invent_output_groups(tmp_path):
    class Source:
        def calibrations(self):
            return {}

        def __iter__(self):
            yield group(0).frame_set
            self.exhausted = True

        def trailing_events(self):
            assert self.exhausted
            return (c.InputEvent(kind=c.InputEventKind.END_OF_INPUT, log_ns=T + 100_000_000),)

    output, diagnostics = tmp_path / "out.csv", tmp_path / "diag.jsonl"
    counts = run_stream(Source(), CameraFilter(initial()), no_evidence,
                        output=output, diagnostics=diagnostics)
    assert counts["groups"] == 1 and len(output.read_text().splitlines()) == 2
    records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert any(item.get("kind") == "end_of_input" for item in records)


def test_raw_recording_cli_fails_clearly_until_frontend_integration(tmp_path):
    script = Path(__file__).resolve().parents[1] / "run.py"
    result = subprocess.run([sys.executable, "-B", str(script)], cwd=tmp_path,
                            text=True, capture_output=True, check=False, timeout=30)
    assert result.returncode != 0
    assert "requires A/B" in result.stderr
    assert list(tmp_path.iterdir()) == []

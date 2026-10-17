"""Synthetic end-to-end smoke run of the Stage 2 core (default ``run_stream`` path).

A test-only frontend chooses each group's evidence by group index and builds it from the
synthetic ``stage4_fakes.gt_pose`` trajectory. Passing these checks is synthetic software
verification of the runner/filter wiring only. It is not a localization accuracy, runtime
or native high-rate result. Files are written only under pytest's ``tmp_path``.
"""

import csv
import json
import math
import time

import numpy as np

import contract_factories as f
import stage4_fakes as s4
from camloc import contracts as c
from camloc.filter import CameraFilter
from run import run_stream

N_GROUPS = 40
RATE_HZ = 10
PREFIX_K = 20
CHANGED_PAYLOAD = b"\xff\xd8\xff\xe0changed_future\xff\xd9"
HEADER = b"timestamp_ns,x,y,yaw"
PATTERN = ("pnp", "out_of_gate", "pnp", "motion", "pnp", "border",
           "pnp", "exception", "pnp", "prediction")
EXPECTED_KINDS = {
    "pnp": [c.EvidenceKind.PNP.value],
    "out_of_gate": [c.EvidenceKind.PREDICTION_ONLY.value],
    "motion": [c.EvidenceKind.GROUND_MOTION.value],
    "border": [c.EvidenceKind.BORDER.value],
    "exception": [c.EvidenceKind.PREDICTION_ONLY.value],
    "prediction": [c.EvidenceKind.PREDICTION_ONLY.value],
}


def group_kind(group_id):
    """Groups 0-11 are in-gate PnP; then the ten-group design pattern repeats."""
    return "pnp" if group_id < 12 else PATTERN[(group_id - 12) % len(PATTERN)]


def initial_pose():
    # The first `fr` exposure, 0.5 ms before the first group time (its `r` exposure).
    return c.InitialPose(timestamp_ns=s4.T0, x=0.0, y=0.0, yaw=0.0)


def road_aux(stamp):
    return c.AuxiliaryGeometry(source=c.AuxSource.ROAD_PATCH, source_ns=stamp, z_m=0.3,
                               roll_rad=0.0, pitch_rad=0.0, road_patch_id="road_patch_0",
                               valid=True, reason="valid")


def pnp(stamp, rng, x_offset=0.0):
    x, y, yaw = s4.gt_pose(stamp)
    noise = rng.uniform(-0.05, 0.05, size=2)
    yaw_noise = math.radians(rng.uniform(-0.1, 0.1))
    return f.make_pose_evidence(
        exposure_ns=stamp,
        T_WB=s4.pose_matrix(x + noise[0] + x_offset, y + noise[1], yaw + yaw_noise, z=0.3),
        inlier_count=64, track_ids=np.arange(64), landmark_ids=np.arange(64) + 100,
        inlier_fraction=0.8, occupied_cells=8, median_error_px=0.5, p90_error_px=1.0,
    )


def make_frontend(estimator, frames, *, noise_seed=0, changed_after=None):
    """Return the group-indexed smoke frontend; groups after ``changed_after`` use a new seed."""
    fr_ns = [s4.image_of(frame, "fr").exposure_ns for frame in frames]
    gated_out = []

    def frontend(frame, prior):
        g = frame.group_id
        seed = noise_seed + (1 if changed_after is not None and g > changed_after else 0)
        rng = np.random.default_rng([seed, g])
        stamp = fr_ns[g]
        kind = group_kind(g)
        if kind == "pnp":
            return f.make_group_evidence(frame_set=frame, pose=pnp(stamp, rng))
        if kind == "out_of_gate":
            pose = pnp(stamp, rng, x_offset=50.0)
            assert not estimator.pose_in_gate(pose)
            gated_out.append(g)
            return f.make_group_evidence(
                frame_set=frame, unused=(c.UnusedEvidence(reason="innovation_gate", evidence=pose),))
        if kind == "motion":
            previous = fr_ns[g - 1]
            delta = f.rigid_inverse(s4.gt_transform(previous, z=0.3)) @ s4.gt_transform(stamp, z=0.3)
            motion = f.make_motion_evidence(previous_ns=previous, current_ns=stamp,
                                            T_Bprev_Bcurr=delta)
            return f.make_group_evidence(frame_set=frame, motion=motion, auxiliary=road_aux(stamp))
        if kind == "border":
            heading = float(prior.state[2])
            normal = np.array([-math.sin(heading), math.cos(heading)])
            x, y, yaw = s4.gt_pose(stamp)
            border = f.make_border_evidence(exposure_ns=stamp, normal_xy=normal,
                                            lateral_coordinate_m=float(normal @ [x, y]), yaw_rad=yaw)
            return f.make_group_evidence(frame_set=frame, border=border)
        if kind == "exception":
            raise c.InputError(f"synthetic frontend failure in group {g}")
        return f.make_group_evidence(frame_set=frame)

    frontend.gated_out = gated_out
    return frontend


def smoke_run(directory, name, *, frames=None, noise_seed=0, changed_after=None):
    frames = s4.synthetic_stream(N_GROUPS, RATE_HZ) if frames is None else frames
    estimator = CameraFilter(initial_pose())
    frontend = make_frontend(estimator, frames, noise_seed=noise_seed, changed_after=changed_after)
    output, diagnostics = directory / f"{name}.csv", directory / f"{name}.jsonl"
    counts = run_stream(frames, estimator, frontend, output=output, diagnostics=diagnostics)
    records = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()]
    return {"frames": frames, "counts": counts, "output": output, "records": records,
            "gated_out": frontend.gated_out}


def group_estimates(records):
    return [r for r in records if r["record"] == "Estimate" and r["group_id"] is not None]


def test_smoke_run_drives_every_evidence_path(tmp_path):
    start = time.perf_counter()
    run = smoke_run(tmp_path, "smoke")
    frames = run["frames"]

    assert len(frames) == N_GROUPS >= 30
    assert all([image.camera_id for image in frame.images] == ["fr", "r"] for frame in frames)
    assert all(b.t_ns - a.t_ns == 100_000_000 for a, b in zip(frames, frames[1:]))
    assert len(str(frames[0].t_ns)) == 19
    assert frames[0].t_ns - initial_pose().timestamp_ns == 500_000

    estimates = group_estimates(run["records"])
    assert [e["group_id"] for e in estimates] == list(range(N_GROUPS))
    kinds = {group_kind(g) for g in range(N_GROUPS)}
    assert kinds == set(EXPECTED_KINDS)
    for estimate in estimates:
        expected = EXPECTED_KINDS[group_kind(estimate["group_id"])]
        assert estimate["evidence_kinds"] == expected, estimate["group_id"]
    assert run["gated_out"] == [g for g in range(N_GROUPS) if group_kind(g) == "out_of_gate"]
    for g in run["gated_out"]:
        assert c.EvidenceKind.PNP.value not in estimates[g]["evidence_kinds"]
        assert estimates[g]["reason"] == "no_evidence"

    raised = [g for g in range(N_GROUPS) if group_kind(g) == "exception"]
    assert raised and raised[-1] == N_GROUPS - 1 and raised[0] < N_GROUPS - 1
    for g in raised:  # prediction-only, and the run continued to the next FrameSet
        assert estimates[g]["evidence_kinds"] == [c.EvidenceKind.PREDICTION_ONLY.value]
        if g + 1 < N_GROUPS:
            assert estimates[g + 1]["evidence_kinds"] == [c.EvidenceKind.PNP.value]

    lines = run["output"].read_bytes().splitlines()
    assert lines[0] == HEADER and len(lines) == N_GROUPS + 1
    assert [int(line.split(b",")[0]) for line in lines[1:]] == [frame.t_ns for frame in frames]
    assert run["counts"] == {"groups": N_GROUPS, "frontend_errors": len(raised),
                             "prediction_only": sum(e["evidence_kinds"] == ["prediction_only"]
                                                    for e in estimates)}
    assert time.perf_counter() - start < 10.0


def test_smoke_csv_passes_unchanged_reference_validation(tmp_path):
    from test_submission import _reference_eval

    ev = _reference_eval()
    run = smoke_run(tmp_path, "smoke")
    frames = run["frames"]
    assert run["output"].read_bytes().splitlines()[0] == HEADER

    # Synthetic GT: 10 ms spacing from the InitialPose time, plus the exact last group time.
    first, last = initial_pose().timestamp_ns, frames[-1].t_ns
    gt_ns = np.unique(np.r_[np.arange(first, last, 10_000_000, dtype=np.int64), last])
    gt_poses = np.array([s4.gt_pose(int(t)) for t in gt_ns])
    gt = {"timestamp_ns": gt_ns, "x": gt_poses[:, 0], "y": gt_poses[:, 1], "yaw": gt_poses[:, 2]}
    stamps = np.array(sorted({image.exposure_ns for frame in frames for image in frame.images}),
                      dtype=np.int64)

    submission = ev.read_csv(run["output"])  # rows are never trimmed, rewritten or re-sorted
    result = ev.validate_submission(submission, gt, stamps)
    assert result["coverage"] >= 0.90
    assert result["evaluated"] == result["submitted"] == N_GROUPS


def test_smoke_diagnostics_are_complete_and_counted(tmp_path):
    run = smoke_run(tmp_path, "smoke")
    records = run["records"]
    by_kind = {}
    for record in records:
        by_kind.setdefault(record["record"], []).append(record)

    assert records[0]["record"] == "Estimate" and records[0]["group_id"] is None
    assert [r["group_id"] for r in by_kind["FrameSet"]] == list(range(N_GROUPS))
    assert [r["frame_set"]["group_id"] for r in by_kind["GroupEvidence"]] == list(range(N_GROUPS))
    estimates = group_estimates(records)
    assert [r["group_id"] for r in estimates] == list(range(N_GROUPS))
    raised = [g for g in range(N_GROUPS) if group_kind(g) == "exception"]
    assert [r["group_id"] for r in by_kind["FrontendError"]] == raised
    assert all(r["exception"] == "InputError" for r in by_kind["FrontendError"])

    assert len(by_kind["RunSummary"]) == 1 and records[-1]["record"] == "RunSummary"
    prediction_only = sum(r["evidence_kinds"] == ["prediction_only"] for r in estimates)
    assert records[-1] == {"record": "RunSummary", "groups": len(by_kind["FrameSet"]),
                           "frontend_errors": len(by_kind["FrontendError"]),
                           "prediction_only": prediction_only}


def test_smoke_run_repeats_exactly_and_its_prefix_ignores_the_future(tmp_path):
    first = smoke_run(tmp_path, "first")
    repeat = smoke_run(tmp_path, "repeat")
    assert first["output"].read_bytes() == repeat["output"].read_bytes()
    assert s4.strip_runtime(first["records"]) == s4.strip_runtime(repeat["records"])
    assert first["counts"] == repeat["counts"]

    # Groups after k get different JPEG payload bytes and a different evidence noise seed.
    base = s4.synthetic_stream(N_GROUPS, RATE_HZ)
    changed_images = s4.synthetic_stream(N_GROUPS, RATE_HZ, payload=CHANGED_PAYLOAD)
    frames = base[:PREFIX_K + 1] + changed_images[PREFIX_K + 1:]
    changed = smoke_run(tmp_path, "changed", frames=frames, changed_after=PREFIX_K)
    assert PREFIX_K + 1 >= 5 and N_GROUPS - PREFIX_K - 1 >= 5
    original_lines = first["output"].read_bytes().splitlines()
    changed_lines = changed["output"].read_bytes().splitlines()
    assert changed_lines[:PREFIX_K + 2] == original_lines[:PREFIX_K + 2]  # header + rows 0..k
    assert changed_lines[PREFIX_K + 2:] != original_lines[PREFIX_K + 2:]  # the change took effect


def test_smoke_rows_stay_near_the_synthetic_trajectory(tmp_path):
    # Filter-wiring sanity bound on synthetic data, NOT a localization result.
    run = smoke_run(tmp_path, "smoke")
    t_initial = initial_pose().timestamp_ns
    with run["output"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    checked = 0
    for row in rows:
        t_ns = int(row["timestamp_ns"])
        if t_ns - t_initial <= c.NS_PER_S:
            continue
        x, y, yaw = s4.gt_pose(t_ns)
        position_error = math.hypot(float(row["x"]) - x, float(row["y"]) - y)
        yaw_error = abs(math.degrees(c.wrap_angle(float(row["yaw"]) - yaw)))
        assert position_error <= 2.0, (t_ns, position_error)
        assert yaw_error <= 5.0, (t_ns, yaw_error)
        checked += 1
    assert checked >= 25

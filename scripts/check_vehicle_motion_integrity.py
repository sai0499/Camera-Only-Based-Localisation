"""Bounded real-recording proof: pose-field isolation and future-motion prefix causality.

Writes fresh ignored MCAP variants and runs the unchanged production CLI. No GT pose is
decoded or used for scoring. The one initial-pose JSON is supplied by the offline harness.
"""
import argparse
import dataclasses
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camloc import contracts as c  # noqa: E402
from camloc.vehicle_motion import _align, _header, decode_sample  # noqa: E402
from scripts.check_step5 import compare, copy_messages, dump, variant  # noqa: E402


def poison(data):
    """Retain only the original header and exactly wz/vx/vy bytes of StateEstimation."""
    _, _, offset, _ = _header(data)
    allowed = _align(_align(offset, 4) + 8, 8) + 12 * 8 + 16
    changed = bytearray(data)
    changed[offset:] = b"\xff" * (len(data) - offset)
    changed[allowed:allowed+24] = data[allowed:allowed+24]
    return bytes(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bag", type=Path, default=ROOT/"c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-images", action="store_true",
                        help="also poison future images of all seven cameras and compare the earlier prefix")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output must be fresh")
    initial = c.read_initial_pose(args.initial_pose)
    start, end = initial.timestamp_ns, initial.timestamp_ns + 30*c.NS_PER_S
    if not c.load_split()["tune"].start_ns <= start < end <= c.load_split()["tune"].end_ns:
        parser.error("audit must stay in tuning")
    args.output.mkdir(parents=True)
    cases = {}
    cutoff = start + 10*c.NS_PER_S
    names = ["baseline", "pose_poison", "future_motion"] + (["future_images"] if args.future_images else [])
    for case in names:
        path = args.output/(case+".mcap")
        motion_digest = hashlib.sha256()

        def select(channel, message):
            return (channel.topic == "/tf_static" or channel.topic.endswith("/camera_info")
                    or start-c.NS_PER_S <= message.log_time < end+500_000_000)

        def transform(schema, channel, message):
            data = message.data
            if channel.topic == "/tf" and case == "pose_poison":
                return b"DYNAMIC_TF_MUST_NOT_BE_DECODED"
            if channel.topic == "/constructor0/state_estimation":
                if case == "pose_poison":
                    data = poison(data)
                sample = decode_sample(data, message.log_time, "state-fields")
                if case == "future_motion" and sample.t_ns >= cutoff:
                    _, _, offset, endian = _header(data)
                    allowed = _align(_align(offset, 4) + 8, 8) + 12 * 8 + 16
                    data = bytearray(data)
                    struct.pack_into(endian+"3d", data, allowed, .25, 10., 0.)
                    data = bytes(data)
                    sample = decode_sample(data, message.log_time, "state-fields")
                motion_digest.update(json.dumps(dataclasses.asdict(sample), sort_keys=True).encode())
            return data

        if case == "future_images":
            receipt = variant(args.output/"baseline.mcap", path, "future", start, cameras=c.CAMERA_IDS)
        else:
            receipt = copy_messages(args.bag, path, select, transform)
            receipt["allowed_motion_sha256"] = motion_digest.hexdigest()
        folder = args.output/case
        command = [sys.executable, "-u", "-B", str(ROOT/"run.py"), "--bag", str(path),
            "--artifacts", str(args.artifacts), "--initial-pose", str(args.initial_pose),
            "--input-recording", "--end-ns", str(end), "--vehicle-motion", "state-fields",
            "--retrieval", "--output", str(folder)]
        with (args.output/(case+".log")).open("wb") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        cases[case] = receipt
        print(case+" complete", flush=True)
    baseline = args.output/"baseline"
    isolation = compare(baseline,args.output/"pose_poison")
    future = compare(baseline,args.output/"future_motion",before_ns=cutoff)
    assert cases["baseline"]["permitted_messages_sha256"] == cases["pose_poison"]["permitted_messages_sha256"]
    assert cases["baseline"]["allowed_motion_sha256"] == cases["pose_poison"]["allowed_motion_sha256"]
    assert cases["pose_poison"]["messages"]["changed"] > 0
    assert cases["future_motion"]["allowed_motion_sha256"] != cases["baseline"]["allowed_motion_sha256"]
    report = {"scope":"fixed 30 s tuning integrity audit; no accuracy scoring or GT decoding",
        "episode":{"start_ns":str(start),"end_ns":str(end)}, "cases":cases,
        "pose_isolation":isolation,"future_motion_prefix":future,
        "passed":isolation["passed"] and future["passed"]}
    if args.future_images:
        report["future_image_prefix"] = compare(baseline, args.output/"future_images", before_ns=cutoff)
        assert cases["future_images"]["changed_images"] > 0
        assert set(cases["future_images"]["changed_camera_ids"]) == set(c.CAMERA_IDS)
        report["passed"] &= report["future_image_prefix"]["passed"]
    dump(args.output/"report.json", report)
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

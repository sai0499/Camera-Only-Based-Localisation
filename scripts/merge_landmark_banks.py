"""Append disjoint-camera fitting rescue coverage without changing the original map rows."""
import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import build_map  # noqa: E402
from camloc import contracts as c, landmark_map as lm, masks  # noqa: E402


def merge(base_path, rescue_path, output):
    base, rescue = (lm.load_landmark_map(path) for path in (base_path, rescue_path))
    if set(base.cameras) & set(rescue.cameras):
        raise c.ContractError("rescue cameras must be disjoint from the original bank")
    build_map.check_reference_count(len(base.keyframe_ids) + len(rescue.keyframe_ids))
    for key in ("source_bag", "split_sha256", "supplied_map_sha256", "fitting_interval_ns",
                "preprocessing_id", "feature_backend", "model_ids", "model_sha256"):
        if base.manifest.get(key) != rescue.manifest.get(key):
            raise c.ContractError(f"rescue bank {key} differs")
    if rescue.mask_id != masks.RESCUE_MASK_ID:
        raise c.ContractError("auxiliary rescue bank requires inspected rescue masks")
    fit = c.load_split()["fit"]
    for bank in (base, rescue):
        if np.any((bank.keyframe_exposure_ns < fit.start_ns) | (bank.keyframe_exposure_ns >= fit.end_ns)):
            raise c.ContractError("bank keyframes outside fitting interval")
    arrays = []
    for bank in (base, rescue):
        with np.load(bank.root / lm.ARRAYS_NAME, allow_pickle=False) as archive:
            arrays.append({key: archive[key] for key in archive.files})
    first, second = arrays
    landmark_offset = int(first["landmark_ids"][-1]) + 1 if len(first["landmark_ids"]) else 0
    frame_offset = int(first["keyframe_ids"][-1]) + 1
    second["landmark_ids"] += landmark_offset
    linked = second["keypoint_landmarks"] >= 0
    second["keypoint_landmarks"][linked] += landmark_offset
    second["keyframe_ids"] += frame_offset
    joined = {key: np.concatenate([first[key], second[key]]) for key in first
              if not key.startswith("mask_") and key != "keypoint_offsets"}
    joined["keypoint_offsets"] = np.r_[first["keypoint_offsets"],
        second["keypoint_offsets"][1:] + first["keypoint_offsets"][-1]]
    for bank, part in zip((base, rescue), arrays, strict=True):
        for camera in bank.cameras:
            joined[f"mask_{camera}"] = part[f"mask_{camera}"]
    manifest = {**base.manifest, "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": "scripts/merge_landmark_banks.py", "cameras": [*base.cameras, *rescue.cameras],
        "mask_id": masks.RESCUE_MASK_ID, "camera_policy": "support-v1",
        "rescue_provenance": {"base_map_id": base.map_id, "rescue_map_id": rescue.map_id,
            "base_arrays_preserved_as_prefix": True,
            "camera_policies_use_no_query_GT": True},
        "statistics": {"keyframes": len(joined["keyframe_ids"]),
            "landmarks": len(joined["landmark_ids"]),
            "keyframes_per_camera": {cam: int(np.count_nonzero(joined["keyframe_cameras"] == cam))
                                     for cam in [*base.cameras, *rescue.cameras]}}}
    for key in ("canonical_K", "T_BC", "raw_camera_geometry"):
        manifest[key] = {**base.manifest.get(key, {}), **rescue.manifest.get(key, {})}
    # The original bank covers the full fit; rescue windows are a strict subset.
    for stamp in rescue.keyframe_exposure_ns:
        if not any(int(a) <= stamp < int(b) for a, b in base.manifest["windows_ns"]):
            raise c.ContractError("rescue keyframe outside declared fitting windows")
    lm.save_landmark_map(output, joined, manifest)
    return lm.load_landmark_map(output).summary()


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rescue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.base, args.rescue, args.output), indent=2))

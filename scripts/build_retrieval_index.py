"""Build the frozen EigenPlaces descriptor index of a fitted bank (owner: B). Offline tool.

Decodes only the bank's eligible keyframe exposures (>= 30 landmark keypoints) from the
manifest's fitting windows, canonicalises them with A's remap, applies the bank's frozen
usable-pixel mask through ``camloc.retrieval.preprocess`` (the same function queries use)
and writes ``retrieval.npz`` + ``retrieval.json`` next to the bank. Existing index files
are never overwritten. No ground truth is read: keyframes are selected by id/exposure from
the map, and only image and calibration topics are decoded.

Example::

    .venv/Scripts/python.exe -B scripts/build_retrieval_index.py \
        --artifacts artifacts/fit-b-integrated-v2
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import build_map  # noqa: E402
from camloc import contracts as c  # noqa: E402
from camloc import geometry as geo  # noqa: E402
from camloc import io as cio  # noqa: E402
from camloc import landmark_map as lm  # noqa: E402
from camloc import retrieval as rt  # noqa: E402


def build(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    bank = Path(args.artifacts).resolve()
    arrays_path, meta_path = bank / rt.INDEX_ARRAYS_NAME, bank / rt.INDEX_META_NAME
    if arrays_path.exists() or meta_path.exists():
        raise SystemExit(f"{bank} already holds a retrieval index; refusing to overwrite")
    landmark_map = lm.load_landmark_map(bank)
    manifest = landmark_map.manifest
    if landmark_map.mask_id != rt.masks.MASK_ID:
        raise SystemExit("bank mask id differs from the current frozen mask")
    bag = Path(args.bag)
    source = manifest["source_bag"]
    if bag.name != source["name"] or bag.stat().st_size != source["bytes"]:
        raise SystemExit("bag name/size differs from the bank's source bag")
    if args.verify_bag_hash and build_map._sha256(bag) != source["sha256"]:
        raise SystemExit("bag SHA-256 differs from the bank's source bag")

    fit_start, fit_end = (int(v) for v in manifest["fitting_interval_ns"])
    windows = [(int(a), int(b)) for a, b in manifest["windows_ns"]]
    ids = rt.eligible_keyframe_ids(landmark_map)
    wanted: dict[str, dict[int, int]] = {}
    for kid in ids:
        frame = landmark_map.keyframe(int(kid))
        if not any(a <= frame.exposure_ns < b for a, b in windows) \
                or not fit_start <= frame.exposure_ns < fit_end:
            raise SystemExit(f"keyframe {int(kid)} lies outside the bank's fitting windows")
        wanted.setdefault(frame.camera_id, {})[frame.exposure_ns] = int(kid)

    print("Reading calibration ...", flush=True)
    calibrations = cio.read_calibrations(bag)
    for camera_id in wanted:
        cal = calibrations[camera_id]
        reason = landmark_map.compatibility(camera_id, cal.K, cal.T_BC, cal.preprocessing_id)
        if reason != "valid":
            raise SystemExit(f"camera {camera_id}: {reason}")
    canonicalizer = geo.make_canonicalizer({cam: calibrations[cam] for cam in wanted})
    model = rt.load_pinned_eigenplaces(args.device)
    print(f"EigenPlaces on {model.device}; {len(ids)} eligible keyframes of "
          f"{landmark_map.keyframe_ids.size}", flush=True)

    found: dict[int, np.ndarray] = {}
    batch: list[np.ndarray] = []
    batch_ids: list[int] = []

    def flush() -> None:
        if batch:
            for kid, desc in zip(batch_ids, model.descriptors(np.concatenate(batch)), strict=True):
                found[kid] = desc
            batch.clear()
            batch_ids.clear()

    for camera_id, exposures in wanted.items():
        mask = landmark_map.mask(camera_id)
        for start, end in windows:
            for image in build_map.iter_images(bag, camera_id, start, end,
                                               calibrations[camera_id]):
                kid = exposures.get(image.exposure_ns)
                if kid is None:
                    continue
                if kid in found or kid in batch_ids:
                    raise SystemExit(f"keyframe {kid} exposure decoded twice")
                observation = canonicalizer.canonicalize(image)
                batch.append(rt.preprocess(np.asarray(observation.image), mask))
                batch_ids.append(kid)
                if len(batch) >= args.batch_size:
                    flush()
                    print(f"  {len(found)}/{len(ids)}", flush=True)
        flush()
    missing = sorted(set(ids.tolist()) - set(found))
    if missing:
        raise SystemExit(f"{len(missing)} keyframe exposures not found, e.g. {missing[:5]}")

    keyframe_ids = np.array(sorted(found), np.int64)
    descriptors = np.stack([found[int(k)] for k in keyframe_ids]).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    camera_ids = np.array([landmark_map.keyframe(int(k)).camera_id for k in keyframe_ids],
                          dtype="<U2")
    provenance = {
        "builder": "build_retrieval_index.py",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": "Fitting-only EigenPlaces reference descriptors; appearance ranking only.",
        "canonical_preprocessing_id": landmark_map.preprocessing_id,
        "keyframes_per_camera": {cam: int(np.count_nonzero(camera_ids == cam))
                                 for cam in sorted(wanted)},
        "fitting_interval_ns": [str(fit_start), str(fit_end)],
        "windows_ns": [[str(a), str(b)] for a, b in windows],
        "source_bag": {**source, "sha256_verified_here": bool(args.verify_bag_hash)},
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("torch", "torchvision", "opencv-python", "numpy")},
        "device": model.device,
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    meta = rt.write_index(bank, landmark_map, descriptors, keyframe_ids, camera_ids,
                          model.checkpoint_sha256, provenance)
    rt.RetrievalIndex.load(bank, landmark_map, model)  # round-trip the bindings
    print(json.dumps({k: meta[k] for k in ("retrieval_id", "keyframes", "descriptor_sha256",
                                           "build_seconds")}, indent=2), flush=True)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", type=Path, required=True, help="fitted bank directory")
    parser.add_argument("--bag", type=Path,
                        default=c.PROJECT_ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verify-bag-hash", action="store_true",
                        help="re-hash the source bag (slow); default checks name and size")
    build(parser.parse_args(argv))


if __name__ == "__main__":
    main()

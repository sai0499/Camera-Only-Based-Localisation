"""Copy a frozen fitting bank with hash-bound raw camera proof; preserve arrays exactly.

For legacy banks only. Reads image calibration/static extrinsics from the original
SHA-256-verified fitting recording, never pose or motion topics. Rebinds an existing
retrieval index without changing descriptors. Outputs must be fresh.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from camloc import contracts as c, io, landmark_map as lm, retrieval as rt  # noqa: E402


def bind(source, output, bag):
    source, output, bag = Path(source), Path(output), Path(bag)
    if output.exists():
        raise c.ContractError("geometry-bound bank output must be fresh")
    bank = lm.load_landmark_map(source)
    provenance = bank.manifest["source_bag"]
    with bag.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != provenance["sha256"] or bag.stat().st_size != provenance["bytes"]:
        raise c.ContractError("raw geometry must come from the original hash-verified fitting recording")
    if bank.manifest["split_sha256"] != c.BUILD_SPLIT_SHA256:
        raise c.ContractError("fitting split differs")
    index = (rt.RetrievalIndex.load(source, bank, None)
             if (source / rt.INDEX_META_NAME).exists() else None)
    calibrations = io.read_calibrations(bag)
    for camera in bank.cameras:
        cal = calibrations[camera]
        if (cal.preprocessing_id != bank.preprocessing_id or not np.array_equal(cal.K, bank.K(camera))
                or not np.array_equal(cal.T_BC, bank.T_BC(camera))):
            raise c.ContractError("fitting calibration differs from the frozen bank")
    manifest = {**bank.manifest, "preprocessing_recipe": io.PREPROCESSING_RECIPE,
        "raw_camera_geometry": {cam: lm.raw_calibration_geometry(calibrations[cam]) for cam in bank.cameras},
        "raw_geometry_provenance": {"source_map_id": bank.map_id, "source_bag_sha256_verified": digest,
                                    "builder": "scripts/bind_map_geometry.py", "map_arrays_changed": False}}
    manifest.pop("map_id")
    manifest["map_id"] = lm._map_id(manifest)
    output.mkdir(parents=True)
    shutil.copyfile(source / lm.ARRAYS_NAME, output / lm.ARRAYS_NAME)
    (output / lm.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    bound = lm.load_landmark_map(output)
    if index is not None:
        rt.write_index(output, bound, index.descriptors, index.keyframe_ids, index.camera_ids,
            index.meta["checkpoint_sha256"], {**{k:v for k,v in index.meta.items() if k != "retrieval_id"},
                "geometry_binding_source_retrieval_id": index.retrieval_id})
        rt.RetrievalIndex.load(output, bound, None)
    assert (output / lm.ARRAYS_NAME).stat().st_size == (source / lm.ARRAYS_NAME).stat().st_size
    return bound.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bag", type=Path, default=ROOT/"c0_2026-10-13_22-53-42_camloc_0.mcap")
    args = parser.parse_args()
    print(json.dumps(bind(args.artifacts, args.output, args.bag), indent=2))

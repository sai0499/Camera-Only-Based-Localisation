"""Rescue-bank integration preserves geometry and identities and rejects incompatible input."""
import json

import numpy as np
import pytest

import vision_fixtures as vf
from camloc import contracts as c, landmark_map as lm, masks
from scripts.merge_landmark_banks import merge


def bank(path, camera, *, rescue=False, source="same"):
    original = vf.Scene().build_map(path.with_name(path.name + "-original"))
    with np.load(original.root / "map.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[f"mask_{camera}"] = arrays.pop("mask_fr")
    arrays["keyframe_cameras"][:] = camera
    fit = c.load_split()["fit"]
    arrays["keyframe_exposure_ns"] = fit.start_ns + np.arange(len(original.keyframe_ids))*100_000_000
    meta = {**original.manifest, "cameras": [camera],
        "canonical_K": {camera: original.K("fr").tolist()},
        "T_BC": {camera: original.T_BC("fr").tolist()},
        "windows_ns": [[str(fit.start_ns), str(fit.end_ns)]],
        "fitting_interval_ns": [str(fit.start_ns), str(fit.end_ns)],
        "source_bag": {"sha256": source},
        "mask_id": masks.RESCUE_MASK_ID if rescue else masks.MASK_ID}
    lm.save_landmark_map(path, arrays, meta)
    return lm.load_landmark_map(path)


def test_append_preserves_original_rows_and_offsets_every_new_identity(tmp_path):
    base = bank(tmp_path/"base", "fr")
    rescue = bank(tmp_path/"rescue", "cl", rescue=True)
    output = tmp_path/"joined"
    merge(base.root, rescue.root, output)
    joined = lm.load_landmark_map(output)
    assert joined.cameras == ("fr", "cl")
    assert joined.manifest["camera_policy"] == "support-v1"
    np.testing.assert_array_equal(joined.points_W[:len(base.landmark_ids)], base.points_W)
    np.testing.assert_array_equal(joined.keyframe_T_WC[:len(base.keyframe_ids)], base.keyframe_T_WC)
    np.testing.assert_array_equal(joined.keyframe_ids[:len(base.keyframe_ids)], base.keyframe_ids)
    offset = int(base.landmark_ids[-1]) + 1
    for frame_id in rescue.keyframe_ids:
        old = rescue.keyframe(int(frame_id))
        new = joined.keyframe(int(frame_id + base.keyframe_ids[-1] + 1))
        linked = old.landmark_ids >= 0
        np.testing.assert_array_equal(new.landmark_ids[linked], old.landmark_ids[linked] + offset)
        np.testing.assert_array_equal(joined.points(new.landmark_ids[linked]), rescue.points(old.landmark_ids[linked]))
    assert json.loads((output/"manifest.json").read_text())["rescue_provenance"]["base_arrays_preserved_as_prefix"]


@pytest.mark.parametrize("camera,source,reason", [("fr", "same", "disjoint"), ("cl", "different", "source_bag")])
def test_invalid_append_fails_before_creating_an_output(tmp_path, camera, source, reason):
    base = bank(tmp_path/"base", "fr")
    rescue = bank(tmp_path/"rescue", camera, rescue=True, source=source)
    with pytest.raises(c.ContractError, match=reason):
        merge(base.root, rescue.root, tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()

"""A camera-specific raw proof permits unrelated mount changes, never remap changes."""
import dataclasses

import numpy as np
import pytest

from camloc import contracts as c, io, landmark_map as lm
from test_geometry import _calibration


def manifest(cal):
    return {"cameras": ["fr"], "preprocessing_id": cal.preprocessing_id,
        "preprocessing_recipe": io.PREPROCESSING_RECIPE,
        "canonical_K": {"fr": cal.K.tolist()}, "T_BC": {"fr": cal.T_BC.tolist()},
        "raw_camera_geometry": {"fr": lm.raw_calibration_geometry(cal)}}


def test_exact_front_camera_binds_despite_global_identity_change():
    original = _calibration()
    query = dataclasses.replace(original, calibration_id="calib:other-rig", preprocessing_id="prep:other-rig")
    bound = lm.bind_calibrations({"fr": query}, manifest(original))["fr"]
    assert bound.preprocessing_id == original.preprocessing_id
    assert bound.calibration_id == query.calibration_id
    np.testing.assert_array_equal(bound.K_raw, query.K_raw)


@pytest.mark.parametrize("field,value", [("D_raw", np.array([.1, 0., 0., 0., 0.])),
    ("raw_width", 1508), ("frame_id", "constructor0/camera_cr")])
def test_changed_raw_pixels_or_frame_are_rejected(field, value):
    original = _calibration()
    query = dataclasses.replace(original, preprocessing_id="prep:changed", **{field: value})
    with pytest.raises(c.ContractError, match="raw preprocessing"):
        lm.bind_calibrations({"fr": query}, manifest(original))


def test_legacy_bank_without_raw_proof_cannot_relabel_another_rig():
    original = _calibration()
    meta = manifest(original)
    del meta["raw_camera_geometry"]
    query = dataclasses.replace(original, preprocessing_id="prep:other-rig")
    with pytest.raises(c.ContractError, match="raw preprocessing"):
        lm.bind_calibrations({"fr": query}, meta)


def test_matching_legacy_identity_is_unchanged():
    original = _calibration()
    assert lm.bind_calibrations({"fr": original}, manifest(original))["fr"] is original


def test_adaptive_mode_leaves_changed_camera_unbound_for_exclusion():
    original = _calibration()
    changed = dataclasses.replace(original, preprocessing_id="prep:different", raw_width=1508)
    result = lm.bind_calibrations({"fr": changed}, manifest(original), strict=False)
    assert result["fr"] is changed and result["fr"].preprocessing_id != original.preprocessing_id

"""Step 7: B's inference stack starts and runs with every network path blocked.

Builds a synthetic map plus a retrieval index, then constructs the pinned DISK/LightGlue
backend, EigenPlaces, the index, the anchor worker and the tracker, and runs one anchor
job, while sockets, urllib and torch.hub downloads raise. Uses the cached pinned weights.
"""

import os
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import contract_factories as f
import vision_fixtures as vf
from camloc import contracts as c
from camloc import features
from camloc import landmark_map as lm
from camloc import masks, vision
from camloc import retrieval as rt


class NetworkUsed(AssertionError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkUsed("network access attempted during offline startup")


@pytest.fixture()
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    import torch.hub

    # load_state_dict_from_url stays: it reads the pinned local cache and only downloads
    # through download_url_to_file, which is blocked.
    monkeypatch.setattr(torch.hub, "download_url_to_file", _blocked)


@pytest.fixture()
def bank(tmp_path, monkeypatch):
    original = lm.save_landmark_map

    def with_production_mask(path, arrays, manifest):
        return original(path, arrays, {**manifest, "mask_id": masks.MASK_ID})

    monkeypatch.setattr(lm, "save_landmark_map", with_production_mask)
    scene = vf.Scene()
    landmark_map = scene.build_map(tmp_path / "bank")
    return scene, landmark_map, tmp_path / "bank"


def test_inference_stack_starts_and_runs_offline(offline, bank):
    scene, landmark_map, bank_dir = bank
    backend = features.FeatureBackend()
    assert Path(os.environ["TORCH_HOME"]).resolve() == \
        (c.PROJECT_ROOT / "artifacts" / "weights").resolve()
    model = rt.load_pinned_eigenplaces()
    ids = rt.eligible_keyframe_ids(landmark_map)
    rng = np.random.default_rng(0)
    descriptors = rng.normal(size=(ids.size, rt.DESCRIPTOR_DIM)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    rt.write_index(bank_dir, landmark_map, descriptors, ids, np.array(["fr"] * ids.size),
                   model.checkpoint_sha256)
    index = rt.RetrievalIndex.load(bank_dir, landmark_map, model)
    worker = vision.DiskAnchorWorker(landmark_map, backend, retrieval=index)
    tracker = vision.VisionTracker(landmark_map)
    observation = scene.observation(4)
    tracker.observe(observation)
    request = f.make_anchor_request(
        job_id=1, source_exposure_ns=scene.t_ns(4), observation=observation,
        snapshot=scene.snapshot(4), reference_ids=(), map_id=landmark_map.map_id)
    result = worker.run(request)
    result.check_against(request)
    assert result.status in (c.AnchorStatus.OK, c.AnchorStatus.FAILED)
    assert result.model_ids == backend.model_ids
    assert set(result.matched_reference_ids) <= set(ids.tolist())


def test_missing_or_corrupt_weights_fail_instead_of_downloading(offline, tmp_path):
    manifest = tmp_path / "config" / "pretrained_models.json"
    manifest.parent.mkdir()
    entry = features.load_model_manifest()
    for model in entry["models"]:
        model["sha256"] = "0" * 64
    import json

    manifest.write_text(json.dumps(entry), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing|SHA-256"):
        features.FeatureBackend(manifest_path=manifest)

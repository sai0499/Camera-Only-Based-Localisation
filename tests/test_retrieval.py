"""EigenPlaces retrieval: vendored network, strict pinned loading, preprocessing, bound index.

Synthetic maps and a randomly initialised checkpoint keep the suite fast and offline; the
real author checkpoint is read only for its key names when present locally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import contract_factories as f
from camloc import contracts as c
from camloc import landmark_map as lm
from camloc import masks
from camloc import retrieval as rt

KEYPOINTS_PER_FRAME = 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map(root: Path, cameras: list[str], landmark_counts: list[int]) -> lm.LandmarkMap:
    """Map with one keyframe per entry: camera ``cameras[i % len]``, ``landmark_counts[i]`` links."""
    n = len(landmark_counts)
    kf_cameras = [cameras[i % len(cameras)] for i in range(n)]
    links = np.full((n, KEYPOINTS_PER_FRAME), -1, np.int64)
    for i, count in enumerate(landmark_counts):
        links[i, :count] = np.arange(count)
    T = np.eye(4)
    total = n * KEYPOINTS_PER_FRAME
    arrays = {
        "landmark_ids": np.arange(KEYPOINTS_PER_FRAME, dtype=np.int64),
        "points_W": np.ones((KEYPOINTS_PER_FRAME, 3)),
        "landmark_observations": np.full(KEYPOINTS_PER_FRAME, 3, np.int32),
        "landmark_median_error_px": np.zeros(KEYPOINTS_PER_FRAME, np.float32),
        "landmark_max_angle_deg": np.full(KEYPOINTS_PER_FRAME, 2.0, np.float32),
        "keyframe_ids": np.arange(n, dtype=np.int64) * 3 + 2,
        "keyframe_cameras": np.array(kf_cameras, dtype="<U2"),
        "keyframe_exposure_ns": np.arange(n, dtype=np.int64) * 100 + 1_000,
        "keyframe_T_WC": np.repeat(T[None], n, axis=0),
        "keypoint_offsets": np.arange(n + 1, dtype=np.int64) * KEYPOINTS_PER_FRAME,
        "keypoints": np.zeros((total, 2), np.float32),
        "keypoint_scores": np.ones(total, np.float32),
        "descriptors": np.zeros((total, 128), np.float16),
        "keypoint_landmarks": links.reshape(-1),
    }
    for cam in cameras:
        arrays[f"mask_{cam}"] = np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    manifest = {
        "cameras": cameras, "calibration_id": "calibration_fr_v1",
        "preprocessing_id": f.PREPROCESSING_ID,
        "canonical_K": {cam: f.canonical_K().tolist() for cam in cameras},
        "T_BC": {cam: f.camera_to_body().tolist() for cam in cameras},
        "mask_id": masks.MASK_ID,
    }
    lm.save_landmark_map(root, arrays, manifest)
    return lm.load_landmark_map(root)


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    d = rng.normal(size=(n, rt.DESCRIPTOR_DIM)).astype(np.float32)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _index(root: Path, landmark_map: lm.LandmarkMap, descriptors: np.ndarray,
           model: rt.RetrievalModel | None = None, sha: str = "0" * 64) -> rt.RetrievalIndex:
    ids = np.asarray(landmark_map.keyframe_ids)
    cams = np.asarray(landmark_map.keyframe_cameras)
    rt.write_index(root, landmark_map, descriptors, ids, cams,
                   model.checkpoint_sha256 if model else sha, {"fitting_interval_ns": ["0", "1"]})
    return rt.RetrievalIndex.load(root, landmark_map, model)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, dict]:
    torch.manual_seed(0)
    state = rt.EigenPlacesNet().state_dict()
    path = tmp_path_factory.mktemp("ckpt") / "synthetic.pth"
    torch.save(state, path)
    return path, _sha(path), state


@pytest.fixture(scope="module")
def model(checkpoint: tuple[Path, str, dict]) -> rt.RetrievalModel:
    path, sha, _ = checkpoint
    return rt.load_eigenplaces(path, sha, device="cpu")


# Feature: stage3-gate-fix-round2, Property 3: frozen, bound, strict retrieval assets
def test_checkpoint_key_set_matches_vendored_network(checkpoint: tuple[Path, str, dict]) -> None:
    path, sha = rt.pinned_checkpoint()
    assert path.name == "ResNet18_512_eigenplaces.pth" and len(sha) == 64
    if path.is_file():
        reference = torch.load(path, map_location="cpu", weights_only=True)
    else:
        reference = checkpoint[2]
    assert set(reference) == set(rt.EigenPlacesNet().state_dict())
    assert {"aggregation.1.p", "aggregation.3.weight", "aggregation.3.bias"} <= set(reference)


# Feature: stage3-gate-fix-round2, Property 3: frozen, bound, strict retrieval assets
def test_strict_round_trip_gives_unit_descriptors(checkpoint, model: rt.RetrievalModel) -> None:
    _, sha, state = checkpoint
    assert model.device == "cpu" and model.checkpoint_sha256 == sha
    assert model.model_id == f"eigenplaces-r18-512:{sha[:12]}" and not model.net.training
    loaded = model.net.state_dict()
    assert all(torch.equal(loaded[k], state[k]) for k in state)
    rng = f.seeded_rng(11)
    batch = rng.normal(size=(2, 3, rt.INPUT_HEIGHT, rt.INPUT_WIDTH)).astype(np.float32)
    out = model.descriptors(batch)
    assert out.shape == (2, 512) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(out, model.descriptors(batch))


# Feature: stage3-gate-fix-round2, Property 3: frozen, bound, strict retrieval assets
def test_load_rejects_hash_key_and_shape_mismatch(tmp_path: Path, checkpoint) -> None:
    path, sha, state = checkpoint
    with pytest.raises(c.ContractError, match="SHA-256"):
        rt.load_eigenplaces(path, "f" * 64, device="cpu")
    with pytest.raises(c.ContractError, match="missing"):
        rt.load_eigenplaces(tmp_path / "absent.pth", sha, device="cpu")
    variants = {
        "extra": {**state, "aggregation.5.weight": torch.zeros(1)},
        "missing": {k: v for k, v in state.items() if k != "aggregation.1.p"},
        "shape": {**state, "aggregation.3.bias": torch.zeros(7)},
    }
    for name, bad in variants.items():
        bad_path = tmp_path / f"{name}.pth"
        torch.save(bad, bad_path)
        with pytest.raises(c.ContractError):
            rt.load_eigenplaces(bad_path, _sha(bad_path), device="cpu")


# Feature: stage3-gate-fix-round2, Property 3: frozen, bound, strict retrieval assets
def test_preprocess_shape_normalization_and_mask() -> None:
    assert rt.PREPROCESSING_ID == f"eigenplaces-r18-512:rgb480x232:imagenet:mask-{masks.MASK_ID}"
    rng = f.seeded_rng(5)
    for _ in range(5):
        image = rng.integers(0, 256, (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3), dtype=np.uint8)
        mask = rng.random((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH)) > 0.05
        out = rt.preprocess(image, mask)
        assert out.shape == (1, 3, 232, 480) and out.dtype == np.float32
        usable = mask.reshape(232, 2, 480, 2).all(axis=(1, 3))
        assert np.all(out[0][:, ~usable] == 0.0)
        cell = image.reshape(232, 2, 480, 2, 3).astype(np.float32).mean(axis=(1, 3)) / 255.0
        expected = ((cell - rt.IMAGENET_MEAN) / rt.IMAGENET_STD).transpose(2, 0, 1)
        np.testing.assert_allclose(out[0][:, usable], expected[:, usable], atol=0.03)
    mean_image = np.broadcast_to(np.round(rt.IMAGENET_MEAN * 255).astype(np.uint8),
                                 (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3)).copy()
    full = np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    assert np.abs(rt.preprocess(mean_image, full)).max() < 0.01
    for bad_image, bad_mask in ((image[:-2], mask[:-2]), (image.astype(np.float32), mask),
                                (image, mask.astype(np.uint8))):
        with pytest.raises(c.ContractError):
            rt.preprocess(bad_image, bad_mask)


# Feature: stage3-gate-fix-round2, Property 2: worker retrieval ranking
def test_rank_is_cosine_top_k_with_id_tie_break(tmp_path: Path) -> None:
    landmark_map = _map(tmp_path / "map", ["fr"], [35] * 16)
    rng = f.seeded_rng(3)
    descriptors = _unit(rng, 16)
    descriptors[9] = descriptors[4]  # exact tie: ids 14 and 29
    descriptors[12] = descriptors[4]
    index = _index(tmp_path / "map", landmark_map, descriptors)
    ids = np.asarray(landmark_map.keyframe_ids)
    for trial in range(30):
        query = descriptors[4] if trial == 0 else _unit(rng, 1)[0]
        got = index.rank(query, "fr")
        sims = descriptors @ query
        expected = tuple(int(k) for k in sorted(ids, key=lambda k: (-sims[np.searchsorted(ids, k)], k)))
        assert got == expected[: c.RETRIEVAL_CANDIDATES]
        assert got == index.rank(query, "fr")
    top = index.rank(descriptors[4], "fr")
    assert top[:3] == tuple(int(ids[i]) for i in (4, 9, 12))
    assert index.rank(descriptors[0], "fr", count=3) == index.rank(descriptors[0], "fr")[:3]


# Feature: stage3-gate-fix-round2, Property 2: worker retrieval ranking
def test_rank_filters_eligibility_and_camera(tmp_path: Path) -> None:
    rng = f.seeded_rng(9)
    counts = [int(v) for v in rng.integers(20, 41, 24)]
    landmark_map = _map(tmp_path / "map", ["fr", "r"], counts)
    eligible = set(rt.eligible_keyframe_ids(landmark_map).tolist())
    assert eligible == {int(k) for k, n in zip(landmark_map.keyframe_ids, counts, strict=True)
                        if n >= 30}
    index = _index(tmp_path / "map", landmark_map, _unit(rng, 24))
    for _ in range(20):
        query = _unit(rng, 1)[0]
        for cam in ("fr", "r"):
            got = index.rank(query, cam)
            assert got and set(got) <= eligible and len(got) <= c.RETRIEVAL_CANDIDATES
            assert all(landmark_map.keyframe(k).camera_id == cam for k in got)
        assert index.rank(query, "fl") == ()


# Feature: stage3-gate-fix-round2, Property 2: worker retrieval ranking
def test_query_uses_model_on_observation(tmp_path: Path, model: rt.RetrievalModel) -> None:
    landmark_map = _map(tmp_path / "map", ["fr"], [35] * 12)
    rng = f.seeded_rng(21)
    index = _index(tmp_path / "map", landmark_map, _unit(rng, 12), model=model)
    image = rng.integers(0, 256, (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3), dtype=np.uint8)
    observation = f.make_observation(image=image)
    expected = index.rank(model.describe(image, landmark_map.mask("fr")), "fr")
    assert index.query(observation) == expected and len(expected) == c.RETRIEVAL_CANDIDATES
    assert index.query(f.make_observation(camera_id="r", image=image)) == ()
    with pytest.raises(c.ContractError, match="preprocessing"):
        index.query(f.make_observation(image=image, preprocessing_id="other_prep"))
    assert index.identity()["retrieval_id"] == index.retrieval_id


# Feature: stage3-gate-fix-round2, Property 3: frozen, bound, strict retrieval assets
def test_index_load_rejects_identity_mismatch(tmp_path: Path, model: rt.RetrievalModel) -> None:
    root = tmp_path / "map"
    landmark_map = _map(root, ["fr"], [35] * 6)
    rng = f.seeded_rng(1)
    _index(root, landmark_map, _unit(rng, 6), model=model)
    with pytest.raises(c.ContractError, match="refusing to overwrite"):
        rt.write_index(root, landmark_map, _unit(rng, 6), landmark_map.keyframe_ids,
                       landmark_map.keyframe_cameras, model.checkpoint_sha256)
    other_map = _map(tmp_path / "other", ["fr"], [36] * 6)
    with pytest.raises(c.ContractError, match="map_id"):
        rt.RetrievalIndex.load(root, other_map, model)
    other_model = rt.RetrievalModel(model.net, "cpu", "e" * 64)
    with pytest.raises(c.ContractError, match="checkpoint_sha256"):
        rt.RetrievalIndex.load(root, landmark_map, other_model)

    meta_path, arrays_path = root / rt.INDEX_META_NAME, root / rt.INDEX_ARRAYS_NAME
    original_meta = meta_path.read_text(encoding="utf-8")
    original_arrays = arrays_path.read_bytes()

    def tampered(field: str, value: object, *, rehash: bool = True) -> None:
        meta = json.loads(original_meta)
        meta[field] = value
        if rehash:
            meta["retrieval_id"] = rt.retrieval_id(meta)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(c.ContractError, match=field):
            rt.RetrievalIndex.load(root, landmark_map, model)
        meta_path.write_text(original_meta, encoding="utf-8")

    tampered("map_npz_sha256", "0" * 64)
    tampered("preprocessing_id", "eigenplaces-r18-512:rgb240x116:imagenet:mask-x")
    tampered("descriptor_sha256", "1" * 64)
    tampered("format", "other-v0")
    meta = json.loads(original_meta)
    meta["fitting_interval_ns"] = ["0", "2"]  # provenance edit without re-binding
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(c.ContractError, match="retrieval_id"):
        rt.RetrievalIndex.load(root, landmark_map, model)
    meta_path.write_text(original_meta, encoding="utf-8")

    with np.load(arrays_path) as archive:
        base = {k: archive[k] for k in archive.files}

    def arrays_variant(match: str, *, rehash: bool, **changes: np.ndarray) -> None:
        arrays = {**base, **changes}
        with arrays_path.open("wb") as handle:
            np.savez(handle, **arrays)
        if rehash:  # a consistently re-bound index must still match the map
            meta = json.loads(original_meta)
            meta["descriptor_sha256"] = rt.descriptor_sha256(
                arrays["descriptors"], arrays["keyframe_ids"], arrays["camera_ids"])
            meta["retrieval_id"] = rt.retrieval_id(meta)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(c.ContractError, match=match):
            rt.RetrievalIndex.load(root, landmark_map, model)
        arrays_path.write_bytes(original_arrays)
        meta_path.write_text(original_meta, encoding="utf-8")

    arrays_variant("descriptor_sha256", rehash=False, descriptors=_unit(rng, 6))
    arrays_variant("not in map", rehash=True, keyframe_ids=base["keyframe_ids"] + 1)
    arrays_variant("camera r", rehash=True, camera_ids=np.array(["r"] * 6, dtype="<U2"))
    arrays_variant("L2-normalized", rehash=True, descriptors=base["descriptors"] * 2)
    assert rt.RetrievalIndex.load(root, landmark_map, model).keyframe_ids.size == 6

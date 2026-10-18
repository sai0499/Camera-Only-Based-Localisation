"""EigenPlaces ResNet18/512 global retrieval over the frozen fitting bank (owner: B).

Vendored architecture (MIT): Gabriele Berton et al., EigenPlaces,
https://github.com/gmberton/EigenPlaces at commit a2969f71d5ea31017443af490b15273ca4c50af1
(``eigenplaces_model/network.py`` and ``eigenplaces_model/layers.py``).

    MIT License

    Copyright (c) 2023 Gabriele Berton, Gabriele Trivigno, Carlo Masone, Barbara Caputo

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Only the ResNet18 / 512-d network is vendored: ``torchvision.models.resnet18(weights=None)``
without its pooling and FC layers, then ``L2Norm -> GeM(p=3) -> Flatten -> Linear(512, 512)
-> L2Norm``. The upstream constructor's ImageNet/CosPlace download and every ``torch.hub``
path are omitted; the author checkpoint is read from the checksum-pinned local cache
(``config/pretrained_models.json``, id ``eigenplaces-r18-512``) with ``weights_only=True``
and a strict state-dict load.

References and queries use the same ``preprocess``: canonical RGB 960x464 -> 480x232
(``INTER_AREA``), [0, 1], ImageNet mean/std, frozen usable-pixel mask zeroed (a 2x2 cell is
usable only when all four source pixels are). The index (``retrieval.npz`` +
``retrieval.json`` next to the bank) is bound to the map, its keyframe ids, the descriptor
content, the checkpoint and the preprocessing; any mismatch raises ``ContractError``.
Similarity only ranks reference ids for geometric verification; it never supplies a pose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from camloc import contracts as c
from camloc import masks

MODEL_ID = "eigenplaces-r18-512"
MODEL_MANIFEST = c.PROJECT_ROOT / "config" / "pretrained_models.json"
DESCRIPTOR_DIM = 512
INPUT_WIDTH, INPUT_HEIGHT = 480, 232
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_LANDMARKS = 30  # the select_references(min_landmarks=30) eligibility rule
INDEX_FORMAT = "camloc-retrieval-index-v1"
INDEX_ARRAYS_NAME = "retrieval.npz"
INDEX_META_NAME = "retrieval.json"


def preprocessing_id(mask_id: str) -> str:
    return f"{MODEL_ID}:rgb{INPUT_WIDTH}x{INPUT_HEIGHT}:imagenet:mask-{mask_id}"


PREPROCESSING_ID = preprocessing_id(masks.MASK_ID)


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# ---------------------------------------------------------------------------------------
# Vendored EigenPlaces layers and network (MIT, see the module header)
# ---------------------------------------------------------------------------------------


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=self.dim)


class GeM(nn.Module):
    def __init__(self, p: float = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class EigenPlacesNet(nn.Module):
    """EigenPlaces GeoLocalizationNet for ResNet18 with a 512-d output (no pretrained init)."""

    def __init__(self) -> None:
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.aggregation = nn.Sequential(
            L2Norm(dim=1), GeM(p=3), Flatten(), nn.Linear(512, DESCRIPTOR_DIM), L2Norm()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregation(self.backbone(x))


# ---------------------------------------------------------------------------------------
# Checkpoint loading and preprocessing
# ---------------------------------------------------------------------------------------


def pinned_checkpoint(manifest_path: Path = MODEL_MANIFEST) -> tuple[Path, str]:
    """(path, sha256) of the ``eigenplaces-r18-512`` entry in the pinned model manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["models"]:
        if entry["id"] == MODEL_ID:
            return (manifest_path.resolve().parents[1] / entry["path"]).resolve(), entry["sha256"]
    raise c.ContractError(f"{manifest_path}: no {MODEL_ID} entry")


class RetrievalModel:
    """A loaded, evaluated EigenPlaces network on one device plus its checkpoint identity."""

    def __init__(self, net: EigenPlacesNet, device: str, checkpoint_sha256: str) -> None:
        self.net = net
        self.device = device
        self.checkpoint_sha256 = checkpoint_sha256
        self.model_id = f"{MODEL_ID}:{checkpoint_sha256[:12]}"

    def descriptors(self, batch: np.ndarray) -> np.ndarray:
        """L2-normalized float32 (N, 512) descriptors of a preprocessed (N, 3, 232, 480) batch."""
        batch = np.asarray(batch, np.float32)
        if batch.ndim != 4 or batch.shape[1:] != (3, INPUT_HEIGHT, INPUT_WIDTH):
            raise c.ContractError("RetrievalModel.descriptors: expected (N, 3, 232, 480)")
        with torch.inference_mode():
            out = self.net(torch.from_numpy(np.ascontiguousarray(batch)).to(self.device))
        result = out.float().cpu().numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise c.ContractError("RetrievalModel.descriptors: non-finite descriptor")
        return result

    def describe(self, canonical_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Descriptor (512,) of one canonical image under the frozen mask."""
        return self.descriptors(preprocess(canonical_rgb, mask))[0]


def load_eigenplaces(path: str | Path, sha256: str, device: str | None = None) -> RetrievalModel:
    """Load a local EigenPlaces R18/512 checkpoint: SHA-256, weights-only, strict, eval, warm.

    ``sha256`` is the expected file hash (``pinned_checkpoint()`` supplies the manifest's).
    A missing file, hash mismatch, non-tensor payload or key mismatch raises ``ContractError``.
    No network or hub access: the network is built with ``weights=None``.
    """
    path = Path(path)
    if not path.is_file():
        raise c.ContractError(f"EigenPlaces checkpoint {path} is missing; run setup.cmd")
    digest = _sha256_file(path)
    if digest != sha256:
        raise c.ContractError(f"EigenPlaces checkpoint {path.name} fails its SHA-256 check")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise c.ContractError("EigenPlaces checkpoint is not a flat tensor state dict")
    net = EigenPlacesNet()
    expected, found = set(net.state_dict()), set(state)
    if expected != found:
        raise c.ContractError(
            f"EigenPlaces checkpoint keys differ: missing {sorted(expected - found)[:5]}, "
            f"unexpected {sorted(found - expected)[:5]}")
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as error:  # shape mismatch
        raise c.ContractError(f"EigenPlaces checkpoint rejected: {error}") from error
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net.eval().to(device)
    model = RetrievalModel(net, device, digest)
    model.descriptors(np.zeros((1, 3, INPUT_HEIGHT, INPUT_WIDTH), np.float32))  # warm-up
    return model


def load_pinned_eigenplaces(device: str | None = None,
                            manifest_path: Path = MODEL_MANIFEST) -> RetrievalModel:
    """``load_eigenplaces`` on the checkpoint and hash pinned in ``config/pretrained_models.json``."""
    path, sha256 = pinned_checkpoint(manifest_path)
    return load_eigenplaces(path, sha256, device)


def preprocess(canonical_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Float32 (1, 3, 232, 480) ImageNet-normalized network input; masked pixels are 0.

    ``canonical_rgb`` is RGB uint8 (464, 960, 3); ``mask`` is the frozen bool (464, 960)
    usable-pixel mask (True = usable). The same function serves references and queries.
    """
    import cv2

    image = np.asarray(canonical_rgb)
    mask = np.asarray(mask)
    if image.shape != (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3) or image.dtype != np.uint8:
        raise c.ContractError("preprocess: expected canonical RGB uint8 (464, 960, 3)")
    if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise c.ContractError("preprocess: expected a bool (464, 960) mask")
    small = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    normalized = (small.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    usable = mask.reshape(INPUT_HEIGHT, 2, INPUT_WIDTH, 2).all(axis=(1, 3))
    normalized[~usable] = 0.0
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


# ---------------------------------------------------------------------------------------
# Frozen descriptor index
# ---------------------------------------------------------------------------------------


def eligible_keyframe_ids(landmark_map: Any, min_landmarks: int = MIN_LANDMARKS) -> np.ndarray:
    """Sorted int64 ids of keyframes with at least ``min_landmarks`` landmark keypoints."""
    ids = [int(k) for k in landmark_map.keyframe_ids
           if int(np.count_nonzero(landmark_map.keyframe(int(k)).landmark_ids >= 0))
           >= min_landmarks]
    return np.array(ids, np.int64)


def descriptor_sha256(descriptors: np.ndarray, keyframe_ids: np.ndarray,
                      camera_ids: np.ndarray) -> str:
    """Content hash of the index arrays (little-endian float32 / int64, UTF-8 camera ids)."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(descriptors, "<f4").tobytes())
    digest.update(np.ascontiguousarray(keyframe_ids, "<i8").tobytes())
    digest.update("\n".join(str(v) for v in camera_ids).encode())
    return digest.hexdigest()


def retrieval_id(meta: dict[str, Any]) -> str:
    payload = {k: v for k, v in meta.items() if k != "retrieval_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "retrieval:" + hashlib.sha256(encoded).hexdigest()[:16]


def validate_index_arrays(descriptors: np.ndarray, keyframe_ids: np.ndarray,
                          camera_ids: np.ndarray) -> None:
    n = keyframe_ids.shape[0] if keyframe_ids.ndim == 1 else -1
    if descriptors.dtype != np.float32 or descriptors.shape != (n, DESCRIPTOR_DIM):
        raise c.ContractError("retrieval descriptors must be float32 (N, 512)")
    if keyframe_ids.dtype != np.int64 or camera_ids.shape != (n,) or n < 1:
        raise c.ContractError("retrieval keyframe_ids/camera_ids must be int64 (N,) / (N,)")
    if np.any(np.diff(keyframe_ids) <= 0) or keyframe_ids[0] < 0:
        raise c.ContractError("retrieval keyframe_ids must be sorted, unique and non-negative")
    if not np.isfinite(descriptors).all() or not np.allclose(
            np.linalg.norm(descriptors, axis=1), 1.0, atol=1e-4):
        raise c.ContractError("retrieval descriptors must be finite and L2-normalized")


def write_index(bank_dir: str | Path, landmark_map: Any, descriptors: np.ndarray,
                keyframe_ids: np.ndarray, camera_ids: np.ndarray, checkpoint_sha256: str,
                provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write ``retrieval.npz`` + ``retrieval.json`` into ``bank_dir``; never overwrites.

    Returns the metadata, whose ``retrieval_id`` hashes every other field (bindings,
    descriptor hash and the caller's provenance such as fitting interval and source bag).
    """
    root = Path(bank_dir)
    descriptors = np.ascontiguousarray(descriptors, np.float32)
    keyframe_ids = np.ascontiguousarray(keyframe_ids, np.int64)
    camera_ids = np.asarray(camera_ids, dtype="<U2")
    validate_index_arrays(descriptors, keyframe_ids, camera_ids)
    meta: dict[str, Any] = {
        **(provenance or {}),
        "format": INDEX_FORMAT,
        "map_id": landmark_map.map_id,
        "map_npz_sha256": landmark_map.manifest["map_npz_sha256"],
        "model_id": MODEL_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_id": preprocessing_id(landmark_map.mask_id),
        "min_landmarks": MIN_LANDMARKS,
        "keyframes": int(keyframe_ids.size),
        "descriptor_sha256": descriptor_sha256(descriptors, keyframe_ids, camera_ids),
    }
    meta["retrieval_id"] = retrieval_id(meta)
    try:
        with (root / INDEX_ARRAYS_NAME).open("xb") as handle:
            np.savez(handle, descriptors=descriptors, keyframe_ids=keyframe_ids,
                     camera_ids=camera_ids)
        with (root / INDEX_META_NAME).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    except FileExistsError as error:
        raise c.ContractError(f"{root}: retrieval index exists; refusing to overwrite") from error
    return meta


class RetrievalIndex:
    """Frozen reference descriptors bound to one map, checkpoint and preprocessing."""

    def __init__(self, landmark_map: Any, model: RetrievalModel | None, meta: dict[str, Any],
                 descriptors: np.ndarray, keyframe_ids: np.ndarray,
                 camera_ids: np.ndarray) -> None:
        self.map = landmark_map
        self.model = model
        self.meta = meta
        self.retrieval_id: str = meta["retrieval_id"]
        self.descriptors = descriptors
        self.keyframe_ids = keyframe_ids
        self.camera_ids = camera_ids
        eligible = set(eligible_keyframe_ids(landmark_map).tolist())
        self._eligible = np.array([int(k) in eligible for k in keyframe_ids], bool)
        for array in (self.descriptors, self.keyframe_ids, self.camera_ids, self._eligible):
            array.setflags(write=False)

    @classmethod
    def load(cls, bank_dir: str | Path, landmark_map: Any,
             model: RetrievalModel | None) -> RetrievalIndex:
        """Load ``retrieval.npz``/``retrieval.json`` from ``bank_dir``; verify every binding."""
        root = Path(bank_dir)
        meta_path, arrays_path = root / INDEX_META_NAME, root / INDEX_ARRAYS_NAME
        if not meta_path.is_file() or not arrays_path.is_file():
            raise c.ContractError(f"{root}: retrieval index missing; run build_retrieval_index")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != {"descriptors", "keyframe_ids", "camera_ids"}:
                raise c.ContractError("retrieval.npz members differ from the index format")
            descriptors = archive["descriptors"]
            keyframe_ids = archive["keyframe_ids"]
            camera_ids = archive["camera_ids"]
        checks = {
            "format": (meta.get("format"), INDEX_FORMAT),
            "retrieval_id": (meta.get("retrieval_id"), retrieval_id(meta)),
            "map_id": (meta.get("map_id"), landmark_map.map_id),
            "map_npz_sha256": (meta.get("map_npz_sha256"),
                               landmark_map.manifest.get("map_npz_sha256")),
            "preprocessing_id": (meta.get("preprocessing_id"), preprocessing_id(landmark_map.mask_id)),
        }
        if landmark_map.mask_id not in masks.SUPPORTED_MASK_IDS:
            raise c.ContractError("retrieval index mask_id mismatch: unsupported frozen mask")
        if model is not None:
            checks["checkpoint_sha256"] = (meta.get("checkpoint_sha256"), model.checkpoint_sha256)
        for name, (found, expected) in checks.items():
            if found != expected:
                raise c.ContractError(f"retrieval index {name} mismatch: {found!r} != {expected!r}")
        validate_index_arrays(descriptors, keyframe_ids, camera_ids)
        if meta.get("descriptor_sha256") != descriptor_sha256(descriptors, keyframe_ids,
                                                              camera_ids):
            raise c.ContractError("retrieval index descriptor_sha256 mismatch")
        if meta.get("keyframes") != int(keyframe_ids.size):
            raise c.ContractError("retrieval index keyframe count mismatch")
        for kid, cam in zip(keyframe_ids, camera_ids, strict=True):
            if not landmark_map.has_keyframe(int(kid)) \
                    or landmark_map.keyframe(int(kid)).camera_id != str(cam):
                raise c.ContractError(f"retrieval keyframe {int(kid)} not in map with camera {cam}")
        return cls(landmark_map, model, meta, descriptors, keyframe_ids, camera_ids)

    def rank(self, descriptor: np.ndarray, camera_id: str,
             count: int = c.RETRIEVAL_CANDIDATES) -> tuple[int, ...]:
        """Top ``count`` eligible same-camera ids by cosine similarity, ties by ascending id."""
        query = np.asarray(descriptor, np.float32).reshape(-1)
        if query.shape != (DESCRIPTOR_DIM,) or not np.isfinite(query).all():
            raise c.ContractError("RetrievalIndex.rank: expected a finite (512,) descriptor")
        rows = np.flatnonzero(self._eligible & (self.camera_ids == camera_id))
        if not rows.size:
            return ()
        similarity = self.descriptors[rows] @ query
        order = rows[np.lexsort((self.keyframe_ids[rows], -similarity))]
        return tuple(int(k) for k in self.keyframe_ids[order[:count]])

    def query(self, observation: c.Observation) -> tuple[int, ...]:
        """Top-10 eligible same-camera reference ids for one canonical observation."""
        if self.model is None:
            raise c.ContractError("RetrievalIndex.query: no model loaded")
        camera_id = observation.camera_id
        if camera_id not in self.map.cameras:
            return ()
        if observation.preprocessing_id != self.map.preprocessing_id:
            raise c.ContractError("RetrievalIndex.query: observation preprocessing differs")
        descriptor = self.model.describe(np.asarray(observation.image), self.map.mask(camera_id))
        return self.rank(descriptor, camera_id)

    def identity(self) -> dict[str, Any]:
        """JSON-safe identity for run records."""
        return {key: self.meta[key] for key in (
            "retrieval_id", "map_id", "checkpoint_sha256", "preprocessing_id",
            "descriptor_sha256", "keyframes")}

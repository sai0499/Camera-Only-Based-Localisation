"""DISK + LightGlue feature backend shared by the map builder and the anchor worker (owner: B).

One fixed configuration (FINAL_BUILD_PLAN section 3) is used offline and online, so map and
query descriptors live in the same space:

* DISK ``depth`` weights, NMS window 5, detection threshold 0, padding to a multiple of 16,
  RGB in [0, 1], no extractor resize. DISK's ``Extractor.extract`` would resize to 1024 px by
  default, so the backend calls the model's ``forward`` directly on the canonical image and
  the returned keypoints stay in canonical OpenCV pixel centres.
* A candidate pool of ``CANDIDATE_POOL`` detections is filtered by the caller's mask and the
  ``MAX_KEYPOINTS`` (1,024) best-scoring unmasked keypoints are retained.
* DISK-specific LightGlue: depth confidence 0.95, width confidence 0.99, filter threshold
  0.1, FlashAttention auto-detection, mixed precision off, no compilation.

Weights come only from the checksum-pinned cache in ``config/pretrained_models.json``;
construction blocks ``torch.hub`` downloads, so a missing cache fails instead of fetching.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from camloc import contracts as c

MODEL_MANIFEST = c.PROJECT_ROOT / "config" / "pretrained_models.json"
CANDIDATE_POOL = 4096
DESCRIPTOR_DIM = 128

EXTRACTOR_CONFIG: dict[str, Any] = {
    "name": "disk",
    "weights": "depth",
    "nms_window_size": 5,
    "detection_threshold": 0.0,
    "pad_if_not_divisible": True,
    "candidate_pool": CANDIDATE_POOL,
    "max_keypoints": c.MAX_KEYPOINTS,
    "resize": None,
    "input": "canonical RGB float32 [0, 1], 960 x 464",
}
MATCHER_CONFIG: dict[str, Any] = {
    "name": "lightglue",
    "features": "disk",
    "depth_confidence": 0.95,
    "width_confidence": 0.99,
    "filter_threshold": 0.1,
    "flash": True,  # used only when the installed torch supports it
    "mp": False,
    "compile": False,
}
_MODEL_IDS = {"disk": "disk-depth", "lightglue": "disk-lightglue"}


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Features:
    """Keypoints of one canonical image, ordered by descending detection score.

    ``keypoints`` float32 (N, 2) canonical OpenCV pixel centres, ``scores`` float32 (N,) and
    ``descriptors`` float32 (N, 128). N <= MAX_KEYPOINTS.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_model_manifest(manifest_path: Path = MODEL_MANIFEST) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def verified_weights(manifest_path: Path = MODEL_MANIFEST) -> dict[str, dict[str, Any]]:
    """Check every pinned checkpoint's size and SHA-256; return entries keyed by model id.

    Raises ``RuntimeError`` naming the missing or corrupted checkpoint; setup downloads,
    inference never does.
    """
    manifest = load_model_manifest(manifest_path)
    root = Path(manifest_path).resolve().parents[1]
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["models"]:
        path = (root / entry["path"]).resolve()
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Pinned checkpoint {entry['id']} is missing; run setup.cmd")
        if _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Pinned checkpoint {entry['id']} fails its SHA-256 check")
        entries[entry["id"]] = {**entry, "resolved_path": str(path)}
    return {"torch_home": str(root / manifest["torch_home"]), **entries}


def model_identities(weights: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    """Opaque model ids for AnchorResult.model_ids: model id plus checkpoint hash prefix."""
    return tuple(
        f"{_MODEL_IDS[name]}:{weights[_MODEL_IDS[name]]['sha256'][:12]}"
        for name in ("disk", "lightglue")
    )


def backend_identity() -> dict[str, Any]:
    """Configuration recorded in map manifests and compared at load time."""
    return {"extractor": dict(EXTRACTOR_CONFIG), "matcher": dict(MATCHER_CONFIG)}


class FeatureBackend:
    """DISK extraction and LightGlue matching on one torch device (CUDA when available)."""

    def __init__(self, device: str | None = None, manifest_path: Path = MODEL_MANIFEST) -> None:
        self.weights = verified_weights(manifest_path)
        os.environ["TORCH_HOME"] = self.weights["torch_home"]
        import torch
        from lightglue import DISK, LightGlue

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blocked = RuntimeError("runtime model download blocked; weights come from setup only")
        with mock.patch("torch.hub.download_url_to_file", side_effect=blocked):
            self._extractor = DISK(
                weights=EXTRACTOR_CONFIG["weights"],
                max_num_keypoints=CANDIDATE_POOL,
                nms_window_size=EXTRACTOR_CONFIG["nms_window_size"],
                detection_threshold=EXTRACTOR_CONFIG["detection_threshold"],
                pad_if_not_divisible=EXTRACTOR_CONFIG["pad_if_not_divisible"],
            ).eval().to(self.device)
            self._matcher = LightGlue(
                features=MATCHER_CONFIG["features"],
                depth_confidence=MATCHER_CONFIG["depth_confidence"],
                width_confidence=MATCHER_CONFIG["width_confidence"],
                filter_threshold=MATCHER_CONFIG["filter_threshold"],
                flash=MATCHER_CONFIG["flash"],
                mp=MATCHER_CONFIG["mp"],
            ).eval().to(self.device)
        self.model_ids = model_identities(self.weights)

    def extract(self, image: np.ndarray, mask: np.ndarray) -> Features:
        """DISK keypoints of a canonical RGB uint8 image, restricted to ``mask`` (True=usable)."""
        torch = self._torch
        if image.shape != (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3) or image.dtype != np.uint8:
            raise c.ContractError("FeatureBackend.extract: expected canonical RGB uint8 (464, 960, 3)")
        if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
            raise c.ContractError("FeatureBackend.extract: expected a bool (464, 960) mask")
        tensor = torch.from_numpy(np.array(image, copy=True)).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)[None]
        with torch.inference_mode():
            out = self._extractor({"image": tensor})
        keypoints = out["keypoints"][0].float().cpu().numpy()
        scores = out["keypoint_scores"][0].float().cpu().numpy()
        descriptors = out["descriptors"][0].float().cpu().numpy()
        return select_keypoints(keypoints, scores, descriptors, mask)

    def match(self, first: Features, second: Features) -> tuple[np.ndarray, np.ndarray]:
        """LightGlue matches between two canonical feature sets.

        Returns int64 (M, 2) index pairs (first, second) and float32 (M,) scores, sorted by
        the first index. Empty inputs give empty outputs.
        """
        if len(first) == 0 or len(second) == 0:
            return np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
        torch = self._torch
        size = torch.tensor([[c.CANONICAL_WIDTH, c.CANONICAL_HEIGHT]], dtype=torch.float32,
                            device=self.device)

        def side(features: Features) -> dict[str, Any]:
            return {
                "keypoints": torch.tensor(features.keypoints, device=self.device)[None],
                "descriptors": torch.tensor(features.descriptors, device=self.device)[None],
                "image_size": size,
            }

        with torch.inference_mode():
            out = self._matcher({"image0": side(first), "image1": side(second)})
        matches = out["matches"][0].cpu().numpy().astype(np.int64)
        scores = out["scores"][0].float().cpu().numpy().astype(np.float32)
        order = np.argsort(matches[:, 0], kind="stable")
        return matches[order], scores[order]

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()


def select_keypoints(
    keypoints: np.ndarray, scores: np.ndarray, descriptors: np.ndarray, mask: np.ndarray
) -> Features:
    """Keep detections on usable pixels, then the MAX_KEYPOINTS best by score (stable order)."""
    keypoints = np.asarray(keypoints, np.float32).reshape(-1, 2)
    scores = np.asarray(scores, np.float32).reshape(-1)
    descriptors = np.asarray(descriptors, np.float32).reshape(-1, DESCRIPTOR_DIM)
    u = np.rint(keypoints[:, 0]).astype(np.int64)
    v = np.rint(keypoints[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < c.CANONICAL_WIDTH) & (v >= 0) & (v < c.CANONICAL_HEIGHT)
    usable = np.zeros(len(keypoints), bool)
    usable[inside] = mask[v[inside], u[inside]]
    index = np.flatnonzero(usable)
    order = index[np.argsort(-scores[index], kind="stable")][: c.MAX_KEYPOINTS]
    return Features(
        keypoints=np.ascontiguousarray(keypoints[order]),
        scores=np.ascontiguousarray(scores[order]),
        descriptors=np.ascontiguousarray(descriptors[order]),
    )

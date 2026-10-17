"""Offline component smoke check, not a camera-localization benchmark."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from unittest.mock import patch

from prepare_weights import valid_weight, weight_path

ROOT = Path(__file__).resolve().parents[1]


def check():
    expected_python = (ROOT / ".python-version").read_text().strip()
    if platform.python_version() != expected_python or sys.prefix == sys.base_prefix:
        raise RuntimeError(f"Use this project's .venv with Python {expected_python}")
    manifest = json.loads((ROOT / "config/pretrained_models.json").read_text())
    for entry in manifest["models"]:
        if not valid_weight(weight_path(ROOT, entry), entry):
            raise RuntimeError(f"Run setup.cmd to prepare {entry['id']}")
    os.environ["TORCH_HOME"] = str(ROOT / manifest["torch_home"])
    import cv2
    import h5py
    import numpy as np
    import pycolmap
    import torch
    from lightglue import DISK, LightGlue

    for name in ("hloc.triangulation", "mcap.reader", "mcap_ros2.decoder", "torchvision"):
        importlib.import_module(name)

    camera = pycolmap.Camera(model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240])
    K = np.asarray(camera.calibration_matrix())
    world = np.array([[-1, -1, 4], [1, -1, 4], [-1, 1, 4], [1, 1, 4],
                      [-1, -1, 6], [1, -1, 6], [-1, 1, 6], [1, 1, 6]], dtype=np.float64)
    rotation = np.array([.03, -.02, .01], dtype=np.float64)
    translation = np.array([.1, -.2, .3], dtype=np.float64)
    pixels, _ = cv2.projectPoints(world, rotation, translation, K, None)
    ok, recovered_r, recovered_t = cv2.solvePnP(world, pixels, K, None)
    if not ok or not np.allclose(recovered_r.ravel(), rotation, atol=1e-6) or not np.allclose(recovered_t.ravel(), translation, atol=1e-6):
        raise RuntimeError("Synthetic OpenCV/pycolmap camera geometry check failed")
    cv2.SIFT_create(nfeatures=128)
    with tempfile.TemporaryDirectory() as folder:
        with h5py.File(Path(folder) / "smoke.h5", "w") as f:
            f.create_dataset("keypoints", data=pixels.astype(np.float32))
            if f["keypoints"].dtype != np.dtype("float32"):
                raise RuntimeError("Float32 keypoint storage check failed")

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    # Downloads belong to setup, never to an offline inference/check process.
    with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("Unexpected runtime model download")):
        extractor = DISK(weights="depth", max_num_keypoints=128).eval().to(device)
        matcher = LightGlue(features="disk", mp=False).eval().to(device)
        with torch.inference_mode():
            image = torch.rand(3, 256, 256, device=device)
            features = extractor.extract(image, resize=None)
            matches = matcher({"image0": features, "image1": features})
            count = len(matches["matches"][0])
            if count == 0:
                raise RuntimeError("DISK/LightGlue same-image component smoke check found no matches")
        if device == "cuda":
            torch.cuda.synchronize()
    eigen = next(e for e in manifest["models"] if e["id"] == "eigenplaces-r18-512")
    state = torch.load(weight_path(ROOT, eigen), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("EigenPlaces checkpoint could not be decoded")
    distributions = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    report = {
        "scope": "Offline component smoke check only; no localizer, map build, accuracy or throughput claim.",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_available": device == "cuda",
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "synthetic_pnp": "passed",
        "hloc_triangulation_import": "passed",
        "hdf5_float32_geometry": "passed",
        "disk_lightglue_same_image_matches": count,
        "feature_check_device": device,
        "feature_check_including_initialization_seconds": time.perf_counter() - started,
        "eigenplaces_checkpoint_decode": "passed; descriptor model integration belongs to development",
        "dependencies": dict(distributions),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256((ROOT / "config/pretrained_models.json").read_bytes()).hexdigest(),
    }
    target = ROOT / ".setup/environment_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dependencies"}, indent=2))
    if device == "cpu":
        print("CPU development is ready. CUDA runtime qualification requires a compatible NVIDIA GPU/driver.")


if __name__ == "__main__":
    check()

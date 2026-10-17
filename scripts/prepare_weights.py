"""Cache the three checksum-pinned public checkpoints; never read project data."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/pretrained_models.json"


def weight_path(root, entry):
    path = (root / entry["path"]).resolve()
    allowed = (root / "artifacts/weights/hub/checkpoints").resolve()
    if path.parent != allowed or len(entry["sha256"]) != 64:
        raise ValueError("Invalid managed checkpoint path or SHA-256")
    return path


def valid_weight(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() == entry["sha256"]


def prepare(root=ROOT):
    manifest = json.loads((root / "config/pretrained_models.json").read_text())
    for entry in manifest["models"]:
        path = weight_path(root, entry)
        if valid_weight(path, entry):
            print(f"Verified cached weight: {entry['id']}", flush=True)
            continue
        if not entry["url"].startswith("https://"):
            raise ValueError("Checkpoint downloads must use HTTPS")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading pinned weight: {entry['id']}", flush=True)
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "TrackCode-Apex-setup"})
        try:
            with urllib.request.urlopen(request, timeout=180) as src, partial.open("wb") as out:
                size = 0
                while chunk := src.read(1024 * 1024):
                    size += len(chunk)
                    if size > entry["bytes"]:
                        raise ValueError(f"Unexpected checkpoint size: {entry['id']}")
                    out.write(chunk)
            if not valid_weight(partial, entry):
                raise ValueError(f"Checkpoint checksum mismatch: {entry['id']}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    prepare()

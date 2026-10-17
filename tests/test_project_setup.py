import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_weights", ROOT / "scripts/prepare_weights.py")
weights = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weights)
export_spec = importlib.util.spec_from_file_location("export_requirements", ROOT / "scripts/export_requirements.py")
exporter = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(exporter)


class SetupTests(unittest.TestCase):
    def test_python_and_direct_dependency_pins_agree_with_lock(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        version = (ROOT / ".python-version").read_text().strip()
        self.assertEqual(project["project"]["requires-python"], "==" + version)
        self.assertEqual(lock["requires-python"], "==" + version)
        packages = {p["name"].lower().replace("_", "-"): p for p in lock["package"]}
        for dep in project["project"]["dependencies"] + project["dependency-groups"]["dev"]:
            if "==" in dep:
                name, expected = dep.split("==")
                self.assertEqual(packages[name.lower().replace("_", "-")]["version"], expected)
        for name, source in project["tool"]["uv"]["sources"].items():
            if "url" in source:
                self.assertEqual(packages[name]["source"]["url"], source["url"])

    def test_model_manifest_is_complete_and_paths_are_confined(self):
        manifest = json.loads((ROOT / "config/pretrained_models.json").read_text())
        self.assertEqual({m["id"] for m in manifest["models"]}, {"disk-depth", "disk-lightglue", "eigenplaces-r18-512"})
        for entry in manifest["models"]:
            self.assertTrue(entry["url"].startswith("https://"))
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)
            weights.weight_path(ROOT, entry)
        with self.assertRaises(ValueError):
            weights.weight_path(ROOT, {"path": "../outside.pth", "sha256": "0" * 64})

    def test_requirement_export_retains_explicit_cuda_sources_and_hashes(self):
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        raw = "torch==2.7.1+cu128 ; sys_platform == 'win32' \\\n    --hash=sha256:old\ntorchvision==0.22.1+cu128 \\\n    --hash=sha256:old\nnumpy==2.3.5\n"
        rendered = exporter.render(raw, lock)
        self.assertIn("torch @ https://download", rendered)
        self.assertIn("torchvision @ https://download", rendered)
        self.assertEqual(rendered.count("cp312-cp312-win_amd64.whl"), 2)
        self.assertIn(" ; sys_platform == 'win32'", rendered)
        self.assertNotIn("sha256:old", rendered)
        self.assertIn("numpy==2.3.5", rendered)

    def test_checkpoint_validation_rejects_same_length_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weight.pth"
            good = b"verified public checkpoint fixture"
            entry = {"bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()}
            self.assertFalse(weights.valid_weight(path, entry))
            path.write_bytes(good)
            self.assertTrue(weights.valid_weight(path, entry))
            path.write_bytes(b"X" * len(good))
            self.assertFalse(weights.valid_weight(path, entry))

    def test_bad_download_preserves_existing_cache_then_verified_repair_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = b"verified checkpoint fixture"
            entry = {"id": "fixture", "path": "artifacts/weights/hub/checkpoints/model.pth",
                     "url": "https://example.com/model.pth", "bytes": len(good),
                     "sha256": hashlib.sha256(good).hexdigest()}
            path = weights.weight_path(root, entry)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"old invalid cache")
            (root / "config").mkdir()
            (root / "config/pretrained_models.json").write_text(json.dumps({"models": [entry]}))
            with patch.object(weights.urllib.request, "urlopen", return_value=io.BytesIO(b"X" * len(good))):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    weights.prepare(root)
            self.assertEqual(path.read_bytes(), b"old invalid cache")
            self.assertFalse(path.with_suffix(".pth.part").exists())
            with patch.object(weights.urllib.request, "urlopen", return_value=io.BytesIO(good)):
                weights.prepare(root)
            self.assertTrue(weights.valid_weight(path, entry))

    @unittest.skipUnless(shutil.which("git"), "Git is only needed for repository exclusion checks")
    def test_git_excludes_generated_data_but_keeps_team_setup_and_supplied_map(self):
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(["git", "init", "-q", temp], check=True)
            shutil.copyfile(ROOT / ".gitignore", Path(temp) / ".gitignore")
            excluded = ["recording.mcap", "videos/camera_fr.mp4", ".venv/pyvenv.cfg", ".setup/cache/a",
                        "evaluation/reference/cache.csv", "evaluation/reference_reconciled/gt.csv",
                        "artifacts/weights/model.pth", "results/submission.csv",
                        "private_guides/TEAMMATE_A_GEOMETRY.md", "private_guides/TEAMMATE_B_VISION.md",
                        "private_guides/TEAMMATE_C_INTEGRATION.md", "private_guides/notes/draft.md"]
            kept = ["uv.lock", "requirements.txt", "setup.cmd", "config/pretrained_models.json",
                    "23-00-38_cut_120m_update/map.mcap", "camloc/__init__.py"]
            result = subprocess.run(["git", "-C", temp, "check-ignore", "--no-index", "--stdin", "-z"],
                                    input="\0".join(excluded + kept) + "\0", text=True,
                                    capture_output=True, check=True)
            self.assertEqual(set(filter(None, result.stdout.split("\0"))), set(excluded))


if __name__ == "__main__":
    unittest.main()

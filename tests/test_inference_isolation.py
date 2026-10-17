"""Inference isolation: ``camloc`` must not reach GT, evaluator, organizer or script code.

Static AST scan of every current ``camloc/**/*.py`` file, an import/text check of the
two shared modules, a scanner self-test and a fresh-interpreter import probe
(Requirement 19). The organizer ``track_frame.py`` helper stays loadable (19.5).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMLOC = ROOT / "camloc"
FORBIDDEN = frozenset({
    "evaluation", "reference_eval", "native_bridge", "evaluate", "extract_gt",
    "convert", "scripts", "tools", "export_initial_pose",
})
# A bare (separator-free) string constant is flagged only for names that are never words.
UNAMBIGUOUS = frozenset({"evaluation", "reference_eval", "native_bridge", "extract_gt", "export_initial_pose"})
LOADERS = frozenset({"import_module", "__import__", "spec_from_file_location", "run_path", "run_module"})
ALLOWED_HELPER = "track_frame.py"
GT_MARKERS = ("reference_reconciled", "gt_tf", "state_estimation")
RUNTIME_FORBIDDEN = FORBIDDEN | {"mcap", "torch", "cv2"}
# Non-stdlib, non-underscore top-level modules the locked numpy imports transitively.
# None today; list each here with a reason if a pinned numpy change adds one.
NUMPY_TRANSITIVE: frozenset[str] = frozenset()


def _segments(text: str) -> list[str]:
    return [s for s in re.split(r"[./\\]", text) if s]


def _strings(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _names_forbidden(texts: list[str]) -> bool:
    """Any string segment of the expression is forbidden, unless one ends with track_frame.py."""
    if any(part.endswith(ALLOWED_HELPER) for t in texts for part in re.split(r"[/\\]", t)):
        return False
    return any(seg in FORBIDDEN for t in texts for seg in _segments(t))


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def scan_source(source: str, label: str = "<src>") -> list[str]:
    """Return one message per forbidden import, literal dynamic load or module/path constant."""
    tree = ast.parse(source, filename=label)
    exempt = _docstring_ids(tree)
    for node in ast.walk(tree):  # constants inside a track_frame.py path expression stay allowed
        if isinstance(node, (ast.BinOp, ast.Call)) and not _names_forbidden(_strings(node)):
            exempt.update(id(n) for n in ast.walk(node) if isinstance(n, ast.Constant))
    hits: list[str] = []
    for node in ast.walk(tree):
        what = None
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if FORBIDDEN.intersection(a.name.split("."))]
            what = f"import {', '.join(bad)}" if bad else None
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            paths = [module]
            if node.level == 0 and module.split(".")[0] != "camloc":
                paths += [f"{module}.{a.name}" for a in node.names]
            if any(FORBIDDEN.intersection(p.split(".")) for p in paths):
                what = f"from {'.' * node.level}{module} import {', '.join(a.name for a in node.names)}"
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            texts = [s for arg in [*node.args, *(k.value for k in node.keywords)] for s in _strings(arg)]
            if name in LOADERS and _names_forbidden(texts):
                what = f"{name}({texts})"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in exempt:
            text = node.value
            if re.fullmatch(r"[\w.:/\\-]+", text):
                segs = _segments(text)
                if (text in UNAMBIGUOUS) if len(segs) == 1 else _names_forbidden([text]):
                    what = f"string constant {text!r}"
        if what:
            hits.append(f"{label}:{node.lineno}: {what}")
    return hits


def _camloc_files() -> list[Path]:
    files = sorted(CAMLOC.rglob("*.py"))
    assert {"__init__.py", "contracts.py", "submission.py"} <= {f.name for f in files}
    return files


def test_camloc_sources_do_not_reach_evaluator_gt_or_scripts() -> None:
    hits = [
        hit
        for path in _camloc_files()
        for hit in scan_source(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix())
    ]
    assert hits == []


@pytest.mark.parametrize("name", ["contracts.py", "submission.py"])
def test_shared_modules_import_only_stdlib_numpy_camloc(name: str) -> None:
    source = (CAMLOC / name).read_text(encoding="utf-8")
    tops: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    allowed = set(sys.stdlib_module_names) | {"__future__", "numpy", "camloc"}
    assert sorted(tops - allowed) == []
    lowered = source.lower()
    assert [marker for marker in GT_MARKERS if marker in lowered] == []


FLAGGED = [
    "import evaluation.reference_eval\n",
    "from tools import evaluate\n",
    "import importlib\nimportlib.import_module('native_bridge')\n",
    "from importlib.util import spec_from_file_location\n"
    "spec_from_file_location('x', 'evaluation/reference_eval.py')\n",
    "__import__('extract_gt')\n",
    "import runpy\nrunpy.run_path('scripts/export_initial_pose.py')\n",
    "GT_PATH = 'evaluation/reference_eval.py'\n",
]
ALLOWED = [
    "import importlib.util\nfrom pathlib import Path\nROOT = Path('.')\n"
    "importlib.util.spec_from_file_location('tf', ROOT / '23-00-38_cut_120m_update' / 'tools' / 'track_frame.py')\n",
    "import numpy\n",
    '"""A docstring may mention evaluation/reference_eval.py."""\nimport json\n',
]


@pytest.mark.parametrize(("source", "flagged"), [(s, True) for s in FLAGGED] + [(s, False) for s in ALLOWED])
def test_scanner_on_synthetic_sources(tmp_path: Path, source: str, flagged: bool) -> None:
    path = tmp_path / "probe_module.py"
    path.write_text(source, encoding="utf-8")
    assert bool(scan_source(path.read_text(encoding="utf-8"), path.name)) is flagged


PROBE = (
    "import json, sys\n"
    "before = set(sys.modules)\n"
    "import camloc.contracts, camloc.submission\n"
    "print(json.dumps(sorted(set(sys.modules) - before)))\n"
)


def test_fresh_interpreter_import_adds_only_stdlib_numpy_camloc() -> None:
    proc = subprocess.run(
        [sys.executable, "-B", "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr
    tops = {name.split(".")[0] for name in json.loads(proc.stdout.strip().splitlines()[-1])}
    assert {"camloc", "numpy"} <= tops
    allowed = set(sys.stdlib_module_names) | {"numpy", "camloc"} | NUMPY_TRANSITIVE
    assert sorted(t for t in tops if t not in allowed and not t.startswith("_")) == []
    assert sorted(tops & RUNTIME_FORBIDDEN) == []

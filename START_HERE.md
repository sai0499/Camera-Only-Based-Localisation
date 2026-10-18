# Start here: TrackCode Apex

This is the shared development workspace for the three-person camera-localization team. A's geometry/input, B's vision/map components through Step 3, and C's contracts/filter/scheduler/recovery are integrated on `dev/implementation`. The raw-camera runner and validator now work; [Stage 3 integration status](task_docs/STAGE3_STATUS.md) records commands and real tuning results. The latest full forward-camera replay fused 537 visual corrections across 2,708 outputs but still had a 125.6 s full-pose gap; Team C Step 3 is **not signed off**. [Stage 2](task_docs/STAGE2_STATUS.md) and [Stage 4](task_docs/STAGE4_STATUS.md) describe C's APIs. The evaluation adapter and native-coordinate reconciliation already work.

## Set up your machine

1. Clone the team's repository into a normal local folder. Use Windows x64; the shared lock targets this platform.
2. Double-click **`setup.cmd`**. No preinstalled Python, uv, CUDA Toolkit or administrator account is required. Internet access is required for the first installation. Allow roughly 20 GB free disk space; the first download includes a roughly 3 GB PyTorch wheel.
3. Open the folder in VS Code and use **`.venv/Scripts/python.exe`**. The committed editor settings select this environment.

The setup installs a project-local uv bootstrap, Python **3.12.13**, the exact packages in `uv.lock`, and the three checksum-pinned pretrained checkpoints. It then checks imports, synthetic camera geometry, HDF5 geometry precision and DISK/LightGlue inference. It can be rerun safely. A compatible NVIDIA driver/GPU enables CUDA; a CPU-only teammate can develop and run CPU checks with the same package versions. CUDA performance must be tested on the competition hardware.

From a terminal, use `setup.cmd --no-pause` or `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1`. Success ends with "Setup complete". Machine-specific results are in `.setup/environment_report.json`. Setup does not change the system Python or install a graphics driver.

## Read before coding

- [AGENTS.md](AGENTS.md): common instructions for every coding agent.
- [Final build plan](task_docs/FINAL_BUILD_PLAN.md): authoritative architecture, interfaces, owners, schedule and acceptance gates.
- [Original dataset README](README.md) and [challenge brief](task_docs/Constructor%20Labs%20-%20Camera-based%20localization%20on%20a%20race%20track.pdf): organizer requirements, preserved unchanged.
- [Evaluation guide](evaluation/README.md): working commands and the reconciled use of the new map.
- [Documentation index](task_docs/README.md): research, audits and retained evidence.

## Three people, one interface

| Owner | Main work | Files |
|---|---|---|
| A: geometry/input | MCAP allowlist, groups, image geometry, borders and route coordinates | `camloc/io.py`, `geometry.py`, `borders.py`, `track_coords.py` |
| B: vision/map | Fixed-pose map, DISK/LightGlue, LK/PnP, retrieval, handoff and ground motion | `build_map.py`, `camloc/vision.py`, `retrieval.py` |
| C: integration | Shared records, filter, deterministic scheduler, runner, validation and demo | `camloc/contracts.py`, `filter.py`, `scheduler.py`, `run.py`, `validate_run.py`, `render_replay.py` |

Each teammate starts a separate branch and gives their agent the corresponding handoff in section 11 of the build plan. C establishes the records first; A/B use those interfaces rather than creating competing runners or estimators. Agree changes to shared records before merging. Do not independently upgrade packages. Get a continuous scored baseline before optional improvements.

The repository owner's work stays on **`dev/implementation`**, with upstream **`origin/dev/implementation`**. Check the active branch before editing, commit tested changes there, and use `git push` to publish to its matching upstream. Teammates keep their own feature branches. Leave `main` unchanged until the owner explicitly requests reviewed integration; a feature-branch push does not merge code into `main`. `AGENTS.md` records this workflow for future coding agents.

## Folder map

```text
camloc/                         localizer package; implementation starts here
scripts/                        environment setup and component checks
tests/                          new project tests
config/                         pinned public-model manifest
pyproject.toml                   direct dependencies and platform/source choices
uv.lock                         complete shared dependency lock; commit this
requirements.txt                hashed export of the same environment
task_docs/FINAL_BUILD_PLAN.md    the build specification
task_docs/research/              supporting research and audits
task_docs/evidence/              small measured audit/setup records
task_docs/figures/               useful geometry diagnostics
evaluation/                     working local evaluator and regression tests
tools/                          original organizer tools; preserve unchanged
23-00-38_cut_120m_update/         supplied static map and new tools; preserve unchanged
artifacts/                      local weights and future fitted maps; ignored by Git
results/                        generated predictions/reports; ignored by Git
videos/                         seven exported videos; local and ignored by Git
.venv/                          recreated development environment; ignored by Git
.setup/                         local Python/bootstrap/cache/report; ignored by Git
```

The original driving MCAP, `metadata.yaml`, `README.md` and challenge PDF keep their paths. Avoid moving hash-bound assets or changing organizer tools to make a test pass.

## Data and Git

Commit the source, setup scripts, `pyproject.toml`, `uv.lock`, `requirements.txt`, model manifest, documentation and small supplied map. Use a private team repository. The `.gitignore` excludes the large recording, videos, generated GT references, environments, weights, fitted maps and results. Virtual environments are recreated locally; do not copy or commit `.venv`.

Transfer `c0_2026-10-13_22-53-42_camloc_0.mcap` privately to each teammate and place it at the project root. Its SHA-256 is `fe5a0b9565b1bc03dd1ad775645722645e135b2f19012cf7bc815a8bcb91d613`. Videos are optional for coding. The new `map.mcap` stays in its organizer folder.

On a fresh clone, after copying the recording, generate local evaluation references once:

```powershell
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py prepare
```

This reads training GT for evaluation only and writes the ignored reconciled bundle. On the original workstation that bundle already exists: do not rebuild over it. The older raw reference cache is retained locally because `reconcile` can reuse it. Inference must never read either reference bundle.

## Daily commands

```powershell
# Run both new tests and the existing evaluation regression tests.
& .\.venv\Scripts\python.exe -B -m pytest

# Repeat the offline environment/component check.
& .\.venv\Scripts\python.exe -B scripts/check_environment.py

# Check code written by the team; organizer/evaluation code is excluded.
& .\.venv\Scripts\ruff.exe check camloc scripts tests
```

Use `uv sync --locked` through setup rather than unpinned `pip install`. `requirements.txt` is a complete hashed `uv pip sync` export with explicit Windows CUDA wheel URLs; prefer setup because it also installs the correct Python and model cache and retains hloc's LightGlue source override. If a dependency change is necessary, C updates `pyproject.toml`, runs the pinned uv `lock`, syncs/tests, then runs `.venv/Scripts/python.exe -B scripts/export_requirements.py` and commits all three dependency files together. Preserve the evaluation's pinned MCAP/NumPy versions unless deliberately rebuilding and validating its manifests.

## Backup and current evidence

The complete pre-organization project is backed up on the original workstation's Desktop as `TrackCode-Apex-backup-20261017-135029.zip`, with a `.sha256` sidecar. It contains all 1,011 original files (13,663,778,923 source bytes), including the recording, all videos, the old environment and reference bundles. Every archived file and the Desktop ZIP copy were verified. [Backup receipt](task_docs/evidence/project_backup.json).

Environment checks prove component installation, not localization accuracy. Small/full fixed-pose builds and complete causal tuning replays have now run; the visual-baseline gate failed. Holdout performance and native/high-frequency workload qualification remain open. See the Stage 3 status before treating any CSV or component FPS as a working localizer.

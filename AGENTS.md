# Coding-agent instructions

Read `START_HERE.md` and `task_docs/FINAL_BUILD_PLAN.md` before changing code. The latter is the single implementation specification. Research under `task_docs/research/` supplies context; do not restart model selection or redesign the architecture independently.

## Git workflow

- The repository owner's implementation branch is `dev/implementation`, tracking `origin/dev/implementation`. Keep their work on this branch. Other teammates use their own assigned feature branches.
- Before editing, check `git status --short --branch` and `git branch --show-current`. Never implement or commit directly on `main`; switch to the assigned development branch first and preserve unrelated changes.
- Commit and push completed, verified implementation changes to the matching development/feature branch. Stage only relevant files; keep private guides, recordings, GT references, weights and generated outputs excluded from Git.
- Do not merge into, commit to or push `main` until the user explicitly requests integration. Do not force-push or rewrite shared history. Pushing a development branch does not merge it into `main`.

## Environment and commands

- Target Windows x64, Python 3.12.13, the committed `uv.lock` and `.venv`.
- Run `setup.cmd --no-pause` if setup is missing. Use `.venv/Scripts/python.exe`, not the MSYS/system Python. Never install globally or into another teammate's environment.
- Run `.venv/Scripts/python.exe -B -m pytest` after relevant changes. Existing evaluator tests must remain valid. Test new geometry/timing behavior with small meaningful fixtures.
- Run `.venv/Scripts/ruff.exe check camloc scripts tests` on new code. Do not reformat organizer/evaluation files to satisfy a style tool.
- Package changes belong to C: update `pyproject.toml`, resolve/test `uv.lock`, export `requirements.txt` using `.venv/Scripts/python.exe -B scripts/export_requirements.py`, and commit them together. Never run an unpinned upgrade as a fix.

## Scope and ownership

- A owns input/geometry/borders/track coordinates. B owns map/vision/retrieval/ground motion. C owns shared records/filter/scheduler/runner/evaluation integration/demo.
- C defines `camloc/contracts.py` first. Propose shared-record changes explicitly; reuse existing records and one estimator/runner.
- Do not create stub implementations that silently return plausible poses. Unsupported/failed evidence must be rejected and diagnosed.
- The current `camloc` package is a scaffold. Evaluation and setup are implemented; localizer accuracy, runtime and high-rate support have not been established.

## Preserve these assets

- Do not modify/move/delete the root recording, `metadata.yaml`, original `README.md`, challenge PDF, `tools/`, or `23-00-38_cut_120m_update/`.
- Keep `task_docs/build_split.json` byte-for-byte unchanged. It and evaluation source/assets are hash-bound; old descriptive profile text is superseded by the active reconciled manifest and evaluation guide.
- Preserve `evaluation/reference/` and `evaluation/reference_reconciled/` locally. They are generated GT/cache and must remain excluded from inference and Git. Rebuild only into a fresh output when deliberately required.
- Write fitted assets/weights under `artifacts/`, predictions/reports under `results/`, temporary work under `tmp/`. Keep large/generated/private inputs out of Git. Do not upload project data to external services.
- The pre-organization Desktop ZIP is a backup, not a development directory. Do not alter it.

## Estimation boundaries

- Inference accepts allowlisted images/calibration/static geometry, declared frozen fitting artifacts and exactly one initial x/y/yaw pose.
- Never decode dynamic `/tf`, live `/constructor0/state_estimation` or GT CSVs in the localizer. Training/evaluation tools remain separate.
- Use integer nanoseconds, exposure stamps, consistent transforms and canonical pixels. No future image evidence, timestamp lookup, GT alignment or offline smoothing in ranked predictions.
- Keep camera-rate LK/current PnP separate from bounded anchor jobs. Never fuse a stale worker pose. Preserve unique landmark/observation identities and explicit loss/recovery behavior.
- Record all weak/rejected/prediction-only outputs. Component FPS or synthetic replay is not native high-frequency localization accuracy.

Work only on the assigned scope, preserve other teammates' edits, and report what changed, what was tested and what remains unverified.

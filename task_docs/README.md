# Project documents

Read [FINAL_BUILD_PLAN.md](FINAL_BUILD_PLAN.md) to build. It is the authoritative architecture and three-person schedule. The original [challenge brief](Constructor%20Labs%20-%20Camera-based%20localization%20on%20a%20race%20track.pdf) and [dataset README](../README.md) remain unchanged.

| Location | Purpose |
|---|---|
| `FINAL_BUILD_PLAN.md` | Current system, records, ownership, thresholds, failure actions and acceptance gates |
| `STAGE2_STATUS.md` | Implemented C-only filter/streaming API, synthetic checks and pending A/B integration |
| `STAGE3_STATUS.md` | Integrated B PR, raw-camera runner/validator, fitting builds, real tuning failures and remaining visual-baseline gate |
| `STAGE4_STATUS.md` | Implemented C anchor scheduler, recovery manager, status machine, synthetic checks and resolved lifecycle decisions |
| `STAGE5_STATUS.md` | Real-recording repeatability, prohibited-input and future-prefix checks, bounded stress cases and runtime telemetry; remaining qualification gates |
| `REPLAY_UI_PLAN.md` | Planned local input/run/monitor workflow and passive 2D/3D replay; no UI or hidden-run qualification claimed |
| `A_C_INTEGRATION_STATUS.md` | A's reviewed merge into C, input/geometry/worker fixes, combined checks and remaining B integration |
| `build_split.json` | Frozen exact fitting/tuning/holdout boundaries and source identities; do not edit |
| `research/FINAL_RESEARCH_REVIEW.md` | Primary-source comparisons and final design reasoning |
| `research/Dataset_Audit.md` | Full recording inventory, camera/geometry/timing observations |
| `research/Requirements_and_Risk_Review.md` | Verified requirements, historical ambiguities and conservative handling |
| `research/ORGANIZER_UPDATE_REVIEW.md` | Additional map, Option B tooling and native-coordinate reconciliation |
| `evidence/` | Small audit, math, provenance, backup and environment-preparation records |
| `figures/` | Two retained geometry diagnostics |

Historical audit evidence describes the state when captured. The current development environment is defined by root `pyproject.toml`/`uv.lock`, `config/pretrained_models.json` and [START_HERE.md](../START_HERE.md). The working coordinate evaluator is documented in [evaluation/README.md](../evaluation/README.md).

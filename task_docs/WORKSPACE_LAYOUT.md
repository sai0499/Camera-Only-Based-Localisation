# Retained workspace and cleanup

The original recording and byte-identical `build_split.json` are the dataset. Fitting,
tuning and holdout are selected by those fixed intervals; separate MCAP copies are not
needed. The supplied static map, original README/metadata/challenge PDF, organizer
tools, both evaluation-reference directories, source, tests and local environment stay.

## Retained inference assets

| Directory under `artifacts/` | Reason to keep | Assisted tuning position RMSE / longest visual gap |
|---|---|---|
| `fit-seven-rescue-v2` | Locked submission; best measured camera-only continuity among retained banks | 0.535 m / 2.7 s |
| `fit-stage3-multilap-v5-rigproof-v2` | Earlier lower-error baseline; best retained holdout RMSE, 0.635 m | 0.530 m / 4.8 s |
| `fit-seven-spatial-v1` | Shortest assisted tuning visual gap; experimental | 0.552 m / 2.2 s |
| `weights` | Three checksum-pinned model checkpoints required for inference | Not a localization run |

The opt-in turn-bias experiment uses `fit-seven-rescue-v2` too: its assisted tuning
RMSE is 0.521 m with a 2.4 s gap. Its worse holdout result leaves it outside the locked
submission. The spatial bank worsens camera-only tuning RMSE to 3.265 m and its gap to
5.6 s. These experimental results are retained for comparison, not silently selected.

Each retained bank contains its own `manifest.json`, `map.npz`, `retrieval.json` and
`retrieval.npz`. Intermediate banks are provenance, not runtime file dependencies.
The viewer still defaults to `fit-seven-rescue-v2`.

## Retained results

The selected assisted tuning, isolated repeat, camera-only ablation, holdout regression,
lighting stress and integrity audits remain under `results/seven-camera-*`. The spatial
and turn-bias comparisons, earlier deadline baselines and the original temporary
visualization remain too. Their predictions and diagnostics support replay and audits.
The deliverable `results/seven-camera-submission-assets-v1.zip` and checksum receipt
remain unchanged. Earlier redundant asset/source ZIPs were removed; source history
and release tags remain in Git, and the alternative banks remain unpacked locally.

`results/workspace-cleanup/historical-evidence*.zip` retains compact predictions, scores,
metadata, audit summaries and temporary analysis commands from removed experiments.
Large diagnostics, synthetic poison/future-message MCAPs, intermediate map arrays and
test fixtures are deliberately excluded. Historical documents can name these removed
experiment paths; consult this archive rather than treating them as current inputs.

`results/workspace-cleanup/plan*.json` records the exact deletion inventory, protected
file hashes and archive checksums. `completed.json` records post-cleanup checks and
any access-restricted folders left untouched. Small initial-pose JSONs and recording
audit notes stay in `tmp/` for existing reproduction commands.

The seven exported MP4s duplicated the original recording and were removed. Their
provenance manifest remains in `videos/`; the UI obtains frames directly from MCAP.
Package caches and the working `.venv`/`.setup` were preserved.

Post-cleanup verification: 1,133 tests, Ruff and UI geometry checks pass. All 46 protected
file hashes match, all three map/retrieval bindings load, and the viewer loads 2,708
retained poses with offline GT and an original MCAP image. A fresh 100-group localization
smoke replay has no processing errors; its first 99 completed groups reproduce the saved
poses and decisions exactly (the EOF partial group is excluded from prefix comparison).
Thirteen access-restricted historical folders were left untouched.

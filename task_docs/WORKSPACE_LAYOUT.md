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

Only these five entries remain, plus `.gitkeep`:

| Path under `results/` | Purpose |
|---|---|
| `submission-evidence-holdout-20261018-v1/` | Fresh whole-holdout replay, prediction CSV, diagnostics, scorer/validator/runtime reports, offline GT viewer and `evidence.zip` |
| `seven-camera-tune-v3/` | One complete selected tuning replay, plots and viewer; also supplies the UI's development initial-pose defaults |
| `seven-camera-submission-assets-v1.zip` | Unchanged deliverable frozen bank and pinned weights |
| `seven-camera-assets-receipt.json` | Deliverable asset checksums |
| `retained-comparisons.zip` | Compact older predictions, scores, run metadata, validation and audit receipts |

The comparison archive includes the earlier baseline, camera-only, spatial, turn-bias,
stress and repeatability results, with their original relative paths. Large historical
diagnostics, runtime traces, duplicate viewers, older cleanup archives and temporary
visualizations were removed. Frozen release receipts keep historical run paths; use
the comparison archive for those compact files and the fresh holdout folder for current
complete evidence. Inference has no dependency on a deleted result directory.

The fresh holdout receipt is `task_docs/evidence/submission_holdout_20261018.json`.
The original broad cleanup is recorded in `task_docs/evidence/workspace_cleanup.json`;
the later results-only cleanup is in `task_docs/evidence/results_cleanup_20261018.json`.
Small initial-pose JSONs and recording audit notes stay in `tmp/` for reproduction.

The seven exported MP4s duplicated the original recording and were removed. Their
provenance manifest remains in `videos/`; the UI obtains frames directly from MCAP.
Package caches and the working `.venv`/`.setup` were preserved.

The broad cleanup passed 1,133 tests, Ruff, all three map/retrieval bindings and a fresh
localization smoke replay. The subsequent results-only cleanup passed 25 UI/setup tests
and verified 89 retained-file hashes, including the original recording, fixed split,
frozen banks/weights, submission ZIP and complete retained runs. The UI still lists both
current runs, attaches holdout GT to all 2,191 poses and reads an original MCAP image.
All 25 superseded result targets were removed; no restricted result folders remain.
Six older access-restricted test folders under `tmp/` remain outside this cleanup's scope.

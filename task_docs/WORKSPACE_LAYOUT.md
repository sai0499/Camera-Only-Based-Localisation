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

Only `results/holdout/` remains, plus `.gitkeep`:

| Path under `results/holdout/` | Purpose |
|---|---|
| `submission.csv` | Exact 2,191 predicted holdout poses; required timestamp_ns,x,y,yaw columns |
| `validation/` | Outputs of the supplied scoring tools and scorer validation record |
| `validated.json` | Output-format, timestamp, coverage and visual-gap checks |
| `runtime.json` | Measured processing-time and memory summary |
| `run_metadata.json` | Source/model identities, input bounds and inference mode |
| `trajectory_vs_ground_truth.png` | Required validation GT comparison on the track map |
| `README.md` | Mode, hardware, metrics and remaining limits |

The challenge PDF's expected-output list was checked directly on pages 2-3. Besides
poses and validation scores it requires source/weights, runtime/hardware, presentation
and a validation visualization. Source and dependencies remain in the repository; the
three frozen banks and pinned weights remain under `artifacts/`. No map or weight was
removed. The redundant ZIPs, old tuning replay, comparison archive and exported GT
copies were removed. Historical receipts retain their original paths and checksums;
the current deliverables above supersede those paths.

The one current interactive replay was moved intact to `tmp/replay_ui/holdout/` so it
remains available for UI work, including raw images and valid auxiliary 3D geometry.
The server discovers this dedicated cache as `preview/holdout`; editable source remains
under `replay_ui/`. This cache is temporary UI data, not a submission deliverable.

The fresh holdout receipt is `task_docs/evidence/submission_holdout_20261018.json`.
The original broad cleanup is recorded in `task_docs/evidence/workspace_cleanup.json`;
the later results-only cleanup is in `task_docs/evidence/results_cleanup_20261018.json`.
The final PDF-aligned trim is in `task_docs/evidence/submission_contents_20261018.json`.
Small initial-pose JSONs and recording audit notes stay in `tmp/` for reproduction.

The seven exported MP4s duplicated the original recording and were removed. Their
provenance manifest remains in `videos/`; the UI obtains frames directly from MCAP.
Package caches and the working `.venv`/`.setup` were preserved.

The broad cleanup passed 1,133 tests, Ruff, map/retrieval binding checks and a localization
smoke replay. The final trim and UI-cache support pass 1,134 tests and Ruff. Original
recording, fixed split, challenge PDF, frozen bank/weight and pose hashes match. The
cached replay loads all 2,191 poses with offline GT and reads an original camera image;
cached downloads/snapshots are covered by HTTP tests. No localization algorithm or
prediction was changed. Presentation preparation is separate from this cleanup.

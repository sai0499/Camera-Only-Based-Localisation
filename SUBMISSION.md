# TrackCode Apex: run the submitted localizer


**Locked release:** Git tag `release-seven-camera-assisted-v1` (commit `6f0c6d7`): bank `artifacts/fit-seven-rescue-v2`, retrieval, and the permitted vehicle velocity/yaw-rate inputs (`--vehicle-motion`), as in the command below. Acceleration is not read. Hashes and reference runs: [release_lock.json](task_docs/evidence/release_lock.json). Camera-only is a fallback; `--camera-turn-bias` is not part of the release.

Use Windows x64 and one CUDA GPU. Run `setup.cmd --no-pause` for the locked Python
3.12.13 environment and declared public checkpoints. The frozen fitting-only bank is
already in `artifacts/fit-seven-rescue-v2/`, and pinned weights are in `artifacts/weights/`.
Provide the frozen bank alongside the source when handing the system to the organizers;
these ignored local assets are not included in Git. Public checkpoint download links
and checksums are in `config/pretrained_models.json`. The redundant local asset ZIP was
removed. Do not include recordings, GT or `.venv` in a source handoff.

Pages 2-3 of the challenge PDF require the test pose CSV, mandatory reproducible source
and weights/download links, validation scoring outputs, measured runtime/hardware,
a validation trajectory video/plot, and a presentation (at most five minutes/six slides).
The current validation deliverables are in `results/holdout/`: `submission.csv`,
`validation/`, `validated.json`, `runtime.json`, `run_metadata.json`, and
`trajectory_vs_ground_truth.png`. Only required outputs are retained in `results/`.

The judge supplies one initial pose JSON, with exactly these keys (replace the example):

```json
{"timestamp_ns":"1791894452091749130","x":0.0,"y":0.0,"yaw":0.0}
```

Use the image exposure timestamp as an integer decimal string, metres and radians.
For a recording with the separate body-velocity topic:

```powershell
& .\.venv\Scripts\python.exe -B run.py --bag C:\path\test.mcap --input-recording --initial-pose C:\path\initial.json --artifacts artifacts/fit-seven-rescue-v2 --retrieval --vehicle-motion validation-twist --runtime --output results/submission-test
```

If the recording provides `/constructor0/state_estimation` with the supported velocity
layout, use `--vehicle-motion state-fields` instead. These modes consume only timestamp,
body vx/vy and yaw rate, as permitted on page 2 of the challenge brief. They skip pose,
orientation, status and track fields. If no permitted motion is supplied, omit the flag
for camera-only; its accuracy is different and is not represented by the table below.
Missing/unsupported requested motion fails explicitly. The output directory must be new.

The submission is `results/submission-test/submission.csv`, with columns
`timestamp_ns,x,y,yaw`. `run.json`, `diagnostics.jsonl` and `runtime.json` record inputs,
identities, rejected/weak/prediction-only outputs and measured timing. Inference never
reads an evaluation reference, dynamic TF or a GT CSV. It uses no future images or
motion samples. The one initial pose is the only external absolute query pose.

The selected bank maps all seven cameras. Front coverage spans the fitting route;
auxiliary coverage targets three weak regions. Camera selection uses mapped support
and recent source-time failures within the existing worker budget. Compatible cameras
are selected automatically; changed camera geometry is excluded and logged. If no
mapped camera remains compatible, startup fails explicitly.

The retained configuration was frozen before its holdout regression. This split had
already been inspected in earlier work, so this is not a new unseen test:

| Motion-assisted result | Tuning (270.8 s) | Holdout regression (219.2 s) |
|---|---:|---:|
| Position RMSE m | 0.535 | 0.655 |
| Lateral RMSE m | 0.289 | 0.330 |
| Yaw RMSE deg | 0.210 | 0.239 |
| Poses within 1 m | 91.5% | 86.5% |
| Coverage, official 1 s bins | 100% | 100% |
| Longest full-visual-pose gap s | 2.7 | 9.8 |

The camera-only tuning ablation improves position RMSE from 12.893 m to **2.554 m**
and passes the unchanged continuity gate with a 2.3 s longest full-pose gap. Its lateral
RMSE is 1.924 m: camera-only accuracy is not represented by the assisted table.

Runtime is measured on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), CUDA 12.8,
Windows 11. The current holdout averages 73.3 ms/output; the earlier camera-only tuning
ablation averaged 68.1 ms/output, including reader and worker waiting.
Startup is separate. These averages do not certify bounded latency or a higher native
camera frequency. The motion-assisted mode still has a 9.8 s full-visual-pose gap on
holdout, so the map is not globally gap-free. The separate unscored camera lap
has no GT and uses an experimental image-derived initialization; it is not the hidden test.

See [camera-map improvements and limits](task_docs/MULTICAMERA_IMPROVEMENTS.md),
[pre-holdout selection receipt](task_docs/evidence/multicam_selection.json),
[earlier lower-lateral-error bank](task_docs/DEADLINE_ACCURACY.md),
[model declarations and notices](config/MODEL_NOTICES.md), and
[offline/live viewer instructions](replay_ui/README.md).

A fresh whole-holdout replay after workspace cleanup reproduces all 2,191 predictions
and localization decisions exactly, with the same 0.655 m RMSE and 9.8 s visual gap.
It averages 73.3 ms/output; frozen assets, weights, source and split hashes are unchanged
during the run. See the [fresh evidence receipt](task_docs/evidence/submission_holdout_20261018.json).
The unchanged pose CSV and scorer/validator reports are now in `results/holdout/`.
The current interactive replay is a temporary UI cache at `tmp/replay_ui/holdout/`;
the old evidence/archive copies were removed. The live UI loads this cache automatically.
See [the current contents receipt](task_docs/evidence/submission_contents_20261018.json).
This is a previously inspected holdout regression, not a new unseen test. Presentation
preparation remains separate from this cleanup.

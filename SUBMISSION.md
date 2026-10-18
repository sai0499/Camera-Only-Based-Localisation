# TrackCode Apex: run the submitted localizer

Use Windows x64 and one CUDA GPU. Run `setup.cmd --no-pause` for the locked Python
3.12.13 environment and declared public checkpoints. Extract the accompanying private
`deadline-submission-assets-v1.zip` at the repository root; it preserves `artifacts/`
paths and supplies the frozen fitting-only landmark/retrieval bank and pinned weights.
Do not copy recordings, GT or `.venv` into the source archive.

The judge supplies one initial pose JSON, with exactly these keys (replace the example):

```json
{"timestamp_ns":"1791894452091749130","x":0.0,"y":0.0,"yaw":0.0}
```

Use the image exposure timestamp as an integer decimal string, metres and radians.
For a recording with the separate body-velocity topic:

```powershell
& .\.venv\Scripts\python.exe -B run.py --bag C:\path\test.mcap --input-recording --initial-pose C:\path\initial.json --artifacts artifacts/fit-stage3-multilap-v5-rigproof-v2 --retrieval --vehicle-motion validation-twist --runtime --output results/submission-test
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

The selected configuration was frozen before this reserved holdout evaluation:

| Motion-assisted result | Tuning (270.8 s) | Holdout (219.2 s) |
|---|---:|---:|
| Position RMSE m | 0.530 | 0.635 |
| Lateral RMSE m | 0.262 | 0.300 |
| Yaw RMSE deg | 0.195 | 0.300 |
| Poses within 1 m | 92.3% | 90.6% |
| Coverage, official 1 s bins | 100% | 100% |
| Average steady file-replay time/output ms | 66.3 | 76.6 |

Measured on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), CUDA 12.8, Windows 11.
Timing includes reader and worker waiting; startup is separate. These averages do not
certify bounded latency or a higher native camera frequency. The motion-assisted mode
still has full-visual-pose gaps up to 7.2 s on holdout. The separate unscored camera lap
has no GT and uses an experimental image-derived initialization; it is not the hidden test.

See [the full accuracy report](task_docs/DEADLINE_ACCURACY.md),
[selection receipt](task_docs/evidence/deadline_selection.json),
[model declarations and notices](config/MODEL_NOTICES.md), and
[offline/live viewer instructions](replay_ui/README.md).

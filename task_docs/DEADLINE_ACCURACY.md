# Deadline accuracy: optional vehicle-motion assistance

Owner authorized taking over B's remaining work on 18 October. The challenge PDF,
page 2, permits car-state accelerations/speed/other fields while prohibiting inference
access to GT position and orientation. This amendment uses the existing visual map,
runner and six-state filter. It adds no dependency, checkpoint or estimator.

## Inputs and inference boundary

Camera-only is still the default. Explicit `--vehicle-motion state-fields` reads only
the header timestamp, body vx/vy and body yaw rate from the original state topic.
The raw CDR parser validates the schema, skips pose/accuracy/status bytes without
unpacking them, and never reads the subsequent track/lane fields. Alternatively,
`--vehicle-motion validation-twist` reads the equivalent body-frame TwistStamped topic.
No dynamic TF, position, orientation, GT CSV or reference cache enters inference.
One initial timestamp/x/y/yaw remains mandatory, supplied by the judges for the hidden run.

Controls must have both a source stamp no later than prediction time and a log arrival
no later than the current image group's latest constituent arrival. Samples older than
100 ms fall back to the original camera filter's prediction. Missing topics, changed
layouts and invalid controls fail explicitly. Full-file prefetch is an I/O optimization;
future values are excluded from prediction by both checks.

SE(2) body-twist integration drives prediction. A Student-t innovation weight with
5 degrees of freedom reduces accepted visual outliers against a fresh motion prior,
including confirmed recovery poses. The existing PnP, image coverage, innovation,
recovery and scheduling gates are unchanged. Controls never count as visual updates:
prediction-only and Lost outputs remain recorded. Motion noise floors are conservative
assumptions, not a calibrated statistical guarantee. A persistently wrong motion prior
can resist a correct visual recovery; different-slip behavior remains a limitation.

## Selection before holdout

[The frozen selection receipt](evidence/deadline_selection.json) binds inference source
hashes, map/index identities and the tuning-only decision before the new holdout run.

| Whole tuning episode | Position RMSE m | Lateral RMSE m | Yaw RMSE deg | Within 1 m |
|---|---:|---:|---:|---:|
| Camera-only v5 baseline | 12.893 | 5.977 | 6.958 | 69.2% |
| Motion propagation, ordinary fusion | 1.409 | 0.854 | 0.458 | 82.2% |
| Selected: motion + robust visual fusion | 0.530 | 0.262 | 0.195 | 92.3% |
| Spatial-feature/12 m pair map + motion | 0.538 | 0.219 | 0.216 | 93.3% |

All have 100% one-second output coverage. The existing v5 visual map is selected for
the lower primary position RMSE. The optional spatial selector keeps 32 features per
4x4 usable-image cell before filling the remaining 1,024-point budget; wider pair
selection preserves the 8-neighbor cap. That fitting-only experiment has 107,029
landmarks/2,456 references versus 114,855/2,456 in v5. Its modest lateral benefit does
not justify replacing the selected bank for this submission.

The selected geometry-bound bank copies v5 map arrays and retrieval descriptors
unchanged. It adds hash-bound proof of the original raw camera calibration, remapping
recipe and selected camera extrinsics. This allows an identical front camera to be
used when an unrelated rear mount changes. Changed selected-camera geometry still
fails, and legacy banks without proof retain the original strict identity requirement.

## Reproduce

After setup and privately transferring the frozen bank, use the single production runner:

```powershell
# Development tuning: the exporter is an offline harness, not part of inference.
& .\.venv\Scripts\python.exe -B scripts/export_initial_pose.py --split tune --out tmp/tune-initial.json
& .\.venv\Scripts\python.exe -B run.py --artifacts artifacts/fit-stage3-multilap-v5-rigproof-v2 --retrieval --initial-pose tmp/tune-initial.json --split tune --vehicle-motion state-fields --runtime --output results/your-tuning-run
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py score results/your-tuning-run/submission.csv --split tune --out results/your-tuning-run/evaluation
& .\.venv\Scripts\python.exe -B validate_run.py --submission results/your-tuning-run/submission.csv --diagnostics results/your-tuning-run/diagnostics.jsonl --bag c0_2026-10-13_22-53-42_camloc_0.mcap --initial-pose tmp/tune-initial.json --split tune --evaluation results/your-tuning-run/evaluation --out results/your-tuning-run/validated.json

# A separate input: use the organizer's initial JSON and the actual available motion topic.
& .\.venv\Scripts\python.exe -B run.py --bag C:\path\test.mcap --input-recording --artifacts artifacts/fit-stage3-multilap-v5-rigproof-v2 --retrieval --initial-pose C:\path\initial.json --vehicle-motion validation-twist --runtime --output results/test-run
```

If the separate input contains the supported state velocity fields instead, select
`state-fields`. If no permitted motion exists, omit the flag and report the camera-only
result separately. Never substitute GT pose or orientation. The initial JSON contains
exactly `timestamp_ns` (decimal string), `x`, `y`, `yaw` (metres/radians).

The selected bank can be reconstructed from the original fitting source and already
frozen v5 bank without changing its learned arrays:

```powershell
& .\.venv\Scripts\python.exe -B scripts/bind_map_geometry.py --artifacts artifacts/fit-stage3-multilap-v5 --output artifacts/your-fresh-geometry-bound-bank
```

Generated maps, checkpoints, recordings and GT stay ignored and transfer privately.
The source-only Git repository is insufficient without the declared frozen artifacts.

## Verification and remaining gates

The 30 s real-recording audit under `results/deadline-motion-integrity-v1` passed:
poisoning all state fields except the exact timestamp/body vx/vy/yaw-rate bytes and
corrupting dynamic TF left all 300 outputs and logical decisions unchanged. Altering
future allowed motion left all 100 preceding outputs and decisions unchanged. The
audit compares framed permitted-image/calibration payload hashes and allowed-control
hashes as well as predictions, so it is non-vacuous. Reproduce with:

```powershell
& .\.venv\Scripts\python.exe -B scripts/check_vehicle_motion_integrity.py --artifacts artifacts/fit-stage3-multilap-v5-rigproof-v2 --initial-pose tmp/tune-initial.json --output results/your-fresh-motion-audit
```

The selected preliminary tuning run has 1,814 full visual updates, 221 Lost outputs and
a 4.800 s longest full-pose gap. Its output/schema/exposure/metric checks pass, but its
unchanged visual-baseline gate fails. Motion-assisted accuracy does not sign off the
camera-only Stage 3 continuity requirement. Runtime is file-replay telemetry until a
separate rate-faithful workload is qualified. The additional camera lap has no absolute
GT or supplied initial pose: its image-derived initialization permits an unscored
lighting/speed stress test only, not an accuracy or hidden-test claim.

## Final frozen runs

| Result | Fresh tuning | Reserved holdout |
|---|---:|---:|
| Outputs / evaluated | 2,708 / 2,708 | 2,191 / 2,191 |
| Position RMSE m | 0.530 | 0.635 |
| Position median / p95 m | 0.301 / 1.165 | 0.448 / 1.250 |
| Lateral / longitudinal RMSE m | 0.262 / 0.461 | 0.300 / 0.559 |
| Yaw RMSE degrees | 0.195 | 0.300 |
| Within 1 m | 92.3% | 90.6% |
| Official 1-second-bin coverage | 100% | 100% |
| Full visual updates / Lost outputs | 1,814 / 221 | 1,283 / 282 |
| Longest full-visual-pose gap s | 4.800 | 7.200 |
| Average steady file-replay time/output ms | 66.3 | 76.6 |
| Group service p95 ms (reader excluded) | 161.0 | 184.5 |

Local folders: `results/deadline-final-tune-v1`, `results/deadline-holdout-v1`.
Both official x/y evaluator versions and the reconciled supplied-map report are retained
under each `evaluation/`. Both `validated.json` files confirm schema, complete exposure
sequence and complete metric sections. The unchanged visual-baseline gate remains false.
The fresh tuning replay matches all 2,708 earlier selected poses and logical decisions
exactly. Inference source hashes match the pre-holdout selection receipt.

Runtime hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported VRAM, driver
617.42, Torch CUDA 12.8, Windows 11, Python 3.12.13. Fresh tuning startup is 21.8 s;
its average includes 3.4 ms reader time and worker joining, with 30.0 ms average for the
separately defined pipeline excluding worker waits. The GPU inference runs are sequential;
small CPU-only offline evaluation/plotting may overlap. See the raw runtime summaries
for memory, percentile and measurement-limit fields.

The lighting/speed recording, `results/daylight-motion-stress-v1`, has body-forward
speeds 15.3–68.0 m/s after the experimental image-derived initial pose. It emits all 644
post-initialization groups, with 172 full visual updates, 291 Lost outputs, no processing
faults and 86.6 ms average steady file-replay time/output. Its longest visual gap is
9.400 s. The original 739 front images include an unlocalized 95-frame prefix, so its
87.1% original-frame output fraction cannot qualify this entire recording. There is no
GT score; more visual updates do not demonstrate an accuracy gain in different lighting.

Each scored folder contains `localisation_vs_ground_truth.png` / `.svg` and an entirely
offline `viewer-with-gt/index.html`. The stress folder has `viewer/index.html` without GT.
These are passive offline outputs and do not alter predictions. Root [SUBMISSION.md](../SUBMISSION.md)
provides the judge-facing single-run command. The private asset ZIP and receipt live
under `results/`; code is published only on `dev/implementation`, not `main`.

[The final measured receipt](evidence/deadline_accuracy.json) retains scorer metrics,
validators, runtime summaries, processing-error checks, all 5,543 control-availability
checks, exact repeatability and integrity results. **1,115 tests pass** in the final
full suite; Ruff, JavaScript syntax, UI geometry and the restarted loopback API pass.
The frozen source matches all three final runs. The split is checked against its exact
SHA-256; protected files, evaluator/tools and dependency locks have no Git changes.

Private `results/deadline-submission-assets-v1.zip` is 802,798,853 bytes, with SHA-256
`3e9d58e102b402d99b7ec5ed56ee9e578c511f71cde8c10d302fd9bbfb21703f`.
Its ZIP integrity check passes and it contains only the frozen inference bank, the
three pinned public checkpoints and an asset hash receipt. No recording, GT CSV or
environment is included. Extract it at the repository root alongside the source.

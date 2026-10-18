# Seven-camera rescue mapping and selection

Owner requested all seven feeds be inspected and used intelligently to improve weak
track locations. The prior bank used only `fr` landmarks. Its reader grouped `fr` and
`r`, but `fl/cl/cr/rl/rr` could not produce mapped poses. Simply receiving seven camera
calibrations was not seven-camera localization.

## Map changes

The existing fitting-only front bank is preserved as an exact prefix, including its
3D points, descriptors, keyframes, pixels, landmark identities and front mask. The new
bank adds six-camera rescue references at three spatially recurring weak regions.
Regions were identified from causal tuning predictions and full-visual-pose gaps, not
tuning GT coordinates. Only fitting GT was used for triangulation and camera poses.
The surveyed track borders and supplied map are unchanged: the improvement is to the
visual landmark map used to localize on that track.

| Bank | Cameras | References | Landmarks |
|---|---:|---:|---:|
| Prior v5 | `fr` | 2,456 | 114,855 |
| First rescue candidate | all 7 | 2,840 | 128,500 |
| Selected extended rescue bank | all 7 | 2,942 | 133,610 |

The added 486 references (81 per auxiliary camera) use fitting windows 265–274,
299–309 and 372–379 seconds after fit start, 5 m / 10-degree keyframe selection,
the existing eight-neighbor cap and 12 m target-baseline pairing. At least three
observations, positive depth, one-degree triangulation angle and two-pixel median
reprojection gates remain unchanged. The append tool rejects overlapping cameras,
incompatible provenance, out-of-fitting observations and reference-count overflow.
Original rows are never truncated, retriangulated or overwritten.

Fitting images at the weak regions showed substantial tyres/sidepods in side views.
The versioned `bodywork-rescue-v1` masks mirror the original front mask for `fl` and
conservatively exclude the lower bodywork in `cl/cr/rl/rr`. The `fr` mask stays identical.
Retrieval preprocessing is bound to the actual mask recipe and the new map identity;
the existing front retrieval descriptors reproduce byte-for-byte in the new index.

## Camera policy

`camera_policy: support-v1` is explicit in the frozen bank manifest. Legacy banks keep
the original frontend and camera grouping. The new runner groups all compatible mapped
cameras, always retaining the front timing stream. It records all calibration exclusions.
An unrelated changed mount cannot disable an otherwise compatible front camera, and an
incompatible camera cannot use the old map's geometry through this fallback.

- Rank available views using nearby heading-compatible reference landmark counts and
  occupied image cells. No fitting lap time or query GT is a search key.
- Decode the front continuously; probe the strongest mapped auxiliary and preserve
  its same-camera image chain through handoff/confirmation. Continue useful auxiliary
  tracking. Cameras without local map support cannot reserve and starve front refresh.
- Reduce priority after failed workers or handoffs inconsistent with the current prior.
  Failure feedback expires in source time. Deterministic ties keep replay reproducible.
- Prefer a valid front pose when current poses agree. Auxiliary views rescue missing
  support; incompatible current poses retain the existing conflict/confirmation rule.
- Use four times the pose variance for auxiliary corrections, including recovery.
  This is a conservative assumption for limited rescue coverage, not measured calibration.
  Select one visual correction per group; do not sum correlated camera poses.

The scheduler still permits one in-flight job, one pending request, three normal or
four recovery jobs per source second, at most two references per job, 512 LK tracks per
camera and 256 handoff points. Every accepted handoff re-solves the current same-camera
image. PnP, innovation, loss/recovery and acceptance thresholds are unchanged.
`CameraSelection` diagnostics record scores, decoded feeds and proposed jobs. The viewer
supports inspecting any selected camera at its exact recorded exposure.

## Whole-tuning results and trade-off

[The frozen selection](evidence/multicam_selection.json) was written before new holdout
regression scores. The pre-existing holdout has already been inspected in earlier work;
a repeated check is labelled a regression, not a new unseen evaluation.

| Result | Previous front bank | Selected seven-camera rescue |
|---|---:|---:|
| Position RMSE m | 0.530 | 0.535 |
| Lateral RMSE m | 0.262 | 0.289 |
| Yaw RMSE degrees | 0.195 | 0.210 |
| Position p95 m | 1.165 | 1.119 |
| Within 0.5 m | 69.1% | 71.8% |
| Within 1 m | 92.3% | 91.5% |
| Official 1-second output coverage | 100% | 100% |
| Full visual updates | 1,814 | 1,974 |
| Lost outputs | 221 | 79 |
| Longest full-visual-pose gap s | 4.800 | 2.700 |
| Gaps above the unchanged 3 s gate | 4 | 0 |
| Validator visual baseline gate | false | true |

Both are velocity/yaw-rate assisted runs over the same frozen tuning interval. All
2,708 outputs have valid image exposure/support checks. The first two seven-camera
candidates scored 0.612 and 0.606 m position RMSE and retained a slightly-over-3 s gap;
they are not selected. Fixing a front-refresh starvation bug, extending fitting rescue
coverage, preferring valid front evidence and conservative auxiliary noise produced
the selected result. The full-tuning camera-only ablation is recorded separately.

One final fitting-only spatial-feature variant was also tested with the same 2,942
reference budget. It used the earlier `grid-reserve-32` front bank and rebuilt matching
auxiliary features. Assisted tuning had 0.552 m position / 0.280 m lateral RMSE,
2,072 visual updates and a 2.200 s longest gap. However, its camera-only ablation had
3.265 m position RMSE, 6.502-degree yaw RMSE and a 5.600 s gap, failing continuity.
It is rejected using tuning results; `fit-seven-rescue-v2` remains the frozen release.

The camera-only ablation confirms this is a visual map/selection improvement:
`results/seven-camera-cameraonly-tune-v1` scores 2.554 m position RMSE (previously
12.893 m), 1.924 m lateral RMSE, 2.673-degree yaw RMSE, 0.465 m median / 5.514 m p95,
72.0% within 1 m and 100% one-second output coverage. It has 1,956 full visual updates,
104 Lost outputs and a 2.300 s longest full-pose gap; its unchanged visual-baseline
validator passes. Average file-replay time is 68.1 ms/output. These camera-only errors
are still above the brief's sub-metre lateral target; the assisted mode is distinct.

This improves continuity and p95 position error. It is not an improvement in every
accuracy metric: lateral/yaw RMSE and within-1 m share are worse than the prior candidate.
The previous bank and its private archive remain available for the lower lateral error.

| Selected tuning usage | Decoded images | Anchor jobs | Selected full poses |
|---|---:|---:|---:|
| `fr` | 2,708 | 727 | 1,729 |
| `fl` | 34 | 4 | 12 |
| `cl` | 10 | 1 | 0 |
| `cr` | 88 | 22 | 23 |
| `r` | 71 | 9 | 0 |
| `rl` | 338 | 52 | 189 |
| `rr` | 99 | 21 | 21 |

All seven are mapped and considered; five supplied accepted poses on this run. Most
rescue came from `rl`. Zero accepted `cl/r` poses are reported as zero, not hidden as
seven equally successful cameras. JPEG decoding totaled 3,348 images, far below decoding
seven images for all 2,708 groups. There were no processing errors. About 87.4 ms/output
was measured during the selected replay, overlapping some CPU verification; the final
isolated repeat supplies the delivery timing.

The isolated repeat `results/seven-camera-final-tune-v1` reproduces the selected assisted
submission byte-for-byte and averages 76.7 ms/output (not rate-qualified).

## Turn-aware camera bias experiment (opt-in, not selected)

`run.py --camera-turn-bias` multiplies a side camera's support score by 2 when it looks
toward the inside of a turn and by 0.5 on the outside, when the causal prior's yaw rate is
at least 0.15 rad/s (`cl`/`rl` look left, `cr`/`rr` right; `fr`, `fl`, `r` unchanged). Same
bank `fit-seven-rescue-v2`, fresh replays, unchanged gates/evaluator/validator:

| Result | Camera-only tune | Assisted tune | Assisted holdout regression |
|---|---|---|---|
| Position RMSE m (baseline -> turn bias) | 2.554 -> 2.619 | 0.535 -> **0.521** | 0.655 -> 0.730 |
| Lateral RMSE m | 1.924 -> 1.732 | 0.289 -> 0.268 | 0.330 -> 0.500 |
| Yaw RMSE deg | 2.673 -> 2.942 | 0.210 -> 0.186 | 0.239 -> 0.314 |
| Within 1 m | 72.0% -> 71.9% | 91.5% -> 92.6% | 86.5% -> 86.1% |
| Longest full-pose gap s | 2.3 -> 2.3 | 2.7 -> 2.4 | 9.8 -> 10.6 |

It helps assisted tuning but is mixed camera-only and worse on the (already inspected)
holdout, so it does not generalise and stays **off by default**; the frozen release is
unchanged. Runs: `results/turnbias-{cameraonly,assisted}-tune-v1`, `results/turnbias-assisted-holdout-v1`.

## Reproduce and audit

```powershell
& .\.venv\Scripts\python.exe -B build_map.py --output artifacts/your-six-camera-rescue --cameras fl cl cr r rl rr --window 265 274 --window 299 309 --window 372 379 --travel-m 5 --heading-deg 10 --rescue-masks --pair-target-baseline-m 12
& .\.venv\Scripts\python.exe -B scripts/merge_landmark_banks.py --base artifacts/fit-stage3-multilap-v5-rigproof-v2 --rescue artifacts/your-six-camera-rescue --output artifacts/your-seven-camera-bank
& .\.venv\Scripts\python.exe -B scripts/build_retrieval_index.py --artifacts artifacts/your-seven-camera-bank
& .\.venv\Scripts\python.exe -B run.py --artifacts artifacts/fit-seven-rescue-v2 --retrieval --initial-pose tmp/tune-initial.json --split tune --vehicle-motion state-fields --runtime --output results/your-seven-camera-tune
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py score results/your-seven-camera-tune/submission.csv --split tune --out results/your-seven-camera-tune/evaluation
& .\.venv\Scripts\python.exe -B validate_run.py --submission results/your-seven-camera-tune/submission.csv --diagnostics results/your-seven-camera-tune/diagnostics.jsonl --bag c0_2026-10-13_22-53-42_camloc_0.mcap --initial-pose tmp/tune-initial.json --split tune --evaluation results/your-seven-camera-tune/evaluation --out results/your-seven-camera-tune/validated.json
& .\.venv\Scripts\python.exe -B scripts/report_camera_support.py --artifacts artifacts/fit-seven-rescue-v2 --run results/your-seven-camera-tune --baseline results/deadline-final-tune-v1 --output results/your-seven-camera-tune/camera_support.json
```

The source and private inference-assets archives are kept separately under `results/`.
No recording, GT, weights or generated map is pushed to Git. Original reference banks,
protected assets, evaluator sources and split bytes remain unchanged.

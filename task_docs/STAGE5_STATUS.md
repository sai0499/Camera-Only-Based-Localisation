# Team C Step 5: real-recording verification and runtime instrumentation

Step 5 verification has started. **Step 3 and Step 5 are not signed off.** The existing
full-tuning continuity failure remains recorded in [STAGE3_STATUS.md](STAGE3_STATUS.md).
Passing these software/integrity checks does not certify localization accuracy or a rate.

Small shareable receipt: [stage5_verification.json](evidence/stage5_verification.json).
Generated MCAP variants, predictions and full diagnostics remain local and ignored.

## A integration check

After fetching `origin`, `origin/darshana` points to `18a93d3fa073993580b0fcf54433c24391abf859`,
an ancestor of the starting development head `93e42c0b3092898836635c6d471f823af879550e`.
There are no outstanding A commits to merge from that branch. `main` remains at
`ce10848b383a5c85f82a41921ead0317e98de3f1`. No B-owned code is changed by this step.

## Implemented checks

- `scripts/check_step5.py` creates fresh private MCAP variants and runs the production CLI
  in a fresh process for each case. Schemas, sequences, log/publish timestamps and channel
  metadata are retained. Prohibited-input variants have a matching framed hash of every
  permitted message payload and timestamp. The original recording is never overwritten.
- Repeatability compares all output timestamps, x/y within 0.00001 m and wrapped yaw
  within 0.0000001 rad, plus selected landmark identities, status, evidence kinds, recovery
  and logical scheduler decisions. Wall-time worker/stall fields are excluded.
- Input isolation removes dynamic `/tf` and the live state topic, then separately corrupts
  their ROS payloads inside valid CRC-checked MCAP containers. The audit never decodes
  those payloads. The child inference process receives only the existing one-pose JSON.
- The causality case replaces both selected cameras with black JPEGs at/after ten seconds
  and compares every earlier completed output and logical decision.
- Fixed image-stress cases start at 15 s: each camera removed for 1/3 s, both cameras
  blacked out for 0.5/1 s, Gaussian blur (15x15, sigma 3), quarter exposure and a white
  glare disk for 1 s. All output groups are retained and validated against the modified
  input; recovery delays and statuses are measured, not assumed to pass.
- `scripts/report_step5_errors.py` is offline only. It uses the organizer's integer-time
  interpolation, 50 ms GT-bracket gate and wrapped yaw convention on the predeclared audit
  interval. It records position/yaw errors and reset errors. Bounded diagnostics are not
  presented as official full-episode scores and GT is never passed to inference.

## Runtime instrumentation

`run.py --runtime` writes `runtime.jsonl` and `runtime.json` separately from evidence.
It measures reader cost, complete group service, elapsed/output and nonwaiting pipeline
cost (reader + service minus logged worker joins, including handoff). Windows process
peak working-set/commit and PyTorch CUDA peak allocated/reserved memory include startup.
Instrumentation preserves the original FrameSetSource protocol and predictions.

`--transport-speed 1` paces original constituent packet arrivals; 2/3/6 accelerates those
arrivals while preserving exposure stamps and source-time anchor budgets. This is a
transport-load test, not native higher-rate imagery or a rate-faithful computational
profile. Deadline-closure wait and physical exposure latency are not instrumented. No
rate is marked qualified by this telemetry tool.
EOF-cancelled daemon work is not joined by this instrumentation; its cost must be handled
explicitly in the final workload qualification rather than inferred from this summary.

## Full-episode integrity results

`results/step5-full-integrity-v1` uses the complete unchanged tuning interval and fresh
processes with the v5 bank/index. Each case emitted all 2,708 expected outputs. The fresh
baseline CSV SHA-256 is `c175017bf10f26810389aa17265b48aa03dc1ede69e80244c24c848d870944c3`,
identical to `tune-stage3-fix-7`: optional telemetry preserved the original predictions.

| Check | Result |
|---|---|
| Fresh repeat process | All 2,708 poses and checked logical decisions match exactly |
| Prohibited topics removed | All 2,708 poses/decisions match; permitted-message hash unchanged |
| 57,528 prohibited payloads corrupted | All 2,708 poses/decisions match; permitted-message hash unchanged |
| 5,214 future image payloads changed | All preceding 100 poses/decisions match exactly |
| Complete-group schema, stamps and support causality | Independently valid for every case |

This certifies these specific repeatability/input/prefix checks, not correctness of every
pose. The baseline still has 1,799 full-pose updates, 214 Lost rows and a 6.200 s maximum
full-pose gap, so its full visual-baseline gate remains failed. The permanent future
blackout deliberately loses localization after the comparison prefix.
Offline full-episode diagnostics reproduce 12.893 m position RMSE and retain the one
recovery reset with error over 10 m. Those failures are not hidden by the integrity passes.

## Bounded real-data preflight

`results/step5-preflight-v2` retains all 14 cases over the first declared 30 seconds of
tuning, using the current front-only v5 bank/index. Each case emitted 300 independently
validated rows, with no frontend/backend faults. Repeatability and both prohibited-input
checks matched all 300 outputs exactly; the future-image check matched the preceding 100
outputs and logical decisions exactly. No full pose was applied during either blackout.

| Case | First full pose after restoration | Position RMSE over fixed 30 s |
|---|---|---|
| Baseline | n/a | 1.266 m |
| Front absent 1 s / 3 s | 0.177 s / 0.177 s | 1.526 / 1.730 m |
| Rear absent 1 s / 3 s | 0.077 s / 0.077 s | 1.266 / 1.266 m |
| Both blacked out 0.5 s / 1 s | 0.677 s / 0.177 s | 1.311 / 1.311 m |
| Blur / exposure / glare, 1 s each | 0.077 / 0.177 / 0.177 s | 1.271 / 1.297 / 1.233 m |

The separate offline `errors-verified.json` binds GT SHA-256 to the existing reference
manifest. It found no reset errors over 10 m in these short cases. Rear absence exercises
input/grouping behavior with this front-only bank; it does not qualify rear localization.
The permanent future blackout intentionally loses localization (106.735 m RMSE); only its
unchanged earlier prefix is an integrity pass. These diagnostics are not a replacement for
the complete tuning episode, later weak places, full robustness testing or holdout.

The first harness attempt (`step5-preflight-v1`) is retained: an exposure-end/log-arrival
margin bug admitted extra rows and independent validation rejected it. The corrected
preflight excludes selected images at/after its declared exposure end while preserving
legitimate late-arriving packets before that end.

## Preliminary paced telemetry

`results/step5-paced-preflight-v1` paces the same first 30 s at original arrival speed.
All 300 poses and checked decisions match the unpaced preflight exactly. Measured reader
mean/p95: 1.41/2.96 ms; complete group service mean/p95: 56.20/164.66 ms; nonwaiting
pipeline mean/p95/p99: 30.30/46.62/58.70 ms. Latest-packet arrival-to-completion proxy
p95 is 384.78 ms (p99 688.79 ms), above the 200 ms 10 Hz residence budget. This flags a
latency concern; it is not a completed five-minute qualification or a native higher-rate
accuracy test. CPU peak working set is 2.56 GB; CUDA peak allocated/reserved is 1.22/1.69 GB.
Full startup is recorded separately in `run.json`; no cold-start output is removed.

## Reproduction

Use fresh outputs and the locally cached private map/index and recording. The one initial
pose must already have been exported by the existing offline tool.

```powershell
# Bounded preflight: real images, explicitly not the complete ranked episode.
& .\.venv\Scripts\python.exe -B scripts/check_step5.py --artifacts artifacts/fit-stage3-multilap-v5 --initial-pose tmp/b_integrated_tune_initial.json --duration-s 30 --output results/step5-preflight-repeat
& .\.venv\Scripts\python.exe -B scripts/report_step5_errors.py --audit results/step5-preflight-repeat --output results/step5-preflight-repeat/errors.json

# Complete tuning: fresh-process repeatability, prohibited-input and future-prefix checks.
& .\.venv\Scripts\python.exe -B scripts/check_step5.py --artifacts artifacts/fit-stage3-multilap-v5 --initial-pose tmp/b_integrated_tune_initial.json --cases repeat prohibited_remove prohibited_poison future --output results/step5-full-repeat

# Preliminary paced transport telemetry; does not itself qualify a rate.
& .\.venv\Scripts\python.exe -B scripts/check_step5.py --artifacts artifacts/fit-stage3-multilap-v5 --initial-pose tmp/b_integrated_tune_initial.json --duration-s 30 --cases baseline --transport-speed 1 --output results/step5-paced-repeat
```

## Remaining gates

- Sustained full-tuning visual support and B's review of the pending vision/map changes.
- Full-episode robustness/error assessment, including false resets, distractors, native
  rear gaps, invalid geometry and injected backend/packet failures. Existing synthetic
  failure tests remain software evidence rather than real-recording accuracy evidence.
- Five-minute paced native workload plus separately labelled rate-faithful 20/30/60 Hz
  compute profiles, queue residence/high-water telemetry and complete latency measurement.
- Configuration freeze before the untouched holdout, then final replay/delivery work.

## Software verification

Whole suite: **1,021 passed**, no skips/xfails, in 169.73 s:
`.venv/Scripts/python.exe -B -m pytest -q --basetemp tmp/step5-full-suite-v1`.
Ruff passed for `camloc scripts tests run.py build_map.py validate_run.py`.
The 27 new tests cover MCAP payload/timing preservation, non-vacuous input mutations,
actual image perturbations, strict comparison failure cases, offline GT interpolation,
processing-error detection, pacing/backpressure and unchanged production-runner output.
The original recording, supplied map and split hashes were rechecked unchanged. No
A/B-owned implementation, shared record, estimator gate, dependency or evaluator changed.

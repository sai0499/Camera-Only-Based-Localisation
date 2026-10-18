# B PR #2 integration and paired verification

Integrated [PR #2](https://github.com/sai0499/Camera-Only-Based-Localisation/pull/2),
`feature/b-step4-visual-support` at `5e1644f70b7afcff1c19c5ebe2c32c1cefc0c624`, into
`dev/implementation`, preserving C's UI/runtime work. Integration base was `e984109`;
B branched from `93e42c0`. `main` is unchanged.

**There is no localization improvement attributable to this PR.** Both paired prediction
CSVs are identical, and both continuity gates still fail. Team C Step 3 remains open.
Receipt: [b_pr2_integration.json](evidence/b_pr2_integration.json).

## What B delivered

- Review accepting C's retrieval, joint handoff, seeded LK carry, adoption consistency and
  3,000-reference cap. B found no defect in those C changes.
- Synthetic delayed-handoff/current-image tests at 10/20/30/60 Hz, immutable earlier evidence
  checks, backend-fault propagation and offline cached-weight startup checks.
- Offline coverage-grid/LK-carry diagnostics, component profiling and FP16 comparisons,
  plus updated pretrained-model notices.
- A default-off stricter coverage experiment and full-image cell diagnostics in `pnp.py`.
  It stays **off**. No existing production gate/default, scheduler cap or contract changed.

The PR does not change `build_map.py`, `camloc/vision.py` or `camloc/retrieval.py`, and includes
no denser fitting bank. The requested weak-place landmark density, near-field coverage and
lower map reprojection error are still outstanding. B's rejected FP16/LK/coverage experiments
are not enabled by integration. B's original laptop measurements remain in [B_STATUS.md](B_STATUS.md)
and are separate from these same-machine paired checks.

## Review fixes made during integration

- FP16 adoption now requires identical query-image sequences and at least one paired accepted
  pose. With no pose evidence, differences are unavailable rather than zero; adoption fails.
- Grid/carry scripts validate every stretch before opening the recording, so a later invalid
  stretch cannot cross the selected fitting/tuning interval. Nonpositive strides are rejected.
- Decode/canonicalization timing excludes the separate offline GT-loading work.
- The offline-startup test must reach nonempty reference matching, rather than merely accept
  any failed result. Focused tests cover all these behavior changes.

## Full tuning comparison

Fresh processes before/after integration used the same `fit-stage3-multilap-v5` frozen bank
(`fitmap:cb52ba980ffd803e`), retrieval index (`retrieval:eff77597458adb60`), exported single initial
pose, complete fixed tuning interval, inference options, evaluator and validator. No map was
rebuilt and no gate was relaxed. Replays were serialized with other GPU work.

| Measure | Before | Integrated |
|---|---:|---:|
| Outputs / output coverage | 2708 / 100% | 2708 / 100% |
| Original x/y RMSE | 12.893 m | 12.893 m |
| Full-pose updates | 1799 | 1799 |
| Lost outputs | 214 | 214 |
| Longest full-pose gap | 6.200 s | 6.200 s |
| Frontend errors / worker faults | 0 / 0 | 0 / 0 |
| Visual-baseline gate | fail | fail |

All 2,708 timestamps and poses are byte-identical; selected evidence identities, statuses,
recovery and logical scheduler decisions match exactly. CSV SHA-256:
`c175017bf10f26810389aa17265b48aa03dc1ede69e80244c24c848d870944c3`.
Both evaluations retain the original, updated x/y and reconciled supplied-map reports.
Both independent validators pass schema/exposure/coverage/metric checks and fail only the
visual-baseline continuity gate. This is the current v5 configuration, not a comparison
between B's historical v1 result and C's v5 result.

## Supplied camera-lap experiment

Repeated the earlier experiment using the exact saved image-derived initial pose at 9.5 s,
the same frozen v5 bank/index and the same temporary front-camera compatibility adapter.
The adapter again checked exact raw/canonical front calibration against the hash-verified
fitting recording. No adapter or automatic initialization was added to the production CLI/UI.

All 644 post-start outputs are byte-identical to the earlier experiment, including
163 full-pose updates, 481 prediction-only outputs,
310 Lost outputs and a 9.200 s longest full-pose gap.
The first 95/739 front frames remain unlocalized. Output/schema/causality checks cover only
the post-start segment. No absolute pose GT exists in that supplied bag, so position/yaw RMSE
cannot be reported. Continuity still fails; a geometric initial estimate is not verified GT.

## Software verification and limits

- Full suite: **1,085 passed in 196.93 s**, no skips/xfails. This includes 30 B tests and 18
  integration diagnostic tests beyond the previous 1,037-test suite.
- Ruff and whitespace checks pass. Python 3.12.13; existing pinned environment/dependencies.
- Original recordings, calibration/map assets, hash-bound split, evaluator/reference caches,
  gates and fitting artifacts were preserved. Generated runs and logs stay ignored locally.
- Runtime receipts are unpaced file-replay telemetry, not live/native high-rate qualification.
  Synthetic handoff tests do not establish real 20/30/60 Hz localization accuracy.
- Holdout was not scored or used to select these changes. Step 3 and Step 5 remain unsigned.

Reproduction outputs: `results/b-pr2-before-v5`, `results/b-pr2-after-v5`,
`results/hidden-lap-experimental-v1` and `results/b-pr2-hidden-after-v5`. The paired runs use
`tmp/b-pr2-integration-initial.json` (tuning); the hidden comparison uses the earlier
`tmp/hidden_experiment_initial.json` via `tmp/b_pr2_hidden_replay.py`.

# Team C Step 3: B integration and the real baseline gate

**18 October deadline update:** optional vehicle-motion assistance reaches 0.530 m
whole-tuning position RMSE, 0.262 m lateral RMSE and 0.195-degree yaw RMSE with 100%
output coverage. See [DEADLINE_ACCURACY.md](DEADLINE_ACCURACY.md) for the owner-authorized
input amendment and frozen selection. Its 4.800 s full-visual-pose gap still fails this
stage's unchanged continuity gate. **Camera-only Step 3 remains open.** The deadline
amendment authorizes independent evaluation of this separate mode; the earlier results
and sequencing guidance below describe the original camera-only build.

B's [PR #1](https://github.com/sai0499/Camera-Only-Based-Localisation/pull/1), head
`b1202d8a27b7581fc6dac153215f10a822238963`, is integrated into `dev/implementation`.
It adds DISK/LightGlue, the fixed-pose fitting builder, frozen map loader, LK/PnP tracker,
anchor correspondence worker and ground motion. Its base was the current A+C commit
`22a2b4f`; none of A/C's existing files conflicted. `main` remains unchanged.

**Integration works; the Step 3 visual-baseline gate has failed and remains open.**
B's PR also explicitly disclosed this gate as unmet. Do not advertise the prediction-only
tuning CSV as camera localization or proceed to holdout/optional demo work on that basis.

## Round 5: B PR #2 verification/tools integration (no accuracy improvement)

[PR #2](https://github.com/sai0499/Camera-Only-Based-Localisation/pull/2), head `5e1644f`,
is integrated on `dev/implementation` with C's newer runtime/UI work preserved. B accepted
C's retrieval, joint handoff, seeded carry and 3,000-reference cap. The PR adds delayed-handoff,
offline-startup, coverage and profiling checks; it does not change map building/retention or
deliver the requested denser fitting bank. Its stricter coverage experiment remains off.

[Integration report](B_PR2_INTEGRATION.md), [receipt](evidence/b_pr2_integration.json).
C fixed diagnostic split-window validation, unsupported FP16 adoption, timing contamination
from GT loading and a weak offline-startup assertion. Full suite: **1,085 passed**, no
skips/xfails; Ruff passes. No production threshold, scheduler cap, shared contract or dependency
was changed; no fitting artifacts were rebuilt.

Fresh paired v5 tuning runs emitted identical CSVs and all 2,708 logical decisions matched:
**12.893 m RMSE, 1,799 full-pose updates, 214 Lost outputs, 6.200 s longest full-pose gap**.
Both pass the other four validator checks and fail the visual-baseline gate. The separate
camera-lap experiment also reproduces all 644 post-start outputs exactly (163 full-pose updates,
310 Lost, 9.200 s longest gap); its initial 95 frames remain unlocalized and it has no GT score.
This demonstrates no regression and no localization improvement from B's PR. The map-support
work at the recurring places remains outstanding; **Team C Step 3 is not signed off**.

## Round 4: 3,000-reference bound and seeded handoff carry (not signed off)

Receipt: [stage3_recheck.json](evidence/stage3_recheck.json). **Gate still open:** on the fresh
fix-7 replay, unchanged `validate_run.py` passes 4 of 5 checks (valid, exposure sequence,
coverage 1.0, metric sections); the visual-baseline gate fails (longest full-pose gap 6.20 s,
4 gaps > 3 s). Suite: **994 passed** (986 existing + 8 new); Ruff passed.

B-owned code changes (no contract, gate, threshold, cap, evaluator, validator or dependency change):

- `build_map.py`: named `MAX_REFERENCES = 3000` and `check_reference_count(n)` replace the
  inline 2,000-reference abort; still reject-not-truncate, not written to the manifest.
- `camloc/vision.py`: bounded per-camera `_prior_log` (≤ `HISTORY_MAX_FRAMES`, written by
  `_remember_prior`, cleared on calibration change); `_motion_guess` extracted from `_guess`
  (identical math); each handoff carry transition whose two exposures both have a logged causal
  prior is LK-seeded with the projected motion, else unseeded. `_carry` already retries seeded
  failures unseeded. Tests: `tests/test_map_build.py`, new `tests/test_stage3_handoff_seed.py`.

Bank `artifacts/fit-stage3-multilap-v5` (`build_map.py --window 0 700.39`, defaults, `fr`):
2,456 refs / 114,855 landmarks / median error 1.48 px, 71 weak (< 30 landmarks), 481.7 s,
`fitmap:cb52ba980ffd803e`; index `retrieval:eff77597458adb60` (2,385 keyframes).

| Measured value | fix-5 (round 3) | **fix-7 (v5 + seeded carry)** |
|---|---|---|
| Full-pose updates (tracking / handoff / resets) / lost rows | 1,755 (1,231 / 512 / 12) / 239 | **1,799 (1,294 / 494 / 11) / 214** |
| Longest full-pose gap / gaps > 3 s | 5.40 s / 4 | **6.20 s / 4** |
| Original = updated x/y RMSE | 9.665 m | **12.893 m** |
| Reconciled map x/y / racing-line s/n RMSE | 9.667 / 9.025 m | **12.895 / 11.905 m** |
| Full-pose error median / p90 | 0.31 / 1.21 m | 0.33 / 1.38 m |
| Wrong-lock rows > 10 m | 2 | **1** (185.48 s reset, 10.1 m) |
| Anchors dispatched / delivered; worker failures | 864 / 824; 40 | 855 / 829; 26 |
| Handoffs accepted / recovery `too_few_correspondences` (whole run) | 624 / 38 | 623 / 35 |
| CSV, exposure sequence, coverage 1.0, metrics / visual gate | pass / fail | pass / **fail** |

fix-7 gaps over 3 s (t, s): 104.28–109.78, 138.88–142.08, 142.38–146.58, 231.88–238.08;
submission SHA-256 `c175017b…0944c3`. Diagnosis (GT used only in a deleted `tmp/` script):
the gaps stay at the recurring places near 104–110 s, 139–147 s and 232–238 s; fix-5's
259–262 s gap closed. The denser bank cut worker failures (40 → 26) and lost rows, but the
seeded carry did not measurably reduce `too_few_correspondences` (38 → 35 over the run; inside
the gaps 17 of 38 handoffs for fix-5 vs 23 of 57 for fix-7, recounted with one script, so
round 3's 19-of-51 tally is not directly comparable). RMSE regressed although wrong locks fell;
neither change is isolated as the cause, so the optional revert replay (fix-8) was not run.

Next options: attribute the seed effect (v1 bank with the new code, or v5 unseeded), per-place
densification at the three recurring places, and handoff image coverage (`poor_image_coverage`
remains the largest recovery-handoff rejection). Each needs a fresh scored, validated run.

## Round 3: multi-lap fitting bank (superseded by Round 4; not signed off)

Receipt: [stage3_recheck.json](evidence/stage3_recheck.json). **No source code changed.** Same
runtime, split, evaluator, validator, gates, thresholds and contracts as round 2; only the
frozen fitting bank and its retrieval index changed. Each run is a fresh process over the full
unchanged tuning interval. Suite: **986 passed**; Ruff passed.

Bank builds (unchanged `build_map.py`, fitting-interval windows and fitting GT only; retrieval
index from `scripts/build_retrieval_index.py --artifacts <bank>`):

| Bank | Build | Refs / landmarks / median err | Weak (< 30 lm) / build time | Map / retrieval |
|---|---|---|---|---|
| `fit-stage3-dual-v1` (fix-4) | round-2 windows, `fr` + `r` | 1,486 / 64,791 | — | `fitmap:41f4f38f269fa64f` / `retrieval:d10c40cdb6185006` (680 `fr` + 702 `r`) |
| `fit-stage3-multilap-v1` (B1, fix-5) | `--window 0 90 --window 321 700.39`, `fr`, 4 m / 5° | 1,807 / 84,691 / 1.48 px | 73 / 360.2 s | `fitmap:634545ba3c704d5b` / `retrieval:acaaf3f6edd7174e` (1,734) |
| `fit-stage3-multilap-v4` (B4, fix-6) | `--window 0 700.39 --travel-m 5.7`, `fr` | 1,909 / 86,251 / 1.49 px | 91 / 381.9 s | `fitmap:618ee99e157e2291` / `retrieval:c1996fe897e054d0` (1,818) |

B4 covers the whole fitting interval [0, 700.39) s at the smallest 0.1 m travel step that stays
under the builder's 2,000-reference cap. Four passes per place at 4 m spacing would need about
2,458 references.

| Measured value | fix-3 (round 2) | fix-4 (dual) | **fix-5 (B1, best)** | fix-6 (B4) |
|---|---|---|---|---|
| Full-pose updates / lost rows | 1,378 / 602 | 1,221 / 721 | **1,755 / 239** | 1,762 / 244 |
| Longest full-pose gap / gaps > 3 s | 11.80 s / 10 | 13.70 s / 9 | **5.40 s / 4** | 7.60 s / 3 |
| Original = updated x/y RMSE | 27.961 m | 37.157 m | **9.665 m** | 9.561 m |
| Reconciled map x/y / racing-line s/n RMSE | 27.967 / 27.878 m | 37.164 / 37.310 m | **9.667 / 9.025 m** | 9.562 / 8.860 m |
| Wrong-lock rows > 10 m | 5 | 10 (9 `r`) | **2** (10.3, 10.4 m) | 3 (10.7–12.4 m) |
| Anchors dispatched / delivered | 910 / 817 | 962 / 860 | 864 / 824 | 854 / 817 |
| Frontend errors / backend faults | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| CSV, exposure sequence, coverage 1.0, metrics | pass | pass | pass | pass |
| Visual-baseline gate | fail | fail | **fail** | fail |

fix-5 gaps over 3 s (t, s): 104.98–109.68, 142.28–146.48, 232.38–237.78, 258.98–261.98.
fix-6: 59.18–63.08, 139.08–146.68, 232.88–237.58. Submission SHA-256: fix-5 `d6055bea…17c3bc`,
fix-6 `df461848…694b65`, fix-4 `3c36db48…1de5eb`; full hashes in the receipt.

- **fix-4 rejected.** The dual bank kept one pass per place. Recovery alternates cameras as
  designed, so rear jobs took up to 85 % of recovery slots in some gaps, and 9 of its 10 wrong
  locks came from `r`. Front full poses fell by 335.
- **Per-place passes.** The round-2 windows gave each recurring gap place (A–D) exactly one
  fitting pass. B1 adds the 321-, 483- and 605-s laps, giving three passes per place and
  multi-lap support in the former holes at 28–34 s and 180–183 s.
- **fix-5 diagnosis.** Retrieval is correct: 79 of 83 recovery jobs had both references within
  15 m of GT. Map support is the limit: near-GT references hold only 31–60 landmarks at place B,
  and 32 of 83 worker jobs failed `pnp_too_few_inliers`/`insufficient_matches`. Of 51 delivered
  handoffs, 11 were accepted, 19 failed `too_few_correspondences` (the worker delivered 30–56
  points but LK carry dropped below 30), 11 failed `poor_image_coverage` at 4 cells (pose error
  0.1–2 m) and 5 were degenerate. No C/B code defect and no slot starvation was found.
- **Selection caveat.** Window selection used tuning-interval diagnostics and fitting-interval
  GT only. These are tuning results, not a holdout result.

Current recommended configuration: bank `artifacts/fit-stage3-multilap-v1` with
`retrieval:acaaf3f6edd7174e`; best run `results/tune-stage3-fix-5` (shortest longest gap; fix-6
has one fewer gap over 3 s but a longer 7.60 s gap). Four of the five validator checks pass;
the continuity gate fails.

Remaining limitation: four gaps of 3.0–5.4 s remain at places B and C, so the unchanged 3 s
gate fails; 9.7 m RMSE is not a certified localizer. A full pass is unlikely from bank windows
alone under the 2,000-reference cap. Concrete next options:

1. Raise the 2,000-reference build cap, with B's agreement, so four passes fit at 4 m spacing.
2. Reduce handoff LK-carry point loss, so delivered 30–56-point worker results keep ≥ 30.
3. Densify the map per place (more retained landmarks at B and C).

Any option needs a fresh replay, fresh evaluator reports and the unchanged validator.

## Superseded: Round 2: global retrieval + joint handoff (not signed off)

Receipt (now replaced by round 3): [stage3_recheck.json](evidence/stage3_recheck.json). Same bank
(`artifacts/fit-b-integrated-v2`), split, evaluator, validator, gates, thresholds and shared
contracts; fresh process, full unchanged tuning interval. Suite: **986 passed**; Ruff passed.
Retrieval artifact `retrieval:294a97a90ef0922c` (680 `fr` references; offline recall@10
within 25 m 98.5 %, @1 within 15 m 90.0 %; ~18 ms/query on RTX 4060).

| Measured value | `tune-stage3-fix-1` | `tune-stage3-fix-2` (retrieval) | `tune-stage3-fix-3` (+ joint handoff) |
|---|---|---|---|
| Full-pose updates (PnP / handoff / reset) | 537 (373/161/3) | 1,328 (953/365/10) | **1,378** (978/390/10) |
| Lost rows | 1,926 | 708 | **602** |
| Longest full-pose gap / gaps > 3 s | 125.60 s / — | 12.80 s / 13 | **11.80 s** / 10 |
| Original = updated x/y RMSE | 323.450 m | 29.586 m | **27.961 m** |
| Reconciled map x/y / racing-line s/n RMSE | 323.500 / 673.008 m | 29.592 / 30.572 m | 27.967 / 27.878 m |
| Anchors dispatched / delivered | 1,014 / 304 | 913 / 813 | 910 / 817 |
| Frontend errors / backend faults | 0 / 0 | 0 / 0 | 0 / 0 |
| CSV, exposure sequence, coverage 1.0, metrics | pass | pass | pass |
| Visual-baseline gate | fail | fail | **fail** |

fix-3 gaps over 3 s (t, s): 1.98–5.28, 5.98–12.18, 28.18–34.38, 62.88–66.08,
104.38–116.18, 138.68–146.68, 160.18–168.08, 180.08–183.48, 232.88–238.48, 258.48–262.18.
Handoff rejections: `poor_image_coverage` 181, `high_median_error` 92,
`too_few_correspondences` 40, `too_few_inliers` 21, `degenerate_geometry` 11. Five rows
are wrong locks (> 10 m). Submission SHA-256 `5f7808ba…d6d9f1`; source hashes in `run.json`
and the receipt.

Changes:

- `camloc/retrieval.py` (new): EigenPlaces ResNet-18 / 512-D global descriptor, vendored
  under MIT, with a frozen, hash-identified reference index.
- `scripts/build_retrieval_index.py` (new): builds that index from the fitting bank.
- `run.py --retrieval`: loads the index and ranks recovery references by descriptor.
- `camloc/scheduler.py` / `camloc/frontend.py`: `GroupContext.settled` and frontend carry of
  a settled candidate across recovery rounds.
- `camloc/vision.py`: worker-side retrieval ranking and the F2 joint handoff.
- **For B:** `camloc/retrieval.py` and the `camloc/vision.py` changes are in B's scope and
  need B's review before they are treated as accepted.

Remaining limitation: the longest gap is 11.8 s and ten gaps exceed the unchanged 3 s bound,
so the gate still fails; 28 m RMSE is not a certified localizer. The residual gaps are
weak-support sections where handoffs reach only 4–5 of the ≥6 required grid cells (rejected
poses have median 2.6 m error). No C-side defect was found; the bank is front-only. Closing
the gate needs B's map/vision support there, then a fresh scored and validated run.

## Superseded: gate-fix recheck, 19 October 2026 (not signed off)

Receipt (now replaced by round 2): [stage3_recheck.json](evidence/stage3_recheck.json). Same bank
(`artifacts/fit-b-integrated-v2`), split, evaluator, validator, gates and thresholds; fresh
process, full unchanged tuning interval, no row cropping. Suite: **963 passed** in 167.19 s
(`--basetemp tmp/stage3-gate-fix-recheck`); Ruff passed.

| Measured value | `tune-stage3-recheck-fr-final` (previous) | `tune-stage3-fix-1` (this run) |
|---|---|---|
| Output / validated rows | 2,708 / 2,708 | 2,708 / 2,708 |
| Fused full-pose updates / prediction-only rows | 2 / 2,706 | **537** / 2,171 |
| Lost rows / longest full-pose gap | 2,695 / 269.716 s | 1,926 / **125.602 s** (t = 1.1–126.7 s) |
| Original x/y RMSE | 5,838.073 m | **323.450 m** |
| Updated x/y / reconciled map x/y RMSE | — | 323.450 / 323.500 m |
| Reconciled racing-line s/n RMSE | — | 673.008 m |
| Anchors created / dispatched / delivered / rejected | — | 1,014 / 1,014 / 304 / 0 |
| Frontend errors / backend faults | 0 / 0 | 0 / 0 |
| Startup / replay elapsed | 24.42 / 57.54 s | 20.18 / 160.88 s |
| CSV, exposure sequence, coverage 1.0, metric sections | pass | pass |
| Visual-baseline gate | fail | **fail** |

Of the 537 updates, 373 are tracking PnP, 161 handoff PnP and 3 confirmed recovery resets.
Undelivered anchor jobs failed with `insufficient_matches` (538), `pnp_too_few_inliers`
(164) and `pnp_pnp_failed` (8). Handoffs were rejected for `poor_image_coverage` (68),
`high_median_error` (25), `too_few_correspondences` (12), `too_few_inliers` (6),
`degenerate_geometry` (2) and `handoff_tracking_failed` (1). Submission SHA-256
`96f18d10…75eaea3`; source hashes are in `run.json` and the receipt.

Fixes in this round:

- `camloc/filter.py`: after the 3 s full-pose loss boundary the velocity is extrapolated
  as a bounded Ornstein–Uhlenbeck process, so loss no longer drives the position off-track.
- `camloc/frontend.py`: stranding-proof recovery search (uncapped, falls back to the whole
  bank, sweeps a tried set), forward reference ordering at startup, removal of the recovery
  throttle and a per-delivery-delay camera hold.
- `camloc/vision.py` (B's file, minimal change reported to B): `_adopt` applies a
  consistency filter so adopted correspondences agree with the accepted pose.
- Hypothesis H4b was not confirmed, so it was not implemented.
- One justified assertion change in `tests/test_filter.py`
  (`test_property_status_precedence_and_process_noise`) to match the bounded loss model.

Remaining limitation: there is still no full pose from 1.1 s to 126.7 s, plus gaps of 22.2,
14.2 and 12.5 s, against the unchanged 3 s continuity bound. Recovery jobs run, but B's map
support there sits in a far-field horizon band that cannot pass the unchanged ≥6-cell /
≤2 px gates. Closing the gate needs B's map/vision support in those sections, then a fresh
scored and independently validated run. The 323 m RMSE is not a usable localizer.

## Superseded: latest recheck, 18 October 2026 (not signed off)

Small shareable receipt (now replaced by the gate-fix recheck): [stage3_recheck.json](evidence/stage3_recheck.json). Generated
maps, full diagnostics and submissions remain local and ignored.

The reviewed fixes now apply the frozen usable-image extent to the PnP grid, retain bounded
LK hypotheses after weak PnP instead of irreversibly pruning below 30, seed subsequent LK
from accepted current-image poses, and preserve a compatible unused handoff as a track
refresh after one filter update. Disagreeing current poses from the same camera now need
confirmation too. Recovery proposals alternate available cameras; rear recovery keeps its
image chain and healthy tracking reserves its staggered rear slot. Road patches now gather
local support around each border, allowing valid wide roads while rejecting distant queries.
These changes preserve the existing shared records, dependencies and PnP/LK thresholds.

`validate_run.py` no longer signs off a baseline after one visual update. It reports the
longest gap between applied full poses, including episode edges, and requires at least
one full pose and no gap exceeding the filter's three-second full-pose loss boundary.
CSV legality/complete exposure coverage remains a separate result; this continuity check
does not certify accuracy or runtime.

Fresh process, original forward bank, unchanged full tuning interval:

| `results/tune-stage3-recheck-fr-final` | Measured value |
|---|---|
| Output / independently validated rows | 2,708 / 2,708 |
| Fused full-pose updates / prediction-only rows | **2 / 2,706** |
| Lost rows / longest full-pose gap | 2,695 / 269.716 s |
| Frontend errors / backend faults | 0 / 0 |
| Original x/y RMSE | **5,838.073 m** |
| Startup / replay elapsed | 24.42 / 57.54 s |
| CSV/evaluation checks / visual-baseline gate | pass / **fail** |

All three unchanged scorer reports are present and hash-bound to the submitted CSV.
This is worse than the initial prediction-only position score: a few early measurements
learn nonzero velocity, then prolonged loss extrapolates it far beyond the track. Complete
row coverage and faster aggregate replay do not make this a usable localizer.

Fitting-only diagnostics isolated weak horizon-concentrated support, repeated-feature
ambiguity, and increasing map-to-track residuals. Lowering spatial gates admitted coherent
wrong poses; larger LK windows, CLAHE, source pruning and subpixel corner refinement did
not sustain the fitting trajectory. Those experimental algorithm changes are excluded
from production. A two-camera fitting bank was also built (1,486 references, 64,791
landmarks); adding the rear bank alone did not close the gate. A fitting-only rear gate
with four cells and at least 60 inliers still gave only 9 updates in 60 groups and
39.151 m RMSE; it is also excluded from production. Local failed experiments
remain under ignored `results/`, with diagnostic scripts under `tmp/`.

Ground-motion qualification also failed on fitting image pairs. A fitted asphalt-plane
offset was not identifiable with usable residuals, so no guessed height, offset or ground
correction is enabled. Border physical-association review and global retrieval remain open.
The remaining acceptance task is a sustained, correctly verified visual trajectory across
the complete tuning episode, followed by fresh scoring and independent validation.

Whole suite: **937 passed**, no skips/xfails, in 125.63 s. Command:
`.venv/Scripts/python.exe -B -m pytest --basetemp tmp/stage3-complete-recheck -q`.
Ruff passed for `camloc scripts tests run.py build_map.py validate_run.py`.
New regressions cover masked spatial gates, weak-PnP pool retention, current-pose LK
initialization, rear recovery, compatible refresh without double fusion, same-camera
disagreement, wide-road support and rejection of a baseline with only early visual updates.

## Reviewed changes

- C's `camloc/frontend.py` assembles A's canonical observations and B's current-image
  evidence. It observes each decoded image before tracking/handoff, applies the frozen
  bodywork mask, uses exposure-specific filter snapshots, chooses one in-gate pose, logs
  rejected/unused evidence, verifies candidates and proposes bounded scheduler jobs.
  Geometric shortlists cycle during unsupported startup; global EigenPlaces retrieval is
  still B's Step 5. It never reads GT, fits on query data or uses time to look up a pose.
- `run.py` now provides the raw-MCAP CLI using one initial pose and the existing
  filter/scheduler/recovery/projector. It checks fitting bounds, map/rig geometry, source
  identities and pinned backend compatibility before replay, preserves weak outputs and
  records code hashes, startup/replay elapsed time and failures separately.
- `validate_run.py` independently rebuilds A's complete camera-group sequence. It checks
  exact CSV format, finite values, yaw radians, increasing int64 exposure stamps, complete
  output sequence/episode coverage, log consistency and image-support causality/membership.
  When supplied an evaluation directory, it requires all three finite metric reports and
  binds them to the submitted CSV hash. Legal CSV validation and a successful visual
  baseline are reported separately.
- The builder now clamps GT interpolation support to the fitting interval, including
  boundary margins; caps each keyframe's actual undirected neighbor count at eight;
  rejects builds exceeding 2,000 references rather than silently truncating coverage;
  and checks fixed intrinsics as well as fixed poses.
- Map format **v2** binds the entire manifest (calibration, masks, models and provenance)
  and arrays to its identity. Shapes, finite geometry/descriptors and rigid poses are
  checked. Older v1 banks must be rebuilt; training-bag identity is provenance and is
  never required to equal a hidden query bag.
- B's tracker now checks observed source/preprocessing identity, clears incompatible
  candidate state on geometry changes, and enforces exactly-once settlement for rejected
  handoffs too. Invalid adoption does not consume a pending handoff; bounded pending
  overflow raises rather than silently forgetting an unsettled result. Ground motion
  rejects stale image pairs. PnP rejects malformed/non-finite point pools.
- Source-worker correspondences remain provisional: a source pose is never fused. Full
  PnP gates apply to the fresh image at handoff. An initially added source-pose gate was
  removed after review because it incorrectly prevented this correspondence handoff.
- Read-only map keypoints are copied into torch tensors before matching. Runtime weight
  downloads remain blocked and checkpoint hashes remain pinned.

Ground-motion corrections remain disabled pending real fitting-image qualification;
measured accepted motion is logged as unused. Borders retain A's diagnostic-only default
pending the manual physical-boundary review. Missing startup z/roll/pitch is never filled
from hidden GT. Rear fitting is now available locally; global retrieval is still absent.
The bank includes B's static bodywork mask, but no per-reference moving-vehicle masks.
Real road-branch/height initialization and the manual border-association review also
remain open; the runner does not invent these prerequisites to enable fallback updates.

## Initial integration results, 18 October 2026

Both fitting builds used the preserved local recording and unchanged split. The 100-image
proof build completed in 56.9 s. The larger corrected build used fitting windows [0,90)
and [483,603) seconds after the fitting start, with the default 4 m/5 degree selection.

| Larger fitting bank | Measured value |
|---|---|
| Local directory | `artifacts/fit-b-integrated-v2` (ignored) |
| Identity | `fitmap:8637237e723b6b8e` |
| References | 744 `fr` images; no rear bank |
| Matched pairs | 2,970; at most eight neighbors per reference |
| Retained landmarks | 32,634, each with at least three observations |
| Median landmark error | 1.387 canonical pixels |
| Fixed-pose maximum deviation | 8.88e-16 |
| Weak references | 64 have fewer than 30 retained landmarks |
| Build elapsed | 214.1 s on this workstation |

The first full tuning attempt is retained at `results/tune-b-integrated-first/`. It used
the overly strict source-pose gate, delivered zero anchors and fused no visual correction.
The corrected handoff attempt is retained at `results/tune-b-integrated-current/`.

| Corrected full tuning replay | Measured value |
|---|---|
| Completed groups / CSV rows | 2,708 |
| Input sequence / episode coverage | Exact group sequence; 100% nominal/organizer coverage |
| Frontend errors / backend faults | 0 / 0 |
| Dispatched / delivered anchor jobs | 1,084 / 17 |
| Handoff rejections | 8 too few inliers, 5 poor image coverage, 4 high median error |
| Fused visual updates | **0**; all 2,708 rows are prediction-only |
| Original x/y position RMSE | **436.929 m** |
| Lateral / longitudinal RMSE | 279.808 / 335.580 m |
| Yaw RMSE | 100.684 degrees |
| Startup / replay elapsed | 25.60 / 180.75 s |

Both original/updated x/y reports and the reconciled supplied-map report were produced by
the unchanged evaluator. The reconciled denominator remains 2,707 versus 2,708 original
evaluated rows; CSV rows were not cropped to alter that boundary behavior. All failures
and the first attempt remain on disk. Aggregate replay throughput is not paced/native
10 Hz qualification, a fast-path latency measurement or a localization accuracy result.

The initial zero-velocity prior falls behind the moving car; limited/clustered mapped
support then fails current-image gates, and the geometric shortlist cannot supply robust
recovery. **Next B acceptance task:** improve fitting coverage/current-image verified
support and demonstrate usable forward handoffs from this allowed initialization. Global
retrieval is the planned later recovery mechanism; it is not present or silently emulated.
Keep gates intact and repeat a fresh full tuning run after the visual fix. C's acceptance
gate closes only with genuine fused visual evidence and a complete scored trajectory.

## Reproduction

Use fresh output names; these commands refuse to overwrite existing artifacts/results.
The first command is an offline training tool. The second exports exactly one allowed
initial pose; only the third is inference. Cached setup weights and private input/reference
assets are required locally and remain excluded from Git.

```powershell
& .\.venv\Scripts\python.exe -B build_map.py --output artifacts/fit-b-repeat --window 0 90 --window 483 603
& .\.venv\Scripts\python.exe -B scripts/export_initial_pose.py --split tune --out tmp/tune-repeat-initial.json
& .\.venv\Scripts\python.exe -B run.py --artifacts artifacts/fit-b-repeat --initial-pose tmp/tune-repeat-initial.json --split tune --output results/tune-repeat
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py score results/tune-repeat/submission.csv --split tune --out results/tune-repeat/evaluation
& .\.venv\Scripts\python.exe -B validate_run.py --submission results/tune-repeat/submission.csv --diagnostics results/tune-repeat/diagnostics.jsonl --bag c0_2026-10-13_22-53-42_camloc_0.mcap --initial-pose tmp/tune-repeat-initial.json --split tune --evaluation results/tune-repeat/evaluation --out results/tune-repeat/validated.json
```

The initial combined suite passed 912 tests. After the integration fixes and the retired
placeholder-CLI assertion was updated, the final full suite passed **927 tests** in
154.41 s with no skips/xfails; Ruff and staged whitespace checks also passed.
Focused integration checks cover actual B
LK/PnP with C's scheduler/filter on rendered known scenes, blackout, current-image
handoff rejection, settlement/identity boundaries, future-image prefix independence,
fitting-only GT support, manifest tampering and submission/metric validation. These
synthetic checks establish software behavior, not real localization accuracy.

At the initial integration, holdout, global/rear recovery, road/border correction
qualification and paced workload gates had not run. The latest recheck above supersedes
that component status. No protected asset, evaluator source, dependency lock or split changed.

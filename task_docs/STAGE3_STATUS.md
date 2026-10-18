# Team C Step 3: B integration and the real baseline gate

B's [PR #1](https://github.com/sai0499/Camera-Only-Based-Localisation/pull/1), head
`b1202d8a27b7581fc6dac153215f10a822238963`, is integrated into `dev/implementation`.
It adds DISK/LightGlue, the fixed-pose fitting builder, frozen map loader, LK/PnP tracker,
anchor correspondence worker and ground motion. Its base was the current A+C commit
`22a2b4f`; none of A/C's existing files conflicted. `main` remains unchanged.

**Integration works; the Step 3 visual-baseline gate has failed and remains open.**
B's PR also explicitly disclosed this gate as unmet. Do not advertise the prediction-only
tuning CSV as camera localization or proceed to holdout/optional demo work on that basis.

## Latest recheck: not signed off

Small shareable receipt: [stage3_recheck.json](evidence/stage3_recheck.json). Generated
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

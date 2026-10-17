# Organizer update: meaning and build impact

Reviewed 17 October 2026. Package: `23-00-38_cut_120m_update`. Measured results are retained in `../evidence/organizer_update_audit.json`. This is a data/tool compatibility review, not a localization benchmark.

**Decision**

Continue building the agreed camera-localization system with the additional map as the canonical static map and target track-coordinate reference. The measured native-reference mismatch is now reconciled by the frozen fitting-only bridge in `../../evaluation/native_bridge.py`. The original native fields and the supplied map are both active in the default `reference_reconciled/` evaluation bundle. Keep both x/y reports and the reconciled supplied-map track report, with their distinct provenance.

The organizer announcement, supplied by the user, says: "Additional data and materials, including the evaluation script for Option B, are available here". Neither the supplied challenge PDF nor the update README defines the complete Options A/B contract, and the user has no further definition. The letter's formal meaning remains unknown. However, the toolkit's purpose is explicit: own-vehicle pose evaluation with x/y/yaw and/or track-relative coordinates. Do not infer that Option B is s/n-only or necessarily a separate localization task. The earlier separation is an evaluation/provenance precaution, not evidence of a different technical problem.

**What "ego localization tools" means**

"Ego" means the vehicle carrying the cameras: our own race car. Its pose is still `constructor0/base_link` in `map`. It does not mean detecting the locations of other cars, and it does not authorize reading the live ground-truth state. The package contains no image-based pose estimator; our planned algorithm must supply predictions.

| File | What it actually does | Role in our build |
|---|---|---|
| `map.mcap` | Stores physical borders and the two static reference routes | Shared geometry and coordinate reference |
| `track_frame.py` | Defines and converts map pose versus route progress/lateral offset | Reuse the coordinate implementation |
| `extract_gt.py` | Reads state estimation to export reference answers | Evaluation/training only; native coordinate compatibility must pass |
| `convert.py` | Converts an existing pose CSV; does not look at images | Adapt our predicted pose to another output representation |
| `evaluate.py` | Compares submitted poses with interpolated GT | Score our existing x/y/yaw output immediately; optional s/n reporting after compatibility checks |
| `README.md` | Documents the same coordinate formats and scoring workflow | Defines this toolkit's interfaces, though not the Option B label |

The clearest evidence is the evaluator's explicit support for `timestamp_ns,x,y[,yaw]` and `timestamp_ns,s,n[,epsi]`, or both. Thus the existing x/y/yaw localizer already supplies an accepted input format for this program. Coordinates s/n/epsi describe the same predicted car pose relative to a route, for example progress 1,200 m, offset 1.5 m to the left, and heading 3 degrees relative to the segment. A coordinate converter does not determine that pose from camera images.

On this second read-through, two synthetic comparisons (nonzero errors through yaw wrapping, and partial coverage) produced identical values for all nine compared x/y/yaw metric fields in the old and new evaluators. This establishes compatibility for those cases, not the organizers' ranking policy. Integrating the updated x/y report does not need to wait for the missing option definition. The native s/n reference mismatch remains a separate, demonstrated limitation.

**What actually arrived**

The folder contains six files: one 1,158,729-byte `map.mcap`, plus `tools/README.md` and four Python files (`track_frame.py`, `extract_gt.py`, `convert.py`, `evaluate.py`). The map contains nine static messages: four border Paths, a border MarkerArray, two reference Paths and their visualization MarkerArrays. It contains no camera frames or moving-car observations.

The README references a different recording, `rosbag_2026-10-12_23-00-38_cut_120m`, described as 35 seconds long. That recording is absent. The existing October 13, 1,194-second bag remains our actual driving dataset. Its seven exported videos and fitting/tuning/holdout time boundaries remain applicable; the folder name is not evidence that a new 120-minute dataset arrived.

**Verified static-map compatibility**

Every XYZ coordinate in all four supplied border marker arrays matches the corresponding original-bag marker coordinate exactly. The marker transforms are identity and their frame is `map`. The counts remain 5,914/6,115 main-track border points and 2,260/2,243 pit-border points. No coordinate-frame alignment or replacement of the camera geometry is indicated by these checks.

The new border Paths duplicate the markers, with one representation detail: main-track Paths omit the repeated closing point, giving 5,913/6,114 points; main-track markers include it. Pit-border Paths and markers are both open. Choose one representation for estimation to avoid counting duplicate map evidence.

The additional authoritative reference Paths are:

| Reference | Topic | Points | XY loop length |
|---|---|---:|---:|
| Racing line | `/constructor0/map_ar/racing_line` | 2,960 | 2,976.875276596 m |
| Whole lap via pit | `/constructor0/map_ar/pitlane_line` | 2,960 | 2,985.489335790 m |

Both form full loops, with the pit route detouring through the pit section. This does not make the physical pit borders closed. Their visualization markers have a different point count; use the Paths when reproducing the organizer's coordinates. The tiny final closing segment is ignored by the supplied helper's 1 mm segment threshold.

In these coordinates, s is distance along the XY polyline modulo its lap length, n is positive to the left, and epsi is car yaw minus the selected segment's heading, wrapped to +/-pi. These are coordinate definitions and static map priors, not measurements of where the current car is. Keep the estimator in x/y/yaw and derive optional coordinates from its own pose. Do not drive n toward zero or read the GT lane/progress into inference. Whether reference-line priors are allowed in the original task still needs the option/rules definition; evaluation and explanatory plotting can use them separately.

**Critical result: the native coordinate version does not match our bag**

I ran the update's unchanged GT extractor on the complete existing bag. It produced 126,574 state rows with the same first/last timestamps as the original split, no non-finite exported fields, and lane counts 13,121 pit / 113,453 track. The documented lane enum is now known: 0 pit, 1 track, 2 mixed, 3 undefined, 4 none. These remain GT labels, not permitted runtime inputs.

The extractor reconstructs each car pose from its recorded reference-segment index, segment distance and lateral offset using the supplied Paths. That compatibility check failed:

| Reference | Median reconstruction error, all rows | Maximum, all rows | Maximum while that route is active |
|---|---:|---:|---:|
| Racing line | 14.62 m | 37.40 m | 37.40 m on lane=1 |
| Pit route | 14.41 m | 58.36 m | 41.19 m on lane=0 |

These are **reference-coordinate reconstruction discrepancies**, not errors of a camera-localization algorithm and not proof that the x/y GT is inaccurate. The physical borders agree exactly while the recorded idx/ds/n values and the supplied reference Paths do not reproduce the car's x/y. The deeper diagnostic below establishes internally consistent native segment geometry incompatible with these Paths; the exact upstream file/version history is still unknown. This is not fixed by applying one global s offset: the discrepancy varies and includes n differences. Even a median s-offset correction leaves maximum reconstruction errors of 17.80 m / 32.00 m.

The README's reported sub-millimetre native-coordinate agreement relates to the material the organizers tested; it cannot be transferred to this October 13 bag. Do not use the extractor's native s/n columns as scoring truth against the supplied map for this recording.

The initial containment path recomputed GT s/n from x/y and later used global per-pose projection. That kept evaluation usable but did not reconcile the original native fields. The active bridge below supersedes that path and restores the organizer's sequence convention. The initial converter check observed maximum XY round-trip residuals of 0.467 m / 0.430 m because the inverse projection can lose an endpoint-aligned component. Preserve original x/y rather than round-tripping predictions through s/n.

**Risk resolution implemented**

The active implementation is in `../../evaluation/native_bridge.py`, `../../evaluation/reference_eval.py` and their tests; see `../../evaluation/README.md`. Prepared reconciled references are in `../../evaluation/reference_reconciled/`. Their manifest binds the source bag, unchanged supplied map, scorers, exact split, adapter, bridge, dependencies and generated CSVs. `reconciliation.json` in that bundle contains the active verification; `../evidence/reference_coordinate_resolution.json` retains the earlier diagnostic history.

The recording contains 2,929 native segment frames per route; the new reference Paths contain 2,960 poses. Every frame has at least 13 fitting observations. Fitting a unit rotation first reproduced positions to within 0.000001 m. Retaining the native arithmetic's tiny scale term improves this to a maximum 0.000000000000342 m across all 126,574 rows, including independent tuning and holdout periods. The recovered scale is within approximately 0.00000004 of one, and consecutive frames form a directed closed polyline. This establishes a usable version conversion, not camera-localization accuracy.

The active conversion is raw `(idx,ds,n)` -> frozen original segment frame -> shared map `(x,y)` -> supplied-map `(s,n,epsi)`. It reconstructs track and pit positions independently and requires them to agree; maximum disagreement is about 0.000000000000683 m. Reconstruction after calibration does not read the source x/y, which is covered by a full-recording regression test. The supplied map is never moved, resampled or aligned to predictions. Its exact start point, lap lengths, left-positive normal and heading convention define the final coordinates.

All reconciled state rows are finite. New-map s/n from the reconstructed native positions agrees with direct organizer-helper conversion within 0.000000000000455 m; epsi agrees exactly. The organizer's timestamped sequence projection is used. For predicted recovery jumps over 100 m the adapter explicitly restarts globally, and reports that restart; the actual GT needs no extra restarts. Direct conversion and the original TF/state agreement therefore support preserving the existing pose frame and camera calibration. The old source cache's `s` and `epsi` columns were synthesized using mismatched Path indices; the active bundle instead retains only original native idx/ds/n plus pose/lane fields in `source_state_native.csv`.

The wrapper retains schema, finite-value, timestamp, exposure, coverage, overlap and provenance checks. It keeps fixed episode GT and requires finite metric sections. Both x/y scorers receive identical TF GT and predictions. The updated evaluator additionally receives reconciled supplied-map GT and directly projected camera estimates, producing `reconciled_supplied_map.txt` and combined `submission_supplied_map.csv`. Bridge tests cover exclusion of non-fitting positions from calibration, negative ds, invalid/uncovered indices, scale preservation, cross-route disagreement, conversion without source x/y, recovery resets and every episode. GT fixtures establish plumbing only. The remaining administrative question is which reference/version defines the ranked Option B contract; the data mismatch itself is now mitigated without dropping the native fields or additional map.

**What the revised tools change**

The updated evaluator accepts x/y and/or racing-line s/n, with optional heading fields and optional pit-route coordinates. It interpolates GT with the same 50 ms maximum gap, unwraps s before interpolation, wraps ds around the lap, and reports biases as well as errors. Racing-line coordinates are scored over all matched poses, including pit driving; optional pit-route coordinates are selected only when both GT interpolation brackets have lane=0. Mixed entry/exit is excluded from that optional pit metric. Reference-line n error and car-heading lateral error are different metrics, especially off the line and in curves.

The CLI loads the map even for x/y-only files, so pass its path explicitly. `convert.py xy2sn` produces a coordinate-only CSV; it does not append x/y. `sn2xy` uses the racing line, not an automatic lane choice. Keep the primary four-column submission and create separate coordinate reports or an explicit timestamp-preserving merge for analysis. The original heading/output requirements remain the default until Option B is defined.

Ten synthetic scorer cases and coordinate self-checks passed. Single-row input and no overlap are handled more gracefully than in the root scorer. Remaining gaps include acceptance of duplicates, NaNs and 1 Hz submissions; a timestamp-only CSV can display coverage without any pose-error report. Preserve independent output validation and verify that the requested metric section exists. The source, map version and GT profile must accompany scores.

The coordinate helper searches +/-200 segment indices around its previous projection, restarting globally only after a time gap greater than one second. It is not a global localization algorithm. A synthetic large-jump check produced 174.51 m reconstruction residual with that local window and essentially zero with a global projection. After a confirmed recovery jump, the adapter must reset its search history and check a global association. Distinguish projection failure from estimator failure and do not force the estimate onto the line to conceal it.

**Concrete changes already made to the plan**

- `../FINAL_BUILD_PLAN.md` retains the six-state x/y filter, DISK/LightGlue, forward/rear camera strategy, border correction and 48-hour build gates. It adds static-map representation handling and an optional track-coordinate adapter.
- `../build_split.json` retains every exact time boundary and the new map path/hash. Its failed compatibility note concerns direct cross-version index reuse; the active reconciled manifest separately records `native_coordinate_bridge.enabled: true`.
- The root TF-based x/y scoring profile remains the continuity reference. The updated x/y report runs alongside it on the same files, and native fields converted into the supplied map provide the track-coordinate report. All profiles have explicit provenance.
- A uses the additional map's identical surveyed borders and exact reference Paths. C reuses the working bridge and evaluator and connects the camera system's output. B's landmark/tracking implementation does not need redesign. No new visual model is needed for this integration.
- The risk review now distinguishes original versus updated scorer behavior, documents the lane enum and adds the newly exposed compatibility question. The dataset audit links this review.

The new package does not resolve frozen GT-anchored visual-map permission, camera preprocessing provenance, runtime hardware/aggregation, the undocumented quality field, or final submission logistics. It also does not establish that all additional Option B inputs may be used for the original task.

**Questions to send to organizers**

1. What is the complete Option B task, and are teams choosing between options or submitting both? Which map priors and output fields are allowed for each?
2. Can they provide the October 12 35-second bag referenced by the update README?
3. Which reference/version defines ranked Option B? We reconciled the October 13 recording's 2,929 native segment frames into the new 2,960-pose Paths using frozen fitting-only geometry. Independent position reconstruction and direct supplied-map conversion agree within about 0.000000000001 m. Is this supplied-map regeneration the intended scoring policy?
4. Which scorer/output/ranking profile should our selected camera-localization entry use?

These questions are prepared for the team; no message has been sent. The missing option definition and reference mismatch do not prevent implementing the original x/y system.

Both organizer tool directories, their READMEs, maps and original challenge materials are preserved. The original `evaluation/reference/` cache and active `evaluation/reference_reconciled/` bundle are retained. A later setup pass replaced `.venv-eval` with the shared root `.venv`; see `../../START_HERE.md`. No new driving clip, localization model, algorithm accuracy, hidden-test result or localization runtime was evaluated.

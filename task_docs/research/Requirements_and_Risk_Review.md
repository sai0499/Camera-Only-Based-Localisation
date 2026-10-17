# Requirements and risk review

Second review, 17 October 2026. This document distinguishes verified facts, conservative implementation decisions, and questions that require the organizers' answer. It supersedes stronger assumptions in the first design wherever noted. A document review can resolve interface mistakes and identify rule conflicts; it cannot establish localization accuracy or eliminate failures under unseen conditions.

Final research addendum: `FINAL_RESEARCH_REVIEW.md` records the primary-source comparison. Use `../FINAL_BUILD_PLAN.md` for the revised bounded matching worker, current-image handoff, EigenPlaces retrieval, one-pass remap, rate-consistent filtering and grouping. Its section 10 assigns every known condition a fixed action, owner and acceptance gate. Earlier questions below remain a historical record; the build no longer waits for an unspecified model choice or risk decision. Empirical performance and unpublished organizer policy remain explicitly unverified.

Organizer-update addendum: `ORGANIZER_UPDATE_REVIEW.md` and the revised `../FINAL_BUILD_PLAN.md` cover `23-00-38_cut_120m_update`. The organizer announcement identifies its evaluator as Option B, whose complete task definition is absent from the local PDF and new README. Its physical borders match the old recording exactly, but native s/n from the old recording fails reconstruction against the new reference Paths. The new scorer supports coordinate formats; it does not establish a replacement of the original task or resolve every competition rule. Historical root-scorer findings below remain version-specific.

Second read-through: "ego localization tools" explicitly concerns the same own-vehicle pose problem and accepts x/y/yaw as well as track coordinates. The missing Option B label definition is not a blocker to running its x/y evaluator alongside the original one. Two synthetic parity checks matched all nine compared metric fields. Do not interpret the earlier separation of scoring profiles as evidence that Option B necessarily requires a different algorithm or s/n-only output.

Implemented reconciliation: `../../evaluation/README.md` documents the active native-coordinate bridge and `reference_reconciled/` bundle. The old recording's 2,929 native segment frames are recovered using fitting data only, including the tiny near-unit scale term. Frozen frames then convert raw idx/ds/n into map positions, which the new 2,960-pose Paths express as s/n/epsi. Both native routes agree within about 0.000000000001 m; independent positions and direct supplied-map conversion agree at the same numerical scale. Native fields and the additional map are actively used. The organizer sequence convention is preserved, with an explicit global restart after a large recovery jump. Both x/y scorers still use identical TF GT; schema/finite/timestamp/exposure/coverage/provenance checks remain. The engineering mismatch is reconciled; the remaining organizer choice is which reference/version defines ranked Option B results. Evidence is in `../../evaluation/reference_reconciled/reconciliation.json` and its manifest; `../evidence/reference_coordinate_resolution.json` retains the earlier diagnostic history.

**Sources and scope**

The five-page challenge PDF was re-extracted and its requirements/rubric pages visually reread. The README, metadata, original exporter/scorer, and embedded ROS message schemas were rechecked. Small full-recording and synthetic scorer result summaries are retained under `../evidence/`. Temporary audit inputs, scripts and package caches were removed after consolidation. Original organizer files and the MCAP were not edited. Use `../FINAL_BUILD_PLAN.md` for current implementation decisions.

The PDF links a separate INTRO PDF through SharePoint. That document is not in this workspace, and the linked URL was inaccessible through the available web reader. Its contents therefore remain unreviewed. The README also lists `track_bounds_check.png`, which is absent; a diagnostic map plot was generated in the first review. Neither omission establishes that a rule has changed.

**Full recording checks now completed**

| Check | Measured result | What this establishes |
|---|---|---|
| Sequential message scan | 345,462 messages; every topic count matches metadata | Complete traversal of this file |
| Full JPEG decoding | All 83,603 decoded; RGB, 1506 x 728; no decode failures | Readability and dimensions, not visual usefulness |
| Exposure header extraction | Fast CDR extraction agrees with the ROS decoder for every image | The audited timestamp reader is consistent |
| MCAP chunk checksums | All 905 stored, nonzero chunk CRCs passed | No chunk corruption detected by the stored checksums |
| MCAP summary checksum | Stored and calculated CRC both 2234336267 | Summary checksum passed |
| MCAP data-section checksum | Stored value is zero | This checksum is absent; do not claim it passed |
| Compression | All 905 chunks use zstd | Blank rosbag compression fields in metadata do not imply uncompressed MCAP chunks |
| Static transforms and borders | Each has 120 messages and one content version, ignoring repeated timestamps | No change within this recording |
| Camera calibration | Each camera has 1,195 messages and one content version, including K/D/R/P, ROI and binning | No calibration changes within this recording |
| State versus dynamic TF | 126,568 exact timestamp matches; identical XYZ and yaw within numerical precision | They share a reference pose, not two independent accuracy measurements |

There are 126,574 state messages and 126,680 dynamic TF poses. Six state timestamps have no exact TF match. Join by timestamps, never array position. All image exposure stamps precede or equal their log times. Across all images, median exposure-to-publication delay is 41.895 ms and median publication-to-log delay is 0.120 ms. Most of the observed approximately 42 ms offset precedes publication; it is not simply disk-recording latency. Use the supplied exposure time, without adding or subtracting a learned average offset.

Evidence: `../evidence/dataset_recheck.json`, including aggregate image timing. Forty-two images were visually reviewed; successful software decoding of all frames is not a visual review of every frame. Per-image temporary arrays and audit scripts were removed during cleanup.

**Two newly identified geometry and reference-quality risks**

The supplied K and the left 3 x 3 block of P differ for every camera. For example, `fr` has K focal length fx = 1349.904, while P has fx = 1174.755; `cl` has 796.970 versus 560.736. The ROS message definition assigns K to raw distorted images and P to processed rectified images. R is identity here, but that does not make the distortion D zero. The README specifies K/D for these images, so the default is the raw K/D model. If images are deliberately undistorted, use the resulting new intrinsics and zero distortion consistently. Do not mix P with the original D or undistort twice. The JPEG format string alone does not establish image preprocessing; verify known-pose overlays and ask the producer if provenance is unclear. [Official ROS CameraInfo definition](https://raw.githubusercontent.com/ros2/common_interfaces/rolling/sensor_msgs/msg/CameraInfo.msg).

The undocumented `pos_accuracy` array changes substantially near the start of the recording. In 1,228 state messages between 43.239 and 54.821 s after the first GT timestamp, at least one component exceeds the diagnostic threshold 10; component maxima over the run are approximately 148.25, 323.80 and 444.34. The embedded schema gives no units or statistical meaning for this field. These values are **not established position errors, standard deviations, or metre-valued uncertainties**. They identify a question for the dataset producer, not proof that the reference trajectory is wrong. The README's roughly 0.1 m / 0.2 degree accuracy statement is not a per-sample guarantee.

Ask what `pos_accuracy` and status values mean, whether this interval has known reference-quality limitations, and whether any official evaluation exclusions exist. Until clarified, retain the official scoring interval and report all errors. Do not remove difficult evaluation frames based on this diagnostic. For fitting calibration/maps, avoid treating the advertised accuracy as a uniform noise constant; inspect sensitivity to this early interval using fitting data only. These fields remain prohibited inference inputs under the conservative policy. Evidence: `../evidence/reference_field_diagnostic.json`.

**Confirmed competition contract**

| Item | Confirmed interpretation | Source |
|---|---|---|
| Target | Pose of `constructor0/base_link` in `map`, x/y metres and yaw radians | README Task/Submission |
| Primary CSV header | Keep `timestamp_ns,x,y,yaw`; updated tooling also accepts optional track-coordinate formats, without a new statement waiving heading/submission requirements | Original README/brief and updated tools README |
| Measurement time | Image exposure `header.stamp`; use integer nanoseconds | README Data/Submission |
| Causality | Estimate at t may not use image data with exposure after t; ranked results cannot be offline-smoothed | PDF pp. 1-2, README |
| Initialization | One first pose is allowed; later test GT poses are not | PDF pp. 1-2, README |
| Output | 10 Hz expected; at least 90% coverage of one-second intervals is the ranking floor | PDF pp. 2-3, README |
| Ground-truth use | Training and validation are allowed; test-time pose access is prohibited except initialization | PDF pp. 1-2, README |
| Pretrained models | Allowed, with declarations of models and public external datasets | PDF p. 2 |
| Source | Mandatory source, dependencies, weights or download location, README, and one reproducible run script | PDF pp. 2-3 |
| Hardware | Must run on a single-GPU workstation; actual benchmark GPU/model and OS are unspecified | PDF p. 2 |
| Presentation | At most five minutes and six slides; results, runtime, failure cases, and visualization required | PDF p. 3 |
| Practical accuracy | Sub-metre lateral accuracy is context, not a guaranteed winning threshold | PDF p. 3 |

Seven available cameras does not explicitly require running seven expensive feature extractors on every frame. Using a selected subset is an implementation choice; the system should still accept the complete supplied input and preserve timing. Camera count does not make the rig a conventional synchronized, rectified stereo pair.

The PDF assigns 25/20/10/10/5 percentage points to position/lateral/heading/under-one-metre fraction/runtime, and 15/10/5 to technical merit/engineering/presentation. It says metrics are normalized across teams but gives no full normalization equation. The local scorer prints position RMSE as the score and does not implement that weighted ranking. There is no defensible local scalar predicting the final ranking without clarification.

**The car-state contradiction is more consequential than it first appeared**

The PDF p. 2 both limits inputs to cameras/calibration/borders and says that speed, acceleration, and other car-state fields can be used. The README treats the entire `/constructor0/state_estimation` topic as ground truth and excludes it from inference.

Its actual embedded schema contains direct x/y/z, roll/pitch/yaw, velocities and accelerations, but also:

- `sn_map_state.current_lane`;
- `track_sn_state` and `pitlane_sn_state`, with reference-curve index/progress, lateral offset `n`, heading error `epsi`, and inside-border indicators;
- additional heading-related fields including `yaw_vel_rad`;
- estimated accuracy/status and other vehicle-state fields.

Removing only `x_m`, `y_m`, and `yaw_rad` does not create a safe camera-only input. Progress/lateral offset can disclose location through a different representation. The implementation default is to deny the entire topic, including nested fields, and deny dynamic `/tf` after the one-pose initialization adapter. Preserve only camera extrinsics from `/tf_static`; the presence of lidar/IMU/GNSS mounting transforms is not the presence of those sensors' measurements or permission to use state estimates derived from them.

If organizers permit a car-state variant, obtain an exact field whitelist and confirm those fields exist in the hidden run. Keep its results explicitly separate from the strict camera-only variant.

**Implementation assumptions tightened by this review**

1. **The frozen landmark map is a rule interpretation awaiting confirmation.** Allowing pose-supervised training strongly suggests that training-derived visual artifacts can be used, but the brief does not explicitly discuss a GT-anchored sparse 3D map or reference-image database at inference. The first design stated this too categorically. Ask specifically about that artifact. Keep a border-plus-causal-visual-tracking path independent of it. Do not make a legality claim from general library practice.
2. **Do not require a six-degree-of-freedom initial pose.** The brief promises timestamp/x/y/yaw. The training bag happens to contain z/roll/pitch, but their availability in the hidden initialization is not promised. Initialize remaining geometry conservatively from supplied map/calibration and visual estimation. The precise initial-pose file/topic interface is unspecified.
3. **Map elevation is data, not a certified dense surface.** Preserve XYZ border points, but the existence of z values does not certify interpolated banking, occlusion, or camera height. Local road-surface estimates need a geometry check and uncertainty. Do not freely optimize calibration and road shape to make an incorrect pose fit.
4. **Physical borders and reference loops have different topology.** Main-track border markers close exactly; pit-border polylines are open, with endpoints over 630 m apart. The update separately supplies closed racing/pit-route reference loops, the latter covering a whole lap through the pit. Do not close physical pit borders or conflate the two reference coordinates. Main border Paths omit the closing duplicate found in markers.
5. **Repeated calibration does not prove hidden-run calibration identity.** Use the calibration carried by each run. Key any learned correction and sparse map by an explicit rig/calibration identifier. A changed extrinsic invalidates geometry built with the old extrinsic.
6. **Group timing must be reproducible.** Exposure-stamp grouping and arrival deadlines are different quantities. Declare a bounded policy; in offline replay use recorded arrival times or another deterministic scheduler rather than CPU-speed-dependent wall-clock grouping. Record the maximum source exposure used for every pose. The chosen image-set output stamp should be confirmed with organizers.
7. **Frame dropout and sensor disagreement must not stop output.** Missing cameras should degrade evidence without deadlocking the reader. Repeated stale frames must not repeatedly reduce uncertainty.
8. **Coverage and confidence are separate.** Produce the expected stream and retain low-confidence predictions in the main score. A confidence filter must not hide difficult intervals, and a successful coverage number is not proof of useful accuracy.
9. **No global trajectory alignment.** The supplied scorer uses predictions directly in `map`; it does not remove translation, rotation, or scale error. Do not report aligned SLAM ATE as the official score.
10. **No fixed lap-time or speed template.** Initial pose does not authorize replaying the training trajectory at a learned clock. Hidden speed and appearance may differ. Training speed extrema are diagnostic, not hard motion limits.

**Legacy root-scorer behavior reproduced with synthetic files**

Eight isolated synthetic cases were run through the unchanged `tools/evaluate.py`; results are retained in `../evidence/scorer_behavior.json`. They are not localization results. Temporary case CSVs were removed during cleanup.

| Case | Observed behavior | Required separate check |
|---|---|---|
| Finite, valid 10 Hz trajectory | Correct zero-error reference case | Retain as scorer plumbing check |
| One submitted row | Raises an indexing exception | Reject unsupported/insufficient output clearly |
| Duplicate rows | Accepted and counted as multiple evaluated poses | Require unique timestamps and one intended pose per output event |
| NaN positions at valid timestamps | Reports 100% coverage while error becomes NaN | Reject all non-finite position/heading values |
| Constant one-metre translation error | Reports 1 m position RMSE | Do not align the estimate to GT after inference |
| Only one pose per second | Reports 100% coverage | Check expected 10 Hz output separately |
| Reversed row order | Sorted and scored normally | Enforce monotonic output for causal replay and reproducibility |
| No timestamps overlapping GT | Raises an empty-array exception | Check overlap and evaluated-row count before reporting metrics |

Further facts established by code inspection: the scorer skips the exact first GT timestamp because it requires a preceding interpolation sample; it accepts the final exact GT timestamp if the bracket is valid; coverage is computed from evaluated timestamps over bins anchored at the first supplied GT time; and `within 1.0 m` means strictly less than one metre. It does not check that a submitted timestamp is an actual image/set timestamp, whether input topics were legal, or whether future images influenced a prediction. Those checks belong around the estimator, not in a modified organizer scorer.

The updated scorer was checked separately with ten synthetic cases. Single-row submissions now work, and no-overlap submissions return without an exception. Duplicate rows, NaNs, and 1 Hz coverage remain accepted; timestamp-only files can report coverage with no pose metrics. Its CLI always loads the map even for x/y scoring. Keep independent validation and version the scorer/GT source. The update documents lane labels 0 pit, 1 track, 2 mixed, 3 undefined, 4 none; optional pit-coordinate scoring selects only pairs of lane=0 GT brackets. The label meanings are now documented, but lane remains GT rather than an inference input.

**Evaluation split refinement**

The initial review used closest approaches to a reference point. This review checked directed crossings of a line through map XY `(81.1478, 363.2207)`, normal to local travel, with a lateral gate and compatible heading. Crossings relative to first GT stamp `1791918157114146050` are approximately:

`90.002711, 319.015864, 483.455241, 603.497522, 702.390798, 860.237619, 975.188391, 1083.634942` seconds.

These support repeatable local split boundaries; they are not official lap identifiers. The first passage is laterally displaced by about seven metres from later passages, so initial/pit transitions must not be treated as identical route geometry. Gate-to-gate interval lengths range from about 98.9 to 229.0 s. Never assume constant lap duration.

Use a fitting cutoff near 700.390798 s, tuning start at 702.390798 s, tuning end near 973.188391 s, and holdout start at 975.188391 s through the end. The two-second guards separate adjacent frames. Save exact integer nanosecond boundaries in the manifest rather than reconstructing them from rounded text. For an image-based episode, choose the first output exposure at or after the boundary and provide its allowed initial pose through the evaluation harness. Keep the GT evaluation window predetermined and include the beginning/end of that episode honestly.

No model fitting or algorithm accuracy evaluation has occurred. General data inspection includes the whole recording, so the held-out interval is reserved from fitting/tuning, not described as completely unseen during inventory.

**Questions requiring an organizer answer**

| Priority | Precise question | Interim decision |
|---|---|---|
| Critical | What is the complete Option B task, are the options alternatives or jointly required, and are its reference lines permitted inference priors for the original camera-localization task? | Keep the original x/y task as the build priority; separate Option B analysis from its ranked pipeline |
| Critical | Which reference/version defines ranked Option B? Direct old-index reuse gives 37.40 m / 58.36 m errors; our frozen fitting-only bridge reconciles the native fields into the new map with about 0.000000000001 m numerical agreement. Is this supplied-map coordinate regeneration the intended policy? | Use the active reconciled supplied-map profile and retain both x/y reports; disclose the reconciliation provenance |
| High | Can you provide the October 12 35-second driving bag referenced in the new tools README? Only map.mcap and tools are present in this update folder. | Preserve the original recording and split; do not infer a new data duration from the folder name |
| Critical | Which exact car-state fields, if any, are legal at inference, and are they supplied in the hidden run? Are nested track-progress/lateral-offset fields excluded? | Deny the whole state-estimation topic |
| Critical | May we build a frozen visual keyframe/3D landmark map using images and GT poses from training laps and use that artifact during test inference? | Keep this extension conditional; maintain a camera/map-only baseline without a prebuilt GT-anchored landmark bank |
| High | How is the first pose supplied, what timestamp/fields are included, and what should happen to exposures before that timestamp? | Require only timestamp/x/y/yaw; do not backfill with a later initial pose |
| High | Is the PDF's weighted rubric authoritative, and what are the normalization and lap-aggregation formulas? | Report every metric separately |
| High | Is a frame the complete 10 Hz camera set or one individual camera image for runtime scoring? What costs and hardware are used? | Measure a complete update including decoding/transfers/matching/fusion/writing and disclose startup separately |
| High | Is the maximum exposure stamp of the used image set an accepted output timestamp, or is a designated reference camera required? | Preserve all support stamps; never use data later than the submitted stamp |
| High | What are the units/meaning of `pos_accuracy` and state-status codes? Are there known reference-quality issues around 43-55 s, and any official exclusion policy? | Do not infer error magnitudes or remove scored samples; diagnose fitting sensitivity separately |
| Practical | Are the JPEGs raw distorted images corresponding to K/D, as the README indicates, or already processed using P? | Use K/D initially, verify overlays, and keep one consistent pixel model |
| Practical | Will the hidden run retain this rig, resolution, optical-frame conventions, map layout and coordinate frame? Will its own calibration/map be included? | Read and validate each run's provided assets; detect incompatible learned artifacts |
| Practical | Can you provide the linked INTRO PDF and the exact submission deadline/timezone, upload method, and artifact-access requirements? | No assumed exact deadline or unreviewed rule changes |

These questions are prepared for the team; no message has been sent to organizers. The supplied PDF's work dates do not specify an exact submission clock time. The final deadline must not be inferred from a local time zone or this review's timestamp.

**Acceptance evidence still required before claiming robustness**

The raw-data and scorer checks remove several avoidable engineering unknowns. Remaining algorithm gates are fitting-only map/geometry verification, independent tuning/holdout metrics, a causal-prefix test, a prohibited-topic independence test, a complete input-to-output runtime measurement, and sustained recovery tests. Hidden lighting/speed generalization, calibration accuracy, and periods with no usable visual evidence remain empirical risks. No review can honestly set those risks to zero.

# Requirements Document

## Introduction

This document derives requirements from the approved shared-contracts design (teammate C, step 1). The feature freezes the shared records, conventions and call boundaries of the TrackCode Apex camera-only localizer in `camloc/contracts.py`, adds the single CSV writer `camloc/submission.py` and the offline initial-pose harness `scripts/export_initial_pose.py`, and tests them. It estimates no pose, changes no dependency and leaves all protected assets unchanged.

## Glossary

- **Contracts_Module**: `camloc/contracts.py`, holding conventions, constants, enums, exceptions, validation helpers, records, `InitialPose`/`EpisodeBounds`, JSON/split functions and Protocols.
- **Submission_Writer**: `camloc/submission.py`, holding `format_row` and `SubmissionWriter`.
- **Initial_Pose_Exporter**: `scripts/export_initial_pose.py`, the offline harness that selects one initial pose from a GT CSV.
- **Split_Loader**: `load_split` and `episodes_from_split` in the Contracts_Module.
- **Isolation_Test**: `tests/test_inference_isolation.py`.
- **Test_Suite**: all tests run by `.venv/Scripts/python.exe -B -m pytest` from the project root.
- **Shared_Contracts_Feature**: the complete set of step-1 files and changes described by the design.
- **Record**: a frozen dataclass defined by the Contracts_Module.
- **Evidence_Record**: `PoseEvidence`, `HandoffEvidence`, `MotionEvidence` or `BorderEvidence`.
- **Geometry_Record**: `RoadPatch`, `AuxiliaryGeometry` or `TrackProjection`.
- **Nanosecond_Value**: a Python `int` in `[0, 2**63 - 1]`.
- **Wrapped_Angle**: a float in the half-open interval (-pi, pi].
- **Rigid_Transform**: a 4 x 4 float64 matrix with orthonormal rotation (tolerance 1e-6), determinant +1 (tolerance 1e-6) and last row [0, 0, 0, 1] (tolerance 1e-9); `T_AB` maps coordinates from frame B into frame A.
- **Group_Time**: `FrameSet.t_ns`, the maximum exposure of the group's images and the CSV timestamp.
- **Source_Exposure**: the exposure of an anchor job's source image.
- **Measurement_Exposure**: the exposure of the current same-camera image on which a pose is solved.
- **Snake_Case_Reason**: a string matching `^[a-z][a-z0-9_]*$` in full.
- **GT**: ground-truth pose data used only by evaluation and the offline harness.
- **Owner_A / Owner_B / Owner_C**: teammates owning geometry/input, vision/map and integration respectively.

## Requirements

### Requirement 1: Integer nanosecond time

**User Story:** As Owner_C, I want one exact time type, so that no timestamp loses precision or silently mixes units.

#### Acceptance Criteria

1. WHEN a value of type `int` (excluding `bool`) or `np.integer` within `[0, 2**63 - 1]` is passed to `as_ns`, THE Contracts_Module SHALL return a Python `int` equal to the value.
2. IF a value of type `bool`, `np.bool_`, any float type (including integral floats such as `1.0`), `str` or any other type is passed to `as_ns`, THEN THE Contracts_Module SHALL raise `ContractError` naming the field.
3. IF a value outside `[0, 2**63 - 1]` is passed to `as_ns`, THEN THE Contracts_Module SHALL raise `ContractError`.
4. WHEN a string fully matching `[0-9]+` within int64 range is passed to `parse_decimal_ns`, THE Contracts_Module SHALL return its integer value.
5. IF a string with a sign, whitespace, trailing newline, underscore, decimal point, exponent or non-ASCII digit, or a non-string, is passed to `parse_decimal_ns`, THEN THE Contracts_Module SHALL raise `ContractError`.
6. THE Contracts_Module SHALL store every timestamp and duration field of every Record as a Nanosecond_Value.

### Requirement 2: Angle convention

**User Story:** As Owner_A and Owner_B, I want one angle convention, so that yaw values compare and serialize identically across modules.

#### Acceptance Criteria

1. WHEN a finite real angle is passed to `wrap_angle`, THE Contracts_Module SHALL return a Wrapped_Angle congruent to the input modulo 2 pi.
2. WHEN an angle already within (-pi, pi] is passed to `wrap_angle`, THE Contracts_Module SHALL return the identical float value.
3. IF a non-finite, `bool` or non-real angle is passed to `wrap_angle`, THEN THE Contracts_Module SHALL raise `ContractError`.
4. WHEN a Record with an angle field is constructed, THE Contracts_Module SHALL store that angle as returned by `wrap_angle`.
5. THE Contracts_Module SHALL return angles that differ from the organizer `track_frame.wrap` result by at most 1e-12 rad modulo 2 pi for inputs with absolute value up to 1e3.

### Requirement 3: Transforms and covariances

**User Story:** As Owner_B, I want transform and covariance checks shared by all records, so that a wrong-direction or non-rigid matrix is caught where it is created.

#### Acceptance Criteria

1. WHEN a finite real 4 x 4 array satisfying the Rigid_Transform tolerances is passed to `check_rigid`, THE Contracts_Module SHALL return a read-only float64 copy.
2. IF an array with another shape, a non-finite entry, a non-orthonormal or reflected rotation block, or a last row outside tolerance is passed to `check_rigid`, THEN THE Contracts_Module SHALL raise `ContractError`.
3. WHEN a finite n x n array that is symmetric within relative tolerance 1e-9 and has minimum eigenvalue at least -1e-9 relative is passed to `check_covariance`, THE Contracts_Module SHALL return a read-only float64 copy.
4. IF a covariance has another shape, a non-finite entry, an asymmetry beyond tolerance or a negative eigenvalue beyond tolerance, THEN THE Contracts_Module SHALL raise `ContractError`.
5. WHERE a covariance is declared definite (`MotionEvidence.covariance`, `BorderEvidence.R`), THE Contracts_Module SHALL require every eigenvalue to be greater than zero.
6. THE Contracts_Module SHALL document in its module docstring that `T_AB` maps coordinates from B into A, that `T_WB = inv(T_CW) @ inv(T_BC)`, that quaternions are stored xyzw and that planar yaw is `atan2(T_WB[1,0], T_WB[0,0])`.

### Requirement 4: Image messages, calibration and canonical observations

**User Story:** As Owner_A, I want exact image and calibration records, so that Owner_B receives one canonical pixel convention.

#### Acceptance Criteria

1. WHEN an `ImageMessage` is constructed, THE Contracts_Module SHALL require a camera id in `CAMERA_IDS`, Nanosecond_Value exposure and arrival stamps, non-empty identity strings and non-empty `bytes` data.
2. WHEN an `Observation` is constructed, THE Contracts_Module SHALL require a C-contiguous `uint8` image of shape (464, 960, 3) and a `bool` valid mask of shape (464, 960), checking only dtype, shape and flags.
3. WHEN an `Observation` is constructed, THE Contracts_Module SHALL mark the image and the mask read-only without copying them.
4. WHEN an `Observation` or `CameraCalibration` is constructed, THE Contracts_Module SHALL require each K to have positive focal lengths, zero skew and last row [0, 0, 1], and `T_BC` to be a Rigid_Transform.
5. WHEN an `Observation` is constructed, THE Contracts_Module SHALL require the canonical principal point to lie within [-0.5, 959.5] x [-0.5, 463.5].
6. WHEN a `CameraCalibration` is constructed, THE Contracts_Module SHALL require a float64 `D_raw` of shape (5,) and positive raw width and height.
7. IF an image or mask has another dtype, shape or memory layout, THEN THE Contracts_Module SHALL raise `ContractError`.

### Requirement 5: Header-only frame sets

**User Story:** As Owner_C, I want grouped image sets with exact identity, so that output timestamps stay identical whichever frontend decodes the images.

#### Acceptance Criteria

1. WHEN a `FrameSet` is constructed, THE Contracts_Module SHALL require a non-empty unique `selected_camera_ids`, at least one image, each camera at most once and images ordered as selected.
2. WHEN a `FrameSet` is constructed, THE Contracts_Module SHALL require `t_ns` to equal the maximum image exposure.
3. WHEN a `FrameSet` is constructed, THE Contracts_Module SHALL require `missing_camera_ids` to equal the selected cameras without an image, in selected order.
4. WHEN a `FrameSet` is constructed, THE Contracts_Module SHALL require closure `complete` exactly when no camera is missing.
5. WHEN a `FrameSet` is constructed, THE Contracts_Module SHALL require `terminal_partial` to be True exactly when closure is `eof`.
6. IF the image exposures of a `FrameSet` span more than 5 ms or its arrivals span more than 40 ms, THEN THE Contracts_Module SHALL raise `ContractError`.
7. WHEN a `FrameSet` carries input events, THE Contracts_Module SHALL require them in non-decreasing `log_ns` order.

### Requirement 6: Accepted and rejected semantics

**User Story:** As Owner_C, I want weak evidence to remain a logged record, so that no rejected evidence is consumed silently.

#### Acceptance Criteria

1. THE Contracts_Module SHALL give every Evidence_Record an `accepted` flag and every Geometry_Record a `valid` flag, each with a `reason`.
2. WHEN a Record with `accepted` or `valid` set to True is constructed, THE Contracts_Module SHALL require the reason `accepted` or `valid` respectively.
3. WHEN a Record with `accepted` or `valid` set to False is constructed, THE Contracts_Module SHALL require a Snake_Case_Reason other than `accepted` and `valid`.
4. WHEN `require_accepted` receives an accepted or valid Record, THE Contracts_Module SHALL return that Record unchanged.
5. IF `require_accepted` receives a rejected or invalid Record, THEN THE Contracts_Module SHALL raise `RejectedEvidenceError` containing the Record type and reason.
6. THE Contracts_Module SHALL enforce only structural rules and leave tuning thresholds (inlier gates, innovation limits, noise floors) to the owners' modules.

### Requirement 7: Correspondence and pose identity

**User Story:** As Owner_B, I want unique keypoint and landmark identities in every record, so that duplicated matches cannot inflate PnP support.

#### Acceptance Criteria

1. WHEN `Correspondences` are constructed, THE Contracts_Module SHALL require equal-length unique non-negative int64 keypoint ids, unique non-negative int64 landmark ids and float32 (N, 2) pixels inside [-0.5, 959.5] x [-0.5, 463.5], with N at most 1024.
2. IF any keypoint id or landmark id repeats within one record, THEN THE Contracts_Module SHALL raise `ContractError`.
3. WHEN a `PoseEvidence` is constructed, THE Contracts_Module SHALL require `inlier_count` to equal the lengths of its unique `track_ids` and unique `landmark_ids`.
4. WHEN an accepted `PoseEvidence` is constructed, THE Contracts_Module SHALL require a Rigid_Transform `T_WB`, non-negative error statistics with median not above p90, `inlier_count` of at least 1 and `occupied_cells` of at least 1.
5. WHEN a `PoseEvidence` is constructed, THE Contracts_Module SHALL require `occupied_cells` in 0..16, `inlier_fraction` in [0, 1], and `inlier_count` at most 512 for source `tracking` and at most 256 for sources `handoff` and `candidate`.

### Requirement 8: Anchor requests and results

**User Story:** As Owner_C, I want anchor jobs to carry frozen inputs and return only correspondences, so that a stale worker pose can never be fused.

#### Acceptance Criteria

1. WHEN an `AnchorRequest` is constructed, THE Contracts_Module SHALL require `delivery_not_before_ns` to be at least `source_exposure_ns + 90_000_000`.
2. WHEN an `AnchorRequest` is constructed, THE Contracts_Module SHALL require the embedded `Observation` to match its camera id, Source_Exposure and preprocessing id, and the embedded `SearchSnapshot` time to lie within 0 to 5 ms after the Source_Exposure.
3. WHEN an `AnchorRequest` is constructed, THE Contracts_Module SHALL require at most two unique non-negative reference ids, purpose `front_anchor` only on camera `fr` and purpose `rear_anchor` only on camera `r`.
4. THE Contracts_Module SHALL define `AnchorResult` without any pose, transform, position or yaw field.
5. WHEN an `AnchorResult` with status `ok` is constructed, THE Contracts_Module SHALL require reason `ok` and source-image `Correspondences` with at least one row.
6. WHEN an `AnchorResult` is constructed, THE Contracts_Module SHALL require at most two unique matched reference ids and at most ten unique remaining candidate ids disjoint from the matched ids.
7. IF `AnchorResult.check_against` receives a request with a different job id, purpose, camera, source group, Source_Exposure, map id or preprocessing id, THEN THE Contracts_Module SHALL raise `ContractError`.

### Requirement 9: Current-image handoff timing

**User Story:** As Owner_B, I want the handoff record to prove its time ordering, so that only current-image PnP enters the estimator.

#### Acceptance Criteria

1. WHEN a `HandoffEvidence` is constructed, THE Contracts_Module SHALL require Source_Exposure strictly before Measurement_Exposure and strictly increasing support stamps within [Source_Exposure, Measurement_Exposure], at most five of them.
2. WHEN an accepted `HandoffEvidence` is constructed, THE Contracts_Module SHALL require support stamps that start at the Source_Exposure and end at the Measurement_Exposure, between 1 and 4 transitions and an age of at most 250 ms.
3. WHEN a `HandoffEvidence` carries correspondences, THE Contracts_Module SHALL require them on the Measurement_Exposure image of the same camera with at most 256 rows.
4. WHEN a `HandoffEvidence` carries a pose, THE Contracts_Module SHALL require that pose to have source `handoff`, the same camera, the Measurement_Exposure and landmark ids contained in the carried correspondences.
5. WHEN an accepted `HandoffEvidence` is constructed, THE Contracts_Module SHALL require correspondences, an accepted pose and map, preprocessing and calibration ids equal to the pose's ids.
6. IF any support stamp lies outside [Source_Exposure, Measurement_Exposure], THEN THE Contracts_Module SHALL raise `ContractError`.

### Requirement 10: Ground-motion and border evidence

**User Story:** As Owner_A and Owner_B, I want motion and border records that state their geometry exactly, so that the estimator never double-counts motion or invents along-track fixes.

#### Acceptance Criteria

1. WHEN a `MotionEvidence` is constructed, THE Contracts_Module SHALL require `previous_ns` strictly before `current_ns`, a planar Rigid_Transform `T_Bprev_Bcurr` when present and a definite 3 x 3 covariance when present.
2. WHEN an accepted `MotionEvidence` is constructed, THE Contracts_Module SHALL require the transform, the covariance, a non-negative ground residual, a road patch id and at least one inlier.
3. WHEN a `BorderEvidence` is constructed, THE Contracts_Module SHALL require a unit `normal_xy`, `conditioning` in [0, 1] and non-negative support counts.
4. WHEN an accepted `BorderEvidence` is constructed, THE Contracts_Module SHALL require a lateral coordinate, a Wrapped_Angle yaw, a definite 2 x 2 `R` and branch `track` or `pit`.
5. THE Contracts_Module SHALL represent border evidence by lateral coordinate and yaw only, with no along-track field.

### Requirement 11: Group evidence as the single update input

**User Story:** As Owner_C, I want one validated input for the filter update, so that each group consumes at most one hierarchy level of current, causal evidence.

#### Acceptance Criteria

1. THE Contracts_Module SHALL define `GroupEvidence` as the only argument of `Estimator.step`.
2. WHEN a `GroupEvidence` is constructed, THE Contracts_Module SHALL require every record in the `pose`, `motion` and `border` slots to be accepted.
3. WHEN a `GroupEvidence` with a pose is constructed, THE Contracts_Module SHALL require the `motion` and `border` slots to be empty.
4. WHEN a `GroupEvidence` with a reset is constructed, THE Contracts_Module SHALL require the `pose`, `motion` and `border` slots to be empty.
5. WHEN a `GroupEvidence` is constructed, THE Contracts_Module SHALL require every consumed, rejected and unused evidence record, and a reset's confirmation, to be measured on this group's image of the same camera.
6. WHEN a `GroupEvidence` is constructed, THE Contracts_Module SHALL require `rejected` entries to be rejected, `unused` entries to be accepted with a Snake_Case_Reason, and no record object to appear twice.
7. WHEN a `GroupEvidence` holds a `PoseEvidence` in its pose slot, THE Contracts_Module SHALL require that pose's source to be `tracking`.
8. WHEN `consumed_kinds` is called, THE Contracts_Module SHALL return `(recovery_reset,)`, `(pnp,)`, `(handoff_pnp,)`, the motion and border kinds in that order, or `(prediction_only,)` according to the occupied slots.

### Requirement 12: State, geometry and recovery records

**User Story:** As Owner_C, I want state and geometry records with explicit validity, so that every output and reset is self-describing.

#### Acceptance Criteria

1. WHEN an `Estimate` or `SearchSnapshot` is constructed, THE Contracts_Module SHALL require a finite float64 state of shape (6,) with a Wrapped_Angle yaw and a symmetric positive semidefinite 6 x 6 covariance.
2. WHEN an `Estimate` is constructed, THE Contracts_Module SHALL require `evidence_kinds` to be one of the eight allowed combinations and `group_id` to be None exactly when the kinds are `(initial_pose,)`.
3. WHEN an `Estimate` is constructed, THE Contracts_Module SHALL require `max_support_exposure_ns` to be None exactly when the kinds are `(initial_pose,)` or `(prediction_only,)`, and otherwise at most `t_ns`.
4. WHEN an `AuxiliaryGeometry` is constructed, THE Contracts_Module SHALL require valid records to have source `pnp` or `road_patch` with a source time and z, roll and pitch, valid `road_patch` records to name a patch id, and source `none` records to be invalid without values.
5. WHEN a valid `RoadPatch` is constructed, THE Contracts_Module SHALL require a plane point, a unit normal with positive z, a non-negative residual and branch `track` or `pit`.
6. WHEN a `TrackProjection` is constructed, THE Contracts_Module SHALL require a route id of `track` or `pit`, and valid projections to have non-negative s, finite n, Wrapped_Angle epsi, a non-negative segment index and a non-negative residual.
7. WHEN a `RecoveryReset` is constructed, THE Contracts_Module SHALL require accepted candidate and confirmation poses, a confirmation at least 50 ms after the candidate and not after `t_ns`, and a non-negative `jump_m`.

### Requirement 13: Immutability and diagnostics

**User Story:** As Owner_C, I want immutable records with loggable summaries, so that emitted outputs never change and every low-trust record can be audited.

#### Acceptance Criteria

1. THE Contracts_Module SHALL define every Record as a frozen, slotted, keyword-only dataclass with tuple-valued sequence fields.
2. WHEN a Record is constructed from small arrays, THE Contracts_Module SHALL store read-only copies, so that later changes to the caller's arrays leave the Record unchanged.
3. IF code assigns to a Record field or writes into a stored array, THEN THE Contracts_Module SHALL let Python or numpy raise an error.
4. WHEN `summary()` is called on any Record, THE Contracts_Module SHALL return a mapping that `json.dumps(..., allow_nan=False)` serializes, with exact integers, enum string values and arrays of at most 64 elements listed.
5. WHEN `summary()` is called on a Record holding an image, a mask, JPEG bytes or an array larger than 64 elements, THE Contracts_Module SHALL include only shape and dtype, or byte length, for that value.
6. THE Contracts_Module SHALL declare `eq=False` on every Record that holds numpy arrays directly or through nested Records.

### Requirement 14: Call boundaries

**User Story:** As Owner_A and Owner_B, I want exact interface definitions, so that we build modules that plug into one runner and one estimator.

#### Acceptance Criteria

1. THE Contracts_Module SHALL define the Protocols `FrameSetSource`, `Canonicalizer`, `RoadGeometry`, `BorderEvidenceSource`, `TrackProjector`, `MappedTracker`, `AnchorWorker`, `GroundMotionEstimator` and `Estimator` with the method names and signatures of the design.
2. THE Contracts_Module SHALL state in each Protocol docstring its owner, call order, timing rule and rejection behavior.
3. THE Contracts_Module SHALL contain no implementation of any Protocol.
4. THE Contracts_Module SHALL describe in its module docstring the decode-set, road-patch, auxiliary-geometry, anchor-delivery, recovery-reset, input-event and diagnostics flows between owners.
5. THE Contracts_Module SHALL define `ContractError` as a subclass of `ValueError`, `RejectedEvidenceError` as a subclass of `ContractError`, and `InputError`.
6. THE Test_Suite SHALL contain executable synthetic usage examples that construct the records used by Owner_A, Owner_B and Owner_C.

### Requirement 15: Initial pose object and JSON

**User Story:** As Owner_C, I want one exact initial-pose object, so that the runner receives the single permitted pose and nothing more.

#### Acceptance Criteria

1. THE Contracts_Module SHALL define `InitialPose` with exactly the fields `timestamp_ns`, `x`, `y` and `yaw`, all finite, with `yaw` stored as a Wrapped_Angle.
2. WHEN `format_initial_pose` is called, THE Contracts_Module SHALL produce a JSON object with exactly those four keys and `timestamp_ns` as a decimal string.
3. WHEN `parse_initial_pose` receives a JSON object with exactly those keys, a JSON integer or `[0-9]+` string timestamp and JSON numbers for `x`, `y` and `yaw`, THE Contracts_Module SHALL return the corresponding `InitialPose`.
4. IF the JSON has a float, `bool`, signed, whitespace-padded or exponent timestamp, a `bool` or string coordinate, a NaN, Infinity or overflowing number, a missing, extra or duplicate key, or a non-object top level, THEN THE Contracts_Module SHALL raise `ContractError`.
5. FOR ALL valid `InitialPose` values, formatting then parsing SHALL produce an equal `InitialPose` with bitwise-identical floats (round-trip property).
6. IF `write_initial_pose` targets an existing path, THEN THE Contracts_Module SHALL raise an error and leave that file unchanged.

### Requirement 16: Episode bounds and split loading

**User Story:** As Owner_C, I want fixed, hash-bound episode bounds, so that no teammate can select a different split or blur fitting and evaluation data.

#### Acceptance Criteria

1. THE Contracts_Module SHALL define `EpisodeBounds` with a name, an inclusive `start_ns` and an exclusive `end_ns` or None, requiring `end_ns > start_ns` when present.
2. WHEN `EpisodeBounds.contains(t)` is called, THE Contracts_Module SHALL return True exactly when `start_ns <= t` and either `end_ns` is None or `t < end_ns`.
3. IF `EpisodeBounds.check_initial_pose` receives a pose whose timestamp lies outside the bounds, THEN THE Contracts_Module SHALL raise `ContractError`.
4. WHEN `load_split` reads `task_docs/build_split.json`, THE Split_Loader SHALL verify SHA-256 `894b41790af50e2f555a34001ef00821164ebc4f2f2589eb2e24280088c09222` before parsing and raise `ContractError` on mismatch.
5. WHEN the split document is parsed, THE Split_Loader SHALL check the time-encoding and interval-convention strings, convert decimal strings exactly to integers, and verify 2 s guards, strict ordering, fit start equal to the first GT stamp and holdout end equal to the last GT stamp plus 1.
6. WHEN `load_split` succeeds, THE Split_Loader SHALL return a read-only mapping of `fit`, `tune`, `holdout` and `full` with the bounds listed in the design.
7. THE Split_Loader SHALL open the split file for reading only.

### Requirement 17: Single submission writer

**User Story:** As Owner_C, I want one CSV writer, so that every submission row uses exact integer time, metres and radians in the scorer's format.

#### Acceptance Criteria

1. WHEN a `SubmissionWriter` is created, THE Submission_Writer SHALL create the file exclusively and write the header `timestamp_ns,x,y,yaw` using `\n` line endings and UTF-8 without BOM.
2. IF the target path already exists, THEN THE Submission_Writer SHALL raise `FileExistsError` and leave the existing file unchanged.
3. WHEN `write` receives a group `Estimate` whose `t_ns` exceeds the previous row's timestamp, THE Submission_Writer SHALL append one row with the timestamp as `str(int)` and `x`, `y`, `yaw` formatted with `.17g`.
4. IF `write` receives an `Estimate` whose `t_ns` does not exceed the previous row's timestamp, or an initialization estimate, THEN THE Submission_Writer SHALL raise `ContractError` and leave the written rows unchanged.
5. IF `format_row` receives a non-integer timestamp, a non-finite value or a yaw outside (-pi, pi], THEN THE Submission_Writer SHALL raise `ContractError`.
6. FOR ALL files written by the Submission_Writer, `evaluation/reference_eval.read_csv` SHALL return identical timestamps and bitwise-identical x, y and yaw values (round-trip property).
7. THE Submission_Writer SHALL expose the number of written rows and the last written timestamp.

### Requirement 18: Offline initial-pose export

**User Story:** As Owner_C, I want an offline harness that exposes exactly one GT pose, so that GT never enters the inference package.

#### Acceptance Criteria

1. WHEN the Initial_Pose_Exporter is invoked with `--split NAME`, THE Initial_Pose_Exporter SHALL take its bounds from `load_split`.
2. WHEN the Initial_Pose_Exporter is invoked with `--start-ns` and optional `--end-ns`, THE Initial_Pose_Exporter SHALL parse them with `parse_decimal_ns`.
3. WHEN the GT file's directory contains `manifest.json`, THE Initial_Pose_Exporter SHALL require that manifest's `files` entry for the GT file to equal the file's SHA-256.
4. WHEN the GT CSV is read, THE Initial_Pose_Exporter SHALL parse timestamps exactly as integers, require strictly increasing timestamps and ignore columns other than `timestamp_ns`, `x`, `y` and `yaw`.
5. WHEN at least one GT row lies within the bounds, THE Initial_Pose_Exporter SHALL write the first such row as initial-pose JSON through `write_initial_pose` and print its timestamp.
6. IF no GT row lies within the bounds, a timestamp is malformed, the manifest check fails or the output path exists, THEN THE Initial_Pose_Exporter SHALL exit with status 1 without writing a file.
7. THE Initial_Pose_Exporter SHALL default `--gt` to `evaluation/reference_reconciled/gt_tf.csv`, require `--out` and export exactly one pose.

### Requirement 19: Inference isolation

**User Story:** As Owner_C, I want an automated isolation check, so that no inference module can read GT or evaluator code.

#### Acceptance Criteria

1. IF any module under `camloc/` imports `evaluation`, `reference_eval`, `native_bridge`, organizer `evaluate`, `extract_gt` or `convert`, `scripts` or `tools` modules, statically or through a literal dynamic import or path-load call, THEN THE Isolation_Test SHALL fail.
2. IF `camloc/contracts.py` or `camloc/submission.py` imports a module other than the standard library, `numpy` and `camloc`, THEN THE Isolation_Test SHALL fail.
3. IF the source text of `camloc/contracts.py` or `camloc/submission.py` contains `reference_reconciled`, `gt_tf` or `state_estimation` in any letter case, THEN THE Isolation_Test SHALL fail.
4. WHEN `camloc.contracts` and `camloc.submission` are imported in a fresh interpreter, THE Isolation_Test SHALL verify that only standard-library, `numpy` and `camloc` modules are added to `sys.modules` beyond a bare-interpreter baseline.
5. THE Isolation_Test SHALL permit a `camloc` module to load the organizer `track_frame` helper.

### Requirement 20: Protected assets, environment and repository workflow

**User Story:** As the repository owner, I want this step to change only its own files, so that hash-bound assets, the shared environment and other teammates' work stay intact.

#### Acceptance Criteria

1. THE Shared_Contracts_Feature SHALL leave the root recording, `metadata.yaml`, `README.md`, the challenge PDF, `tools/`, `23-00-38_cut_120m_update/`, `task_docs/build_split.json`, `evaluation/` and `private_guides/` byte-for-byte unchanged.
2. THE Shared_Contracts_Feature SHALL add no dependency and leave `pyproject.toml`, `uv.lock` and `requirements.txt` unchanged.
3. THE Test_Suite SHALL write files only under pytest `tmp_path`.
4. WHEN the Shared_Contracts_Feature is complete, THE Test_Suite SHALL pass, including the 22 pre-existing tests, and `.venv/Scripts/ruff.exe check camloc scripts tests` SHALL report no violations.
5. WHEN the Shared_Contracts_Feature is committed, THE Shared_Contracts_Feature SHALL be committed on `dev/implementation` with only its step-1 source and test files staged and pushed to `origin/dev/implementation`, excluding `.kiro/`, `private_guides/`, results, artifacts and data.

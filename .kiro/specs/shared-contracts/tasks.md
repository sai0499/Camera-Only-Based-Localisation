# Implementation Plan: shared-contracts

## Overview

Implement the design in Python 3.12 with stdlib and numpy only. `camloc/contracts.py` grows in ordered slices (foundation, input records, state/geometry records, evidence, anchor/recovery/group, initial pose/episodes, Protocols), each followed by its tests. Then the single CSV writer, the offline initial-pose exporter and the isolation test are added, the full suite and ruff are run, and only the step-1 files are committed to `dev/implementation`. Property tests are seeded `numpy.random.default_rng` loops with at least 100 iterations and a `# Feature: shared-contracts, Property N: ...` tag; `hypothesis` is not used. All commands run from the project root with `.venv/Scripts/python.exe -B -m pytest` and `.venv/Scripts/ruff.exe check camloc scripts tests`.

## Tasks

- [ ] 1. Contracts foundation
  - [x] 1.1 Create the `camloc/contracts.py` foundation
    - Confirm `git branch --show-current` is `dev/implementation` and `git status --short` shows no unrelated changes before editing
    - Module docstring with the design's conventions table in short form; never write the tokens `reference_reconciled`, `gt_tf` or `state_estimation`
    - All constants, the ten enums (`StrEnum`) and `ContractError`, `RejectedEvidenceError`, `InputError`
    - `as_ns`, `parse_decimal_ns` (`re.fullmatch(r"[0-9]+")`), `wrap_angle` (`math.remainder`, `-pi -> pi`), `check_rigid`, `check_covariance(definite=...)`, `check_camera_id`, reason checks, internal float/array/id helpers that copy small arrays read-only
    - `_Record` base with `__slots__ = ()` and `summary()`, `summarize`, generic `require_accepted`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.4, 6.5, 6.6, 13.4, 13.5, 14.5, 20.1, 20.2_
  - [x] 1.2 Create `tests/contract_factories.py` and example tests in `tests/test_contracts_primitives.py`
    - Factories: seeded rng helper, random rotation (QR with determinant fix), rigid builder, closed-form rigid inverse
    - Examples: `True`, `np.bool_(True)`, `1.0`, `"1"`, `-1`, `2**63` rejected by `as_ns`; organizer wrap returns `-pi` at `-pi` while `wrap_angle` returns `pi`; NaN/inf angles rejected; constants equal the build-plan values; enum string values; exception hierarchy
    - _Requirements: 1.2, 1.3, 2.3, 14.5_
  - [x] 1.3 Write property test for nanosecond type discipline
    - **Property 1: Nanosecond type discipline**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.6**
  - [x] 1.4 Write property test for strict decimal stamps
    - **Property 2: Strict decimal stamps**
    - **Validates: Requirements 1.4, 1.5**
  - [x] 1.5 Write property test for the canonical angle wrap
    - **Property 3: Canonical angle wrap**
    - **Validates: Requirements 2.1, 2.2, 2.4, 2.5, 7.4**
  - [ ]* 1.6 Write property test for rigid transform acceptance
    - **Property 4: Rigid transform acceptance**
    - **Validates: Requirements 3.1, 3.2, 4.4**
  - [ ]* 1.7 Write property test for covariance acceptance
    - **Property 5: Covariance acceptance**
    - **Validates: Requirements 3.3, 3.4, 3.5**

- [x] 2. Input records
  - [x] 2.1 Add `ImageMessage`, `InputEvent`, `CameraCalibration`, `Observation` and `FrameSet` to `camloc/contracts.py`
    - Frozen, slotted, keyword-only dataclasses; `eq=False` for array holders; images and masks checked by dtype/shape/flags only, marked read-only, never copied
    - FrameSet rules: ordered unique cameras, `t_ns == max exposure`, derived missing ids, closure/`terminal_partial` consistency, 5 ms / 40 ms spreads, ordered input events
    - _Requirements: 1.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 13.1, 13.6_
  - [x] 2.2 Write example tests in `tests/test_contracts_records.py` for input records
    - Add valid input-record builders to `tests/contract_factories.py`
    - Observation: wrong dtype, shape, Fortran order, non-contiguous view, mask dtype; `np.shares_memory` with the source and `flags.writeable is False`; K skew/negative focal/principal point bounds; `bytearray` data rejected; `D_raw` shape
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_
  - [x] 2.3 Write property test for FrameSet invariants
    - **Property 6: FrameSet invariants**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**

- [ ] 3. State and geometry records
  - [x] 3.1 Add `AuxiliaryGeometry` (with `unavailable(reason)`), `RoadPatch`, `SearchSnapshot`, `TrackProjection` and `Estimate` to `camloc/contracts.py`
    - Yaw and other angle fields normalized with `wrap_angle`; 6-state float64 copy; PSD covariance; valid/reason rules; route ids from `ROUTE_IDS`
    - Estimate: allowed `evidence_kinds` combinations, `group_id is None` iff `(initial_pose,)`, support-stamp rule
    - _Requirements: 2.4, 6.1, 6.2, 6.3, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 13.1, 13.6_
  - [ ]* 3.2 Write example tests in `tests/test_contracts_records.py` for geometry records
    - Add geometry/state builders to `tests/contract_factories.py`
    - RoadPatch: downward or non-unit normal, unknown branch when valid, negative residual; TrackProjection: bad route id, negative s or index, missing values when valid
    - _Requirements: 12.5, 12.6_
  - [ ]* 3.3 Write property test for Estimate consistency
    - **Property 15: Estimate consistency**
    - **Validates: Requirements 12.1, 12.2, 12.3**

- [ ] 4. Evidence records
  - [x] 4.1 Add `Correspondences`, `PoseEvidence`, `HandoffEvidence`, `MotionEvidence` and `BorderEvidence` to `camloc/contracts.py`
    - Unique int64 ids, float32 pixels in canonical bounds, count caps (1024, 512, 256, 16 cells), accepted/reason rules
    - Handoff stamp ordering, chain endpoints, transitions, 250 ms age, current-image correspondences and pose, landmark subset, id agreement
    - Planar motion transform and definite 3 x 3 covariance; border unit normal, conditioning in [0, 1], definite 2 x 2 R, resolved branch when accepted
    - _Requirements: 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2, 10.3, 10.4, 10.5_
  - [ ]* 4.2 Write example tests in `tests/test_contracts_evidence.py`
    - Add evidence builders to `tests/contract_factories.py`
    - Accepted pose without `T_WB`, with median > p90 or zero cells is rejected; a one-inlier accepted pose constructs (no tuning gate in contracts); count boundaries 512/513 and 256/257; accepted motion/border missing fields; no along-track border field; `require_accepted` on geometry records
    - _Requirements: 6.6, 7.4, 7.5, 9.5, 10.2, 10.4, 10.5_
  - [ ]* 4.3 Write property test for acceptance flag and reason consistency
    - **Property 7: Acceptance flag and reason consistency**
    - **Validates: Requirements 6.2, 6.3, 6.4, 6.5**
  - [ ]* 4.4 Write property test for unique correspondence identities
    - **Property 8: Unique correspondence identities**
    - **Validates: Requirements 7.1, 7.2, 7.3**
  - [ ]* 4.5 Write property test for handoff time ordering
    - **Property 11: Handoff time ordering**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.6**
  - [ ]* 4.6 Write property test for motion and border geometry
    - **Property 12: Motion and border geometry**
    - **Validates: Requirements 10.1, 10.3**

- [x] 5. Anchor, recovery and group records
  - [x] 5.1 Add `AnchorRequest`, `AnchorResult` (with `check_against`), `RecoveryReset`, `UnusedEvidence` and `GroupEvidence` (with `t_ns`, `group_id`, `consumed_kinds()`) to `camloc/contracts.py`
    - AnchorResult has no pose, transform, position or yaw field
    - GroupEvidence follows the design's validation algorithm: accepted consumed slots, PnP-first hierarchy, reset exclusivity, current-image rule for every record, rejected/unused partition, no duplicate record objects
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 12.7_
  - [x] 5.2 Write example tests in `tests/test_group_evidence.py`
    - Add anchor/group builders to `tests/contract_factories.py`
    - AnchorResult field names contain no pose-like names; `ok` needs reason `ok` and correspondences; three or duplicate reference ids rejected; `front_anchor` on `r` rejected; candidate or handoff-source `PoseEvidence` in the pose slot rejected; reset with a pose rejected
    - _Requirements: 8.3, 8.4, 8.5, 11.7_
  - [ ]* 5.3 Write property test for anchor request timing and consistency
    - **Property 9: Anchor request timing and consistency**
    - **Validates: Requirements 8.1, 8.2**
  - [ ]* 5.4 Write property test for anchor result identity
    - **Property 10: Anchor result identity**
    - **Validates: Requirements 8.6, 8.7**
  - [ ]* 5.5 Write property test for group hierarchy and slot discipline
    - **Property 13: Group hierarchy and slot discipline**
    - **Validates: Requirements 11.2, 11.3, 11.4, 11.6, 11.8**
  - [ ]* 5.6 Write property test for current-image causality
    - **Property 14: Current-image causality**
    - **Validates: Requirements 11.5**
  - [ ]* 5.7 Write property test for auxiliary geometry and reset rules
    - **Property 16: Auxiliary geometry and reset rules**
    - **Validates: Requirements 12.4, 12.7**
  - [ ]* 5.8 Write property test for immutability in `tests/test_contracts_records.py`
    - **Property 17: Immutability**
    - **Validates: Requirements 13.2, 13.3**
  - [ ]* 5.9 Write property test for JSON-safe summaries in `tests/test_contracts_records.py`
    - **Property 18: JSON-safe summaries**
    - **Validates: Requirements 13.4, 13.5**

- [ ]* 6. Checkpoint - Ensure all tests pass
  - Run `.venv/Scripts/python.exe -B -m pytest` and `.venv/Scripts/ruff.exe check camloc scripts tests`; ensure all tests pass, ask the user if questions arise.

- [x] 7. Initial pose and episodes
  - [x] 7.1 Add `InitialPose`, `EpisodeBounds` (`contains`, `check_initial_pose`), `parse_initial_pose`, `format_initial_pose`, `read_initial_pose`, `write_initial_pose`, `episodes_from_split` and `load_split` to `camloc/contracts.py`
    - JSON hardening: `parse_constant`, duplicate-key `object_pairs_hook`, exact key set, type checks, finiteness after parsing; exclusive-create UTF-8 writer with a decimal-string timestamp
    - `load_split` reads bytes read-only, verifies `BUILD_SPLIT_SHA256`, then `episodes_from_split` checks encoding strings, exact integers, 2 s guards, ordering, first/last GT bounds and returns a `MappingProxyType` of `fit`, `tune`, `holdout`, `full`
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7_
  - [x] 7.2 Write example tests in `tests/test_initial_pose.py`
    - Real `load_split()` returns the design's four bounds; the file hash is unchanged after loading; a modified copy in `tmp_path` fails the hash; synthetic documents passed to `episodes_from_split` fail on wrong guard, order, encoding and end bound; the mapping is read-only; `write_initial_pose` refuses an existing path in `tmp_path`
    - _Requirements: 15.6, 16.4, 16.5, 16.6, 16.7_
  - [ ]* 7.3 Write property test for the initial-pose JSON round trip
    - **Property 19: Initial-pose JSON round trip**
    - **Validates: Requirements 15.1, 15.2, 15.3, 15.5**
  - [ ]* 7.4 Write property test for initial-pose JSON rejection
    - **Property 20: Initial-pose JSON rejection**
    - **Validates: Requirements 15.4**
  - [ ]* 7.5 Write property test for episode bounds semantics
    - **Property 21: Episode bounds semantics**
    - **Validates: Requirements 16.1, 16.2, 16.3**

- [x] 8. Protocol call boundaries and usage notes
  - [x] 8.1 Add the nine `@runtime_checkable` Protocols to `camloc/contracts.py` and complete the module docstring
    - `FrameSetSource`, `Canonicalizer`, `RoadGeometry`, `BorderEvidenceSource`, `TrackProjector`, `MappedTracker` (`observe`, `track`, `handoff`, `settle`, `verify_candidate`, `reset`), `AnchorWorker`, `GroundMotionEstimator`, `Estimator` (`snapshot(t_ns)`, `step`) with `...` bodies and owner/call-order/timing/rejection docstrings
    - Docstring: decode-set, road-patch, auxiliary-geometry, anchor-delivery, recovery-reset, input-event and diagnostics flows, proposed entry points, pointer to `tests/test_contract_examples.py`
    - _Requirements: 11.1, 14.1, 14.2, 14.3, 14.4_
  - [x] 8.2 Write `tests/test_contract_examples.py` with copyable usage examples
    - A: `ImageMessage`, `FrameSet`, `Observation`, `RoadPatch`, `BorderEvidence`, `TrackProjection`; B: rejected and accepted `PoseEvidence`, `AnchorResult`, `HandoffEvidence`, `MotionEvidence`; C: `AnchorRequest`, `GroupEvidence`, `Estimate`, `require_accepted`, summaries to JSON
    - `inspect.signature` checks of every Protocol method; an AST check that Protocol bodies hold only a docstring or `...`; `isinstance` conformance of test-only fakes that return rejected evidence; docstring mentions each flow
    - _Requirements: 3.6, 6.1, 11.1, 13.1, 13.6, 14.1, 14.2, 14.3, 14.4, 14.6_

- [x] 9. Single submission writer
  - [x] 9.1 Create `camloc/submission.py`
    - `HEADER`, pure `format_row`, `SubmissionWriter` (exclusive create with `newline=""`, header on open, strictly increasing `t_ns`, rejects initialization estimates, `rows`, `last_t_ns`, `flush`, `close`, context manager); imports only stdlib, numpy and `camloc.contracts`
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.7_
  - [x] 9.2 Write example tests in `tests/test_submission.py`
    - Raw bytes start with `timestamp_ns,x,y,yaw\n`, contain no `\r` and no BOM; an existing path raises `FileExistsError` and keeps its content; load `reference_eval` through a helper that inserts `evaluation/` into `sys.path` inside the test; do not modify `tests/contract_factories.py`
    - _Requirements: 17.1, 17.2, 20.3_
  - [ ]* 9.3 Write property test for the submission CSV round trip
    - **Property 22: Submission CSV round trip**
    - **Validates: Requirements 17.3, 17.6, 17.7**
  - [ ]* 9.4 Write property test for submission ordering and rejection
    - **Property 23: Submission ordering and rejection**
    - **Validates: Requirements 17.4, 17.5**

- [x] 10. Offline initial-pose exporter
  - [x] 10.1 Create `scripts/export_initial_pose.py`
    - `main(argv) -> int`, `resolve_bounds`, `verify_manifest` (sibling `manifest.json` `files[<name>]` SHA-256), `select_initial_pose`; mutually exclusive `--split`/`--start-ns`, optional `--end-ns`, default `--gt`, required `--out`
    - Import `camloc.contracts` through `importlib` after inserting the project root into `sys.path` (ruff E402 clean); exact integer stamps, strictly increasing rows, extra columns ignored, exclusive write, exit 1 without writing on any data/integrity error
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7_
  - [x] 10.2 Write example tests in `tests/test_initial_pose_export.py`
    - Load the script with `importlib.util.spec_from_file_location`; synthetic GT and manifest files in `tmp_path`: extra `z` column omitted, duplicate/decreasing/float stamps rejected, manifest mismatch and missing entry rejected, no in-bounds row, existing `--out`, malformed `--start-ns 1e9`; `--split tune` with synthetic GT stamps inside the real tune bounds
    - Real bundle under `unittest.skipUnless((ROOT / "evaluation/reference_reconciled/manifest.json").exists(), ...)`: expected pose computed from the CSV, manifest check passes, output only in `tmp_path`
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.6, 18.7, 20.3_
  - [ ]* 10.3 Write property test for exporter selection
    - **Property 24: Exporter selects the first in-bounds GT row**
    - **Validates: Requirements 18.4, 18.5, 18.7**

- [x] 11. Inference isolation
  - [x] 11.1 Write `tests/test_inference_isolation.py`
    - AST-scan every `camloc/**/*.py`: no `Import`/`ImportFrom` of `evaluation`, `reference_eval`, `native_bridge`, `evaluate`, `extract_gt`, `convert`, `scripts`, `tools` or `export_initial_pose`; no literal `importlib.import_module`, `__import__`, `spec_from_file_location` or `runpy` call naming them; no non-docstring string constant naming them as a module or path; organizer `track_frame` stays allowed
    - `camloc/contracts.py` and `camloc/submission.py`: imports limited to `sys.stdlib_module_names`, `numpy` and `camloc`; case-insensitive raw-text scan for `reference_reconciled`, `gt_tf`, `state_estimation`
    - Scanner self-test on synthetic sources in `tmp_path` (forbidden forms flagged, `track_frame` loading allowed); subprocess import of both modules compared with a bare-interpreter `sys.modules` baseline
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 20.3_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Run `.venv/Scripts/python.exe -B -m pytest` (22 existing plus new tests) and `.venv/Scripts/ruff.exe check camloc scripts tests`; confirm `git status --short` lists only the new step-1 files and `.kiro/`, `git diff --stat` is empty for protected paths and dependency files, and `build_split.json` still hashes to `BUILD_SPLIT_SHA256`; record skipped real-data tests as skipped. Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 20.1, 20.2, 20.3, 20.4_

- [ ] 13. Commit and push the step-1 files
  - [-] 13.1 Commit on `dev/implementation` and push to `origin/dev/implementation`
    - Stage by name only: `camloc/contracts.py`, `camloc/submission.py`, `scripts/export_initial_pose.py`, `tests/contract_factories.py` and the new `tests/test_*.py` files; never stage `.kiro/`, `private_guides/`, `results/`, `artifacts/`, `tmp/` or data
    - Check `git diff --cached --name-only`, commit with a concise message (for example `feat(contracts): shared records, CSV writer and initial-pose harness`), run `git push origin dev/implementation`; do not touch `main`, force-push or skip hooks
    - _Requirements: 20.5_

## Notes

- Every test sub-task is required: the user asked for core contract tests to be mandatory, so no sub-task carries the optional `*` marker.
- Each property sub-task implements exactly one design property as one seeded loop of at least 100 iterations; failures report the iteration index and inputs.
- Protocol bodies stay `...` with docstrings. No task implements A/B modules, `filter.py`, `scheduler.py` or `run.py`, and no fake that returns a plausible pose is added outside tests.
- The "Proposed shared-record changes" and "Open questions" sections of `design.md` need A/B agreement before integration; changes after agreement go through C.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1"] },
    { "id": 2, "tasks": ["1.3", "2.2", "3.1"] },
    { "id": 3, "tasks": ["1.4", "2.3", "4.1"] },
    { "id": 4, "tasks": ["1.5", "5.1"] },
    { "id": 5, "tasks": ["5.2", "7.1", "9.1"] },
    { "id": 6, "tasks": ["7.2", "8.1", "9.2", "10.1", "11.1"] },
    { "id": 7, "tasks": ["8.2", "10.2"] },
    { "id": 8, "tasks": ["12"] },
    { "id": 9, "tasks": ["13.1"] }
  ]
}
```

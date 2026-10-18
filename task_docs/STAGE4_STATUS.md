# Team C Stage 4: anchor scheduler, recovery and status machine

This builds on [STAGE2_STATUS.md](STAGE2_STATUS.md). The source of implementation
requirements remains `FINAL_BUILD_PLAN.md`. This file records what is implemented and the
integration boundary; it is not a localization result. Everything A/B provide is reached
only through the `AnchorWorker`, `MappedTracker` and `TrackProjector` Protocols, exercised
with test-only fakes in `tests/stage4_fakes.py` and B's real tracker/worker in
`tests/test_b_c_integration.py`. A's input/geometry/route adapter and B's Step 1-3 vision
components are integrated. [STAGE3_STATUS.md](STAGE3_STATUS.md) records the raw-camera
runner, real fitting bank and failed tuning baseline gate. Visual evidence must come from
the declared frontend; no fabricated pose provider is supplied.

Implemented:

- `camloc.scheduler.AnchorScheduler(worker, map_id=..., policy=None, threaded=False, clock=...)`: at most
  one In_Flight_Job and one latest Pending_Request. Creation, replacement, expiry, rate and
  trigger decisions use source time only; the injected clock is read only for the worker
  watchdog and runtime fields. Purpose order recovery > front > rear > warm, one request per
  group, rolling per-purpose and total caps (normal/recovery). Delivery happens at the first
  group with `t_ns >= delivery_not_before_ns`. Synchronous and background worker modes
  give the same logical schedule. A backend fault disables the scheduler for the rest of the
  run without raising; stale results are never fused.
  `GroupContext.rear_in_flight` covers any rear-camera job, including recovery, so the
  frontend preserves its image chain through delivery. Recovery proposals alternate
  available cameras within the existing four-job ceiling; healthy front tracking reserves
  the staggered rear slot and decodes its in-flight chain.
  Both modes use one daemon worker thread with a bounded wait. A stuck native/GPU call is
  abandoned, never reused, and disables further jobs; it cannot block process shutdown.
  Priority applies to proposals that pass their rate caps: a capped front proposal cannot
  block an eligible rear/warm proposal. All proposals must match the current source ID.
- `camloc.recovery.RecoveryManager(policy=None)`: holds at most one candidate
  (an accepted `tracking` pose or handoff pose logged unused with reason `innovation_gate`).
  It confirms into a `RecoveryReset` only from a Confirmation_Pose at least
  `RECOVERY_CONFIRM_MIN_NS` later, with matching identities and calibration, and within
  inclusive motion bounds. Timeout, replacement, discard, rejection and confirmation are
  each logged as one append-only `RecoveryEvent`. It reads only `estimator.snapshot`.
  The runner checks innovation before this decision, so a selected but out-of-gate pose
  cannot falsely announce tracking reacquisition and erase the held candidate.
- `camloc.filter.CameraFilter`: full Tracking/Degraded/Recovering/Lost status machine and
  a confirmed `RecoveryReset` that re-initializes state at its confirmation exposure.
  `step(evidence, recovery_search_active=...)` only selects Recovering over Degraded. A
  reset at or before the last fused time is rejected.
- `run.run_stream(..., scheduler=, recovery=, tracker=, projector=)`: with none supplied,
  the unchanged Stage 2 loop (`_run_basic`) runs and its output is byte-identical. Otherwise
  `_run_stage4` passes the scheduler's `GroupContext` to the frontend, routes evidence
  through the recovery manager, settles handoffs (ADOPT/CANDIDATE/DISCARD), resets the
  tracker/projector after a reset, and writes scheduler and recovery diagnostics. An optional
  projector now writes both current route reports per output; failures produce ProjectionError
  diagnostics without changing the pose CSV.
  Cleanup closes jobs and drains diagnostics before JSONL closes on both normal and fatal
  paths. Fatal source/filter/tracker errors preserve the written prefix, cancel remaining
  jobs with `run_aborted`, and append `RunAborted`; only a completed run gets `RunSummary`.
  A compatible, unused current-image handoff (within 1 m/2 degrees of the selected pose)
  can refresh mapped tracks after the selected pose commits. It is settled as `ADOPT`
  without a second filter update; incompatible/stronger-pose-only unused results are discarded.

Notable implementation choices:

- Fault classification checks the watchdog first, then the other fault kinds.
- Recovery reports `identity_mismatch` and `calibration_mismatch` as separate rejection
  reasons.
- A request dropped by the rate limit at dispatch counts in both `anchor_rate_limited`
  and `anchor_expired`.
- `RecoveryError` is a diagnostic record name only; there is no contracts class for it.
- Without a scheduler, `recovery_search_active` is always False, so Recovering is never
  fabricated.

## Resolved decisions, 18 October 2026

Tracking-by-age now requires an applied full pose within 0.5 s and current accepted ground
tracking. Prediction-only and border-only groups use Degraded/Recovering until Lost applies.
InitialPose is not counted as an accepted full pose. No shared-record change is needed.

1. **Keep the evaluator boundary and every camera output.** The unchanged evaluator uses
   `searchsorted(side="left")`, so a prediction exactly at its first GT timestamp has no
   preceding bracket and is excluded from `evaluated`. It remains in `submitted` and the
   original CSV. Report both counts; never shift timestamps, remove the row or move the
   episode to manufacture equality. When genuine GT brackets every camera group, the
   counts agree. `test_first_gt_boundary_exclusion_never_changes_camera_output` covers
   the exact-boundary case, and the existing smoke test covers fully bracketed output.
2. **Stage a confirming handoff as CANDIDATE, then reset.** After the filter commits the
   confirmed reset, settle its current confirming handoff as `CANDIDATE`, then call
   `tracker.reset(reset)` once. The tracker promotes the tracks for `reset.confirmation`
   and clears the old hypothesis/candidate metadata. `DISCARD` loses the current refresh;
   `ADOPT` followed by an older-candidate promotion can restore stale tracks. Ordinary
   in-gate handoff fusion still uses `ADOPT`; unrelated/rejected handoffs use `DISCARD`.
   Candidate verification and tracking-source confirmations must likewise retain their
   current checked tracks for reset. The projector resets before the new route report.

## Required tracker lifecycle for B

The previous proposal is now an explicit C-owned protocol requirement:
`MappedTracker.drop_candidate(reason: str) -> None`. It clears only the uncommitted track
set, preserving active tracks and bounded image history. It must be safe with no candidate.
C calls it once when recovery times out, rejects/replaces a candidate for inconsistent
motion, or reacquires normal tracking. It runs after the group commits and before settling
a replacement handoff. A confirmed reset promotes its confirmation instead of dropping it.

The runner rejects a tracker without this method before creating output files. B must
implement it and the documented `reset` promotion semantics before integration. Data-record
fields, calibration/preprocessing identities and the Estimator protocol are unchanged by
this protocol revision. Tests use a stateful tracker to check which current set is promoted.

Normal/recovery anchor ceilings remain 3/4 jobs per source second. Warm retrieval must share
an eligible anchor slot: an empty `reference_ids` requests retrieval inside that job. B/C's
frontend must schedule that work within the existing budget, rather than adding a fourth
normal job outside the cap.

Border corrections remain disabled pending physical-border review. The acceptance path now
counts image-supported samples/contours, rejects a lost third contour even at high overall
retention, and checks conditioning on the actual matched set at the selected pose. A
projected point with no compatible image edge is not counted as support.

## Remaining implementation gates

- B's Step 1-3 map, tracker, worker and ground-motion implementation are integrated;
  the production frontend/raw-MCAP CLI and independent run validator are assembled.
  Real tuning replay still loses mapped support. See `STAGE3_STATUS.md` for the current
  measured gate; passing synthetic lifecycle tests does not close that gate.
- Global EigenPlaces retrieval remains B's later step. The current geometric shortlist
  cannot recover once the prediction has drifted outside its bounded spatial search.
- Frozen fitting-only body-to-road height, bodywork/traffic masks and physical-border image
  association still need fitting/review. Ground/border startup must reject missing geometry.
- Run-level diagnostics validation is implemented; paced latency/queue qualification and
  replay/demo tooling remain subsequent C work. Reuse the existing offline scorer; no evaluator changes or GT
  callbacks belong in the inference path. An aborted CSV alone is not a completed-run gate.
- End-to-end tuning has been measured and fails the continuous visual-baseline gate.
  Loss/recovery quality, competition GPU runtime and native higher-rate support remain open.

## Verification boundary

Prior A+C recheck, 18 October: **865 passed**, none skipped or xfailed; Ruff and whitespace
checks passed. Command: `.venv/Scripts/python.exe -B -m pytest -q --basetemp tmp/stage4_full_recheck`.
`tests/test_stage4_decisions.py` covers the resolved lifecycle, exact evaluator boundary,
eligible-job priority and fatal-run cleanup. `tests/test_borders.py` covers actual matched
support and a positive synthetic lateral/yaw recovery while preserving the default stop rule.

Original Stage 4 gate at `fc0d512`: `.venv/Scripts/python.exe -B -m pytest -q`: **767 passed**,
none skipped or xfailed. See [A+C integration status](A_C_INTEGRATION_STATUS.md) for current checks.
New/changed Stage 4 files: `tests/test_scheduler.py` 23, `tests/test_recovery.py` 20,
`tests/test_runner_stage4.py` 13, `tests/test_smoke_run.py` 5, `tests/test_filter.py` 51
(112 total). `.venv/Scripts/ruff.exe check camloc scripts tests run.py`: all checks passed.
The only pre-existing test removed is `test_stage_four_reset_is_explicitly_unsupported`,
superseded by the implemented reset. `tests/test_inference_isolation.py`, the split file,
evaluation sources, organizer assets and dependency pins are unchanged.

This is synthetic software and timing verification only. It makes no claim of localization
accuracy, runtime performance or native high-rate support. Those gates remain open until
the integrated visual path passes real-recording validation. Current results and the latest
whole-suite verification are recorded in `STAGE3_STATUS.md`.

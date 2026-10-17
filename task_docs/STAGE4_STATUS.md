# Team C Stage 4: anchor scheduler, recovery and status machine

This builds on [STAGE2_STATUS.md](STAGE2_STATUS.md). The source of implementation
requirements remains `FINAL_BUILD_PLAN.md`. This file records what is implemented and the
integration boundary; it is not a localization result. Everything A/B provide is reached
only through the `AnchorWorker`, `MappedTracker` and `TrackProjector` Protocols, exercised
with test-only fakes in `tests/stage4_fakes.py`. No production module returns a pose of its
own. `camloc/contracts.py` is unchanged.

Implemented:

- `camloc.scheduler.AnchorScheduler(worker, map_id=..., policy=None, threaded=False, clock=...)`: at most
  one In_Flight_Job and one latest Pending_Request. Creation, replacement, expiry, rate and
  trigger decisions use source time only; the injected clock is read only for the worker
  watchdog and runtime fields. Purpose order recovery > front > rear > warm, one request per
  group, rolling per-purpose and total caps (normal/recovery). Delivery happens at the first
  group with `t_ns >= delivery_not_before_ns`. Synchronous and single-thread executor modes
  give the same logical schedule. A backend fault disables the scheduler for the rest of the
  run without raising; stale results are never fused.
- `camloc.recovery.RecoveryManager(policy=None)`: holds at most one candidate
  (an accepted `tracking` pose or handoff pose logged unused with reason `innovation_gate`).
  It confirms into a `RecoveryReset` only from a Confirmation_Pose at least
  `RECOVERY_CONFIRM_MIN_NS` later, with matching identities and calibration, and within
  inclusive motion bounds. Timeout, replacement, discard, rejection and confirmation are
  each logged as one append-only `RecoveryEvent`. It reads only `estimator.snapshot`.
- `camloc.filter.CameraFilter`: full Tracking/Degraded/Recovering/Lost status machine and
  a confirmed `RecoveryReset` that re-initializes state at its confirmation exposure.
  `step(evidence, recovery_search_active=...)` only selects Recovering over Degraded. A
  reset at or before the last fused time is rejected.
- `run.run_stream(..., scheduler=, recovery=, tracker=, projector=)`: with none supplied,
  the unchanged Stage 2 loop (`_run_basic`) runs and its output is byte-identical. Otherwise
  `_run_stage4` passes the scheduler's `GroupContext` to the frontend, routes evidence
  through the recovery manager, settles handoffs (ADOPT/CANDIDATE/DISCARD), resets the
  tracker/projector after a reset, and writes scheduler and recovery diagnostics.
  `scheduler.close()` runs in `finally`; diagnostics are written only on a normal finish.

Notable implementation choices:

- Fault classification checks the watchdog first, then the other fault kinds.
- Recovery reports `identity_mismatch` and `calibration_mismatch` as separate rejection
  reasons.
- A request dropped by the rate limit at dispatch counts in both `anchor_rate_limited`
  and `anchor_expired`.
- `RecoveryError` is a diagnostic record name only; there is no contracts class for it.
- Without a scheduler, `recovery_search_active` is always False, so Recovering is never
  fabricated.

## Open questions

1. Tracking-by-age requires an actual applied Full_Pose. The InitialPose fallback is used
   only for the 3 s Lost rule. This keeps
   `test_border_updates_only_lateral_and_yaw_and_is_source_time_limited` (which expects
   Degraded) valid. Confirm, or amend Req 8.2 to say "the latest applied Full_Pose".
2. `validate_submission` brackets with `searchsorted(side="left")`, so a row at exactly
   `tg[0]` is not evaluated. The synthetic smoke test starts GT at the InitialPose time,
   0.5 ms before the first group, so `evaluated == submitted`. Req 2.1 needs amending to
   "first GT timestamp <= first group t_ns".
3. A HandoffEvidence whose nested pose confirms a reset is settled DISCARD (the estimate
   kinds are `(recovery_reset,)`), followed by `tracker.reset(reset)`. Say if ADOPT is
   preferred.

## Protocol proposal for A/B

`MappedTracker` has no call to drop a held candidate after a recovery timeout or motion
discard. Currently a later `verify_candidate` pose with no held candidate is logged as
`verification_rejected` (`no_candidate_held`) and ignored. Proposal: add
`MappedTracker.drop_candidate(reason: str) -> None`, called by the runner when the recovery
manager discards or times out its candidate. This is a proposal only; `contracts.py` is not
changed until A/B agree.

## Verification boundary

`.venv/Scripts/python.exe -B -m pytest -q`: **767 passed**, none skipped or xfailed.
New/changed Stage 4 files: `tests/test_scheduler.py` 23, `tests/test_recovery.py` 20,
`tests/test_runner_stage4.py` 13, `tests/test_smoke_run.py` 5, `tests/test_filter.py` 51
(112 total). `.venv/Scripts/ruff.exe check camloc scripts tests run.py`: all checks passed.
The only pre-existing test removed is `test_stage_four_reset_is_explicitly_unsupported`,
superseded by the implemented reset. `tests/test_inference_isolation.py`, the split file,
evaluation sources, organizer assets and dependency pins are unchanged.

This is synthetic software and timing verification only. It makes no claim of localization
accuracy, runtime performance or native high-rate support. Those gates remain open until
A/B's frontend, worker and tracker are integrated and measured on the recording.

# A + C integration, 18 October 2026

This records merge `a4b485b`. Subsequent lifecycle/protocol decisions and the latest test
gate are recorded in [STAGE4_STATUS.md](STAGE4_STATUS.md).

A's `origin/darshana` at `7fe7de0` is merged into the owner's `dev/implementation`
on top of C's `fc0d512`. The merge preserves both histories and C's newer contracts.
No shared records, public protocol signatures, dependency pins, evaluation sources,
organizer assets or frozen split were changed. `main` remains separate.

## Fixes

- Input timers advance before attaching new packets. Late support is logged and cannot
  recreate an emitted timestamp or reorder startup/capacity output. Every pending group
  keeps its opening exposure tolerance/deadline. The cadence window is bounded to 16
  intervals, capacity closures preserve groups, and coverage gaps/EOF keep recorded times.
- Repeated camera/static calibration is checked for compatibility. Unsupported distortion,
  changed camera parents and invalid quaternions fail explicitly. Calibration identity
  includes dimensions, frame IDs, raw intrinsics/distortion and extrinsics. Supplied
  calibration dictionaries must cover the seven camera identities correctly.
- The reader reuses A's quaternion conversion. Canonicalization rejects mismatched camera
  frames and masks pixels whose bilinear interpolation would include an outside sample.
- Road patches use contiguous same-branch neighborhoods and monotonic opposing-border
  association; they reject insufficient/collinear support rather than fitting across a
  nearby crossing or hairpin arm.
- Border search uses eight edge-orientation bins, per-contour sample caps/equal weights,
  mask/depth checks and a frozen support denominator. Lost samples receive a penalty;
  candidates cannot win by hiding support. The source limits attempts to 5 Hz and rejects
  exposure-mismatched/future projection geometry. The 200-sample/85-candidate caps remain.
  Corrections stay disabled by default pending the required physical-border image review.
- The route adapter keeps the actual previous segment instead of globally reprojecting
  the previous XY at every call. Streaming projections match the unchanged organizer
  helper's sequence behavior at a crossing, with global reset after recovery or a >1 s gap.
- Both scheduler modes use one bounded daemon worker. The synchronous API also has a real
  watchdog wait; a stuck worker cannot block output forever or prevent process exit. A
  timed-out native/GPU call cannot be forcibly killed: it is abandoned, its result ignored,
  and subsequent jobs disabled for the run. Resources held by it last until it returns or
  the process exits; backend restart is not attempted.
- Prediction-only/border-only output does not inherit Tracking from recent PnP age alone.
  Tracking by age requires current accepted ground tracking and an applied full pose within
  0.5 s. Lost precedence, covariance growth and explicit Recovering behavior are retained.
- Recovery rejects a newly held candidate with incompatible current-image calibration.
  Confirmation keeps the existing identity/calibration checks and diagnostic reasons.
- The runner writes both route projections after resetting projection history when needed.
  Projection failures are diagnosed separately and do not rewrite the pose CSV.

## Verification

The pinned `.venv` is used throughout. Validation includes the complete existing evaluator
suite, A's actual-recording calibration/grouping checks, new timing/calibration/mask/crossing
regressions, stuck-worker process-exit checks, and an A-reader -> canonicalizer -> C-runner
-> A-projector test. That integration test supplies no B pose, so all outputs remain
explicitly prediction-only/Lost, including an intentionally bad JPEG group.

Commands:

```powershell
.venv/Scripts/python.exe -B -m pytest -q --basetemp tmp/merge_final_check
.venv/Scripts/ruff.exe check camloc scripts tests run.py
git diff --check
```

Final gate: **849 passed**, none skipped or xfailed, in 104.40 s. Ruff and the whitespace
check passed. The frozen split SHA-256 remains
`894b41790af50e2f555a34001ef00821164ebc4f2f2589eb2e24280088c09222`.

## Boundary for B

Use the merged A modules and the current `camloc/contracts.py`. Automatic calibration and
preprocessing IDs changed intentionally (`canonical-remap-v2-full-bilinear-mask` digest).
Rebuild reference features/maps fitted against the previous IDs/masks; never relabel an
old artifact as compatible. Camera pixels still use the prescribed one-pass OpenCV remap
and half-pixel K. Body-to-road height remains a frozen fitting artifact, with no zero-height
default or hidden GT startup.

B's actual tracker/worker/retrieval/ground-motion modules, the production evidence-building
frontend and raw-MCAP CLI are still pending. No fabricated provider was added. The later
Stage 4 recheck resolved candidate cleanup and confirming-handoff promotion: B must implement
the current `drop_candidate(reason)` and `reset(reset)` requirements in
`STAGE4_STATUS.md`. Map fitting, the manual border-association gate, end-to-end accuracy,
runtime and native high-rate qualification remain unverified.

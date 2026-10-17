# Team C Stage 2: filter and streaming core

The independent C implementation can proceed while A/B finish their modules. The source
of implementation requirements remains `FINAL_BUILD_PLAN.md`. This file records what is
implemented and the integration boundary; it is not a localization result.

Implemented:

- `camloc.filter.CameraFilter(initial_pose)`: six-state CV prediction, continuous-time Q,
  prescribed initial covariance, wrapped PnP innovations, Joseph covariance updates,
  quality-inflated measurement floors and source-time shared-map information scaling.
- PnP-first selection with a defensive innovation gate. C's frontend should call
  `estimator.pose_in_gate(pose)` before assembling GroupEvidence, so an out-of-gate pose
  can be recorded as unused and ground/border fallback can be selected in that group.
- Lateral/yaw border updates, 5 Hz exposure-time limit and branch-preserving rejection.
  Border evidence never counts as full-position confirmation.
- Ground motion composed at its previous exposure, then advanced to group time. Map-axis
  velocity/yaw-rate smoothing uses `1-exp(-dt/0.2)`. World/body tangent-plane transforms
  include the cached auxiliary roll/pitch. The delta is never added on top of CV motion.
- A cache of at most 32 exposure states per camera, at most 0.25 s old. Earlier-camera
  cache entries exclude evidence from later exposures in the same group. A motion delta
  with missing/stale history, missing geometry or an intervening fused update is declined
  with a reason; it does not overwrite newer absolute/border information.
- `run.run_stream`: consumes supplied FrameSets, calls C's frontend assembly, steps once,
  writes every completed group to the existing exclusive-create CSV writer, and writes
  initialization, groups, evidence, frontend errors, estimates and trailing input events
  to separate JSONL. Frontend failures produce explicitly logged prediction-only outputs.
  Structural input/filter errors stop explicitly. Episode/calibration changes are checked.

The process-noise multiplier is 1 while Tracking, 4 while Degraded/Recovering, and 10
while Lost. These are documented engineering defaults, not fitted uncertainty. A consumed
full pose sets Tracking; usable fallback sets Degraded; missing geometry, >1 s without
usable visual evidence, or >3 s without full-position confirmation sets Lost. The subsequent
[Stage 4 implementation](STAGE4_STATUS.md) adds confirmed recovery resets and active-search
Recovering states. Prediction-only output is Degraded/Recovering until a Lost rule applies;
recent full-pose age alone does not establish usable tracking.

## Call boundary

```python
from camloc.filter import CameraFilter
from run import run_stream

estimator = CameraFilter(initial_pose)  # exactly one allowed timestamp/x/y/yaw object
counts = run_stream(
    source, estimator, build_evidence,
    output="results/submission.csv", diagnostics="results/diagnostics.jsonl",
    episode=episode,
)
```

`source` is A's FrameSet iterable. `build_evidence(frame_set, prior)` is C's main-thread
assembly function using A/B's records and interfaces; it must return GroupEvidence for
that exact FrameSet. The supplied prior is at group time. Use `estimator.snapshot(t_ns)`
for a current exposure's prior and `estimator.exposure_snapshot(camera_id, exposure_ns)`
for its cached preceding exposure. Snapshot queries do not change the estimator. Never
retime a later group posterior to make an earlier exposure snapshot.

New optional shared field proposed for A/B review: `GroupEvidence.auxiliary`, default
None, supplies checked road geometry without a competing filter mutation API. PnP
geometry enters with the selected pose. Missing body-to-road height must be represented
as unavailable geometry; this filter never assumes a zero height or reads extra GT fields.

The caller creates the output directory. Both output files must be new. Returned counts
include frontend failures and prediction-only groups and must be retained in reporting.

## What still requires A/B

No raw-MCAP frontend is wired, no fitted bank is loaded, and no real camera-based tuning
trajectory is produced. Executing `run.py` directly fails with this explanation instead
of pretending that a diagnostic prediction-only stream is a localization baseline.
The programmatic streaming API is exercised with test-only synthetic frontends.

After A/B arrive: agree the optional geometry field, connect their reader/canonicalizer,
tracker and checked road/border/motion paths, validate artifact identities, and obtain the
first real scored baseline. Step 3 reuses the existing evaluator unchanged. Stage 4 adds
the deterministic bounded anchor scheduler, candidate lifecycle and confirmed recovery.

## Verification boundary

The complete suite passes **689 tests**, including **44 new filter/stream checks**, and
Ruff passes for `camloc scripts tests run.py`. The split file and dependency pins remain
unchanged. These are software/math checks on the independent C implementation.

`tests/test_filter.py` covers actual F/Q composition and filter prediction at
10/20/30/60/120 Hz, angle wrapping, PSD/symmetric covariance, source-time PnP scaling,
border observability/rate/branch behavior, motion composition/EMA/tilt, bounded history,
intervening updates and exposure causality. `tests/test_stream_runner.py` covers synthetic
stream failures, output identity, repeatability, completed prefixes, episode/calibration
boundaries, file preservation, trailing events and the unavailable raw-recording CLI.

Run `.venv/Scripts/python.exe -B -m pytest`, then
`.venv/Scripts/ruff.exe check camloc scripts tests run.py`. Synthetic checks do not
establish real input independence, accuracy, runtime, native higher-rate tracking or
camera failure/recovery performance. Those gates remain open until integrated and measured.

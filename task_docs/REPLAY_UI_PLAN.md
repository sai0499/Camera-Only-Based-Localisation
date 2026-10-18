# Localisation UI implementation plan

Status: first implementation available, 18 October 2026; see [UI_STATUS.md](UI_STATUS.md).
Owner: C. Implements the presentation/run workflow
in [FINAL_BUILD_PLAN.md section 9](FINAL_BUILD_PLAN.md); it does not replace the estimator
specification or close Stage 3/5 gates. The user has authorised UI work while B improves
visual support. This document retains the intended scope; status records verified work
and remaining features. UI readiness is separate from localisation qualification.

## Intended workflow

**Select input -> validate -> start localisation -> monitor -> review in 2D/3D -> export.**

The estimator runs once. The UI observes its output and never edits a pose, selects its
visual evidence, changes a gate or supplies GT. Replay controls change the viewer clock
only. The identical command must also run without opening a browser.

Design for hidden evaluation input without requiring its GT. The README allows one
initial timestamp/x/y/yaw, but the exact judges' launch interface, recording packaging
and private scoring process are not confirmed. This UI is our local run/review interface,
not a claim that judges will use it or a new official submission format.

## First screen: configure a run

- Select a local MCAP, its one initial-pose JSON (or enter those four fields), the frozen
  map/retrieval artifact directory and a fresh output directory. Select local paths;
  nothing is uploaded. Show the supplied static map and supported cameras.
- Offer fixed **Tuning** and **Holdout** presets plus **New recording**. New recording
  starts at the supplied initial pose and runs to EOF, with optional explicit bounds.
  Holdout remains an explicit action; the UI must not run it automatically.
- Validate file readability, initial pose, camera/calibration compatibility, map/weight
  identities and writable fresh output. Fail visibly on incompatible assets; never
  substitute a different calibration or silently rebuild/fall back to another map.
- Present a short run summary and Start. GT is not a configuration field for inference.
  The result directory records input identity, initial pose, artifact/code IDs and settings.
- Model inputs remain the allowed images/calibration/static geometry/frozen artifacts.
  Bundled dynamic `/tf` and live vehicle state remain excluded even if present in the bag.

**Required runner change:** `run.py` currently requires `--split tune|holdout` and loads
fixed episode timestamps. Add a mutually exclusive explicit-input mode using the existing
`EpisodeBounds` record (`end_ns=None` already means EOF). Keep existing preset behaviour,
training-artifact provenance checks and the frozen `build_split.json` unchanged. Reuse
the existing reader, frontend, estimator and submission writer; add no second localizer.
The validator must distinguish known development splits from arbitrary input and must
not apply training camera-stamp/GT membership checks to a new recording.

## Screen layout

See the [layout sketch](figures/replay_ui_layout.svg). A precise engineering workspace:
large central scene, compact evidence inspector, synchronised bottom timeline. The user's
design contract supersedes the original dark sketch: warm paper, graphite grid rules and
a signal-red estimate. Barlow Condensed 600/700 headings pair with IBM Plex Sans
400/500/600 body text. Use line patterns/markers as well as colour, visible keyboard
focus, units and compact legends. No gradients, shadows or generic card grids.

The header always shows run identity, input name, Running/Completed/Failed/Stopped,
latest processed exposure, outputs and the estimator's separate Tracking/Degraded/
Recovering/Lost state. A completed process does not imply successful localisation.
Indicate the selected replay time separately from the newest available output.

Four workspaces share one selection and time cursor:

1. **Monitor:** default during processing. Current trajectory/heading, evidence source,
   camera availability, output count, source-time progress, stalls and errors. Follow
   latest output by default; pause/scrub releases follow mode without pausing inference.
2. **2D replay:** equal-scale map, surveyed borders, track/pit reference paths, trajectory
   trail, current car heading, optional separately labelled GT and an error connector.
   Follow car or fit map; show only past trail during monitoring. Full-route overview is
   a separately labelled completed-replay option. Mark losses, resets and evidence gaps.
3. **3D scene:** orbit/follow/top/side cameras; XYZ surveyed borders and optional frozen
   landmark cloud. Show current body/rig geometry only where recorded and valid. It is
   a sparse engineering scene, not a reconstructed photorealistic track.
4. **Diagnostics:** selected-frame explanation, source/current handoff timestamps,
   accepted/rejected evidence, inliers/fraction, median/p90 reprojection residuals, grid
   coverage, recovery history and scheduler timeline. Show values against unchanged
   gates, and jump directly to losses, resets, rejected handoffs or slow groups.

Optional synchronised front/rear image panels sit next to the scene. Show exact camera,
exposure and missing/late state; never substitute a neighbouring image. Canonical point
overlays must use canonical images, matching preprocessing/calibration and actual saved
coordinates. A reprojection computed for display is labelled predicted projection, not
observed feature evidence. Keep camera extraction optional and outside the inference path.

The bottom area has play/pause, step one output, replay speed, time entry, follow-latest
and event navigation. Space toggles playback; arrows step outputs. Synchronized plots
show evidence/status and processing time; after optional GT attachment, add position,
car-frame lateral/longitudinal and wrapped heading errors. Keep reference-line n/epsi
separate from car-frame errors. UI animation rate and localizer throughput are separate.

## Ground truth and honest 3D

- Hidden/no-GT mode shows estimates, map context, evidence and timings. Accuracy is
  explicitly unavailable, never zero. Attach a reference only in offline Review after
  the inference run; verify its identity/interval and use the existing scorer conventions.
- Preserve all outputs and known failures. No alignment, smoothing, automatic scoring
  interval selection or deletion of difficult frames. Changing the visible window cannot
  silently change the saved full-run score; label any selected-window statistics.
- Current submissions and prepared comparison GT contain x/y/yaw, not z. They cannot
  establish a measured 3D trajectory error. Default comparative trajectory stays in 2D.
- Auxiliary z/roll/pitch are logged with validity/source/time. In 3D, display those values
  with their age and provenance, and leave unavailable intervals broken. Never show
  invented z=0 as measured height or connect unknown heights as a continuous trajectory.
- A road-projected marker may be offered as a clearly labelled display projection. It
  is not estimated vertical position or GT. Preserve supplied XYZ and equal metre scales;
  any vertical exaggeration is explicit and defaults to 1x.
- The filter covariance is uncalibrated. Any ellipse is labelled model covariance,
  not a probability/confidence guarantee. Do not turn inlier ratio into confidence %.

## Data already available and gaps

Verified against the fresh temporary tuning run (2,708 outputs). Inputs remain private
under `results/`, `artifacts/` and the original recording; do not embed them in Git.

- `submission.csv`: exact output timestamp/x/y/yaw; canonical source for displayed poses.
- `diagnostics.jsonl`: FrameSet, GroupEvidence, Estimate, TrackProjection, scheduler,
  recovery, tracker and input events. Pose/Handoff records are nested in GroupEvidence;
  parse the real nesting rather than expecting standalone records.
- `runtime.jsonl` and `runtime.json`: group service, reader and wait measurements. Detailed
  timing is available only when requested. Missing timing remains unavailable. Observing
  an output file is not a sensor-to-output latency measurement.
- `run.json`: run settings, hashes, map identities, counters, startup/replay summary.
- Map `points_W` and surveyed XYZ: real 3D context, loaded only through verified assets.
- Images: byte payloads are not stored in diagnostics. Export selected camera images in
  a separate allowlisted reader from the input recording, keyed by camera/exposure.
- Exact track/pixel arrays: some small arrays are saved, but larger arrays are only
  shape/dtype summaries. Existing logs cannot recover every observed feature overlay.
  Show counts first. Detailed overlays need an opt-in bounded debug sidecar and a separate
  overhead check; agree any B-owned logging hook with B, and do not alter shared records.
- Offline GT: optional evaluation bundle only. Do not send its path, data or callbacks to
  the estimator subprocess. Keep scoring tools and UI metrics outside `camloc` inference.

## Implementation structure

Use a local browser UI: buildless HTML/CSS/JavaScript with Plotly for 2D plots and a
CPU Canvas orthographic renderer for real XYZ geometry. This avoids the unreliable
WebGL export observed during browser verification and keeps 3D off the inference GPU. Reuse
the current plotting approach and avoid adding an npm build, cloud service or database
for this scope. Plotly supports [3D scatter scenes](https://plotly.com/javascript/3d-scatter-plots/)
and [incremental chart updates](https://plotly.com/javascript/plotlyjs-function-reference/).
Bundle JavaScript locally; no CDN, external map tiles, analytics or project uploads.

Plotly 7.1.0 was already locked transitively through hloc; the UI now declares it as a
direct pinned dependency. C updated `pyproject.toml`, `uv.lock` and exported
`requirements.txt` together without changing resolved versions. Use Python
standard-library HTTP/subprocess support initially.

Planned files:

- `render_replay.py`: validate/export a saved run to a versioned viewer bundle; also offer
  a passive completed-run path that launches no estimator. Preserve unknown optional
  records and diagnose corrupt/missing mandatory data.
- `scripts/serve_replay.py`: loopback-only run launcher/observer; one inference subprocess
  at a time, invoked with an argument list and the project interpreter. Start returns
  promptly; the browser only polls completed records. No shell command field or arbitrary
  file-serving route. Serve only selected generated viewer assets and run metadata.
- `replay_ui/`: static UI source and one shared replay state/controller. Mode changes do
  not re-run inference. Do not keep separate 2D and 3D copies of pose/clock selection.
- `tests/test_replay_export.py`, `tests/test_run_input.py` and focused browser checks:
  data/runner correctness and visible controls, not screenshots of invented results.

The UI data model keeps absolute nanoseconds as decimal strings; join/index with Python
integers or JavaScript BigInt. Derive relative seconds only after integer subtraction.
Never round epoch nanoseconds through JavaScript Number. Keep frame/group identities,
missing camera markers, accepted/rejected distinctions and source/current times.

Read appended output incrementally with a bounded buffer, tolerate an incomplete final
JSONL line during a running job, and distinguish a partial live record from corruption
in a finished run. Do not repeatedly parse entire diagnostics files. Use a cursor for
incremental updates, small moving-marker updates, optional landmark level of detail and
lazy image loading. Display progress as processed exposure/output count when total is
unknown. Camera image exporting/rendering must not contend with a ranked inference GPU
run by default. Store viewer output below `results/<run>/viewer/`.

If a stop action is included, it must terminate only the owned job, retain its partial
outputs and mark the run stopped. Prefer graceful cancellation; any forced stop leaves
an incomplete run, never a successful submission. Completed, failed and interrupted
runs remain reopenable. Record UI/export versions separately from estimator provenance.

## Delivery order and acceptance

**1. Input/run foundation and passive viewer.** Add generic input mode, preflight and
run lifecycle; completed-run loading plus a basic 2D trajectory/status timeline. Pass:
current tuning CLI remains unchanged, an input with unrelated epoch works from its one
initial pose to EOF without GT files, incompatible calibration fails explicitly, and
displayed poses/timestamps match every CSV row. Test failed/partial runs and repeat Start.

**2. Run monitoring.** Launch, process state, incremental timeline and follow/pause/seek.
Pass: the same input produces the same timestamps/poses with UI attached or detached;
closing the browser does not kill inference; missing image/partial line/worker fault
cannot freeze the UI or fabricate a good state. Report observer overhead separately.

**3. Diagnostic 2D review.** Add optional images, existing evidence details, event jumps,
offline GT comparison, error plots and exports. Pass: exposure synchronisation and
integer-time interpolation fixtures; zero GT access from inference; known loss/error
spikes stay visible; counters agree with saved diagnostics and scorer outputs.

**4. 3D context.** Add real XYZ geometry, bounded landmark cloud and valid auxiliary
body/rig poses. Pass: map axes/metres, transforms and height gaps verified with a small
known-geometry fixture; no vertical accuracy claim without actual GT height; usable 2D
fallback if WebGL is unavailable. Measure scene performance on the project laptop.

**5. Judge/demo package.** One documented local launch path, offline assets, fresh-input
smoke test, screenshot/report export, failure/recovery replay and separate run success
versus localisation-gate indicators. UI readiness never signs off the visual baseline.

Do not start full frontend implementation before these input/clock/data boundaries are
captured in small fixtures. Do not add camera calibration editors, map rebuilding,
threshold tuning, remote multi-user hosting or automatic hidden-set scoring to this UI.
The next implementation slice is 1, followed by 2; B's updated map should be selectable
as another verified frozen artifact without redesigning the UI.

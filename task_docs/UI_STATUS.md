# APEX UI: first implementation

**18 October automatic-input update:** judges now choose an MCAP in the browser and
press Run, with no path/pose/map/motion configuration. The UI extracts one supported
map-frame initial pose, selects the frozen release bank and detects permitted motion.
Embedded GT is exported only after inference exits and is used by the supplied scorer
and offline viewer. Minimal deliverables go to `results/ui-<id>/`; raw cache/GT/uploads
stay in `tmp/replay_ui/`. The original training recording is identified by SHA and uses
the fixed holdout. Manual controls remain under Advanced.

Verification: **1,144 tests passed**, Ruff and JavaScript/geometry checks passed. A real
browser picker/upload/run completed a 59-pose recording with automatic scoring and GT
comparison. A whole automatic holdout replay reproduced all 2,191 frozen predictions
exactly (0.655 m RMSE); its postprocessor was rerun after fixing an integration keyword.
See [automatic input receipt](evidence/automatic_ui_20261018.json). Unseen organizer
topic/schema variants and hidden-test accuracy remain unverified. The historical
implementation notes below describe the earlier manual interface.

18 October 2026. C-owned local run/replay interface. Stage 3/5 localisation gates remain
open; UI completion does not qualify accuracy, hidden data or real-time performance.

Launch `launch_ui.cmd`, then open <http://127.0.0.1:8765>. Full usage:
[replay_ui/README.md](../replay_ui/README.md).

## Implemented

- Dense paper/graphite interface with signal red, exposed rules and flat controls.
  Offline Barlow Condensed 600/700 and IBM Plex Sans 400/500/600 fonts and OFL notices.
- Local recording/artifact/initial-pose configuration, compatibility preflight, one
  production subprocess per UI server, live output monitoring and completed-run loading.
- Existing Tuning/Holdout presets are retained. `run.py --input-recording` supports
  another recording's own clock, its one initial pose and optional exact bounds/EOF.
  This reuses `EpisodeBounds`, the reader, filter, frontend, scheduler and submission
  writer. The UI defaults to Tuning for the known development recording and requires
  explicit bounds or the Holdout preset before crossing its reserved interval.
- Exact-nanosecond joins, incremental file tails, incomplete live-line handling, late
  diagnostic patches and visible corrupt-output errors. CSV remains pose authority.
  No camera-rate data or shared A/B records were changed.
- Equal-scale 2D trajectory/heading/borders; Lost markers, shared replay cursor, playback,
  step/scrub, follow-latest/follow-car, fit-map and event navigation.
- Selected pose/evidence, inliers, reprojection/grid support, rejected poses, camera
  exposure stamps, runtime timeline and recovery/tracker/error ledger.
- Raw front image inspection after completion, selected by exact camera, exposure and
  recorded arrival; no adjacent-frame substitution or invented feature overlay.
- Optional offline GT attachment only after completion. Integer interpolation, 50 ms
  bracket checks, wrapped heading, position and car-frame lateral/longitudinal errors;
  no alignment, smoothing, score-window cropping or GT path in the inference command.
- Real XYZ borders and valid auxiliary height in a rotatable, zoomable orthographic
  3D scene. A CPU Canvas renderer avoids unreliable WebGL capture observed in this
  Chrome session; height gaps remain gaps and no 3D GT error is claimed.
- Exact CSV export, locally saved scene PNGs and a standalone offline replay bundle
  through `render_replay.py`. Generated data stays under ignored `results/`.
- Loopback binding, strict static/export paths, same-origin ephemeral token for actions,
  bounded request/line sizes, no arbitrary shell field, CDN, analytics or project upload.

## Verification

- Browser-launched new-recording mode, bounded to the first 20 s from the tuning initial
  pose: `results/ui-edd897df42`, **200 outputs**, zero frontend/worker faults. Every CSV
  row matches the corresponding saved full-tuning baseline prefix exactly. That is a
  bounded smoke test, not a new full localisation or rate qualification.
- Full saved run: all **2,708** outputs load and offline comparison reproduces
  **12.892825533 m RMSE**, including known drift/losses. Reference matching is diagnostic;
  the existing official scorer remains separate and unchanged.
- Browser checks: setup/start/monitor/completion, saved selection, camera image loading,
  offline GT comparison, 2D/3D switching, 3D drag, saved 2D/3D PNGs, an API-free
  offline bundle, and the 390 px layout without horizontal overflow.
- Final Python regression suite: **1036 passed**, no skips/xfails, in 179.32 s.
  Command: `.venv/Scripts/python.exe -B -m pytest -q --basetemp tmp/ui-ledger-final-suite`.
  Shareable receipt: [ui_verification.json](evidence/ui_verification.json).
- `node tests/check_ui_geometry.cjs`: known XYZ fixture verifies metre scale, axis signs
  and broken paths through unavailable height. Both frontend scripts pass syntax checks.
- Ruff passes for `camloc scripts tests run.py render_replay.py`. Plotly is promoted
  from its existing transitive lock entry to a direct 7.1.0 pin; no resolved dependency
  version changed. Lock sync and requirements export pass offline.

## Remaining scope and limits

The current 3D scene has surveyed borders and auxiliary poses, not a dense road surface
or a full landmark-cloud inspector. Detailed observed pixel tracks cannot be recovered
where historical diagnostics store only array shape/dtype; a bounded opt-in sidecar
would require separate B coordination/overhead checks. The first camera preview is front
only. Live non-pausable sensor ingestion, forced cancellation, multi-user hosting and
calibrated uncertainty are outside this first UI slice.

Hidden recordings have not been supplied. The rig/static-map compatibility gates still
apply, and organisers' exact launch/transfer format is unconfirmed. `validate_run.py`
remains the unchanged development-split gate; a generic input run is not given an
official score or visual-baseline signoff by the UI. Sustained observer-overhead and
broader browser/device qualification remain open. Holdout was not run.

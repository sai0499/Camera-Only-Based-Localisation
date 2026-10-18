# APEX local workbench

Double-click `launch_ui.cmd` at the project root, or run:

```powershell
& .\.venv\Scripts\python.exe -B scripts/serve_replay.py --port 8765
```

Open <http://127.0.0.1:8765>. The server binds only to loopback. All JavaScript and
fonts are local; there is no CDN, cloud upload, npm build or analytics.

The workbench opens a saved tuning run when available. Choose **New run** for a local
MCAP, frozen artifact directory and one initial pose. Paths can be absolute or relative
to the repository. Keep `timestamp_ns` quoted in JSON to preserve exact nanoseconds.
New-recording mode starts at that pose and runs to EOF, or to optional explicit bounds.
Tuning/holdout presets retain their frozen intervals. Production verifies all artifact
hashes during startup; a different rig/calibration is rejected rather than adapted.

**Motion input** defaults to camera-only. Choose **Training state velocity whitelist**
for the original development recording, or **Validation velocity topic** when the input
contains `/constructor0/validation/twist`. This explicit mode uses only timestamp and
body vx/vy/yaw rate, never recorded pose, orientation, status or track coordinates.
The real hidden test still uses the judges' one initial pose. The supplied camera lap
is an unscored lighting/speed stress test with an experimental image-derived initial
pose. Geometry-bound banks accept an unchanged selected camera even if an unused
camera's mount differs; they require exact raw calibration and preprocessing proof.

During a run, **Follow latest** follows newly written outputs. Pausing/stepping/scrubbing
changes the viewer only; closing the browser does not stop inference. Only one owned
inference job may run at a time. The UI does not currently offer forced cancellation.
Camera image inspection is enabled after completion and uses exact camera/exposure/log
timestamps from that run; it does not resample images or recompute feature evidence.

**2D / Track** shows equal-scale metre coordinates, surveyed borders, current heading
and the trail up to the selected output. **3D / Geometry** uses a CPU orthographic renderer for real border XYZ and
valid auxiliary height, preserving gaps; it does not manufacture a 3D GT trajectory.
**Evidence / Log** navigates recorded tracker/recovery/error events. Rejections and
height provenance are in the selected-output inspector.

Attach an optional GT CSV only after completion. It is read by the offline viewer,
never sent to the estimator. The comparison uses existing integer-time interpolation,
50 ms bracket checks and wrapped heading errors. The header RMSE refers to all matched
outputs, not the visible replay prefix; it is a diagnostic, not the official scorer.
Use `evaluation/reference_eval.py` separately for a ranked development-split report.

Space toggles replay; arrow keys step when a form control is not focused. Plot clicks
seek; **Next loss** jumps to the next loss onset. Export the exact prediction CSV or a
scene PNG (saved below the run's `viewer-snapshots/` directory). Animation speed and UI
refresh rate do not measure localizer throughput.

Export an independent offline bundle into a fresh ignored result directory:

```powershell
& .\.venv\Scripts\python.exe -B render_replay.py --run results/your-run --output results/your-run/viewer
# Optional offline GT comparison:
& .\.venv\Scripts\python.exe -B render_replay.py --run results/your-run --output results/your-run/viewer-with-gt --gt path/to/gt.csv
```

Open its `index.html`. Keep the folder together; it includes private replay data and
must not be committed or uploaded. The bundle is passive: launching runs, attaching
new GT and fetching raw camera images require the local server.

Design: warm paper/graphite with signal red, flat grid rules, no shadows or gradients.
Headings use Barlow Condensed 600/700; body/controls use IBM Plex Sans 400/500/600.
The bundled fonts have SIL OFL notices and source hashes in `fonts/`.

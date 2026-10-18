# APEX local workbench

Double-click `launch_ui.cmd` at the project root, or run:

```powershell
& .\.venv\Scripts\python.exe -B scripts/serve_replay.py --port 8765
```

Open <http://127.0.0.1:8765>. The server binds only to loopback. All JavaScript and
fonts are local; there is no CDN, cloud upload, npm build or analytics.

The judge workflow is **New run → Choose MCAP → Run localization**. The browser
uploads the file to this computer only. No path, map, split, pose JSON or motion setting
is required. The frozen seven-camera rescue bank is selected automatically. One
map-frame initial pose is extracted from a dedicated initial-pose topic, or the first
supported embedded reference pose. Supported velocity/yaw-rate data is detected
automatically; recordings without it use camera-only inference. Acceleration is not read.
Only that one initial pose and the existing velocity whitelist reach the localizer.
Production still verifies frozen artifact hashes and camera compatibility at startup.

After inference finishes, embedded ground truth is extracted separately and the supplied
scorer runs automatically. The trajectory/error plots gain a GT comparison and
**Download validation** becomes available alongside **Download CSV**. Minimal pose,
validation and runtime deliverables are saved to `results/ui-<id>/`; uploaded inputs,
raw logs and offline GT remain in ignored `tmp/replay_ui/`.

Supported reference sources include direct `map → constructor0/base_link` dynamic TF,
named ground-truth/validation pose topics using ROS 2 PoseStamped,
PoseWithCovarianceStamped or Odometry, and the development StateEstimation layout.
Frames and schemas are checked; missing/unsupported initialization fails explicitly.
Uploading the byte-identical original training recording automatically selects the
fixed holdout interval, preventing replay of fitting data as a new test.

The UI has no manual pose, map, interval or motion controls. Missing/unsupported
initialization fails explicitly. Direct CLI tooling remains available for development.
Motion assistance reads only timestamp and body vx/vy/yaw rate. The supplied camera lap
is an unscored lighting/speed stress test with an experimental image-derived initial
pose. Geometry-bound banks accept an unchanged selected camera even if an unused
camera's mount differs; they require exact raw calibration and preprocessing proof.

During a run, **Follow latest** follows newly written outputs. Pausing/stepping/scrubbing
changes the viewer only; closing the browser does not stop inference. Only one owned
inference job may run at a time. The UI does not currently offer forced cancellation.
Camera image inspection is enabled after completion and uses exact camera/exposure/log
timestamps from that run; it does not resample images or recompute feature evidence.
Use **Inspect camera** to choose any feed recorded in that output. A `support-v1` bank
selectively decodes mapped auxiliary views, records camera-priority decisions in the
evidence log and excludes changed camera geometry individually. The default artifact
choice is the seven-camera rescue bank when it is available locally.

The retained development replay lives at `tmp/replay_ui/holdout/` and appears as
`preview/holdout` in the run selector. The server supports downloads, camera images and
snapshots for this cache as well as newly generated runs. `results/holdout/` contains
only the required pose/validation/runtime files and trajectory plot. Edit UI source in
`replay_ui/`, not the generated `viewer-with-gt/` copy inside the temporary cache.

**2D / Track** shows equal-scale metre coordinates, surveyed borders, current heading
and the trail up to the selected output. **Evidence / Log** navigates recorded events.
The inspector shows pose/evidence, current camera exposures and the selected recorded
image. Camera previews refresh throughout playback and settle on the selected frame
when paused or scrubbed. The 3D view and rejection/height panel have been removed.

Embedded GT is attached automatically after completion. The comparison uses existing
integer-time interpolation, 50 ms bracket checks and wrapped heading errors. The header
RMSE refers to all matched outputs, not the visible replay prefix; it is a diagnostic,
separate from the supplied scorer available through Download validation.

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
must not be committed or uploaded. The bundle is passive: launching runs and fetching raw camera images require the local server.

Design: warm paper/graphite with signal red, flat grid rules, no shadows or gradients.
Headings use Barlow Condensed 600/700; body/controls use IBM Plex Sans 400/500/600.
The bundled fonts have SIL OFL notices and source hashes in `fonts/`.

## Input browsing, ground truth and camera images

- The MCAP file picker works through the browser. All input configuration is automatic.
- Automatic runs attach only their own MCAP's embedded reference after completion; there is no manual GT attachment control.
- The recorded image of the chosen camera is always shown for the selected output (falls back to the pose camera or `fr` when that camera is missing). Images load only for completed runs so inference is not contended.

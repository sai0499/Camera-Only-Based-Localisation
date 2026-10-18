# TrackCode APEX — race-track localization

Estimate the car's position and heading from a ROS 2 MCAP recording and replay the
result against its embedded ground truth. The local web interface handles input
selection, initialization, localization, scoring and pose/validation downloads.

**Start here:** copy the supplied frozen map into the project, then double-click
**[start_project.cmd](start_project.cmd)**. It installs/checks the locked environment
and public weights, starts the UI and opens <http://127.0.0.1:8765> in your browser.
Keep its console window open while using the application. Start only one UI server.

## Requirements and supplied assets

- Windows 10/11 x64. The locked environment uses Python **3.12.13**.
- Internet for first-time setup and checkpoint downloads; roughly **20 GB free disk**,
  plus space for MCAP uploads and outputs. Subsequent localization runs are local.
- A compatible NVIDIA GPU/driver for the measured performance. Our reference machine
  uses an **RTX 4060 Laptop GPU, 8 GB**, on Windows 11. Setup does not install drivers.
  CPU component checks are supported; CPU localization throughput is not qualified.
- The separately supplied frozen map directory. **A Git clone alone is insufficient:**
  setup downloads pretrained weights but does not download or rebuild this fitted map.

Place the complete directory at these exact paths, without renaming its contents:

```text
artifacts/
  fit-seven-rescue-v2/
    manifest.json
    map.npz
    retrieval.json
    retrieval.npz
```

Use the supplied four files together. Their hashes, camera geometry and retrieval
identity are checked during localization startup. Do not build a map from test data.
Pinned pretrained checkpoints are prepared in `artifacts/weights/` by setup; their
download URLs, sizes and SHA-256 hashes are in
[config/pretrained_models.json](config/pretrained_models.json). Preserve the
[model/code notices](config/MODEL_NOTICES.md).

No preinstalled Python, uv, ROS installation, CUDA Toolkit, npm or administrator
account is required by the project setup. First-time downloads can take several
minutes. The environment and caches remain inside this project.

## One-click setup and launch

1. Clone/download the submitted **`dev/implementation`** branch and extract it into a
   normal local folder. Copy the frozen map directory above into `artifacts/`.
2. Double-click **`start_project.cmd`**. It checks the map files, runs `setup.cmd`,
   verifies dependencies/checkpoints and component smoke checks, and launches the UI.
3. If the browser opens before the server is ready, refresh once. The URL is
   **http://127.0.0.1:8765**. Setup errors remain visible in the console.

For later launches, double-click **[launch_ui.cmd](launch_ui.cmd)** and open that URL.
To install/check the environment separately, double-click **[setup.cmd](setup.cmd)**.
Rerunning setup is safe; it uses the committed dependency lock and checksum-pinned
weights. It never installs packages globally. Do not run multiple launchers at once.

## Run a test recording

1. Click **New run** and **Choose file**; select the test `.mcap` from your computer.
2. Click **Run localization**. No manual map, pose, interval or motion configuration
   is needed. Upload is to this computer's loopback server, with no cloud upload.
3. Monitor the run. **Follow latest** follows newly produced poses. Camera plus
   supported body velocity/yaw rate is selected automatically; recordings without
   motion data use cameras alone. Acceleration is not consumed.
4. After localization finishes, embedded GT is evaluated separately. The 2D trajectory,
   error timeline and position RMSE then show the comparison when GT is available.
5. Click **Export prediction CSV** and **Download validation results**. The validation
   download is the supplied scorer's text report. Full minimal deliverables also
   appear under `results/ui-<id>/`.

The MCAP must contain supported compressed camera images, camera calibration/static
transforms and a supported map-frame initial pose or reference pose stream. The UI
extracts **exactly one initial x/y/yaw pose**. Supported reference sources include
`map -> constructor0/base_link` dynamic TF, named ground-truth/validation pose topics
with ROS 2 PoseStamped, PoseWithCovarianceStamped or Odometry, and the development
StateEstimation layout. Supported motion sources are `/constructor0/validation/twist`
and the whitelisted `/constructor0/state_estimation` velocity layout.

Unknown initialization schemas/frames, invalid motion schemas, corrupted files or
incompatible map geometry fail explicitly. A recording with a supported initial pose
but no GT can localize; it cannot produce a GT accuracy score. Automatic detection is
limited to the supported formats, not arbitrary ROS topic names or custom schemas.

The original development recording is recognized by content hash and runs only its
fixed holdout interval. The separately exported `holdout-evaluation.mcap`, when supplied,
contains that interval and can be selected directly. Keep its original timestamps.
The original training recording is not required to run a separately supplied test.

## Replay and downloads

- **2D / Track:** estimated trajectory, embedded GT, surveyed borders and selected pose.
  Scroll to zoom, drag to pan, and use **Fit map** to reset. Manual navigation disables
  Follow car so the plot stays where you move it.
- **Playback:** play/pause, speed selection, stepping, timeline seeking and Next loss.
  Recorded camera previews update during completed-run replay; select any recorded
  camera. Camera inspection waits until inference completes to avoid contention.
- **Inspector:** selected pose, camera exposures and evidence appear alongside the
  plot on wide screens. **Evidence / Log** shows recorded tracking/recovery events.
- Closing the browser does not cancel localization. Keep the server console open.
  Stop the server with **Ctrl+C** after the run and scoring finish.

```text
results/ui-<id>/
  submission.csv          # timestamp_ns,x,y,yaw
  runtime.json            # measured runtime and memory
  run_metadata.json       # inputs, map/model identities, mode and provenance
  validated.json          # structural, exposure-sequence and visual-gap checks
  validation_status.json  # post-run evaluation status
  validation/score.txt    # supplied scorer's accuracy/coverage report
```

Timestamps are integer nanoseconds, x/y are metres in the map frame, and yaw is radians.
Published results are Git-trackable. Uploaded MCAP copies, raw diagnostics, logs and
offline GT stay under ignored `tmp/replay_ui/`; do not distribute those private caches.
Embedded GT trajectories are never passed to the localization subprocess. Only the
one initialization pose and permitted motion whitelist cross into inference.

## Method and measured results

The frozen fitting-only map covers seven cameras. DISK features, LightGlue matching
and geometric PnP verification provide absolute visual poses. LK/current-image PnP
tracks mapped points between bounded anchor jobs; EigenPlaces retrieves recovery
references. One temporal filter integrates visual evidence and, when available,
causal body velocity/yaw rate. Rejected evidence and prediction-only outputs are logged.
There is no test-time map fitting, GT alignment or offline smoothing of predictions.

Reference whole-holdout regression, **2,191 poses / 219.2 seconds**, camera + motion:

| Metric | Result |
|---|---:|
| Position RMSE | **0.655 m** |
| Lateral RMSE | **0.330 m** |
| Yaw RMSE | **0.239 degrees** |
| Positions within 1 m | **86.5%** |
| Official coverage, 1 s bins | **100%** |
| Reference average runtime | **73.33 ms per synchronized camera group** |

Runtime includes reader/service waiting; startup is separate. It is per synchronized
camera group, not per individual camera image. Fresh runs vary: consult that run's
`runtime.json` field `elapsed_per_output_ms` and its own scorer report. The UI's
processing readout describes the selected group, not the whole-run average.

This holdout was inspected previously and is a regression, not an unseen hidden-test
claim. Its longest full-visual-pose gap is **9.8 seconds**, so the unchanged visual
continuity gate still fails despite valid output and 100% scoring coverage. Motion
propagation is not counted as a visual update. Camera-only accuracy differs from the
assisted table. Hidden-test schemas, accuracy and native real-time rate remain unverified.
See the [reference receipt](task_docs/evidence/submission_holdout_20261018.json) and
[camera/map report](task_docs/MULTICAMERA_IMPROVEMENTS.md).

## Troubleshooting and development

- **Missing frozen map:** obtain the supplied `fit-seven-rescue-v2` directory and copy
  all four files into `artifacts/`. Public pretrained weights do not replace the map.
- **Setup/download error:** check internet/disk space and rerun `setup.cmd`. See the
  console and `.setup/environment_report.json`. Do not use an unpinned package upgrade.
- **Port already in use:** reuse the existing local UI. Stop its console before
  starting another server. Refresh the page after restarting the server.
- **No initial pose / incompatible calibration:** use a supported recording with its
  original calibration and a map-frame initial pose; the system does not invent a pose.
- **Validation pending/failed:** wait for postprocessing; inspect the run's
  `validation_status.json` and the local log under `tmp/replay_ui/inputs/<id>/`.
- **No camera image:** complete inference first and retain its uploaded MCAP/cache.

For developers, use the project interpreter and pinned environment:

```powershell
& .\.venv\Scripts\python.exe -B -m pytest
& .\.venv\Scripts\ruff.exe check camloc scripts tests
& .\.venv\Scripts\python.exe -B scripts/serve_replay.py --port 8765
```

The latest full regression suite passed **1,144 tests**; UI-only follow-up checks also
passed. These checks do not replace measured recording accuracy. No setup rerun on a
fresh judge machine is implied by those results.

Further details: [UI guide](replay_ui/README.md), [CLI submission guide](SUBMISSION.md),
[evaluation guide](evaluation/README.md), [build specification](task_docs/FINAL_BUILD_PLAN.md),
and [original organizer dataset README](README.md), preserved unchanged.

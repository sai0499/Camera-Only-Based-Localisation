# TrackCode APEX — race-track localization

Estimate the car's position and heading from a ROS 2 MCAP recording and replay the
result against its embedded ground truth. The local web interface handles input
selection, initialization, localization, scoring and pose/validation downloads.

**Start here:** place the original training MCAP in the project root, then double-click
**[start_project.cmd](start_project.cmd)**. It installs/checks the locked environment
and public weights, builds the frozen map if missing, starts the UI and opens
<http://127.0.0.1:8765> in your browser.
Keep its console window open while using the application. Start only one UI server.

## Requirements and supplied assets

- Windows 10/11 x64. The locked environment uses Python **3.12.13**.
- Internet for first-time setup and checkpoint downloads; roughly **20 GB free disk**,
  plus space for MCAP uploads and outputs. Subsequent localization runs are local.
- A compatible NVIDIA GPU/driver for the measured performance. Our reference machine
  uses an **RTX 4060 Laptop GPU, 8 GB**, on Windows 11. Setup does not install drivers.
  CPU component checks are supported; CPU localization throughput is not qualified.
- For a first map build, the original training recording named
  `c0_2026-10-13_22-53-42_camloc_0.mcap` in the project root. It is approximately
  7.53 GB and is transferred separately under the organizer's dataset access rules;
  **the recording is not tracked in Git**. A dataset already present in your local
  checkout can be used directly. A test/holdout clip cannot replace it.
- Keep the committed `task_docs/build_split.json` and supplied
  `23-00-38_cut_120m_update/map.mcap` unchanged. The build verifies their hashes.
- Alternatively, a complete prebuilt release map can be supplied at the paths below;
  a valid existing bank is reused without rebuilding or requiring the training MCAP.

The automatic build creates these files. A supplied prebuilt bank uses the same paths:

```text
artifacts/
  fit-seven-rescue-v2/
    manifest.json
    map.npz
    retrieval.json
    retrieval.npz
```

Keep the four generated/supplied files together. Their hashes, camera geometry and retrieval
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
   normal local folder. Place the original training MCAP at the project root, or
   supply the complete prebuilt bank under `artifacts/fit-seven-rescue-v2/`.
2. Double-click **`start_project.cmd`**. It runs `setup.cmd`, verifies dependencies,
   checkpoints and component smoke checks, then builds/verifies the map and launches
   the UI. A first map build can take many minutes on a GPU; it is not instant.
3. If the browser opens before the server is ready, refresh once. The URL is
   **http://127.0.0.1:8765**. Setup errors remain visible in the console.

For later launches, double-click **[launch_ui.cmd](launch_ui.cmd)** and open that URL.
To install/check the environment separately, double-click **[setup.cmd](setup.cmd)**.
Rerunning setup is safe; it uses the committed dependency lock and checksum-pinned
weights. It never installs packages globally. Do not run multiple launchers at once.

## Build the frozen map from the training dataset

The launcher invokes the existing builders through
[scripts/build_submission_assets.py](scripts/build_submission_assets.py). To prepare
assets without opening the UI, first run `setup.cmd`, then:

```powershell
& .\.venv\Scripts\python.exe -u -B scripts/build_submission_assets.py
```

The source recording must have SHA-256
`fe5a0b9565b1bc03dd1ad775645722645e135b2f19012cf7bc815a8bcb91d613`.
The script refuses another dataset, including an uploaded test or the holdout clip.
It uses the published recipe, with times relative to the fixed fitting interval:

1. Build the front-camera bank over **0?700.39 seconds**, with 4 m / 5-degree
   keyframe selection and eight-neighbor matching.
2. Build the other six cameras over fitting windows **265?274, 299?309 and 372?379
   seconds**, using 5 m / 10-degree selection, rescue masks and 12 m target-baseline
   pairing.
3. Merge the disjoint camera banks using the existing 3,000-reference bound and
   fitting/provenance checks.
4. Build and verify the EigenPlaces retrieval index, then publish the completed bank
   to `artifacts/fit-seven-rescue-v2/`.

All triangulation pose support is limited to the fitting interval. Tuning, holdout
and test GT are not used for fitting. Neither `evaluation/reference/` nor
`evaluation/reference_reconciled/` is needed by this build. The supplied static map
is read for the existing provenance/geometry contract; it is not the visual bank.

Intermediates are kept in ignored `artifacts/submission-map-build/`. Existing frozen
banks are verified and reused, never overwritten. If a build fails, its partial work
is preserved; read the error and rename that work directory before retrying. A partial
or invalid final bank also fails explicitly rather than being silently replaced.
Allow additional disk space for fitting intermediates.

A fresh build has its own manifest/map identity. GPU/library numerical differences
may affect its landmarks, so the historical RMSE table below is **not a guarantee for
an untested rebuild**. Validate the rebuilt bank on the fixed development split
before quoting its accuracy. The locked supplied bank remains the exact measured
reference; rebuilding does not overwrite it when it is already present.

### Evaluation-reference folders

The automatic UI reads its uploaded MCAP's embedded GT only after inference ends;
no pre-generated reference folders are needed. For development scoring through
`evaluation/reference_eval.py`, the active reference can be generated separately
from the original recording after setup:

```powershell
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py prepare
```

This creates `evaluation/reference_reconciled/` when absent and refuses to overwrite
an existing reference. `evaluation/reference/` is a legacy source cache and is not
required for this preparation. These folders contain evaluation GT, stay excluded
from Git, and must never become inference inputs.

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
After the map has been built, the original training recording is not required to run
a separately supplied test.

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

- **Missing frozen map:** run `start_project.cmd` with the original training MCAP
  at the project root. It builds the map automatically. A complete supplied prebuilt
  bank can also be used; pretrained weights alone do not replace the map.
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

The latest full regression suite passed **1,149 tests**; UI-only follow-up checks also
passed. These checks do not replace measured recording accuracy. No setup rerun on a
fresh judge machine is implied by those results.

Further details: [UI guide](replay_ui/README.md), [CLI submission guide](SUBMISSION.md),
[evaluation guide](evaluation/README.md), [build specification](task_docs/FINAL_BUILD_PLAN.md),
and [original organizer dataset README](task_docs/DATASET_README.md), preserved unchanged.

# Final localization research and design decisions

Review: 17 October 2026. Scope: a three-person, 48-hour camera-only race-track localization build. This review updates `../FINAL_BUILD_PLAN.md`; it does not begin implementation of the localizer. Source files, both organizer tool folders, all recordings, the challenge PDF and the existing coordinate reconciliation remain unchanged.

Environment-preparation follow-up: this research snapshot predates installation. The current shared lock, cached weights and component checks are described in `../../START_HERE.md`; its later setup evidence does not establish localizer performance.

## Decision

Keep geometric localization against a small fitted visual map, DISK/LightGlue, tracked mapped points, conservative border corrections and the six-state filter. Change the execution design: expensive matching runs at a bounded rate in a separate worker; the CPU tracking/PnP path processes camera groups. Add compact place retrieval before complete loss, one-pass canonical image remapping, unique correspondence checks, time-consistent filtering and a measured operating envelope for higher input rates. The additional map remains the canonical surveyed geometry and coordinate reference; the native-coordinate bridge remains active in evaluation.

This is a more complete deployment specification, not evidence of improved race-track accuracy. No CUDA feature extractor, mapper or camera localizer has been benchmarked on this machine. The detected device is an RTX 4060 Laptop GPU with 8,188 MiB and driver 617.42; there are 16 logical CPU processors. The existing evaluation environment lacks PyTorch, OpenCV and pycolmap. Research cannot establish their runtime or accuracy before installation and implementation.

## What the external evidence supports

Only primary papers, author repositories and maintainer documentation support the choices below. Published image-matching and urban-localization results are not measurements on Yas Marina. The table gives decisions now; it is not an invitation to continue experimenting with every model during the hackathon.

| Component reviewed | Evidence and relevance | Final decision |
|---|---|---|
| LightGlue | The authors provide adaptive matching and feature-specific weights for DISK, ALIKED, SIFT and SuperPoint. FlashAttention and adaptive depth address matcher cost. Their headline FPS is not a complete camera-localization pipeline. [Repository](https://github.com/cvg/LightGlue), [paper](https://arxiv.org/abs/2306.13643) | Keep DISK-compatible LightGlue; use it on bounded anchor jobs, not every input image. Keep the feature-specific checkpoint and native detector NMS. |
| ALIKED | A supported compact alternative with an hloc extractor. The implementation uses torchvision's deformable-convolution operation, which adds a binary compatibility check. It is not established as more accurate on these night images. [hloc adapter](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/extractors/aliked.py), [implementation](https://raw.githubusercontent.com/cvg/LightGlue/main/lightglue/aliked.py) | Do not change the default extractor. Moving expensive detection off the fast path addresses frequency without changing the map's descriptor space. |
| XFeat + LighterGlue | The authors release a genuinely lightweight feature pipeline. Their Aachen table gives different outcomes at different thresholds; it is not a universal accuracy improvement over DISK. An August 2026 reproduction reports sensitivity in downstream Aachen results. [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.pdf), [code](https://github.com/verlab/accelerated_features), [reproduction preprint](https://arxiv.org/abs/2608.09519) | Exclude from this build's critical path. Do not replace a night-oriented mapping backend purely on an extractor FPS claim. |
| 2026 sparse-matching analysis | A February preprint highlights detector/matcher compatibility and problems from nearby or differently sampled keypoints. Its cross-detector findings do not prove a fault in native DISK-to-DISK matching. [Preprint](https://arxiv.org/html/2602.08430v1) | Preserve trained preprocessing/NMS, remove duplicate observations, and never mix descriptor/checkpoint types. No fine-tuning. |
| UltraMatch, September 2026 | The paper reports attractive speed/memory results; the author repository explicitly states code and weights are not released. [Author repository](https://github.com/JiajunLe/UltraMatch) | Exclude. No unavailable component is required by the plan. |
| EfficientLoFTR and LiftFeat | Author implementations offer semi-dense matching or geometry-aware local descriptors. Neither establishes a drop-in advantage for the existing sparse, fixed-pose map on this track. [EfficientLoFTR](https://github.com/zju3dv/efficientloftr), [LiftFeat](https://github.com/lyp-deeplearning/LiftFeat) | Exclude new matcher integrations and monocular-depth dependencies from the 48-hour baseline. |
| EigenPlaces | Author checkpoints include ResNet18 with 256/512-dimensional descriptors; code is MIT. This supplies a small, explicit retrieval model. [Code](https://github.com/gmberton/EigenPlaces), [checkpoint interface](https://github.com/gmberton/EigenPlaces/blob/main/hubconf.py) | Replace the planned NetVLAD recovery default with EigenPlaces ResNet18/512. Cache reference descriptors and use a CPU dot-product index for this small map. This is a footprint/integration choice, not a claim of highest retrieval recall. |
| SALAD and MegaLoc | Strong retrieval alternatives with released implementations; their transformer backbones add a larger model than the selected ResNet18. MegaLoc also has an hloc adapter. [SALAD](https://github.com/serizba/salad), [MegaLoc model](https://github.com/gmberton/MegaLoc/blob/main/megaloc_model.py), [hloc configurations](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/extract_features.py) | Do not add another retrieval backbone. The selected model's failure is handled through geometry and explicit recovery status, not an unplanned model cascade. |
| Dense learned SLAM and DPVO | MASt3R-SLAM targets dense, globally consistent monocular SLAM; DPVO is a separate monocular VO/SLAM architecture. Their headline rates do not establish a 60 Hz, metric, fixed-map localizer on this laptop. [MASt3R-SLAM](https://arxiv.org/abs/2412.12392), [DPVO](https://github.com/princeton-vl/DPVO) | Do not replace the metric fixed-map pipeline or add a second SLAM engine. No dense foundation model, online bundle adjustment or trajectory smoothing is required. |
| PoseLib | Provides robust absolute-pose estimation with LO-RANSAC and nonlinear refinement, but adds another pose API/build dependency. [Author repository](https://github.com/PoseLib/PoseLib) | Keep OpenCV PnP/refinement already needed by the tracker. Correct correspondence identity, timestamps and geometry before changing solvers. |

These decisions avoid confusing component benchmarks with system guarantees. A newer method was considered only if its released implementation, map interface, calibration assumptions, hardware demands and license were relevant to the task.

## Concrete improvements adopted

### 1. Separate camera-rate tracking from sparse map refresh

The old plan allowed up to six expensive matches per input group. If used literally at 60 Hz, that permits 360 pairs per second. That is a bad workload contract regardless of matcher choice.

The revised plan gives one GPU worker a normal budget of three anchor jobs per second and a recovery ceiling of four total jobs per second. Each job matches at most two references. Retrieval runs inside those jobs and its cost counts toward their budget. One job is in flight and one latest request is pending. The camera path tracks at most 512 mapped/ground points and solves current-exposure PnP; it does not wait for fresh learned descriptors on every frame. This is our engineering inference from the existing sparse-map architecture, not a published speed result.

Anchor results contain 2D-to-3D identities at their source exposure. They are handed forward through buffered images to the current exposure, validated again, and used to solve a current pose. An old absolute pose is never applied as a current measurement. Logical delivery and main-thread dispatch prevent worker speed from changing which pending request is selected. The replay rule, queue limits and failure behavior are specified in the build plan.

A timestamp-only check also supports the 90 ms minimum delivery rule: all 11,943 eligible native front images reach the next exposure at 99.972-100.050 ms. An exact 100 ms cutoff would defer 344 of them to a second frame because of microsecond cadence variation. This avoids an artificial extra-frame delay; it is not a measured matching or localization latency.

### 2. Improve recovery without replacing the local feature bank

Use one compact EigenPlaces descriptor at 480 x 232 RGB, with documented normalization and the exact same preprocessing for fitting references and queries. Geometry still determines pose. Retrieval only proposes camera-compatible reference IDs. Retain the best ten candidates, verify at most two per job, and require a second geometrically valid group at least 50 ms later before accepting a distant reset.

Retrieval is warm at 1 Hz and runs immediately at the next eligible anchor slot when local support fails, up to 2 Hz during recovery. There is no compulsory one-second wait before beginning recovery. These are scheduling choices; recovery duration and false-reset rate remain mandatory measurements.

The author's EigenPlaces constructor initializes from CosPlace before loading EigenPlaces weights. The offline loader must construct the same architecture and load the complete local EigenPlaces state strictly, without invoking that redundant network download. This was found by reading the [author implementation](https://raw.githubusercontent.com/gmberton/EigenPlaces/main/eigenplaces_model/eigenplaces_network.py). Vendor the relevant MIT architecture/layers with notice; do not use unpinned remote `torch.hub` calls during a run.

### 3. Make estimator behavior consistent across input rates

The old discrete acceleration injection used a fixed acceleration variance every frame. Its accumulated velocity variance over one second falls from 3.6 at 10 Hz to 0.6 at 60 Hz when its numeric setting is unchanged. A faster camera would therefore make the motion model artificially more confident.

Use continuous white-acceleration spectral density q and, for each position/velocity pair, `Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]`. The same construction applies to yaw/yaw-rate. The maintainer's [reference implementation](https://raw.githubusercontent.com/rlabbe/filterpy/master/filterpy/common/discretization.py) distinguishes this from discrete per-interval acceleration noise. No FilterPy runtime dependency is needed.

The new design starts with translational q = 3.6 m^2/s^3 and angular q = (pi/3)^2 * 0.1 rad^2/s^3, retaining the old 10 Hz velocity-variance scale. Local numerical checks accumulate covariance at 10/20/30/60/120 Hz and agree within 9e-15 over one second. This verifies the mathematical design contract, not a filter implementation.

Replace fixed 0.5 motion smoothing with `alpha = 1-exp(-dt/0.2s)`. Use seconds for all loss/recovery timeouts. For repeated PnP observations from the same map, start with `R_eff = R_base * max(1, 0.1s/dt_since_last_accepted_PnP)`, plus the existing calibration floors. This conservatively bounds information per second from correlated mapped observations. It is not a claim of calibrated uncertainty; tune the declared noise constants on tuning only.

### 4. Reduce image and association overhead without weakening geometry

Cache a direct raw-to-960 x 464 remap using raw K/D and the correctly scaled target K. This performs distortion correction and downsampling in one interpolation. Its ideal rays match the previous two-stage pixel geometry; the pixel values are not numerically identical, so rebuild features and triangulation under a new preprocessing ID. The API supports a target camera matrix and output size. [OpenCV calibration/remapping documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

Keep keypoint coordinates and current image measurements in float32, world geometry/calibration in float64, and cached descriptors in float16 where supported. hloc's default half export can cast keypoints too, so export keypoints explicitly at float32. [hloc exporter](https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/hloc/extract_features.py).

Use each query keypoint and each landmark at most once in a PnP pool. Repeated matches through multiple reference images are not extra evidence. Retain spatial coverage, positive-depth, residual and degeneracy checks. Predict the next locations of known 3D points to initialize LK, then apply forward/backward checks; the prediction is a search aid, not accepted evidence. [OpenCV LK interface](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

Use LightGlue's native adaptive settings and automatic FlashAttention. Keep mixed precision and compilation off for the initial correctness baseline; enable them only through the fixed equivalence gate. The maintainers note hardware-dependent pruning overhead; do not assume pruning helps at 1,024 points. [Configuration and implementation](https://raw.githubusercontent.com/cvg/LightGlue/main/lightglue/lightglue.py).

### 5. Use measured packet behavior in grouping

The additional audit read only primary/rear message timing. There are 11,944 front-right images, 11,939 rear images and exactly five unmatched front images within the 5 ms exposure window. Paired log-arrival spread is median 4.140 ms, p95 11.699 ms, p99 15.299 ms and maximum 36.113 ms. These are original 10 Hz transport measurements, not measurements of a native 60 Hz camera.

At a hypothetical 60 Hz half-period timeout (8.333 ms), 17.68% of these recorded pairs would arrive too far apart. The final contract therefore uses exposure tolerance `min(5ms, 0.2*T)` and arrival deadline `min(40ms, 1.2*T)`, with at most four pending exposure-ordered groups. At 60 Hz the deadline is 20 ms, above the measured p99; it does not guarantee the same jitter distribution at a different input rate. A missing/late secondary never substitutes the next period's image. Late support is logged and excluded from the committed group.

## Completed evidence versus development gates

Completed now: primary-source review; full brief reread; host GPU detection; exact native-to-new-map reconciliation from the previous stage; packet-pair audit; analytic ray consistency; covariance/smoothing time-consistency checks. Measurements are retained in `../evidence/final_design_checks.json` and the reconciled evaluation bundle.

`../evidence/dependency_source_review.json` records exact upstream revisions, Python 3.12 Windows wheel metadata, CUDA wheel availability and official model release assets. These checks establish availability, not interoperability. No weights were downloaded and no localizer environment was installed. Section 6 of the build plan fixes the candidate versions and the 90-minute fallback decision; the successful setup smoke test must produce the actual complete dependency/weight lock. Keep the working evaluation environment separate.

Development gates are executable acceptance criteria with owners and fixed failure actions in `../FINAL_BUILD_PLAN.md`: GPU/library smoke test, small-map triangulation, held-out localization, asynchronous handoff, camera-rate throughput, causal prefixes, prohibited-topic independence, photometric/dropout stress and reproducible packaging. They are not unspecified research questions. The source-data rate is 10 Hz; native 30/60 Hz accuracy cannot be certified using duplicated images or accelerated replay.

No design can remove the physical loss of information from fully occluded cameras, indefinite motion without features, or unobservable along-track position on a uniform straight. The engineering response is specified: reject weak evidence, grow uncertainty, search for recovery and keep failures in the output/metrics. Calling those conditions solved without testing would undermine the judge-facing claims. The same applies to organizers' unpublished rules: package both coordinate representations, declare fitted assets, deny live GT/state input and preserve a strict border/ground-motion mode; do not invent a missing official decision.

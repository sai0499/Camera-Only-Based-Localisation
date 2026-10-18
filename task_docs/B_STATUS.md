# Teammate B status: Steps 4-8 and review of C's changes to B files

Branch `feature/b-step4-visual-support` from `dev/implementation` `93e42c0`, 18 October 2026.
All measurements below were taken on this machine (GTX 1650 4 GB, i5-9300H), not on the
RTX 4060 reference machine. Offline diagnostics read GT only for scoring; `camloc/` reads none.

**Summary.** C's Stage 3 changes to B files are accepted after review; no defects found.
Step 4's delayed-anchor pass test, Step 7's offline-startup check and Step 6's profile and
fixed fp16 comparison are done. Three measured alternatives were rejected and production
behaviour is unchanged: fp16 (fails the plan's rule), looser LK carry (more wrong poses, no
more accepted handoffs) and a stricter usable-only coverage rule (RMSE 9.44 -> 18.68 m). The
fix-5 configuration reproduces here (9.44 m RMSE, longest full-pose gap 5.5 s, 5 gaps > 3 s);
the Step 3 visual-baseline gate is **still open**.

## Review of C's changes to B files

| Change | Verdict | Basis | Notes |
|---|---|---|---|
| `camloc/retrieval.py` (EigenPlaces R18/512, index) | accept | code reading, C's recall figures | Vendored MIT notice and pinned revision; strict weights-only load with hash check; identical `preprocess` for references and queries; index bound to map id, map arrays hash, checkpoint, preprocessing and mask; camera and >= 30-landmark eligibility filters; no GT or time lookup. Used only for unsupported (recovery) proposals; healthy anchors keep geometric references (`frontend._proposal_refs`). |
| F2 joint handoff (`vision.py`) | accept | code reading, C's tests | Fires only after `track` rejected on the same image; landmark ids unique (active rows of handoff landmarks excluded); <= 256 rows; added track ids above the handoff's; the joint set is what settles. |
| Seeded handoff carry (`_prior_log`) | accept | code reading, C's tests | Only causal priors at already observed exposures; bounded deque; cleared on calibration change; unseeded retry on failure. C's own replay did not isolate a benefit. |
| `MAX_REFERENCES = 3000` (`build_map.py`) | accept | code reading | Still rejects instead of truncating; not a gate. The 2,456-reference v5 bank loads in ~6 s and the full worker keyframe cache stays bounded at 256 entries. |
| Usable-extent coverage grid (`pnp.py`) | accept | measurement (below) | Usable-only poses are worse than poses covering both grids, but reverting or tightening loses continuity: a stricter usable-only rule doubled RMSE in a full replay. Kept unchanged; the stricter rule stays a default-off knob. |
| Exactly-once settle, adoption consistency filter, PnP input checks | accept | code reading | No defect found. |

No defects requiring a code fix were found in C's changes.

## Coverage-grid measurement

`scripts/b_grid_check.py`, bank `fit-stage3-multilap-v1` + retrieval, tuning stretches
[40,70], [100,115], [135,150], [228,242] s (the recurring gap places), 370 anchor jobs. Every
source PnP and its one-delivery handoff was solved with the cell gate disabled; solves passing
all other gates were classified by the full-image and usable-extent 4 x 4 grids
(`results/b/grid-check-tune-1/`).

| Class | Count | Median error | p90 | Wrong (> 2 m) |
|---|---|---|---|---|
| both grids >= 6 cells | 284 | 0.23 m | 0.94 m | 1.1 % |
| usable-only (full < 6) | 175 | 0.63 m | 1.50 m | 7.4 % |
| full-only | 0 | — | — | — |
| neither | 80 | 1.70 m | 5.09 m | 45 % |

Within usable-only: full-image cells 3-4 give 11-25 % wrong; 5 give 3.8 %. Usable cells
>= 8 give 0 % wrong (57 poses, median 0.49 m). Reverting to the full-image grid would drop 175
mostly correct poses at exactly the places where support is weakest, so the change stays.
`PnPGates.min_usable_cells_without_full` (default 0 = today's rule) requires >= N usable cells
when the full-image grid has < 6. N = 8 was replayed on the full tuning episode
(`tune-b-cov8`, table below): it removed every full-pose row > 10 m but cut full-pose updates
from 1,794 to 1,433, lengthened the longest gap from 5.5 to 9.2 s and raised x/y RMSE from 9.44
to 18.68 m. The default therefore stays 0. `PnPSolution.full_image_cells` now reports the
full-image count for diagnostics.

## Handoff carry probe

`scripts/b_handoff_probe.py`, same bank, stretches [40,55], [100,115], [135,150], [228,242],
[255,264] s, 312 unseeded single-delivery handoffs (`results/b/handoff-probe-tune-1/`).

| Variant | Accepted | Wrong > 2 m | Median error | Carried median |
|---|---|---|---|---|
| base (21 px, 4 levels, FB 1 px) | 183 | 6 | 0.37 m | 106 |
| FB 2 px | 182 | 10 | 0.35 m | 110 |
| window 31, 5 levels | 183 | 10 | 0.54 m | 111 |
| window 31, 5 levels, FB 2 px | 177 | 9 | 0.54 m | 112 |

One-transition loss in base: 75 % of source points kept; 21 % fail forward/backward > 1 px, 4 %
LK status, < 0.1 % mask. Loosening LK keeps a few more points but adds wrong poses and no
accepted handoffs, so production LK is unchanged. Handoff rejections are dominated by PnP gates
on what is carried (`poor_image_coverage`, `high_median_error`, `degenerate_geometry`), i.e.
weak map support at those places, not by LK.

## Step 4: delayed anchors at 10/20/30/60 Hz

`tests/test_step4_delayed_handoff.py` (25 cases, B's real `VisionTracker` on rendered scenes):
results delivered at source + 90 ms, 150 ms and up to 250 ms are re-solved on the current image
(error < 0.05 m, never the source pose) with 1-4 transitions at every rate; > 0.25 s is
`stale_result`; an unobserved source is `missing_image_chain`; earlier evidence records stay
field-for-field unchanged after a later delivery; a backend exception propagates (C maps it to
`backend_fault`) instead of becoming a result. Rear rescue remains C's frontend scheduling;
C's measured front+rear bank (`fit-stage3-dual-v1`) was worse (37 m RMSE, 9 of 10 wrong locks
from `r`), so the recommended configuration stays front-only.

## Step 6: profile (fitting clip, GTX 1650)

`scripts/b_profile_vision.py --episode fit --start-s 640 --end-s 690 --stride 4 --fp16`,
500 images, 125 anchor jobs (`results/b/profile-fit-1/profile.json`).

| Stage | Mean | p50 | p95 |
|---|---|---|---|
| Decode + one-pass remap (A) | 25.1 ms | — | — |
| Retrieval query (EigenPlaces) | 38.1 ms | 36.8 ms | 50.7 ms |
| DISK extraction | 208.8 ms | 208.4 ms | 212.2 ms |
| LightGlue match (per reference) | 106.4 ms | 121.6 ms | 149.3 ms |
| Source PnP | 6.4 ms | 6.3 ms | 9.2 ms |
| Whole anchor job (2 references) | 483 ms | 466 ms | 581 ms |
| `observe` (gray history) | 0.6 ms | 0.6 ms | 1.2 ms |
| `track` (LK + PnP) | 40.5 ms | 35.6 ms | 84.1 ms |
| `handoff` (carry + PnP) | 40.7 ms | 40.2 ms | 76.9 ms |

Startup: calibration read 31.6 s (A's reader over the bag), map load 5.7 s, DISK/LightGlue
3.96 s, EigenPlaces + index 5.5 s. Peak GPU allocation 794 MiB (budget 6 GiB). All design
budgets are missed on this GPU (anchor 80 ms, fast path 8/13 ms, handoff 6 ms); the fast-path
numbers include CPU contention from the concurrent worker and are not paced qualification.

**fp16 comparison** (plan rule): accepted rate equal (0.92), but pose difference up to 1.09 m /
0.60 deg (limit 0.02 m / 0.05 deg) and p95 anchor latency 1,443 ms vs 581 ms (the GTX 1650 has
no fp16 tensor cores). Fails three of four checks; production stays fp32.

## Step 7: offline startup

`tests/test_offline_startup.py`: the pinned backend, EigenPlaces, a retrieval index, the
worker and the tracker start and run one anchor job with sockets, `urllib` and
`torch.hub.download_url_to_file` blocked; corrupted checkpoint hashes fail instead of
downloading. `TORCH_HOME` resolves to `artifacts/weights`.

## Step 8: notices

`config/MODEL_NOTICES.md` now records the vendored EigenPlaces architecture and how each
pretrained model is used (no training on project data).

## Tuning replays

Same commands as `STAGE3_STATUS.md`, fresh output directories, tuning episode, unchanged
contracts/gates/evaluator/validator.

| Run | Bank / change | x/y RMSE | Full-pose updates | Longest gap | Gaps > 3 s | Full-pose rows > 10 m | Gate |
|---|---|---|---|---|---|---|---|
| fix-5 (C, RTX 4060) | multilap-v1 | 9.665 m | 1,755 | 5.40 s | 4 | 2 | fail |
| `tune-b-repro-fix5` | multilap-v1, unchanged code | 9.441 m | 1,794 | 5.50 s | 5 | 3 | fail |
| `tune-b-cov8` (diagnostic) | + usable-only rule N = 8 | 18.681 m | 1,433 | 9.20 s | 8 | 0 | fail |

Submission SHA-256: `tune-b-repro-fix5`
`eea6db965b384e43d1caca87df12737b37e0c1c39568d909d682689701eff778`, `tune-b-cov8`
`ed54d96ac79482c395f64cbdec7e0bb7badd47412029e5a957ba5f125a870f67` (results stay local and
ignored).
Both runs pass the other four validator checks (valid CSV, exposure sequence, coverage 1.0,
metric sections). `tune-b-cov8` used `scripts/b_run_variant.py`, which patches the gate in-process
and records `b_variant.json`; it is a diagnostic ablation, not a recommended configuration.
Recommended configuration is unchanged: `fit-stage3-multilap-v1` with
`retrieval:acaaf3f6edd7174e`, production code as on this branch (all new knobs off).

`tune-b-repro-fix5` gaps (s after start): 106.4-109.5, 139.0-142.0, 142.2-146.6,
231.9-237.4, 258.7-261.9. Inside the gaps: front tracking mostly has no active tracks; 77
recovery jobs ran (17 `pnp_too_few_inliers`, 7 `insufficient_matches`); 34 of 50 handoffs
carried fewer than 30 points. Handoff rejections over the run: `poor_image_coverage` 73,
`high_median_error` 47, `too_few_correspondences` 34, `degenerate_geometry` 25. Anchor jobs
stall the scheduler on this GPU (853 of 853), which costs wall time but not ranked results.

## Remaining limitations and next steps

- The visual-baseline gate is open: five gaps of 3.0-5.5 s remain at the same places.
- The limit is map support at those places (few, clustered landmarks; repeated structure),
  not LK carry. Next B options: per-place densification with extra fitting passes or a finer
  keyframe step only in those windows, and rear support with stricter rear gates.
- Ground motion remains disabled (C's qualification on fitting pairs failed).
- Runtime budgets need the RTX 4060 (or better) to be meaningful.

## C integration review and paired verification

PR #2 is integrated into `dev/implementation` with C's subsequent runtime/UI changes
preserved. [B_PR2_INTEGRATION.md](B_PR2_INTEGRATION.md) and
[the receipt](evidence/b_pr2_integration.json) record the separate local verification.
C fixed four diagnostic issues: zero-evidence/mismatched-image FP16 adoption, out-of-split
stretch requests, GT-loading time included in image timing, and an offline test that could
pass without reaching reference matching. Production gate defaults remain unchanged;
the optional stricter coverage rule stays off.

Full suite: 1,085 passed. Full paired v5 tuning replays are byte-identical (12.893 m RMSE,
6.2 s longest full-pose gap); the supplied camera-lap experimental segment is also identical.
This PR supplies verification tools and review decisions, not the requested landmark
densification or a localization improvement. The visual-baseline gate remains open.

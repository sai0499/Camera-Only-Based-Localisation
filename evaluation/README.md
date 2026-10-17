# Reconcile the recording with the additional map

The additional `23-00-38_cut_120m_update/map.mcap` is the target map for track coordinates. Its reference Paths and the organizer's coordinate helper and evaluator remain unchanged. The recording's original native fields are actively used through this conversion:

```text
recorded idx, ds, n -> frozen native segment frame -> shared map x/y -> supplied-map s, n, epsi
```

The recording uses 2,929 native segment frames per route, while the supplied Paths contain 2,960 poses. Reusing an old index directly in the new Path produces the large discrepancy. `native_bridge.py` recovers the original segment frames from fitting-period `(idx,ds,n,x,y)` pairs, freezes them, and converts later native coordinates without reading their x/y. Both the track and pit coordinates must reconstruct the same world position. The supplied map then defines the final progress, lateral offset and heading error.

Every native segment has at least 13 fitting observations. A near-unit similarity retains the native coordinate arithmetic's tiny scale term (approximately 1 +/- 0.00000004). Independent reconstruction errors are around double-precision roundoff, and final supplied-map coordinates agree with direct use of the organizer helper to about 0.000000000001 m. These numbers establish coordinate compatibility, not camera-localization accuracy.

## Working commands

From the project root in PowerShell:

```powershell
# Already completed. Uses the earlier validated raw cache; no bag re-decoding needed.
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py reconcile

# Repeat the regression and full-recording plumbing checks.
& .\.venv\Scripts\python.exe -B -m unittest discover -s evaluation -p 'test_*.py' -v

# Once the camera system produces a tuning-episode prediction CSV:
& .\.venv\Scripts\python.exe -B evaluation/reference_eval.py score results/tune/submission.csv --split tune --out results/tune/evaluation
```

The default active reference is **`evaluation/reference_reconciled/`**. Existing output directories are rejected, so rebuilding requires a fresh `--out` and a matching `--reference` when scoring. `prepare` can build this same reconciled reference from the original MCAP without the earlier cache. On another machine, run root `setup.cmd` for the shared Python 3.12.13 environment, then privately copy the training recording and use `prepare` if the ignored local reference bundle is absent. The minimal `evaluation/requirements.txt` retains the five evaluation version pins; prefer the shared root lock. Use `-B` to prevent caches in the organizer's tool directories.

## Active reference artifacts

| File in `reference_reconciled/` | Purpose |
|---|---|
| `source_state_native.csv` | Original x/y/yaw/lane plus raw track/pit idx, ds, n. No coordinates synthesized using the wrong index basis |
| `native_reference_bridge.npz` | Frozen fitting-only segment origins, axes and fitting counts; no trajectory lookup table |
| `gt_state_reconciled.csv` | GT reconstructed from native fields and expressed against the supplied map; 126,574 rows |
| `reconciliation.json` | Independent per-episode reconstruction, cross-route agreement and direct supplied-map conversion agreement |
| `gt_tf.csv` | Original TF-based primary GT, 126,680 rows; unchanged pose frame |
| `camera_stamps.npz` | Exposure timestamps for all seven cameras; no images |
| `manifest.json` | Source/map/adapter/bridge/dependency hashes, fitting interval, active bridge and projection conventions |
| `PLUMBING_ONLY.txt` | GT compared with itself; explicitly not camera-localization results |

The earlier `reference/` is retained as a reproducible source cache, not the active scoring profile. Its `UNSAFE` name concerned coordinates synthesized by applying the new map to old indices. The raw `idx,ds,n` fields themselves are consistent and are now reconciled. The initial `s` and `epsi` columns in that cache were computed using the mismatched Path; the bridge uses the original `idx,ds,n` instead.

Original data, the map and both organizer tool folders are preserved. The bridge calibrates on the predetermined fitting interval only. Tuning/holdout positions are used to verify compatibility, never to fit the bridge. GT-derived bridge data stays in the evaluation harness; the camera estimator never reads the live state-estimation topic or these GT files. Its predictions are projected directly onto the new map from its own estimated pose.

## Coordinate convention and scoring

The organizer's `Line.xy_to_sn` runs with timestamps, matching `convert.py`'s sequence convention. Gaps over one second reset its search as supplied. Since prediction CSVs do not yet carry recovery events, steps over 100 m also reset globally; this avoids the demonstrated window failure after a large recovery jump. This extra reset is applied to GT and predictions alike and is recorded in diagnostics. No such extra reset is needed for the actual GT recording. Replace the jump heuristic with explicit estimator recovery events when available.

Use the supplied Paths' XY arc lengths, start points, left-positive n and wrapped epsi. Neither reference line is moved, resampled or fitted to predictions. Keep original x/y/yaw: polyline s/n has endpoint ambiguities, so inverse projection must not replace an estimated pose. The bridge reconstructs using the original segment index directly, including negative ds; converting old indices to an arc scalar first can lose up to about 0.5 m at endpoints.

`score` requires an explicit fixed `fit`, `tune`, `holdout` or `full` episode and keeps the complete episode GT. It checks exact integer nanoseconds, finite required pose values, increasing unique timestamps, membership in actual camera exposures, valid GT interpolation, >=90% coverage and all reference identities. Output intervals are reported separately from coverage; camera rate is still expected. No alignment to GT or cropping to successful predictions is performed.

Each result directory contains both original/updated x/y reports against identical TF GT, **`reconciled_supplied_map.txt`** from the updated evaluator, and exact episode/prediction CSVs. **`submission_supplied_map.csv`** contains x/y/yaw and both track-coordinate sets accepted by the updated evaluator. `validation.json` records `native_coordinate_bridge_used: true`, `raw_native_fields_used_for_GT: true` and `unconverted_native_indices_used: false` to distinguish active reconciliation from direct cross-version index reuse.

The additional data is fully used for this coordinate/evaluation path. The system plan also uses its identical surveyed borders as the canonical static geometry. The remaining organizer question is which reference/version defines the ranked Option B contract; this locally validated reconciliation cannot establish that administrative choice. Camera-localizer accuracy, inference causality and runtime still require implementation and measurement.

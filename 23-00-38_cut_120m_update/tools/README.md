# Ego localization tools — x/y/yaw and track (s, n) coordinates

```bash
pip install mcap mcap-ros2-support numpy
cd tools
python3 extract_gt.py ../rosbag_2026-10-12_23-00-38_cut_120m ../map.mcap gt.csv      # ground truth
python3 convert.py xy2sn ../map.mcap my_xy.csv my_sn.csv                              # x,y,yaw -> s,n,epsi
python3 convert.py sn2xy ../map.mcap my_sn.csv my_xy.csv                              # s,n,epsi -> x,y,yaw
python3 evaluate.py gt.csv my_xy.csv        # or my_sn.csv (both column sets may be in one file)
```

## Files
| file | content |
|---|---|
| `../rosbag_.../` | the data (ROS 2 bag, 35 s). Ground truth = `/constructor0/state_estimation` (= `/tf` map -> `constructor0/base_link`) |
| `../map.mcap` | physical borders `rr` (track + pit lane, Path + markers), racing line, pit-lane line |
| `track_frame.py` | (s, n) definitions, projection x/y -> (s, n) and back |
| `extract_gt.py`, `convert.py`, `evaluate.py` | see above |

## (s, n) frame
- Reference lines (`nav_msgs/Path`, frame `map`, closed loops, the last pose repeats the first):
  - `/constructor0/map_ar/racing_line` — the line the car's state estimation computes (s, n) against;
  - `/constructor0/map_ar/pitlane_line` — the same loop, but going through the pit lane (it coincides with the
    racing line outside the pit section).
- `s` — arc length along the polyline from its first pose, m, in [0, lap): racing line 2976.875 m, pit-lane line
  2985.489 m. `n` — signed distance from the polyline segment, m, **positive to the left** of the driving
  direction. `epsi` = car yaw − segment heading, rad.
- These are exactly the car's own (s, n) (`sn_map_state`): car (s, n) -> x, y reproduces the car position to
  < 0.1 mm, and `convert.py xy2sn` of `/tf` reproduces the car (s, n) to 5e-5 m.
- `gt.csv` has both: `s, n, epsi` (racing line, always) and `s_pit, n_pit, epsi_pit` (pit-lane line, always);
  `lane` says where the car was: 0 pit lane, 1 track, 2 mixed (entry/exit), 3 undefined, 4 none.

## Scoring
Ground truth is interpolated at each submitted `timestamp_ns` (gap > 50 ms -> pose skipped).
- x/y: position RMSE (score), error split into lateral / longitudinal in the GT car frame, yaw RMSE.
- s/n: RMSE of (ds, dn) (score), ds wrapped over the lap (crossing the start line is fine), dn, epsi.
  The racing-line s/n is scored on every pose; `s_pit, n_pit` (optional) on GT pit-lane poses only.
- coverage: share of 1 s bins with at least one pose, >= 90 % to be ranked.

Note: s/n and x/y errors are equal when the car is on the line and parallel to it; away from it they differ
slightly (ds is measured along the line, lon along the car heading; on a curve ds scales by 1/(1 − κ·n)).

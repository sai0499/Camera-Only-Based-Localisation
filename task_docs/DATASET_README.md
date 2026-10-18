# Camera localization dataset — Yas Marina North, constructor0, 2026-10-13 22-53-42

## Task

Estimate the pose of the race car (frame `constructor0/base_link`) in the track frame `map`
from its **7 cameras only**: position x, y and heading yaw, at camera rate.

You may use: the camera images, their intrinsics (`camera_info`), the camera extrinsics (`/tf_static`)
and the track borders (`/constructor0/track_bounds`, a map prior).
The ego pose topics (`/tf`, `/constructor0/state_estimation`) are the **ground truth**: use them for
development, training and scoring, not as an input of the method — except the very first pose of the
run, which may be used as the initial pose.

## Data

One mcap file (ROS 2, CDR, message schemas embedded), 1194 s, 7.0 GiB. Night session, the car drives
laps at up to ~40 m/s and passes through the pit lane. The recording is trimmed to the driving part
(first motion − 5 s … last motion + 3 s).

| topic | type | rate | content |
|---|---|---|---|
| `/constructor0/sensor/camera_{fl,fr,cl,cr,r,rl,rr}/compressed_image` | sensor_msgs/CompressedImage | 10 Hz | jpeg 1506x728; `header.stamp` = exposure time |
| `/constructor0/sensor/camera_*/camera_info` | sensor_msgs/CameraInfo | 1 Hz | K, D (plumb_bob) |
| `/tf_static` | tf2_msgs/TFMessage | every 10 s | `constructor0/base_link` -> every sensor (cameras, lidars, radars, gnss, imu) |
| `/tf` | tf2_msgs/TFMessage | 106 Hz | `map` -> `constructor0/base_link` — **ground truth** |
| `/constructor0/state_estimation` | sd_localization_msgs/StateEstimation | 106 Hz | same pose + velocities, accelerations, status — **ground truth** |
| `/constructor0/track_bounds` | visualization_msgs/MarkerArray | every 10 s | LINE_STRIPs in `map`: `track_left`, `track_right`, `pitlane_left`, `pitlane_right` |

Cameras: `fl`/`fr` front (narrow, fx ≈ 2600 / 1350), `cl`/`cr` front-side wide, `rl`/`rr` rear-side,
`r` rear. Camera frames `constructor0/camera_*` are optical frames (z forward, x right, y down).
`base_link`: x forward, y left, z up (REP 103).

Latched topics (tf_static, camera_info, track_bounds) are repeated, so they are present at any start
offset and `ros2 bag play` works without QoS overrides. Foxglove opens the file directly; for
`ros2 bag play` of `state_estimation` the `sd_localization_msgs` package is needed (all other topics
are standard ROS messages).

Accuracy notes
- Ground truth is the car's state estimation (RTK GNSS + lidar map matching + IMU/odometry fusion),
  roughly 0.1 m / 0.2° on this track.
- Camera extrinsics were calibrated against the lidar on 2026-10-12/14. Pitch is reliable for all
  cameras; yaw is reliable for `r` and `rl`, about ±1° for `fl`/`fr`; `cl`/`cr` keep the previous
  calibration. Roll and translation are not calibrated (mount values).
- Track borders are the surveyed borders from the map loaded on the car (same `map` frame as the GT).

## Reading the data

```bash
pip install mcap mcap-ros2-support numpy
python3 tools/extract_gt.py c0_2026-10-13_22-53-42_camloc_0.mcap gt.csv     # timestamp_ns,x,y,z,yaw
```

```python
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
with open('c0_2026-10-13_22-53-42_camloc_0.mcap', 'rb') as f:
    for schema, channel, message, msg in make_reader(f, decoder_factories=[DecoderFactory()]) \
            .iter_decoded_messages(topics=['/constructor0/sensor/camera_fl/compressed_image']):
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        jpeg = bytes(msg.data)          # cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
```

## Submission

A CSV file with a header line:

```
timestamp_ns,x,y,yaw
1791918157176982265,7.1728,-9.9093,0.126255
...
```

- `timestamp_ns` — `header.stamp` of the image (or image set) the pose was computed from, ns.
- `x`, `y` — position of `constructor0/base_link` in `map`, metres; `yaw` — heading, radians.
- At least one pose per second over the whole run (camera rate, 10 Hz, is expected).
- The method must be causal: a pose at time t may only use images with stamps ≤ t.

## Metric

```bash
python3 tools/evaluate.py gt.csv submission.csv
```

For every submitted pose the ground truth is linearly interpolated at `timestamp_ns` (poses where the
GT has a gap > 50 ms are skipped). The error is split into the GT car frame: **longitudinal** (along the
heading) and **lateral** (across it).

| metric | meaning |
|---|---|
| **position RMSE [m]** | √mean(dx² + dy²) — **the score, lower is better** |
| lateral / longitudinal RMSE [m] | error across / along the car; lateral matters most for racing |
| position median / p95 [m] | typical and worst-case error |
| yaw RMSE [deg] | heading error, wrapped to ±180° |
| within 0.25 / 0.5 / 1.0 m [%] | share of poses within the radius |
| coverage [%] | share of the 1 s intervals of the run with at least one pose; **≥ 90 % to be ranked** |

Example (GT + 0.3 m lateral noise + 1° yaw noise):

```
coverage (1 s bins)    100.0 %   (>= 90 % to be ranked)
position RMSE  [m]     0.304   <- SCORE (lower is better)
position median/p95    0.211 / 0.591
lateral RMSE   [m]     0.300
longitudinal RMSE [m]  0.049
yaw RMSE     [deg]     1.007
within 0.25 m          58.5 %
within 0.50 m          90.5 %
within 1.00 m          99.9 %
```

## Files

| file | |
|---|---|
| `c0_2026-10-13_22-53-42_camloc_0.mcap`, `metadata.yaml` | the dataset (ROS 2 bag) |
| `tools/extract_gt.py` | ground truth to CSV |
| `tools/evaluate.py` | scoring |
| `track_bounds_check.png` | track borders with the GT trajectory |

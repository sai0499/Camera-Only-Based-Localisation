#!/usr/bin/env python3
"""Score a camera-localization submission against the ground truth.

    python3 evaluate.py gt.csv submission.csv

submission.csv: timestamp_ns,x,y,yaw  (header line required; pose of constructor0/base_link in frame
map at camera time timestamp_ns -- use the header.stamp of the image(s) the pose was computed from).

GT is interpolated linearly at each submitted timestamp (yaw on the unwrapped angle). Errors are split
into the GT body frame: longitudinal (along the car heading) and lateral (across it).
"""
import sys
import numpy as np

MAX_GT_GAP_NS = 50_000_000        # GT gap around a submitted stamp larger than this -> stamp ignored
BIN_NS = 1_000_000_000            # coverage: share of 1 s bins of the GT window with >= 1 prediction
MIN_COVERAGE = 0.90               # below this the submission is not ranked


def load(path, cols):
    d = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    return {c: np.asarray(d[c], dtype=np.int64 if c == 'timestamp_ns' else float) for c in cols}


def main(gt_path, sub_path):
    gt = load(gt_path, ['timestamp_ns', 'x', 'y', 'yaw'])
    sb = load(sub_path, ['timestamp_ns', 'x', 'y', 'yaw'])
    o = np.argsort(gt['timestamp_ns']); gt = {k: v[o] for k, v in gt.items()}
    o = np.argsort(sb['timestamp_ns']); sb = {k: v[o] for k, v in sb.items()}
    tg, ts = gt['timestamp_ns'], sb['timestamp_ns']

    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    n_sub, ts, i = len(ts), ts[ok], i[ok]
    sx, sy, syaw = sb['x'][ok], sb['y'][ok], sb['yaw'][ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yaw_u = np.unwrap(gt['yaw'])
    gx = gt['x'][i - 1] + a * (gt['x'][i] - gt['x'][i - 1])
    gy = gt['y'][i - 1] + a * (gt['y'][i] - gt['y'][i - 1])
    gyaw = yaw_u[i - 1] + a * (yaw_u[i] - yaw_u[i - 1])

    dx, dy = sx - gx, sy - gy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(np.angle(np.exp(1j * (syaw - gyaw))))

    bins = np.arange(tg[0], tg[-1], BIN_NS)
    covered = np.unique(np.searchsorted(bins, ts, side='right') - 1)
    coverage = len(covered) / max(len(bins), 1)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')

    print(f'poses submitted        {n_sub}   evaluated {len(ts)}')
    print(f'coverage (1 s bins)    {coverage * 100:.1f} %   (>= {MIN_COVERAGE * 100:.0f} % to be ranked)')
    print(f'position RMSE  [m]     {rms(pos):.3f}   <- SCORE (lower is better)')
    print(f'position median/p95    {np.median(pos):.3f} / {np.percentile(pos, 95):.3f}')
    print(f'lateral RMSE   [m]     {rms(lat):.3f}')
    print(f'longitudinal RMSE [m]  {rms(lon):.3f}')
    print(f'yaw RMSE     [deg]     {rms(dyaw):.3f}')
    for thr in (0.25, 0.5, 1.0):
        print(f'within {thr:4.2f} m          {np.mean(pos < thr) * 100:.1f} %')
    if coverage < MIN_COVERAGE:
        print('NOT RANKED: coverage below threshold')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""Score an ego-pose submission against the ground truth, in map x/y/yaw and/or track (s, n) coordinates.

    python3 evaluate.py gt.csv submission.csv [../map.mcap]

submission.csv (header line required), any of:
  timestamp_ns,x,y[,yaw]          pose of constructor0/base_link in frame map
  timestamp_ns,s,n[,epsi]         w.r.t. the racing line            (see track_frame.py)
  ...,s_pit,n_pit[,epsi_pit]      w.r.t. the pit-lane line (scored on pit-lane GT rows only)
To score an x/y submission in (s, n) too: python3 convert.py xy2sn ../map.mcap submission.csv submission_sn.csv

For every submitted pose the GT is interpolated linearly at timestamp_ns (s on the lap-unwrapped value, angles
unwrapped); poses where the GT has a gap > 50 ms are skipped.
  x/y : error split in the GT car frame -- longitudinal (along the heading) / lateral (across it)
  s/n : ds = s_sub - s_gt wrapped to +-lap/2 (along the line), dn = n_sub - n_gt (across the line)
"""
import csv, sys
import numpy as np

MAX_GT_GAP_NS = 50_000_000        # GT gap around a submitted stamp larger than this -> stamp ignored
BIN_NS = 1_000_000_000            # coverage: share of 1 s bins of the GT window with >= 1 prediction
MIN_COVERAGE = 0.90               # below this the submission is not ranked
PIT_LANE = 0


def read(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    t = np.array([int(x['timestamp_ns']) for x in r], dtype=np.int64)
    o = np.argsort(t, kind='stable')
    return t[o], {k: np.array([float(x[k]) for x in r])[o] for k in r[0] if k != 'timestamp_ns'}


def rms(v):
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')


def wrapd(a):
    return np.degrees(np.angle(np.exp(1j * a)))


def report(name, pos, a_err, b_err, a_name, b_name, ang=None, ang_name=None):
    print(f'--- {name} ---')
    print(f'position RMSE  [m]     {rms(pos):.3f}   <- SCORE (lower is better)')
    print(f'position median/p95    {np.median(pos):.3f} / {np.percentile(pos, 95):.3f}')
    print(f'{a_name:22s} {rms(a_err):.3f}   bias {np.mean(a_err):+.3f}')
    print(f'{b_name:22s} {rms(b_err):.3f}   bias {np.mean(b_err):+.3f}')
    if ang is not None:
        print(f'{ang_name:22s} {rms(ang):.3f}')
    for thr in (0.25, 0.5, 1.0):
        print(f'within {thr:4.2f} m          {np.mean(pos < thr) * 100:.1f} %')


def main(gt_path, sub_path, lap=None, lap_pit=None):
    tg, g = read(gt_path)
    ts, sb = read(sub_path)

    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    n_sub, ts, i = len(ts), ts[ok], i[ok]
    sb = {k: v[ok] for k, v in sb.items()}
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    lerp = lambda v: v[i - 1] + a * (v[i] - v[i - 1])

    bins = np.arange(tg[0], tg[-1], BIN_NS)
    coverage = len(np.unique(np.searchsorted(bins, ts, side='right') - 1)) / max(len(bins), 1)
    print(f'poses submitted        {n_sub}   evaluated {len(ts)}')
    print(f'coverage (1 s bins)    {coverage * 100:.1f} %   (>= {MIN_COVERAGE * 100:.0f} % to be ranked)')
    if not len(ts):
        return

    if 'x' in sb and 'y' in sb:
        gyaw = lerp(np.unwrap(g['yaw']))
        dx, dy = sb['x'] - lerp(g['x']), sb['y'] - lerp(g['y'])
        lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
        lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
        report('map x/y', np.hypot(dx, dy), lat, lon, 'lateral RMSE   [m]', 'longitudinal RMSE [m]',
               wrapd(sb['yaw'] - gyaw) if 'yaw' in sb else None, 'yaw RMSE     [deg]')

    for suf, lap_len, sel, label in (('', lap, np.ones(len(ts), bool), 'racing line s/n (all poses)'),
                                     ('_pit', lap_pit, None, 'pit-lane line s/n (GT lane = pit)')):
        if 's' + suf not in sb or 'n' + suf not in sb:
            continue
        if sel is None:
            sel = (g['lane'][i - 1] == PIT_LANE) & (g['lane'][i] == PIT_LANE)
            if not sel.any():
                print(f'--- {label} --- no pit-lane GT poses in the evaluated window')
                continue
        gs = lerp(np.unwrap(g['s' + suf], period=lap_len))
        ds = (sb['s' + suf] - gs + lap_len / 2) % lap_len - lap_len / 2
        dn = sb['n' + suf] - lerp(g['n' + suf])
        de = wrapd(sb['epsi' + suf] - lerp(np.unwrap(g['epsi' + suf]))) if 'epsi' + suf in sb else None
        report(label, np.hypot(ds, dn)[sel], dn[sel], ds[sel], 'n (lateral) RMSE  [m]', 's (along) RMSE    [m]',
               de[sel] if de is not None else None, 'epsi RMSE     [deg]')
    if coverage < MIN_COVERAGE:
        print('NOT RANKED: coverage below threshold')


if __name__ == '__main__':
    gt, sub = sys.argv[1:3]
    # lap lengths of both lines are needed to wrap s; read them from map.mcap next to the tools
    import os
    from track_frame import load_lines
    mp = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'map.mcap')
    L = load_lines(mp)
    main(gt, sub, L['track'].lap, L['pit'].lap)

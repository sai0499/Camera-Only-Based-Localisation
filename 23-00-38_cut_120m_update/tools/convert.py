#!/usr/bin/env python3
"""Convert poses between map x/y/yaw and track (s, n, epsi) coordinates using map.mcap.

    python3 convert.py xy2sn ../map.mcap poses_xy.csv poses_sn.csv   # timestamp_ns,x,y[,yaw] -> s,n,epsi,s_pit,n_pit,epsi_pit
    python3 convert.py sn2xy ../map.mcap poses_sn.csv poses_xy.csv   # timestamp_ns,s,n[,epsi]  -> x,y,yaw (racing line)

xy2sn projects every pose on the racing line and on the pit-lane line. Poses are sorted by time and each one is
searched near the previous projection (as the car does); a gap > 1 s restarts with a global search.
"""
import csv, sys
import numpy as np
from track_frame import load_lines


def read(path):
    with open(path) as f:
        r = list(csv.DictReader(f))
    t = np.array([int(x['timestamp_ns']) for x in r], dtype=np.int64)
    o = np.argsort(t, kind='stable')
    cols = {k: np.array([float(x[k]) for x in r])[o] for k in r[0] if k != 'timestamp_ns'}
    return t[o], cols


def write(path, t, cols):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_ns'] + list(cols))
        for i in range(len(t)):
            w.writerow([t[i]] + [f'{cols[k][i]:.6f}' for k in cols])


def main(mode, map_mcap, src, dst):
    lines = load_lines(map_mcap)
    t, c = read(src)
    if mode == 'xy2sn':
        out = {}
        for name, suf in (('track', ''), ('pit', '_pit')):
            s, n, e, _ = lines[name].xy_to_sn(c['x'], c['y'], c.get('yaw'), t)
            out.update({'s' + suf: s, 'n' + suf: n, 'epsi' + suf: e})
    elif mode == 'sn2xy':
        x, y, yaw = lines['track'].sn_to_xy(c['s'], c['n'], c.get('epsi'))
        out = {'x': x, 'y': y, 'yaw': yaw}
    else:
        raise SystemExit('mode: xy2sn | sn2xy')
    write(dst, t, out)
    print(f'{len(t)} poses {mode} -> {dst}')


if __name__ == '__main__':
    main(*sys.argv[1:5])

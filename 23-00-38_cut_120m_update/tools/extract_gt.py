#!/usr/bin/env python3
"""Ground truth (car state estimation) in map x/y/yaw and in track (s, n) coordinates.

    pip install mcap mcap-ros2-support numpy
    python3 extract_gt.py ../rosbag_2026-10-12_23-00-38_cut_120m ../map.mcap gt.csv

gt.csv: timestamp_ns,x,y,yaw,lane,s,n,epsi,s_pit,n_pit,epsi_pit
  timestamp_ns      header stamp of /constructor0/state_estimation
  x, y, yaw         pose of constructor0/base_link in frame map (m, rad)
  lane              0 pit lane, 1 track, 2 mixed (entry/exit), 3 undefined, 4 none -- as decided by the car
  s, n, epsi        w.r.t. the racing line   (/constructor0/map_ar/racing_line), always defined
  s_pit, n_pit, ... w.r.t. the pit-lane line (/constructor0/map_ar/pitlane_line), always defined
(s, n) are the car's own projection (sn_map_state); see track_frame.py for the definitions.
"""
import csv, glob, os, sys
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from track_frame import load_lines, wrap

SE_TOPIC = '/constructor0/state_estimation'


def main(bag, map_mcap, out):
    if os.path.isdir(bag):
        bag = glob.glob(os.path.join(bag, '*.mcap'))[0]
    lines = load_lines(map_mcap)
    rows, check = [], {'track': [], 'pit': []}
    with open(bag, 'rb') as f:
        for _, _, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=[SE_TOPIC]):
            sm, row = m.sn_map_state, [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, m.x_m, m.y_m, m.yaw_rad,
                                       int(m.sn_map_state.current_lane)]
            for name, st in (('track', sm.track_sn_state.sn_state), ('pit', sm.pitlane_sn_state.sn_state)):
                L = lines[name]
                if st.idx >= len(L.p):                      # projection not initialised
                    row += [np.nan] * 3
                    continue
                s = L.from_car(st.idx, st.ds)
                x, y, _ = L.sn_to_xy(s, st.n)
                check[name].append(np.hypot(x[0] - m.x_m, y[0] - m.y_m))
                row += [float(s), st.n, float(wrap(m.yaw_rad - L.heading[st.idx]))]
            rows.append(row)
    with open(out, 'w', newline='') as o:
        w = csv.writer(o)
        w.writerow(['timestamp_ns', 'x', 'y', 'yaw', 'lane', 's', 'n', 'epsi', 's_pit', 'n_pit', 'epsi_pit'])
        for r in rows:
            w.writerow([r[0]] + [f'{v:.4f}' for v in r[1:3]] + [f'{r[3]:.6f}', r[4]] +
                       [f'{v:.4f}' if j % 3 != 2 else f'{v:.6f}' for j, v in enumerate(r[5:])])
    lanes = dict(zip(*np.unique([r[4] for r in rows], return_counts=True)))
    print(f'{len(rows)} poses -> {out}   lanes {lanes}')
    for name, e in check.items():
        print(f'  car (s, n) on {name:5s} line -> x, y vs car x, y: max {max(e) if e else float("nan"):.4f} m')


if __name__ == '__main__':
    main(*sys.argv[1:4])

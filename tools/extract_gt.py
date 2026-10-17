#!/usr/bin/env python3
"""Ground truth from the dataset: map -> constructor0/base_link from /tf.

    pip install mcap mcap-ros2-support numpy
    python3 extract_gt.py ../c0_2026-10-13_22-53-42_camloc_0.mcap gt.csv

gt.csv: timestamp_ns,x,y,z,yaw   (header stamp, metres in frame map, yaw rad)
"""
import csv, math, sys
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def main(bag, out):
    n = 0
    with open(bag, 'rb') as f, open(out, 'w', newline='') as o:
        w = csv.writer(o)
        w.writerow(['timestamp_ns', 'x', 'y', 'z', 'yaw'])
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in r.iter_decoded_messages(topics=['/tf']):
            for t in msg.transforms:
                if t.header.frame_id != 'map' or t.child_frame_id != 'constructor0/base_link':
                    continue
                q, p = t.transform.rotation, t.transform.translation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                w.writerow([t.header.stamp.sec * 10**9 + t.header.stamp.nanosec,
                            f'{p.x:.4f}', f'{p.y:.4f}', f'{p.z:.4f}', f'{yaw:.6f}'])
                n += 1
    print(f'{n} poses -> {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

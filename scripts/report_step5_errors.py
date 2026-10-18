"""Offline only: summarize Step 5 saved predictions and reset errors against fixed tuning GT.

Never imported by inference. Uses the organizer's interpolation/error convention and the
predeclared audit bounds; bounded diagnostics are not an official full-episode score.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def errors(gt, sub):
    """Organizer convention: searchsorted(left), <=50 ms brackets, linear unwrapped yaw."""
    tg, ts = gt['timestamp_ns'], sub['timestamp_ns']
    index = np.searchsorted(tg, ts)
    valid = (index > 0) & (index < len(tg))
    index = np.clip(index, 1, len(tg) - 1)
    valid &= tg[index] - tg[index - 1] <= 50_000_000
    i, stamps = index[valid], ts[valid]
    a = (stamps - tg[i - 1]) / (tg[i] - tg[i - 1])
    yaw = np.unwrap(gt['yaw'])
    truth = [values[i - 1] + a * (values[i] - values[i - 1]) for values in (gt['x'], gt['y'], yaw)]
    dx, dy = sub['x'][valid] - truth[0], sub['y'][valid] - truth[1]
    position = np.hypot(dx, dy)
    yaw_error = np.degrees(np.angle(np.exp(1j * (sub['yaw'][valid] - truth[2]))))
    return stamps, position, yaw_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--gt', type=Path, default=ROOT / 'evaluation/reference_reconciled/gt_tf.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from scripts.export_initial_pose import verify_manifest

    verified_gt = verify_manifest(args.gt)
    spec = importlib.util.spec_from_file_location('step5_organizer', ROOT / 'tools/evaluate.py')
    organizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(organizer)
    report = json.loads((args.audit / 'report.json').read_text())
    gt = organizer.load(args.gt, ['timestamp_ns', 'x', 'y', 'yaw'])
    # Fix the interval from the audit manifest, never from successful prediction rows.
    episode = report['episode']
    selected = ((gt['timestamp_ns'] >= episode['start_ns'])
                & (gt['timestamp_ns'] < episode['end_ns']))
    gt = {key: value[selected] for key, value in gt.items()}
    if len(gt['timestamp_ns']) < 2:
        raise ValueError('no GT support in declared audit interval')
    result = {'scope': report['scope'], 'episode': episode,
              'official_full_episode_scoring': False, 'cases': {}}
    with args.gt.open('rb') as handle:
        result['gt_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    result['gt_manifest_verified'] = verified_gt is not None
    for case in report['cases']:
        folder = args.audit / case
        sub = organizer.load(folder / 'submission.csv', ['timestamp_ns', 'x', 'y', 'yaw'])
        stamps, position, yaw = errors(gt, sub)
        resets = []
        with (folder / 'diagnostics.jsonl').open() as handle:
            for line in handle:
                record = json.loads(line)
                if record.get('record') == 'Estimate' and 'recovery_reset' in record['evidence_kinds']:
                    index = int(np.searchsorted(stamps, record['t_ns']))
                    if index < len(stamps) and stamps[index] == record['t_ns']:
                        resets.append({'timestamp_ns': int(stamps[index]), 'position_error_m': float(position[index])})
        result['cases'][case] = {'matched_rows': len(stamps),
            'position_rmse_m': float(np.sqrt(np.mean(position**2))) if len(stamps) else None,
            'position_median_m': float(np.median(position)) if len(stamps) else None,
            'position_p95_m': float(np.percentile(position, 95)) if len(stamps) else None,
            'yaw_rmse_deg': float(np.sqrt(np.mean(yaw**2))) if len(stamps) else None,
            'recovery_resets': resets,
            'resets_over_10m': sum(reset['position_error_m'] > 10 for reset in resets)}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

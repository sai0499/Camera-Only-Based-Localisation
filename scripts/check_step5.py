"""C's real-MCAP repeatability, input-isolation, causality and image-stress checks.

Creates fresh, private MCAP variants and invokes the unchanged production CLI in fresh
processes. Only selected image payloads are decoded here; prohibited payloads are copied,
removed or replaced by an invalid ROS payload inside a valid CRC-checked MCAP container.
The original recording, maps, evaluator and runtime gates are never modified.

Default: the complete tuning episode. --duration-s is a bounded preflight, never a ranked
trajectory or a full-episode acceptance result. Stress locations/parameters and numeric
comparison tolerances below are fixed before executing the comparisons. No holdout access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camloc import contracts as c, io  # noqa: E402
from validate_run import validate  # noqa: E402

CASES = ('baseline', 'repeat', 'prohibited_remove', 'prohibited_poison', 'future',
         'fr_drop_1', 'fr_drop_3', 'r_drop_1', 'r_drop_3',
         'blackout_0.5', 'blackout_1', 'blur', 'exposure', 'glare')
STRESS_START_S = 15
FUTURE_START_S = 10
POSITION_ATOL_M = 1e-5
YAW_ATOL_RAD = 1e-7


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def copy_messages(source, output, select, transform=None):
    """Copy raw messages, preserving log/publish stamps, sequence, schemas and metadata."""
    from mcap.reader import make_reader
    from mcap.writer import Writer

    counts = Counter()
    digest = hashlib.sha256()
    with Path(source).open('rb') as source_file, Path(output).open('xb') as target:
        reader, writer = make_reader(source_file, validate_crcs=True), Writer(target)
        writer.start(profile='ros2', library='camloc-step5-audit')
        schemas, channels = {}, {}
        for schema, channel, message in reader.iter_messages(log_time_order=True):
            if not select(channel, message):
                continue
            data = message.data if transform is None else transform(schema, channel, message)
            if data is None:
                counts['removed'] += 1
                continue
            counts['changed'] += data != message.data
            if channel.id not in channels:
                schema_id = 0
                if schema is not None:
                    if schema.id not in schemas:
                        schemas[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
                    schema_id = schemas[schema.id]
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding,
                                                                schema_id, channel.metadata)
            writer.add_message(channels[channel.id], message.log_time, data,
                               message.publish_time, message.sequence)
            counts[channel.topic] += 1
            if channel.topic not in io.PROHIBITED_TOPICS:
                # Framed binary fingerprint: permitted payloads and message timing/order.
                topic = channel.topic.encode()
                digest.update(len(topic).to_bytes(4, 'little') + topic)
                for value in (message.log_time, message.publish_time, message.sequence, len(data)):
                    digest.update(value.to_bytes(8, 'little'))
                digest.update(data)
        writer.finish()
    return {'messages': dict(counts), 'permitted_messages_sha256': digest.hexdigest(),
            'mcap_sha256': sha256(output)}


def stress_window(case, initial_ns):
    if case == 'future':
        return initial_ns + FUTURE_START_S * c.NS_PER_S, None
    seconds = float(case.rsplit('_', 1)[-1]) if case.startswith(('fr_drop_', 'r_drop_', 'blackout_')) else 1
    start = initial_ns + STRESS_START_S * c.NS_PER_S
    return start, start + round(seconds * c.NS_PER_S)


def image_change(case, image):
    import cv2

    if case.startswith('blackout_') or case == 'future':
        return np.zeros_like(image)
    if case == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 3)
    if case == 'exposure':
        return np.rint(image.astype(np.float32) * .25).astype(np.uint8)
    if case == 'glare':
        result = image.copy()
        height, width = image.shape[:2]
        cv2.circle(result, (width // 2, height // 3), width // 6, (255, 255, 255), -1)
        return result
    raise ValueError(f'unknown image case {case}')


def variant(source, output, case, initial_ns):
    from mcap_ros2._dynamic import serialize_dynamic
    from mcap_ros2.decoder import DecoderFactory

    decoder, encoders = DecoderFactory(), {}
    begin, end = stress_window(case, initial_ns)
    changed_exposures = []

    def transform(schema, channel, message):
        if channel.topic in io.PROHIBITED_TOPICS:
            if case == 'prohibited_remove':
                return None
            if case == 'prohibited_poison':
                return b'PROHIBITED_PAYLOAD_MUST_NOT_BE_DECODED'
        camera = next((cam for cam in (c.PRIMARY_CAMERA, c.SECONDARY_CAMERA)
                       if channel.topic == io.image_topic(cam)), None)
        if camera is None or case in ('baseline', 'repeat', 'prohibited_remove', 'prohibited_poison'):
            return message.data
        decode = decoder.decoder_for(channel.message_encoding, schema)
        if decode is None:
            raise c.ContractError('selected image has no supported ROS decoder')
        decoded = decode(message.data)
        stamp = io._stamp_ns(decoded.header)
        if stamp < begin or (end is not None and stamp >= end):
            return message.data
        if case.startswith(('fr_drop_', 'r_drop_')):
            if camera == case.split('_')[0]:
                changed_exposures.append(stamp)
                return None
            return message.data
        import cv2

        image = cv2.imdecode(np.frombuffer(bytes(decoded.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise c.InputError('audit input image cannot be decoded')
        ok, jpeg = cv2.imencode('.jpg', image_change(case, image), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise c.InputError('audit JPEG encoding failed')
        if schema.id not in encoders:
            encoders[schema.id] = serialize_dynamic(schema.name, schema.data.decode())[schema.name]
        decoded.data = jpeg.tobytes()
        changed_exposures.append(stamp)
        return encoders[schema.id](decoded)

    receipt = copy_messages(source, output, lambda channel, message: True, transform)
    receipt['changed_images'] = len(changed_exposures)
    receipt['first_changed_exposure_ns'] = min(changed_exposures, default=None)
    if case not in ('baseline', 'repeat') and not (receipt['messages'].get('changed', 0)
                                                  or receipt['messages'].get('removed', 0)):
        raise c.ContractError(f'{case}: no input changed; refusing a vacuous check')
    return receipt


def read_poses(path, before_ns=None):
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        if next(handle).strip() != 'timestamp_ns,x,y,yaw':
            raise c.ContractError('unexpected CSV header')
        for line in handle:
            stamp, x, y, yaw = line.strip().split(',')
            if before_ns is None or int(stamp) < before_ns:
                rows.append((int(stamp), float(x), float(y), float(yaw)))
    return rows


def decisions(path, before_ns=None):
    """Runtime-independent status/evidence/scheduling identity, grouped by output exposure."""
    records, group_times = [], {}
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get('record')
            if kind == 'FrameSet':
                group_times[record['group_id']] = record['t_ns']
            if kind not in ('Estimate', 'GroupEvidence', 'SchedulerEvent', 'RecoveryEvent'):
                continue
            group = record['frame_set']['group_id'] if kind == 'GroupEvidence' else record.get('group_id')
            if group is None or (before_ns is not None and group_times.get(group, before_ns) >= before_ns):
                continue
            if kind == 'SchedulerEvent' and record.get('kind') == 'stall':
                continue
            if kind == 'Estimate':
                records.append((kind, group, record['status'], record['branch'], record['evidence_kinds'],
                                record['reason'], record['max_support_exposure_ns']))
            elif kind == 'GroupEvidence':
                pose = record['pose']
                if pose is not None and pose['record'] == 'HandoffEvidence':
                    pose = pose['pose']
                reset = record['reset']
                records.append((kind, group, None if pose is None else (
                    pose['camera_id'], pose['exposure_ns'], pose['source'], pose['landmark_ids']),
                    None if reset is None else (reset['reset_id'], reset['confirmation']['exposure_ns'])))
            else:
                records.append((kind, group, {key: record.get(key) for key in (
                    'kind', 'job_id', 'purpose', 'camera_id', 'source_exposure_ns',
                    'delivery_not_before_ns', 'reason', 'candidate_exposure_ns')}))
    return records


def compare(first, second, before_ns=None):
    a, b = read_poses(Path(first) / 'submission.csv', before_ns), read_poses(Path(second) / 'submission.csv', before_ns)
    if not a or [row[0] for row in a] != [row[0] for row in b]:
        return {'passed': False, 'reason': 'empty or different timestamp sequences'}
    aa, bb = np.array([row[1:] for row in a]), np.array([row[1:] for row in b])
    position = np.linalg.norm(aa[:, :2] - bb[:, :2], axis=1)
    yaw = np.abs((aa[:, 2] - bb[:, 2] + math.pi) % (2 * math.pi) - math.pi)
    same_decisions = decisions(Path(first) / 'diagnostics.jsonl', before_ns) == decisions(
        Path(second) / 'diagnostics.jsonl', before_ns)
    return {'passed': bool(position.max() <= POSITION_ATOL_M and yaw.max() <= YAW_ATOL_RAD and same_decisions),
            'rows_compared': len(a), 'max_position_difference_m': float(position.max()),
            'max_yaw_difference_rad': float(yaw.max()), 'same_logical_decisions': same_decisions}


def stress_report(folder, case, initial_ns):
    begin, end = stress_window(case, initial_ns)
    full_after, lost, reset_stamps = [], [], []
    statuses, full_during = Counter(), 0
    with (Path(folder) / 'diagnostics.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record') != 'Estimate' or record.get('group_id') is None:
                continue
            stamp = record['t_ns']
            full = any(kind in record['evidence_kinds'] for kind in ('pnp', 'handoff_pnp', 'recovery_reset'))
            if stamp >= begin and (end is None or stamp < end):
                statuses[record['status']] += 1
                full_during += full
            if stamp >= begin and record['status'] == 'lost':
                lost.append(stamp)
            if 'recovery_reset' in record['evidence_kinds']:
                reset_stamps.append(stamp)
            if end is not None and stamp >= end and full:
                full_after.append(stamp)
    return {'stress_start_ns': begin, 'stress_end_ns': end, 'lost_rows_after_start': len(lost),
            'statuses_during_stress': dict(statuses), 'full_pose_updates_during_stress': full_during,
            'full_pose_returned_after_stress': bool(full_after),
            'first_full_pose_delay_s': None if not full_after else (full_after[0] - end) / c.NS_PER_S,
            'recovery_reset_stamps_ns': reset_stamps,
            'accuracy_and_false_reset_assessment': 'requires separate offline scorer/GT analysis'}


def processing_errors(diagnostics):
    counts = Counter()
    with Path(diagnostics).open(encoding='utf-8') as handle:
        for line in handle:
            kind = json.loads(line).get('record', '')
            if kind.endswith('Error') or kind == 'RunAborted':
                counts[kind] += 1
    return dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bag', type=Path, default=ROOT / 'c0_2026-10-13_22-53-42_camloc_0.mcap')
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--initial-pose', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, help='bounded tuning preflight; omitted = complete tuning')
    parser.add_argument('--transport-speed', type=float, help='pace original arrival times; transport-load test only')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args(argv)
    initial, tune = c.read_initial_pose(args.initial_pose), c.load_split()['tune']
    tune.check_initial_pose(initial)
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or args.duration_s < 20):
        parser.error('--duration-s must be finite and at least 20 seconds')
    end = tune.end_ns if args.duration_s is None else min(tune.end_ns, initial.timestamp_ns + round(args.duration_s * c.NS_PER_S))
    episode = c.EpisodeBounds(name='tune', start_ns=tune.start_ns, end_ns=end)
    args.output.mkdir(parents=True, exist_ok=False)
    selected = list(dict.fromkeys(('baseline', *args.cases)))
    base = args.output / 'baseline.mcap'
    calibration = {io.info_topic(cam) for cam in c.CAMERA_IDS} | {'/tf_static'}
    from mcap_ros2.decoder import DecoderFactory

    decoder = DecoderFactory()

    def clip_images(schema, channel, message):
        if channel.topic in (io.image_topic('fr'), io.image_topic('r')):
            decode = decoder.decoder_for(channel.message_encoding, schema)
            if decode is None:
                raise c.ContractError('selected image has no supported ROS decoder')
            if io._stamp_ns(decode(message.data).header) >= end:
                return None
        return message.data

    # Keep a log-time margin at both exposure bounds. Calibration messages are preserved
    # from the entire bag, so the canonical geometry/identity is the production geometry.
    receipt = copy_messages(args.bag, base, lambda channel, message: channel.topic in calibration
        or initial.timestamp_ns - 500_000_000 <= message.log_time < end + 500_000_000, clip_images)
    report = {'scope': 'complete tuning' if end == tune.end_ns else 'bounded real-tuning preflight; not a ranked run',
        'episode': episode.summary(), 'initial_pose': initial.summary(), 'artifacts': str(args.artifacts),
        'source_bag': str(args.bag), 'baseline_input': receipt,
        'position_atol_m': POSITION_ATOL_M, 'yaw_atol_rad': YAW_ATOL_RAD,
        'stress_start_s': STRESS_START_S, 'future_start_s': FUTURE_START_S, 'cases': {}}
    for case in selected:
        print(f'Starting {case}', flush=True)
        bag = base if case in ('baseline', 'repeat') else args.output / f'{case}.mcap'
        changed = None if bag == base else variant(base, bag, case, initial.timestamp_ns)
        folder = args.output / case
        command = [sys.executable, '-B', str(ROOT / 'run.py'), '--bag', str(bag.resolve()),
            '--artifacts', str(args.artifacts.resolve()), '--retrieval', '--runtime', '--initial-pose',
            str(args.initial_pose.resolve()), '--split', 'tune', '--output', str(folder.resolve())]
        if args.transport_speed is not None:
            command += ['--transport-speed', str(args.transport_speed)]
        with (args.output / f'{case}.log').open('x', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=900, check=False)
        result = {'returncode': process.returncode, 'input_changes': changed}
        if process.returncode == 0:
            source = io.open_frame_sets(bag, initial_pose=initial, episode=episode)
            result['validation'] = validate(folder / 'submission.csv', folder / 'diagnostics.jsonl', source, episode)
            metadata = json.loads((folder / 'run.json').read_text())
            result['runtime'] = json.loads((folder / 'runtime.json').read_text())
            result['processing_errors'] = processing_errors(folder / 'diagnostics.jsonl')
            result['no_processing_faults'] = (metadata['frontend_errors'] == 0
                and metadata['anchor_faulted'] == 0 and not result['processing_errors'])
            if case in ('repeat', 'prohibited_remove', 'prohibited_poison', 'future'):
                before = changed['first_changed_exposure_ns'] if case == 'future' else None
                if case.startswith('prohibited_') and changed['permitted_messages_sha256'] != receipt['permitted_messages_sha256']:
                    raise c.ContractError('prohibited-topic mutation changed permitted messages')
                result['comparison'] = compare(args.output / 'baseline', folder, before)
            elif case != 'baseline':
                result['stress'] = stress_report(folder, case, initial.timestamp_ns)
        report['cases'][case] = result
        # Preserve partial evidence on interruption; final report is written exclusively below.
        (args.output / 'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f'Finished {case}: {json.dumps(result)}', flush=True)
    report['baseline_has_visual_updates'] = report['cases']['baseline'].get('validation', {}).get('full_pose_updates', 0) > 0
    report['integrity_checks_passed'] = report['baseline_has_visual_updates'] and all(result['returncode'] == 0 and result.get('no_processing_faults', False)
        and result.get('comparison', {'passed': True})['passed'] for result in report['cases'].values())
    report['blackout_evidence_checks_passed'] = all(result['stress']['full_pose_updates_during_stress'] == 0
        for case, result in report['cases'].items() if case.startswith('blackout_') and 'stress' in result)
    report['integrity_checks_passed'] &= report['blackout_evidence_checks_passed']
    report['stage5_signed_off'] = False
    report['remaining'] = ['full fixed stress matrix and offline error/false-reset analysis',
                           'paced five-minute latency/queue/memory qualification at each claimed rate']
    dump(args.output / 'report.json', report)
    return 0 if report['integrity_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

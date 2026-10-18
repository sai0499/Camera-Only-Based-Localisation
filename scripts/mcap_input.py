"""UI-only MCAP initialization and post-run evaluation; never imported by camloc."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from camloc import contracts as c, vehicle_motion as vm

INITIAL_TOPICS = {'/initialpose', '/initial_pose', '/constructor0/initial_pose',
                  '/constructor0/initialpose', '/constructor0/validation/initial_pose'}
POSE_TYPES = {'geometry_msgs/msg/PoseStamped', 'geometry_msgs/msg/PoseWithCovarianceStamped',
              'nav_msgs/msg/Odometry'}
GT_TOPICS = {'/ground_truth', '/ground_truth/pose', '/ground_truth/odometry',
             '/constructor0/ground_truth', '/constructor0/ground_truth/pose',
             '/constructor0/validation/pose', '/constructor0/validation/odometry'}


def _pose(message, kind):
    if kind == 'tf2_msgs/msg/TFMessage':
        for transform in message.transforms:
            if transform.header.frame_id.lstrip('/') == 'map' and transform.child_frame_id.lstrip('/') == 'constructor0/base_link':
                yield from _coordinates(transform.header, transform.transform.translation, transform.transform.rotation)
    elif message.header.frame_id.lstrip('/') == 'map':
        if kind == 'sd_localization_msgs/msg/StateEstimation':
            yield _checked(message.header, message.x_m, message.y_m, message.yaw_rad)
        else:
            if kind == 'nav_msgs/msg/Odometry' and message.child_frame_id.lstrip('/') not in ('base_link', 'constructor0/base_link'):
                return
            pose = message.pose if kind == 'geometry_msgs/msg/PoseStamped' else message.pose.pose
            yield from _coordinates(message.header, pose.position, pose.orientation)


def _coordinates(header, position, q):
    values = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in values) or abs(sum(v*v for v in values) - 1) > 1e-3:
        raise ValueError('Embedded pose has an invalid quaternion')
    yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
    yield _checked(header, position.x, position.y, yaw)


def _checked(header, x, y, yaw):
    stamp = c.as_ns(header.stamp.sec * c.NS_PER_S + header.stamp.nanosec)
    if not 0 <= header.stamp.nanosec < c.NS_PER_S:
        raise ValueError('Invalid embedded pose timestamp')
    initial = c.InitialPose(timestamp_ns=stamp, x=x, y=y, yaw=yaw)
    return {'timestamp_ns': str(stamp), 'x': initial.x, 'y': initial.y, 'yaw': initial.yaw}


def poses(bag, source, start_ns=0, end_ns=None):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with Path(bag).open('rb') as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for schema, _, _, decoded in reader.iter_decoded_messages(topics=[source['topic']],
                start_time=max(0, start_ns-c.NS_PER_S),
                end_time=None if end_ns is None else end_ns+c.NS_PER_S):
            if schema.name != source['kind']:
                raise ValueError('Embedded pose schema changed')
            for pose in _pose(decoded, schema.name):
                stamp = int(pose['timestamp_ns'])
                if stamp >= start_ns and (end_ns is None or stamp < end_ns):
                    yield pose


def inspect_recording(bag, bank):
    """Read schema inventory and exactly one initialization pose, not a GT trajectory."""
    from mcap.reader import make_reader

    bag, bank = Path(bag), Path(bank)
    manifest = json.loads((bank/'manifest.json').read_text())
    original = manifest['source_bag']
    training = False
    if bag.stat().st_size == original['bytes']:
        with bag.open('rb') as handle:
            training = hashlib.file_digest(handle, 'sha256').hexdigest() == original['sha256']
    bounds = c.load_split()['holdout'] if training else None
    start = 0 if bounds is None else bounds.start_ns
    end = None if bounds is None else bounds.end_ns
    with bag.open('rb') as handle:
        summary = make_reader(handle).get_summary()
        if summary is None:
            raise ValueError('MCAP has no readable indexed summary')
        channels = [(channel, summary.schemas.get(channel.schema_id)) for channel in summary.channels.values()]
    sources, initial_sources = [], []
    for channel, schema in channels:
        if schema is None or schema.encoding != 'ros2msg' or channel.message_encoding != 'cdr':
            continue
        source = {'topic': channel.topic, 'kind': schema.name}
        if channel.topic in INITIAL_TOPICS and schema.name in POSE_TYPES:
            initial_sources.append(source)
        elif ((channel.topic == '/tf' and schema.name == 'tf2_msgs/msg/TFMessage')
                or (channel.topic == '/constructor0/state_estimation' and schema.name == 'sd_localization_msgs/msg/StateEstimation')
                or (channel.topic in GT_TOPICS and schema.name in POSE_TYPES)):
            sources.append(source)
    sources.sort(key=lambda s: (0 if s['topic'] == '/tf' else
                               2 if s['topic'] == '/constructor0/state_estimation' else 1, s['topic']))
    reference = next((s for s in sources if next(poses(bag, s, start, end), None) is not None), None)
    initial = None
    initial_source = None
    for source in [*initial_sources, *([reference] if reference else [])]:
        initial = next(poses(bag, source, start, end), None)
        if initial is not None:
            initial_source = source
            break
    if initial is None:
        raise ValueError('No supported initial pose in this MCAP. Expected a map-frame initial pose, map-to-base /tf, or ground-truth pose topic.')
    motion = None
    for mode in ('validation-twist', 'state-fields'):
        candidates = [(channel, schema) for channel, schema in channels if channel.topic == vm.TOPICS[mode]]
        if candidates:
            for _, schema in candidates:
                vm._check_schema(schema, mode)
            motion = mode
            break
    return {'initial_pose': initial, 'initial_source': initial_source, 'reference': reference,
            'mode': 'holdout' if training else 'input', 'vehicle_motion': motion,
            'training_recording': training}


def finish_recording(bag, folder, inputs, inspection, root):
    """Called only after the localizer exits; GT never becomes a runner argument."""
    from camloc.io import open_frame_sets
    from validate_run import validate

    folder, inputs, root = Path(folder), Path(inputs), Path(root)
    status = folder/'evaluation_status.json'
    status.write_text(json.dumps({'state': 'evaluating'})+'\n')
    try:
        meta = json.loads((folder/'run.json').read_text())
        initial = c.parse_initial_pose(json.dumps(inspection['initial_pose']))
        original_bounds = c.EpisodeBounds(name=meta['episode']['name'],
            start_ns=int(meta['episode']['start_ns']),
            end_ns=None if meta['episode']['end_ns'] is None else int(meta['episode']['end_ns']))
        # Expected groups still scan the original interval to EOF: deriving the coverage
        # endpoint cannot conceal any missing camera group at the recording tail.
        bounds = original_bounds
        if bounds.end_ns is None:
            with (folder/'submission.csv').open() as handle:
                stamps = [int(r['timestamp_ns']) for r in csv.DictReader(handle)]
            cadence = sorted(b-a for a,b in zip(stamps, stamps[1:]))
            period = cadence[len(cadence)//2] if cadence else c.NS_PER_S//10
            bounds = c.EpisodeBounds(name='input', start_ns=bounds.start_ns, end_ns=stamps[-1]+period)
        groups = open_frame_sets(bag, initial_pose=initial, episode=original_bounds,
                                 cameras=tuple(meta['selected_cameras']))
        result = validate(folder/'submission.csv', folder/'diagnostics.jsonl', groups, bounds)
        (folder/'validated.json').write_text(json.dumps(result, indent=2)+'\n')
        reference = inspection['reference']
        if reference is None:
            status.write_text(json.dumps({'state': 'unavailable', 'reason': 'No supported ground-truth topic; localization completed.'})+'\n')
            return
        rows = {}
        for pose in poses(bag, reference, original_bounds.start_ns, original_bounds.end_ns):
            stamp = int(pose['timestamp_ns'])
            if stamp in rows and rows[stamp] != pose:
                raise ValueError('Ground truth contains conflicting duplicate timestamps')
            rows[stamp] = pose
        if len(rows) < 2:
            raise ValueError('At least two ground-truth poses are needed for scoring')
        gt = inputs/'ground_truth.csv'
        with gt.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['timestamp_ns', 'x', 'y', 'yaw'])
            writer.writeheader()
            writer.writerows(rows[t] for t in sorted(rows))
        evaluation = folder/'validation'
        evaluation.mkdir(exist_ok=True)
        with (evaluation/'score.txt').open('w') as log:
            subprocess.run([sys.executable, '-B', str(root/'tools/evaluate.py'), str(gt),
                            str(folder/'submission.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
        status.write_text(json.dumps({'state': 'completed', 'gt': str(gt.resolve()),
                                     'source': reference, 'ground_truth_rows': len(rows)})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'state': 'failed', 'reason': str(error)})+'\n')
    finally:
        output = root/'results'/folder.name
        output.mkdir(parents=True, exist_ok=False)
        for name in ('submission.csv', 'runtime.json', 'validated.json'):
            if (folder/name).is_file():
                shutil.copyfile(folder/name, output/name)
        shutil.copyfile(folder/'run.json', output/'run_metadata.json')
        if (folder/'validation').is_dir():
            shutil.copytree(folder/'validation', output/'validation')
        state = json.loads(status.read_text())
        state.pop('gt', None)
        (output/'validation_status.json').write_text(json.dumps(state, indent=2)+'\n')

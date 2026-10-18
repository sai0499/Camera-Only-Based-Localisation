"""Automatic UI input reads one initial pose; evaluation remains after inference."""
import io
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
from mcap_ros2.writer import Writer
from camloc.vehicle_motion import STATE_PREFIX

from scripts.mcap_input import finish_recording, inspect_recording
from scripts.serve_replay import Workspace, make_server

HEADER = '''
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
'''
POSE = '''std_msgs/Header header
geometry_msgs/Pose pose
''' + HEADER + '''
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
'''
TWIST = '''std_msgs/Header header
Twist twist
''' + HEADER + '''
================================================================================
MSG: geometry_msgs/Twist
Vector3 linear
Vector3 angular
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
'''


def message(stamp, x=1., frame='map'):
    return {'header': {'stamp': {'sec': stamp//10**9, 'nanosec': stamp%10**9}, 'frame_id': frame},
            'pose': {'position': {'x': x, 'y': 2., 'z': 0.},
                     'orientation': {'x': 0., 'y': 0., 'z': 0., 'w': 1.}}}


def bag(path, *, initial=False, future=50., topic='/ground_truth', frame='map', motion=False):
    with Writer(str(path)) as writer:
        schema = writer.register_msgdef('geometry_msgs/msg/PoseStamped', POSE)
        if initial:
            writer.write_message('/initial_pose', schema, message(10**9, x=7.), log_time=10**9)
        for i in range(3):
            stamp = 10**9+i*10_000_000
            writer.write_message(topic, schema, message(stamp, x=1. if i==0 else future, frame=frame), log_time=stamp)
        if motion:
            schema = writer.register_msgdef('geometry_msgs/msg/TwistStamped', TWIST)
            value = {'header': message(10**9)['header'],
                     'twist': {'linear': {'x': 20., 'y': 0., 'z': 0.},
                               'angular': {'x': 0., 'y': 0., 'z': .1}}}
            value['header']['frame_id'] = 'constructor0/base_link'
            writer.write_message('/constructor0/validation/twist', schema, value, log_time=2*10**9)
    return path


@pytest.fixture
def bank(tmp_path):
    p = tmp_path/'bank'
    p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'source_bag': {'bytes': 0, 'sha256': 'not-training'}}))
    return p


def test_automatic_initial_pose_cannot_depend_on_later_gt(tmp_path, bank):
    a = inspect_recording(bag(tmp_path/'a.mcap', future=50), bank)
    b = inspect_recording(bag(tmp_path/'b.mcap', future=5000), bank)
    assert a == b
    assert a['initial_pose'] == {'timestamp_ns': '1000000000', 'x': 1., 'y': 2., 'yaw': 0.}
    assert a['mode'] == 'input' and a['vehicle_motion'] is None


def test_explicit_initial_pose_and_supported_motion_are_automatic(tmp_path, bank):
    result = inspect_recording(bag(tmp_path/'a.mcap', initial=True, motion=True), bank)
    assert result['initial_pose']['x'] == 7.
    assert result['initial_source']['topic'] == '/initial_pose'
    assert result['reference']['topic'] == '/ground_truth'
    assert result['vehicle_motion'] == 'validation-twist'


def test_explicit_ground_truth_precedes_state_estimation_pose(tmp_path, bank):
    path = tmp_path/'mixed.mcap'
    with Writer(str(path)) as writer:
        pose_schema = writer.register_msgdef('geometry_msgs/msg/PoseStamped', POSE)
        writer.write_message('/ground_truth', pose_schema, message(10**9), log_time=10**9)
        state_schema = writer.register_msgdef('sd_localization_msgs/msg/StateEstimation', '\n'.join(STATE_PREFIX)+HEADER)
        state = {'header': message(10**9)['header'], 'se_status': 0, 'se_state': 0,
                 'x_m': 99., 'y_m': 2., 'z_m': 0., 'roll_rad': 0., 'pitch_rad': 0., 'yaw_rad': 0.,
                 'pos_accuracy': [0.,0.,0.], 'vel_accuracy': [0.,0.,0.],
                 'wx_radps': 0., 'wy_radps': 0., 'wz_radps': 0., 'vx_mps': 20., 'vy_mps': 0., 'vz_mps': 0.}
        writer.write_message('/constructor0/state_estimation', state_schema, state, log_time=10**9+1)
    result = inspect_recording(path, bank)
    assert result['initial_pose']['x'] == 1.
    assert result['reference']['topic'] == '/ground_truth'
    assert result['vehicle_motion'] == 'state-fields'


@pytest.mark.parametrize('options', [{'topic': '/unrelated'}, {'frame': 'odom'}])
def test_missing_or_wrong_frame_pose_is_rejected(tmp_path, bank, options):
    with pytest.raises(ValueError, match='initial pose'):
        inspect_recording(bag(tmp_path/'a.mcap', **options), bank)


def test_upload_streams_exact_file_and_rejects_traversal_and_partial_content(tmp_path):
    w = Workspace(tmp_path)
    data = bag(tmp_path/'input.mcap').read_bytes()
    result = w.upload(io.BytesIO(data), len(data), 'judges.mcap')
    assert (tmp_path/result['bag']).read_bytes() == data
    with pytest.raises(ValueError, match='MCAP'):
        w.upload(io.BytesIO(data), len(data), '../escape.mcap')
    with pytest.raises(ValueError, match='interrupted'):
        w.upload(io.BytesIO(data[:20]), len(data), 'partial.mcap')
    with pytest.raises(ValueError, match='MCAP'):
        w.upload(io.BytesIO(b'not a bag'), 9, 'invalid.mcap')
    assert len(list(w.replay_cache.glob('uploads/*/*.mcap'))) == 1


def test_upload_endpoint_requires_local_session_and_returns_uploaded_file(tmp_path):
    w = Workspace(tmp_path)
    data = bag(tmp_path/'input.mcap').read_bytes()
    server = make_server(0, w)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/api/upload'
        request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/octet-stream',
                                                                  'X-File-Name': 'judges.mcap'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 403
        request.add_header('X-Replay-Token', server.token)
        with urllib.request.urlopen(request, timeout=5) as response:
            uploaded = json.load(response)
        assert (tmp_path/uploaded['bag']).read_bytes() == data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_automatic_start_has_no_configuration_or_gt_argument(tmp_path, bank, monkeypatch):
    w = Workspace(tmp_path)
    monkeypatch.setattr(w, 'config', lambda: {'default_artifacts': 'bank'})
    start = Mock(return_value={'id': 'started'})
    monkeypatch.setattr(w, 'start', start)
    assert w.start_auto({'bag': str(bag(tmp_path/'input.mcap'))}) == {'id': 'started'}
    settings = start.call_args.args[0]
    assert set(settings) == {'bag', 'artifacts', 'retrieval', 'mode', 'initial_pose', 'vehicle_motion'}
    assert settings['retrieval'] and settings['initial_pose']['x'] == 1.
    assert start.call_args.kwargs['inspection']['reference']['topic'] == '/ground_truth'


def test_automatic_run_never_attaches_unrelated_development_reference(tmp_path):
    w = Workspace(tmp_path)
    inputs = w.replay_cache/'inputs/job'
    inputs.mkdir(parents=True)
    (inputs/'inspection.json').write_text('{}')
    folder = w.replay_cache/'job'
    folder.mkdir()
    r = SimpleNamespace(folder=folder, state='completed', comparison=None,
                        meta={'split': 'tune'}, attach_gt=Mock())
    w.auto_gt(r)
    assert r.evaluation['state'] == 'pending'
    r.attach_gt.assert_not_called()
    gt = inputs/'ground_truth.csv'
    (folder/'evaluation_status.json').write_text(json.dumps({'state': 'completed', 'gt': str(gt)}))
    w.auto_gt(r)
    r.attach_gt.assert_called_once_with(gt)


def test_completed_run_extracts_gt_scores_and_publishes_only_outputs(tmp_path, bank, monkeypatch):
    source = bag(tmp_path/'input.mcap')
    inspection = inspect_recording(source, bank)
    folder = tmp_path/'tmp/replay_ui/job'
    folder.mkdir(parents=True)
    inputs = tmp_path/'tmp/replay_ui/inputs/job'
    inputs.mkdir(parents=True)
    meta = {'groups': 3, 'episode': {'name': 'input', 'start_ns': 10**9, 'end_ns': 1_030_000_000},
            'selected_cameras': ['fr']}
    (folder/'run.json').write_text(json.dumps(meta))
    (folder/'submission.csv').write_text('timestamp_ns,x,y,yaw\n1005000000,1,2,0\n1015000000,1,2,0\n1025000000,1,2,0\n')
    (folder/'diagnostics.jsonl').write_text('')
    from camloc import io as camera_io
    monkeypatch.setattr(camera_io, 'open_frame_sets', create_autospec(camera_io.open_frame_sets, return_value=iter(())))
    monkeypatch.setattr('validate_run.validate', lambda *args: {'valid': True})
    def scorer(command, **kwargs):
        assert command[-2] == str(inputs/'ground_truth.csv')
        assert (inputs/'ground_truth.csv').read_text().count('\n') == 4
        kwargs['stdout'].write('position RMSE [m] 0.0\n')
    monkeypatch.setattr('scripts.mcap_input.subprocess.run', scorer)
    finish_recording(source, folder, inputs, inspection, tmp_path)
    status = json.loads((folder/'evaluation_status.json').read_text())
    assert status['state'] == 'completed', status
    output = tmp_path/'results/job'
    assert (output/'submission.csv').read_bytes() == (folder/'submission.csv').read_bytes()
    assert (output/'validation/score.txt').is_file()
    assert not (output/'ground_truth.csv').exists() and not (output/'diagnostics.jsonl').exists()

"""The audit must preserve permitted MCAP messages and detect meaningful replay changes."""

import json
import math

import cv2
import numpy as np
import pytest
from mcap.reader import make_reader
from mcap_ros2._dynamic import serialize_dynamic
from mcap_ros2.decoder import DecoderFactory

from camloc import contracts as c, io
from scripts import check_step5 as audit
from scripts.report_step5_errors import errors
from test_io import _write_fixture


def fixture_bag(tmp_path, *, jpeg=False):
    path = tmp_path / 'source.mcap'
    _write_fixture(path, images=[(camera, sec * c.NS_PER_S, sec * c.NS_PER_S + 40_000_000)
        for sec in (29, 30, 31, 35, 36, 37, 38) for camera in ('fr', 'r')], include_prohibited=True)
    if not jpeg:
        return path
    decoder = DecoderFactory()
    image = np.tile(np.arange(200, dtype=np.uint8)[None, :, None], (100, 1, 3))
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok

    def valid_image(schema, channel, message):
        if not channel.topic.endswith('/compressed_image'):
            return message.data
        decoded = decoder.decoder_for(channel.message_encoding, schema)(message.data)
        decoded.data = encoded.tobytes()
        return serialize_dynamic(schema.name, schema.data.decode())[schema.name](decoded)

    result = tmp_path / 'jpeg.mcap'
    audit.copy_messages(path, result, lambda channel, message: True, valid_image)
    return result


@pytest.mark.parametrize('case', ['prohibited_remove', 'prohibited_poison'])
def test_prohibited_mutation_preserves_permitted_bytes_timing_and_calibration(tmp_path, case):
    source = fixture_bag(tmp_path)
    baseline, changed = tmp_path / 'base.mcap', tmp_path / f'{case}.mcap'
    original = audit.copy_messages(source, baseline, lambda channel, message: True)
    receipt = audit.variant(baseline, changed, case, 20 * c.NS_PER_S)
    assert original['permitted_messages_sha256'] == receipt['permitted_messages_sha256']
    assert receipt['messages']['removed' if case.endswith('remove') else 'changed'] == 2
    assert io.read_calibrations(source)['fr'].calibration_id == io.read_calibrations(changed)['fr'].calibration_id
    with changed.open('rb') as handle:
        records = list(make_reader(handle, validate_crcs=True).iter_messages())
    prohibited = [message for _, channel, message in records if channel.topic in io.PROHIBITED_TOPICS]
    assert len(prohibited) == (0 if case.endswith('remove') else 2)
    if prohibited:
        assert all(message.data == b'PROHIBITED_PAYLOAD_MUST_NOT_BE_DECODED' for message in prohibited)


@pytest.mark.parametrize('case', ['future', 'blackout_0.5', 'blackout_1', 'blur', 'exposure', 'glare',
                                  'fr_drop_1', 'fr_drop_3', 'r_drop_1', 'r_drop_3'])
def test_image_variants_change_only_the_declared_exposures(tmp_path, case):
    source = fixture_bag(tmp_path, jpeg=True)
    output = tmp_path / f'{case}.mcap'
    receipt = audit.variant(source, output, case, 20 * c.NS_PER_S)
    assert receipt['changed_images'] > 0
    assert receipt['first_changed_exposure_ns'] == (30 if case == 'future' else 35) * c.NS_PER_S
    with source.open('rb') as handle:
        before = [(ch.topic, msg.log_time, msg.publish_time, msg.sequence, msg.data)
                  for _, ch, msg in make_reader(handle).iter_messages()]
    with output.open('rb') as handle:
        after = [(ch.topic, msg.log_time, msg.publish_time, msg.sequence, msg.data)
                 for _, ch, msg in make_reader(handle, validate_crcs=True).iter_messages()]
    def early(row):
        return row[1] < receipt['first_changed_exposure_ns']
    assert list(filter(early, before)) == list(filter(early, after))
    if '_drop_' in case:
        camera = case.split('_')[0]
        assert len(after) < len(before)
        assert [row for row in before if row[0] != io.image_topic(camera)] == [
            row for row in after if row[0] != io.image_topic(camera)]
    else:
        assert [row[:4] for row in before] == [row[:4] for row in after]


def replay(folder, *, x=0., yaw=0., status='tracking', stamp=100):
    folder.mkdir()
    (folder / 'submission.csv').write_text(f'timestamp_ns,x,y,yaw\n{stamp},{x},0,{yaw}\n')
    record = {'record': 'Estimate', 'group_id': 0, 'status': status, 'branch': 'unknown',
              'evidence_kinds': ['pnp'], 'reason': 'accepted', 'max_support_exposure_ns': stamp}
    (folder / 'diagnostics.jsonl').write_text(json.dumps({'record': 'FrameSet', 'group_id': 0,
        't_ns': stamp}) + '\n' + json.dumps(record) + '\n')
    return folder


def test_comparison_uses_declared_tolerance_wrapped_yaw_and_logical_decisions(tmp_path):
    before = replay(tmp_path / 'a', yaw=math.pi)
    close = replay(tmp_path / 'b', x=1e-6, yaw=-math.pi)
    assert audit.compare(before, close)['passed']
    far = replay(tmp_path / 'c', x=.1)
    assert not audit.compare(before, far)['passed']
    other_status = replay(tmp_path / 'd', yaw=math.pi, status='lost')
    assert not audit.compare(before, other_status)['passed']
    different_stamp = replay(tmp_path / 'e', stamp=101)
    assert not audit.compare(before, different_stamp)['passed']
    assert not audit.compare(before, close, before_ns=100)['passed']


def test_no_change_cannot_pass_a_vacuous_future_check(tmp_path):
    source = fixture_bag(tmp_path)
    with pytest.raises(c.ContractError, match='no input changed'):
        audit.variant(source, tmp_path / 'no_future.mcap', 'future', 100 * c.NS_PER_S)


def test_comparison_detects_changed_selected_landmarks_with_identical_csv(tmp_path):
    before, after = replay(tmp_path / 'a'), replay(tmp_path / 'b')
    for folder, landmark in ((before, 1), (after, 2)):
        evidence = {'record': 'GroupEvidence', 'frame_set': {'group_id': 0}, 'reset': None,
                    'pose': {'record': 'PoseEvidence', 'camera_id': 'fr', 'exposure_ns': 100,
                             'source': 'tracking', 'landmark_ids': [landmark]}}
        with (folder / 'diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps(evidence) + '\n')
    assert not audit.compare(before, after)['same_logical_decisions']


def test_output_refuses_overwrite(tmp_path):
    source = fixture_bag(tmp_path)
    with pytest.raises(FileExistsError):
        audit.copy_messages(source, source, lambda channel, message: True)


def test_processing_failure_is_not_hidden_by_successful_csv_output(tmp_path):
    path = tmp_path / 'diagnostics.jsonl'
    path.write_text('\n'.join(json.dumps({'record': kind}) for kind in (
        'Estimate', 'ProjectionError', 'DecodeError', 'ProjectionError', 'RunAborted'))+'\n')
    assert audit.processing_errors(path) == {'ProjectionError': 2, 'DecodeError': 1, 'RunAborted': 1}


def test_offline_errors_preserve_integer_time_yaw_wrap_and_gt_gap_gate():
    start = 1_791_918_859_000_000_000
    gt = {'timestamp_ns': np.array([start, start+10_000_000, start+100_000_000], np.int64),
          'x': np.array([0., 2., 4.]), 'y': np.zeros(3),
          'yaw': np.array([math.pi-.1, -math.pi+.1, -math.pi+.2])}
    sub = {'timestamp_ns': np.array([start, start+5_000_000, start+50_000_000], np.int64),
           'x': np.array([0., 1., 3.]), 'y': np.zeros(3), 'yaw': np.array([0., math.pi, 0.])}
    stamps, position, yaw = errors(gt, sub)
    assert stamps.tolist() == [start+5_000_000]
    assert position[0] == 0 and abs(yaw[0]) < 1e-12

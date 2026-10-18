"""Optional telemetry preserves frames and source-time behavior; pacing never drops work."""

import json

import pytest

import contract_factories as f
from camloc import contracts as c, runtime


class Source:
    def __init__(self, frames):
        self.frames = frames

    def __iter__(self):
        return iter(self.frames)

    def trailing_events(self):
        return ()

    def calibrations(self):
        return {}


@pytest.mark.parametrize('speed', [None, 1., 6.])
def test_runtime_preserves_frames_and_charges_backpressure(tmp_path, monkeypatch, speed):
    clock = [0]
    sleeps = []
    monkeypatch.setattr(runtime.time, 'perf_counter_ns', lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += round(seconds * c.NS_PER_S)

    monkeypatch.setattr(runtime.time, 'sleep', sleep)
    frames = [f.make_frame_set(group_id=k, images=(f.make_image_message(
        exposure_ns=f.EXPOSURE_NS + k * 100_000_000,
        arrival_ns=f.EXPOSURE_NS + k * 100_000_000 + 40_000_000),), selected_camera_ids=('fr',))
        for k in range(4)]
    source = runtime.RuntimeSource(Source(frames), tmp_path / 'runtime.jsonl', transport_speed=speed)
    assert isinstance(source, c.FrameSetSource)
    delivered = []
    for frame in source:
        delivered.append(frame)
        clock[0] += 70_000_000  # includes a 20 ms worker join for group 0
    assert all(a is b for a, b in zip(delivered, frames, strict=True))
    assert source.trailing_events() == ()
    records = [json.loads(line) for line in source.output.read_text().splitlines()]
    assert len(records) == 4 and all(r['group_service_ns'] == 70_000_000 for r in records)
    log = tmp_path / 'diagnostics.jsonl'
    log.write_text(json.dumps({'record': 'SchedulerEvent', 'kind': 'stall', 'group_id': 0,
                              'stall_ns': 20_000_000}) + '\n')
    report = source.summary(log)
    assert not report['rate_qualified']
    assert report['group_service']['mean_ms'] == 70
    assert report['nonwaiting_pipeline']['mean_ms'] == 65
    if speed == 1:
        assert len(sleeps) == 3
    else:
        assert sleeps == []


@pytest.mark.parametrize('speed', [0., -1., float('nan'), float('inf')])
def test_invalid_pacing_speed_is_rejected(tmp_path, speed):
    with pytest.raises(ValueError, match='finite and positive'):
        runtime.RuntimeSource(Source([]), tmp_path / 'runtime.jsonl', transport_speed=speed)


def test_windows_peak_memory_is_positive():
    result = runtime.process_peak_memory()
    if result is not None:
        assert result['peak_working_set_bytes'] > 0 and result['peak_commit_bytes'] > 0


def test_instrumentation_preserves_production_runner_predictions(tmp_path):
    import stage4_fakes as s4
    from test_runner_stage4 import anchor_frontend, run, scheduler_for

    frames = s4.synthetic_stream(12, 10)
    original = run(tmp_path, frames, anchor_frontend(), scheduler=scheduler_for(s4.FakeWorker()))
    timed = runtime.RuntimeSource(Source(frames), tmp_path / 'runtime.jsonl')
    measured = run(tmp_path, timed, anchor_frontend(), scheduler=scheduler_for(s4.FakeWorker()))
    assert measured['csv'] == original['csv']
    assert measured['counts']['groups'] == len(timed.samples) == 12

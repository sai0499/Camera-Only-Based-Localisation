"""A's real reader/remap/projector feeding C, without fabricated B visual evidence."""
import json

import pytest

from camloc import contracts as c
from camloc import geometry, io, track_coords
from camloc.filter import CameraFilter
from run import run_stream
from test_geometry import _jpeg_bytes
from test_io import _write_fixture


@pytest.mark.parametrize("bad_image", [False, True])
def test_a_input_and_route_diagnostics_flow_through_c(tmp_path, monkeypatch, bad_image):
    from mcap_ros2.writer import Writer

    payload = _jpeg_bytes(1506, 728)
    original = Writer.write_message
    start = 100 * c.NS_PER_S

    def write(writer, **kwargs):
        if kwargs["topic"].endswith("/compressed_image"):
            kwargs["message"]["data"] = (b"invalid-jpeg" if bad_image
                                         and kwargs["log_time"] == start else payload)
        original(writer, **kwargs)

    monkeypatch.setattr(Writer, "write_message", write)
    images = [(camera, start + i * 100_000_000 + offset, start + i * 100_000_000 + offset)
              for i in range(3) for camera, offset in (("fr", 0), ("r", 500_000))]
    bag = tmp_path / "images.mcap"
    _write_fixture(bag, images=images, include_prohibited=True)
    initial = c.InitialPose(timestamp_ns=start, x=0., y=0., yaw=0.)
    episode = c.EpisodeBounds(name="full", start_ns=start, end_ns=None)
    source = io.open_frame_sets(bag, initial_pose=initial, episode=episode)
    canonicalizer = geometry.make_canonicalizer(source.calibrations())
    projector = track_coords.load_track_projector(c.PROJECT_ROOT / "23-00-38_cut_120m_update/map.mcap")

    def frontend(frame, prior):
        for image in frame.images:
            observation = canonicalizer.canonicalize(image)
            assert observation.image.shape == (464, 960, 3)
            assert not observation.image.flags.writeable
        return c.GroupEvidence(frame_set=frame, pose=None, motion=None, border=None,
                               reset=None, rejected=(), unused=())

    csv, diagnostics = tmp_path / "output.csv", tmp_path / "diagnostics.jsonl"
    counts = run_stream(source, CameraFilter(initial), frontend, output=csv,
                        diagnostics=diagnostics, episode=episode, projector=projector)
    records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    projections = [r for r in records if r["record"] == "TrackProjection"]
    assert len(csv.read_text().splitlines()) == 4
    assert counts == {"groups": 3, "frontend_errors": int(bad_image), "prediction_only": 3}
    assert len(projections) == 6
    assert [r["route_id"] for r in projections] == ["track", "pit"] * 3
    assert all(r["status"] == "lost" for r in records if r["record"] == "Estimate")
    assert not any(r["record"] == "ProjectionError" for r in records)

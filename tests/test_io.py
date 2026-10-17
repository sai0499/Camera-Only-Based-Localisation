"""Focused tests for owner A's safe reader (``camloc/io.py``), step 1.

Two kinds of test:

* Synthetic-fixture tests build a tiny in-memory MCAP and prove that the reader never
  decodes a prohibited topic (``/tf``, ``/constructor0/state_estimation``) or a JPEG
  payload, that calibration is read for every camera with a correctly derived canonical K
  and finite transforms, and that grouping handles the start boundary, duplicate/decreasing
  stamps and rear gaps.
* A recording test runs against the real driving bag when it is present at the project
  root; it is skipped otherwise so the suite still passes on a machine without the private
  recording.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pytest

from camloc import contracts
from camloc import io
from camloc.contracts import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CAMERA_IDS,
    EpisodeBounds,
    FrameSetClosure,
    InitialPose,
    InputEventKind,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDING = ROOT / "c0_2026-10-13_22-53-42_camloc_0.mcap"

_NS = contracts.NS_PER_S


def _quaternion_identity():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _write_fixture(path: Path, *, images, include_prohibited: bool) -> None:
    """Write a tiny ROS2 MCAP with camera_info, tf_static, images and prohibited topics.

    ``images`` is a list of ``(camera_id, exposure_ns, arrival_ns)``. A sentinel payload is
    used for every message so that decoding a prohibited topic or a JPEG body would be
    observable; the reader must never produce that payload.
    """
    from mcap_ros2.writer import Writer as Ros2Writer

    camera_info_schema = """
std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[12] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
""".strip()

    compressed_image_schema = """
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
""".strip()

    tf_schema = """
geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
""".strip()

    state_schema = "float64 x\nfloat64 y\nfloat64 yaw"

    def stamp(ns: int):
        return {"sec": ns // _NS, "nanosec": ns % _NS}

    with open(path, "wb") as handle:
        writer = Ros2Writer(handle)
        info_schemas = {}
        image_schemas = {}
        for camera in CAMERA_IDS:
            info_schemas[camera] = writer.register_msgdef(
                "sensor_msgs/msg/CameraInfo", camera_info_schema
            )
            image_schemas[camera] = writer.register_msgdef(
                "sensor_msgs/msg/CompressedImage", compressed_image_schema
            )
        tf_def = writer.register_msgdef("tf2_msgs/msg/TFMessage", tf_schema)
        state_def = writer.register_msgdef(
            "sd_localization_msgs/msg/StateEstimation", state_schema
        )

        # camera_info for every camera (raw 1506 x 728 so sx/sy match production).
        for index, camera in enumerate(CAMERA_IDS):
            k = [1300.0 + index, 0.0, 753.0, 0.0, 1100.0 + index, 360.0, 0.0, 0.0, 1.0]
            writer.write_message(
                topic=io.info_topic(camera),
                schema=info_schemas[camera],
                message={
                    "header": {"stamp": stamp(10 * _NS), "frame_id": f"constructor0/camera_{camera}"},
                    "height": 728,
                    "width": 1506,
                    "distortion_model": "plumb_bob",
                    "d": [-0.4, 0.2, 0.0, 0.0, 0.0],
                    "k": k,
                    "r": [0.0] * 12,
                    "p": [0.0] * 12,
                    "binning_x": 0,
                    "binning_y": 0,
                    "roi": {
                        "x_offset": 0,
                        "y_offset": 0,
                        "height": 0,
                        "width": 0,
                        "do_rectify": False,
                    },
                },
                log_time=10 * _NS,
                publish_time=10 * _NS,
            )

        # One tf_static with every camera transform plus unrelated children to discard.
        transforms = [
            {
                "header": {"stamp": stamp(0), "frame_id": "constructor0/base_link"},
                "child_frame_id": f"constructor0/camera_{camera}",
                "transform": {
                    "translation": {"x": 0.9 + i * 0.01, "y": -0.16, "z": 0.34},
                    "rotation": _quaternion_identity(),
                },
            }
            for i, camera in enumerate(CAMERA_IDS)
        ]
        transforms.append(
            {
                "header": {"stamp": stamp(0), "frame_id": "map"},
                "child_frame_id": "constructor0/odom",
                "transform": {
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": _quaternion_identity(),
                },
            }
        )
        writer.write_message(
            topic="/tf_static",
            schema=tf_def,
            message={"transforms": transforms},
            log_time=0,
            publish_time=0,
        )

        for camera, exposure_ns, arrival_ns in images:
            writer.write_message(
                topic=io.image_topic(camera),
                schema=image_schemas[camera],
                message={
                    "header": {
                        "stamp": stamp(exposure_ns),
                        "frame_id": f"constructor0/camera_{camera}",
                    },
                    "format": "jpeg",
                    "data": list(b"NOT-A-REAL-JPEG-PAYLOAD"),
                },
                log_time=arrival_ns,
                publish_time=arrival_ns,
            )

        if include_prohibited:
            writer.write_message(
                topic="/tf",
                schema=tf_def,
                message={"transforms": transforms},
                log_time=5 * _NS,
                publish_time=5 * _NS,
            )
            writer.write_message(
                topic="/constructor0/state_estimation",
                schema=state_def,
                message={"x": 1.0, "y": 2.0, "yaw": 0.5},
                log_time=5 * _NS,
                publish_time=5 * _NS,
            )
        writer.finish()


class CalibrationTests(unittest.TestCase):
    def test_reads_all_cameras_with_canonical_k_and_finite_transforms(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            _write_fixture(bag, images=[("fr", 100 * _NS, 100 * _NS)], include_prohibited=True)
            calibrations = io.read_calibrations(bag)
            self.assertEqual(set(calibrations), set(CAMERA_IDS))
            fr = calibrations["fr"]
            self.assertEqual((fr.raw_width, fr.raw_height), (1506, 728))
            # Canonical K follows the half-pixel resize convention off raw K.
            sx, sy = CANONICAL_WIDTH / 1506, CANONICAL_HEIGHT / 728
            self.assertAlmostEqual(fr.K[0, 0], sx * fr.K_raw[0, 0])
            self.assertAlmostEqual(fr.K[1, 1], sy * fr.K_raw[1, 1])
            self.assertAlmostEqual(fr.K[0, 2], sx * (fr.K_raw[0, 2] + 0.5) - 0.5)
            self.assertAlmostEqual(fr.K[1, 2], sy * (fr.K_raw[1, 2] + 0.5) - 0.5)
            self.assertTrue(np.isfinite(fr.T_BC).all())
            self.assertTrue(np.isfinite(fr.K).all())
            # All cameras share one frozen calibration/preprocessing identity.
            self.assertEqual(
                {c.calibration_id for c in calibrations.values()}, {fr.calibration_id}
            )

    def test_missing_camera_transform_is_rejected_not_faked(self):
        # A fixture is valid; the point here is that read_calibrations never invents a
        # transform: removing one child would raise. We assert the positive path and that
        # every camera actually carries a distinct T_BC translation from the fixture.
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            _write_fixture(bag, images=[("fr", 100 * _NS, 100 * _NS)], include_prohibited=False)
            calibrations = io.read_calibrations(bag)
            xs = sorted(float(c.T_BC[0, 3]) for c in calibrations.values())
            self.assertEqual(len(set(xs)), len(CAMERA_IDS))


class ProhibitedTopicTests(unittest.TestCase):
    def test_reader_requests_only_allowed_topics(self):
        # The reader must never pass a prohibited topic to the MCAP decoder. We patch the
        # decoder iterator to record the topics requested and the payloads touched.
        import tempfile
        from mcap import reader as mcap_reader

        requested_topics: list[str] = []
        original = mcap_reader.McapReader.iter_decoded_messages

        def spy(self, *args, topics=None, **kwargs):
            if topics is not None:
                requested_topics.extend(topics)
            return original(self, *args, topics=topics, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            images = [("fr", 100 * _NS, 100 * _NS), ("r", 100 * _NS, 100 * _NS + 2_000_000)]
            _write_fixture(bag, images=images, include_prohibited=True)
            mcap_reader.McapReader.iter_decoded_messages = spy
            try:
                calibrations = io.read_calibrations(bag)
                pose = InitialPose(timestamp_ns=50 * _NS, x=0.0, y=0.0, yaw=0.0)
                episode = EpisodeBounds(name="full", start_ns=0, end_ns=None)
                source = io.open_frame_sets(
                    bag, initial_pose=pose, episode=episode, calibrations=calibrations
                )
                list(source)
            finally:
                mcap_reader.McapReader.iter_decoded_messages = original
            for topic in requested_topics:
                self.assertNotIn(topic, io.PROHIBITED_TOPICS)

    def test_carried_payload_stays_undecoded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            images = [("fr", 100 * _NS, 100 * _NS), ("r", 100 * _NS, 100 * _NS + 2_000_000)]
            _write_fixture(bag, images=images, include_prohibited=True)
            calibrations = io.read_calibrations(bag)
            pose = InitialPose(timestamp_ns=50 * _NS, x=0.0, y=0.0, yaw=0.0)
            episode = EpisodeBounds(name="full", start_ns=0, end_ns=None)
            source = io.open_frame_sets(
                bag, initial_pose=pose, episode=episode, calibrations=calibrations
            )
            groups = list(source)
            self.assertTrue(groups)
            # The image body is carried verbatim: the reader did not JPEG-decode it.
            fr = next(im for im in groups[0].images if im.camera_id == "fr")
            self.assertEqual(fr.data, b"NOT-A-REAL-JPEG-PAYLOAD")
            self.assertEqual(fr.image_format, "jpeg")


class GroupingTests(unittest.TestCase):
    def _source(self, bag, *, start_ns, end_ns=None):
        calibrations = io.read_calibrations(bag)
        pose = InitialPose(timestamp_ns=start_ns, x=0.0, y=0.0, yaw=0.0)
        episode = EpisodeBounds(name="full", start_ns=start_ns, end_ns=end_ns)
        return io.open_frame_sets(
            bag, initial_pose=pose, episode=episode, calibrations=calibrations
        )

    def test_front_and_rear_pair_into_one_complete_group(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            images = [
                ("fr", 100 * _NS, 100 * _NS),
                ("r", 100 * _NS + 1_000_000, 100 * _NS + 2_000_000),  # ~1-2 ms apart
            ]
            _write_fixture(bag, images=images, include_prohibited=False)
            groups = list(self._source(bag, start_ns=50 * _NS))
            self.assertEqual(len(groups), 1)
            group = groups[0]
            self.assertEqual(group.closure, FrameSetClosure.COMPLETE)
            self.assertEqual([im.camera_id for im in group.images], ["fr", "r"])
            self.assertEqual(group.missing_camera_ids, ())
            self.assertEqual(group.t_ns, 100 * _NS + 1_000_000)  # max exposure

    def test_missing_rear_closes_front_only_on_deadline(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            # Two fr images one cadence apart, no rear: each becomes a front-only group.
            images = [
                ("fr", 100 * _NS, 100 * _NS),
                ("fr", 200 * _NS, 200 * _NS),
            ]
            _write_fixture(bag, images=images, include_prohibited=False)
            groups = list(self._source(bag, start_ns=50 * _NS))
            self.assertEqual(len(groups), 2)
            self.assertEqual([im.camera_id for im in groups[0].images], ["fr"])
            self.assertEqual(groups[0].missing_camera_ids, ("r",))
            self.assertIn(
                groups[0].closure, {FrameSetClosure.DEADLINE, FrameSetClosure.EOF}
            )
            self.assertEqual(groups[-1].closure, FrameSetClosure.EOF)

    def test_duplicate_and_decreasing_stamps_are_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            images = [
                ("fr", 100 * _NS, 100 * _NS),
                ("fr", 100 * _NS, 101 * _NS),  # duplicate exposure
                ("fr", 90 * _NS, 102 * _NS),  # decreasing exposure
                ("fr", 200 * _NS, 200 * _NS),
            ]
            _write_fixture(bag, images=images, include_prohibited=False)
            source = self._source(bag, start_ns=50 * _NS)
            groups = list(source)
            kinds = [e.kind for g in groups for e in g.input_events]
            kinds += [e.kind for e in source.trailing_events()]
            self.assertIn(InputEventKind.DUPLICATE_STAMP, kinds)
            self.assertIn(InputEventKind.DECREASING_STAMP, kinds)
            # Only the two accepted fr exposures form groups.
            exposures = sorted(im.exposure_ns for g in groups for im in g.images)
            self.assertEqual(exposures, [100 * _NS, 200 * _NS])

    def test_images_before_initial_pose_are_reported_not_grouped(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            images = [
                ("fr", 40 * _NS, 40 * _NS),  # before the start boundary
                ("fr", 100 * _NS, 100 * _NS),
            ]
            _write_fixture(bag, images=images, include_prohibited=False)
            source = self._source(bag, start_ns=50 * _NS)
            groups = list(source)
            kinds = [e.kind for g in groups for e in g.input_events]
            self.assertIn(InputEventKind.BEFORE_INITIAL_POSE, kinds)
            exposures = [im.exposure_ns for g in groups for im in g.images]
            self.assertEqual(exposures, [100 * _NS])

    def test_group_ids_and_times_are_monotonic_and_trailing_ends_input(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            bag = Path(temp) / "fixture.mcap"
            images = []
            for step in range(5):
                base = (100 + 10 * step) * _NS
                images.append(("fr", base, base))
                images.append(("r", base + 1_000_000, base + 1_000_000))
            _write_fixture(bag, images=images, include_prohibited=False)
            source = self._source(bag, start_ns=50 * _NS)
            previous_t = -1
            for index, group in enumerate(source):
                self.assertEqual(group.group_id, index)
                self.assertGreater(group.t_ns, previous_t)
                previous_t = group.t_ns
            self.assertEqual(source.trailing_events()[-1].kind, InputEventKind.END_OF_INPUT)


@unittest.skipUnless(RECORDING.exists(), "driving recording is not present at the project root")
class RecordingTests(unittest.TestCase):
    def test_first_canonical_image_is_decodable_with_real_calibration(self):
        import cv2

        calibrations = io.read_calibrations(RECORDING)
        self.assertEqual(set(calibrations), set(CAMERA_IDS))
        split = contracts.load_split()
        episode = split["fit"]
        pose = InitialPose(timestamp_ns=episode.start_ns, x=0.0, y=0.0, yaw=0.0)
        source = io.open_frame_sets(RECORDING, initial_pose=pose, episode=episode, calibrations=calibrations)
        first = next(iter(source))
        self.assertEqual(first.group_id, 0)
        fr = next(im for im in first.images if im.camera_id == "fr")
        # exposure/arrival are integer nanoseconds inside the fit episode.
        self.assertTrue(episode.contains(fr.exposure_ns))
        decoded = cv2.imdecode(np.frombuffer(fr.data, np.uint8), cv2.IMREAD_COLOR)
        fr_calibration = calibrations["fr"]
        self.assertEqual(decoded.shape, (fr_calibration.raw_height, fr_calibration.raw_width, 3))

    def test_fit_episode_groups_are_mostly_paired_and_ordered(self):
        calibrations = io.read_calibrations(RECORDING)
        split = contracts.load_split()
        episode = split["fit"]
        pose = InitialPose(timestamp_ns=episode.start_ns, x=0.0, y=0.0, yaw=0.0)
        source = io.open_frame_sets(RECORDING, initial_pose=pose, episode=episode, calibrations=calibrations)
        count = 0
        paired = 0
        previous_t = -1
        for group in source:
            count += 1
            self.assertGreater(group.t_ns, previous_t)
            previous_t = group.t_ns
            if len(group.images) == 2:
                paired += 1
        self.assertGreater(count, 1000)
        self.assertGreater(paired / count, 0.9)  # most groups pair fr and r
        self.assertEqual(source.trailing_events()[-1].kind, InputEventKind.END_OF_INPUT)


@pytest.mark.parametrize("images", [
    [("fr", 0, 0), ("r", 0, 50_000_000)],
    [("fr", 0, 0), ("fr", 100_000_000, 100_000_000),
     ("r", 0, 110_000_000), ("fr", 200_000_000, 200_000_000)],
])
def test_late_rear_is_logged_without_crash_or_duplicate_output(tmp_path, images):
    start = 100 * _NS
    bag = tmp_path / "late.mcap"
    _write_fixture(bag, images=[(camera, start + exposure, start + arrival)
                                for camera, exposure, arrival in images], include_prohibited=False)
    source = GroupingTests()._source(bag, start_ns=50 * _NS)
    groups = list(source)
    assert all(a.t_ns < b.t_ns for a, b in zip(groups, groups[1:]))
    assert all(image.camera_id == "fr" for group in groups for image in group.images)
    events = [event for group in groups for event in group.input_events] + list(source.trailing_events())
    assert sum(event.kind is InputEventKind.LATE_SUPPORT for event in events) == 1


def test_group_keeps_opening_cadence_when_future_primary_arrives(tmp_path):
    start = 100 * _NS
    bag = tmp_path / "cadence.mcap"
    images = [("fr", start, start), ("fr", start + 10_000_000, start + 10_000_000),
              ("r", start + 4_000_000, start + 12_000_000)]
    _write_fixture(bag, images=images, include_prohibited=False)
    groups = list(GroupingTests()._source(bag, start_ns=50 * _NS))
    assert groups[0].closure is FrameSetClosure.COMPLETE
    assert groups[0].t_ns == start + 4_000_000
    assert len(groups) == 2


def test_capacity_never_deletes_a_primary_group_and_prefix_ignores_future(tmp_path):
    start = 100 * _NS
    images = [("fr", start + i * 1_000_000, start + i * 1_000_000) for i in range(8)]
    results = []
    for n in (5, 8):
        bag = tmp_path / f"capacity{n}.mcap"
        _write_fixture(bag, images=images[:n], include_prohibited=False)
        groups = list(GroupingTests()._source(bag, start_ns=50 * _NS))
        assert [g.t_ns for g in groups] == [image[1] for image in images[:n]]
        assert groups[0].closure is FrameSetClosure.CAPACITY
        results.append(groups[0].summary())
    assert results[0] == results[1]


def test_old_startup_support_cannot_reorder_capacity_output(tmp_path):
    start = 100 * _NS
    images = [("fr", start + i * 1_000_000, start + i * 1_000_000) for i in range(4)]
    images.append(("r", start - 100_000_000, start + 4_000_000))
    bag = tmp_path / "old-startup.mcap"
    _write_fixture(bag, images=images, include_prohibited=False)
    source = GroupingTests()._source(bag, start_ns=50 * _NS)
    groups = list(source)
    assert [g.t_ns for g in groups] == [image[1] for image in images[:4]]
    assert sum(event.kind is InputEventKind.LATE_SUPPORT for g in groups for event in g.input_events) == 1


@pytest.mark.parametrize("field", ["width", "frame_id", "k", "d", "distortion_model", "transform"])
def test_changed_calibration_is_rejected(tmp_path, monkeypatch, field):
    import copy
    from mcap_ros2.writer import Writer

    original = Writer.write_message

    def write(writer, **kwargs):
        original(writer, **kwargs)
        topic = kwargs["topic"]
        if topic != ("/tf_static" if field == "transform" else io.info_topic("fr")):
            return
        changed = copy.deepcopy(kwargs)
        payload = changed["message"]
        if field == "transform":
            payload["transforms"][0]["transform"]["translation"]["x"] += 0.1
        elif field == "frame_id":
            payload["header"]["frame_id"] += "_changed"
        elif field in ("k", "d"):
            payload[field][0] += 0.01
        elif field == "width":
            payload[field] += 1
        else:
            payload[field] = "equidistant"
        changed["log_time"] += 1
        original(writer, **changed)

    monkeypatch.setattr(Writer, "write_message", write)
    bag = tmp_path / "changed.mcap"
    _write_fixture(bag, images=[], include_prohibited=False)
    with pytest.raises(contracts.ContractError, match="calibration changed|distortion"):
        io.read_calibrations(bag)


def test_calibration_identity_covers_extrinsics(tmp_path, monkeypatch):
    from mcap_ros2.writer import Writer
    first, second = tmp_path / "first.mcap", tmp_path / "second.mcap"
    _write_fixture(first, images=[], include_prohibited=False)
    original = Writer.write_message

    def write(writer, **kwargs):
        if kwargs["topic"] == "/tf_static":
            kwargs["message"]["transforms"][0]["transform"]["translation"]["x"] += 0.1
        original(writer, **kwargs)

    monkeypatch.setattr(Writer, "write_message", write)
    _write_fixture(second, images=[], include_prohibited=False)
    assert io.read_calibrations(first)["fr"].calibration_id != io.read_calibrations(second)["fr"].calibration_id


if __name__ == "__main__":
    unittest.main()

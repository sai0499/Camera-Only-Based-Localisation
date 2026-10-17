"""Owner A, step 1: the safe MCAP reader and camera calibration (``camloc/io.py``).

This module turns the driving recording into frozen per-camera calibration and header-only
``FrameSet`` groups, without ever decoding a JPEG payload, dynamic ``/tf`` or the live
``/constructor0/state_estimation`` topic. It implements the ``FrameSetSource`` protocol of
``camloc.contracts`` and the input-side conventions of ``task_docs/FINAL_BUILD_PLAN.md``
sections 1 and 2.

What is read
============

Calibration   ``/constructor0/sensor/camera_<id>/camera_info`` gives raw K/D and the raw
              size; ``/tf_static`` gives the rigid camera-to-body ``T_BC`` (parent
              ``constructor0/base_link``, child ``constructor0/camera_<id>``). The first
              available payload of each is frozen; later payloads are validated for
              compatibility, never silently switched. All seven cameras are checked, but
              image headers are only grouped for the selected cameras.
Images        ``/constructor0/sensor/camera_<id>/compressed_image`` headers only. The
              compressed ``data`` bytes are carried through undecoded inside
              ``ImageMessage`` so the canonicalizer (owner A, ``geometry.py``, step 2)
              decodes exactly the selected images later.

What is never read
==================

``/tf`` (dynamic) and ``/constructor0/state_estimation`` (live GT) are excluded from every
decode call: inference must not touch them even to filter their fields. ``/track_bounds``
and non-camera ``/tf_static`` children are discarded.

Canonical K
===========

A derives the canonical pinhole K from raw K (never from ``camera_info.P``) with the
OpenCV half-pixel resize convention and the fixed canonical size 960 x 464:
``sx = 960 / raw_width``, ``sy = 464 / raw_height``, ``fx' = sx * fx``, ``fy' = sy * fy``,
``cx' = sx * (cx + 0.5) - 0.5``, ``cy' = sy * (cy + 0.5) - 0.5``. Downstream distortion is
zero. ``geometry.py`` builds the matching remap; this module only supplies the raw
calibration and the derived canonical K inside ``CameraCalibration``.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camloc import contracts
from camloc.contracts import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CAMERA_IDS,
    MAX_ARRIVAL_SPREAD_NS,
    MAX_EXPOSURE_SPREAD_NS,
    MAX_PENDING_GROUPS,
    PRIMARY_CAMERA,
    SECONDARY_CAMERA,
    CameraCalibration,
    EpisodeBounds,
    FrameSet,
    FrameSetClosure,
    ImageMessage,
    InitialPose,
    InputEvent,
    InputEventKind,
)

# ---------------------------------------------------------------------------------------
# Topic names of the driving recording (see metadata.yaml). Prohibited topics are listed
# explicitly so a reader never decodes them, even to discard fields.
# ---------------------------------------------------------------------------------------

_SENSOR_PREFIX = "/constructor0/sensor/camera_"
_IMAGE_SUFFIX = "/compressed_image"
_INFO_SUFFIX = "/camera_info"
_TF_STATIC_TOPIC = "/tf_static"

PROHIBITED_TOPICS = frozenset({"/tf", "/constructor0/state_estimation"})

_BODY_FRAME = "constructor0/base_link"
_CAMERA_CHILD_PREFIX = "constructor0/camera_"

# Cadence (plan section 1): T is the median of the last 16 available valid primary-camera
# exposure intervals, initially 100 ms. Exposure tolerance min(5 ms, 0.2 T); arrival
# deadline min(40 ms, 1.2 T). The absolute caps live in contracts.
_INITIAL_CADENCE_NS = 100_000_000
_CADENCE_WINDOW = 16
_EXPOSURE_TOL_FRACTION = 0.2
_ARRIVAL_DEADLINE_FRACTION = 1.2


def image_topic(camera_id: str) -> str:
    """Return the compressed-image topic of a camera id."""
    return f"{_SENSOR_PREFIX}{contracts.check_camera_id(camera_id)}{_IMAGE_SUFFIX}"


def info_topic(camera_id: str) -> str:
    """Return the camera_info topic of a camera id."""
    return f"{_SENSOR_PREFIX}{contracts.check_camera_id(camera_id)}{_INFO_SUFFIX}"


def _camera_of_sensor_topic(topic: str, suffix: str) -> str | None:
    """Return the camera id of a sensor topic with the given suffix, or None."""
    if not (topic.startswith(_SENSOR_PREFIX) and topic.endswith(suffix)):
        return None
    camera = topic[len(_SENSOR_PREFIX) : -len(suffix)]
    return camera if camera in CAMERA_IDS else None


def _stamp_ns(header: Any) -> int:
    """Return a ROS ``header.stamp`` as integer nanoseconds, validated by ``as_ns``."""
    stamp = header.stamp
    return contracts.as_ns(int(stamp.sec) * contracts.NS_PER_S + int(stamp.nanosec), "exposure_ns")


def _canonical_k(k_raw: np.ndarray, raw_width: int, raw_height: int) -> np.ndarray:
    """Return the canonical pinhole K derived from raw K and the raw size (never from P)."""
    sx = CANONICAL_WIDTH / raw_width
    sy = CANONICAL_HEIGHT / raw_height
    fx, fy = float(k_raw[0, 0]), float(k_raw[1, 1])
    cx, cy = float(k_raw[0, 2]), float(k_raw[1, 2])
    return np.array(
        [
            [sx * fx, 0.0, sx * (cx + 0.5) - 0.5],
            [0.0, sy * fy, sy * (cy + 0.5) - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rigid_from_translation_quaternion(translation: Any, quaternion: Any) -> np.ndarray:
    """Return the 4x4 rigid transform from a ROS translation and an xyzw quaternion.

    ROS stores the quaternion as ``(x, y, z, w)`` (scalar last), matching the module's xyzw
    convention. Reuse the shared geometry conversion; ``check_rigid`` later enforces
    orthonormality.
    """
    from camloc.geometry import quaternion_xyzw_to_matrix

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_matrix(
        float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w))
    T[0, 3] = float(translation.x)
    T[1, 3] = float(translation.y)
    T[2, 3] = float(translation.z)
    return T


@dataclass(frozen=True, slots=True)
class _RawCalibration:
    """First-available raw camera_info payload of one camera (before T_BC is known)."""

    frame_id: str
    raw_width: int
    raw_height: int
    k_raw: np.ndarray
    d_raw: np.ndarray


def _camera_info(decoded: Any) -> _RawCalibration:
    if decoded.distortion_model != "plumb_bob" or len(decoded.d) != 5:
        raise contracts.ContractError("camera_info: expected five-coefficient plumb_bob distortion")
    return _RawCalibration(
        frame_id=str(decoded.header.frame_id), raw_width=int(decoded.width),
        raw_height=int(decoded.height), k_raw=np.asarray(decoded.k, dtype=np.float64).reshape(3, 3),
        d_raw=np.asarray(decoded.d, dtype=np.float64),
    )


def _identity(prefix: str, payload: bytes) -> str:
    """Return a short deterministic, opaque identity string from a payload digest."""
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:16]}"


def read_calibrations(
    bag: str | Path,
    *,
    calibration_id: str | None = None,
    preprocessing_id: str | None = None,
) -> Mapping[str, CameraCalibration]:
    """Read the frozen first-available calibration of all seven cameras from one recording.

    Decodes only ``camera_info`` and ``/tf_static``; it never opens an image payload, the
    dynamic ``/tf`` topic or the live state topic. Each camera needs a ``camera_info``
    payload and a ``base_link -> camera_<id>`` static transform; a camera missing either
    raises ``ContractError``. ``calibration_id`` and ``preprocessing_id`` are opaque owner
    identities; when omitted, a deterministic digest-based id is derived so the same
    recording always yields the same identity.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    bag = Path(bag)
    info_topics = {info_topic(camera): camera for camera in CAMERA_IDS}
    raw: dict[str, _RawCalibration] = {}
    transforms: dict[str, np.ndarray] = {}

    with open(bag, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        topics = sorted(info_topics) + [_TF_STATIC_TOPIC]
        for _schema, channel, _message, decoded in reader.iter_decoded_messages(topics=topics):
            if channel.topic in PROHIBITED_TOPICS:  # defensive; never requested above
                continue
            if channel.topic == _TF_STATIC_TOPIC:
                for transform in decoded.transforms:
                    child = transform.child_frame_id
                    if (
                        child.startswith(_CAMERA_CHILD_PREFIX)
                        and (camera := child[len(_CAMERA_CHILD_PREFIX) :]) in CAMERA_IDS
                    ):
                        if transform.header.frame_id != _BODY_FRAME:
                            raise contracts.ContractError(f"tf_static: camera parent changed for {camera}")
                        matrix = contracts.check_rigid(_rigid_from_translation_quaternion(
                            transform.transform.translation, transform.transform.rotation
                        ), "tf_static camera transform")
                        if camera in transforms and not np.allclose(
                            transforms[camera], matrix, rtol=0.0, atol=1e-9
                        ):
                            raise contracts.ContractError(f"tf_static: calibration changed for {camera}")
                        transforms.setdefault(camera, matrix)
                continue
            camera = info_topics[channel.topic]
            info = _camera_info(decoded)
            if camera in raw:
                previous = raw[camera]
                if ((info.frame_id, info.raw_width, info.raw_height)
                        != (previous.frame_id, previous.raw_width, previous.raw_height)
                        or not np.array_equal(info.k_raw, previous.k_raw)
                        or not np.array_equal(info.d_raw, previous.d_raw)):
                    raise contracts.ContractError(f"camera_info: calibration changed for {camera}")
            else:
                raw[camera] = info

    missing_info = [camera for camera in CAMERA_IDS if camera not in raw]
    if missing_info:
        raise contracts.ContractError(f"camera_info missing for cameras {missing_info}")
    missing_tf = [camera for camera in CAMERA_IDS if camera not in transforms]
    if missing_tf:
        raise contracts.ContractError(f"tf_static camera transform missing for {missing_tf}")

    info_digest = hashlib.sha256(b"camloc:canonical-remap-v2-full-bilinear-mask")
    for camera in CAMERA_IDS:
        info = raw[camera]
        info_digest.update(f"{camera}:{info.frame_id}:{info.raw_width}:{info.raw_height}".encode())
        for array in (info.k_raw, info.d_raw, transforms[camera]):
            info_digest.update(array.astype("<f8").tobytes())
    digest = info_digest.hexdigest()
    calib_id = calibration_id or f"calib:{digest[:16]}"
    prep_id = preprocessing_id or f"prep:canonical-{CANONICAL_WIDTH}x{CANONICAL_HEIGHT}:{digest[:8]}"

    calibrations: dict[str, CameraCalibration] = {}
    for camera in CAMERA_IDS:
        info = raw[camera]
        calibrations[camera] = CameraCalibration(
            camera_id=camera,
            calibration_id=calib_id,
            preprocessing_id=prep_id,
            frame_id=info.frame_id,
            raw_width=info.raw_width,
            raw_height=info.raw_height,
            K_raw=info.k_raw,
            D_raw=info.d_raw,
            K=_canonical_k(info.k_raw, info.raw_width, info.raw_height),
            T_BC=transforms[camera],
        )
    return dict(calibrations)


@dataclass(slots=True)
class _PendingImage:
    """One selected image header waiting to be grouped (payload kept undecoded)."""

    camera_id: str
    exposure_ns: int
    arrival_ns: int
    source_id: str
    frame_id: str
    image_format: str
    data: bytes


@dataclass(slots=True)
class _Group:
    """A forming header-only group keyed by its primary (fr) exposure window."""

    anchor_exposure_ns: int
    first_arrival_ns: int
    deadline_ns: int
    exposure_tol_ns: int
    images: dict[str, _PendingImage]

    def accepts(self, camera: str, exposure_ns: int, arrival_ns: int) -> bool:
        stamps = [image.exposure_ns for image in self.images.values()] + [exposure_ns]
        return (camera not in self.images and arrival_ns <= self.deadline_ns
                and max(stamps) - min(stamps) <= self.exposure_tol_ns)


class _FrameSetSource:
    """Header-only FrameSet grouping over one recording; see ``open_frame_sets``."""

    def __init__(
        self,
        bag: Path,
        *,
        initial_pose: InitialPose,
        episode: EpisodeBounds,
        cameras: tuple[str, ...],
        calibrations: Mapping[str, CameraCalibration],
    ) -> None:
        self._bag = bag
        self._initial_pose = initial_pose
        self._episode = episode
        self._selected = cameras
        self._calibrations = dict(calibrations)
        self._trailing: tuple[InputEvent, ...] | None = None

    def calibrations(self) -> Mapping[str, CameraCalibration]:
        """Frozen first-available calibration of all seven cameras."""
        return dict(self._calibrations)

    def trailing_events(self) -> tuple[InputEvent, ...]:
        """Events after the last FrameSet; valid only after iteration is exhausted."""
        if self._trailing is None:
            raise contracts.ContractError(
                "trailing_events() is available only after the FrameSetSource is exhausted"
            )
        return self._trailing

    def _start_boundary_ns(self) -> int:
        """Lowest exposure that may enter a group: the initial-pose timestamp."""
        return self._initial_pose.timestamp_ns

    def __iter__(self) -> Iterator[FrameSet]:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory

        selected = self._selected
        primary = PRIMARY_CAMERA if PRIMARY_CAMERA in selected else selected[0]
        start_ns = self._start_boundary_ns()
        end_ns = self._episode.end_ns

        topics = [image_topic(camera) for camera in selected]
        topic_camera = {image_topic(camera): camera for camera in selected}

        group_id = 0
        events: list[InputEvent] = []  # anomalies accumulated for the next FrameSet
        pending: list[_Group] = []
        last_exposure: dict[str, int] = {}
        cadence_intervals: deque[int] = deque(maxlen=_CADENCE_WINDOW)
        last_primary_exposure: int | None = None
        last_output_ns: int | None = None
        late_through: dict[str, int] = {}
        last_arrival_ns = self._episode.start_ns

        def cadence_ns() -> int:
            if not cadence_intervals:
                return _INITIAL_CADENCE_NS
            intervals = sorted(cadence_intervals)
            mid = len(intervals) // 2
            return (intervals[mid] + intervals[(len(intervals) - 1) // 2]) // 2

        def make_frame_set(group: _Group, closure: FrameSetClosure) -> FrameSet:
            nonlocal group_id, events, last_output_ns
            images = tuple(
                ImageMessage(
                    camera_id=item.camera_id,
                    exposure_ns=item.exposure_ns,
                    arrival_ns=item.arrival_ns,
                    source_id=item.source_id,
                    frame_id=item.frame_id,
                    image_format=item.image_format,
                    calibration_id=self._calibrations[item.camera_id].calibration_id,
                    data=item.data,
                )
                for camera in selected
                if (item := group.images.get(camera)) is not None
            )
            stamp = max(image.exposure_ns for image in images)
            if last_output_ns is not None and stamp <= last_output_ns:
                raise contracts.ContractError("reader produced non-increasing group times")
            if last_output_ns is not None and stamp - last_output_ns > contracts.NS_PER_S:
                events.append(InputEvent(kind=InputEventKind.COVERAGE_GAP,
                                         log_ns=last_arrival_ns, exposure_ns=stamp,
                                         reference_ns=last_output_ns))
            frame_set = FrameSet(
                group_id=group_id,
                t_ns=stamp,
                selected_camera_ids=selected,
                images=images,
                missing_camera_ids=tuple(
                    camera for camera in selected if camera not in group.images
                ),
                closure=closure,
                terminal_partial=closure is FrameSetClosure.EOF,
                input_events=tuple(events),
            )
            group_id += 1
            last_output_ns = stamp
            for camera in selected:
                boundary = (stamp if camera in group.images else
                            max(stamp, group.anchor_exposure_ns + group.exposure_tol_ns))
                late_through[camera] = max(late_through.get(camera, -1), boundary)
            events = []
            return frame_set

        def emit_oldest(closure: FrameSetClosure) -> FrameSet:
            group = pending.pop(0)
            return make_frame_set(group, closure)

        with open(self._bag, "rb") as handle:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            for _schema, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
                if channel.topic in PROHIBITED_TOPICS:  # defensive; never requested
                    continue
                camera = topic_camera[channel.topic]
                exposure_ns = _stamp_ns(decoded.header)
                arrival_ns = contracts.as_ns(int(message.log_time), "arrival_ns")
                last_arrival_ns = arrival_ns

                # Advance recorded timers before this packet can supply any evidence.
                while pending and arrival_ns > pending[0].deadline_ns:
                    complete = len(pending[0].images) == len(selected)
                    yield emit_oldest(FrameSetClosure.COMPLETE if complete else FrameSetClosure.DEADLINE)

                # Start boundary and episode end: only images in [initial_pose, episode end).
                if exposure_ns < start_ns:
                    events.append(
                        InputEvent(
                            kind=InputEventKind.BEFORE_INITIAL_POSE,
                            log_ns=arrival_ns,
                            camera_id=camera,
                            exposure_ns=exposure_ns,
                            reference_ns=start_ns,
                        )
                    )
                    continue
                if end_ns is not None and exposure_ns >= end_ns:
                    events.append(
                        InputEvent(
                            kind=InputEventKind.AFTER_EPISODE_END,
                            log_ns=arrival_ns,
                            camera_id=camera,
                            exposure_ns=exposure_ns,
                            reference_ns=end_ns,
                        )
                    )
                    continue

                # Reject duplicate/decreasing per-camera exposure stamps without shrinking
                # uncertainty; never substitute a later exposure for a missing one.
                previous = last_exposure.get(camera)
                if previous is not None and exposure_ns <= previous:
                    kind = (
                        InputEventKind.DUPLICATE_STAMP
                        if exposure_ns == previous
                        else InputEventKind.DECREASING_STAMP
                    )
                    events.append(
                        InputEvent(
                            kind=kind,
                            log_ns=arrival_ns,
                            camera_id=camera,
                            exposure_ns=exposure_ns,
                            reference_ns=previous,
                        )
                    )
                    continue
                last_exposure[camera] = exposure_ns
                compatible = any(group.accepts(camera, exposure_ns, arrival_ns) for group in pending)
                before_pending = (pending and exposure_ns < pending[0].anchor_exposure_ns
                                  - pending[0].exposure_tol_ns)
                if ((last_output_ns is not None and exposure_ns <= last_output_ns)
                        or before_pending
                        or (exposure_ns <= late_through.get(camera, -1) and not compatible)):
                    events.append(InputEvent(kind=InputEventKind.LATE_SUPPORT, log_ns=arrival_ns,
                                             camera_id=camera, exposure_ns=exposure_ns,
                                             reference_ns=(last_output_ns if last_output_ns is not None
                                                           else pending[0].anchor_exposure_ns)))
                    continue

                if camera == primary:
                    if last_primary_exposure is not None:
                        interval = exposure_ns - last_primary_exposure
                        if interval > 0:
                            cadence_intervals.append(interval)
                    last_primary_exposure = exposure_ns

                cadence = cadence_ns()
                exposure_tol = min(MAX_EXPOSURE_SPREAD_NS, int(_EXPOSURE_TOL_FRACTION * cadence))
                arrival_deadline = min(
                    MAX_ARRIVAL_SPREAD_NS, int(_ARRIVAL_DEADLINE_FRACTION * cadence)
                )

                # Attach to a compatible open group, else open a new one. A camera appears at
                # most once per group; a second image of the same camera opens a new group.
                placed = False
                for group in sorted(pending, key=lambda g: abs(exposure_ns - g.anchor_exposure_ns)):
                    if group.accepts(camera, exposure_ns, arrival_ns):
                        group.images[camera] = _PendingImage(
                            camera_id=camera,
                            exposure_ns=exposure_ns,
                            arrival_ns=arrival_ns,
                            source_id=f"{camera}:{exposure_ns}",
                            frame_id=decoded.header.frame_id,
                            image_format=str(decoded.format),
                            data=bytes(decoded.data),
                        )
                        placed = True
                        break
                if not placed:
                    # Capacity: before opening a fifth pending group, close/emit the oldest.
                    if len(pending) >= MAX_PENDING_GROUPS:
                        yield emit_oldest(FrameSetClosure.CAPACITY)
                    pending.append(
                        _Group(
                            anchor_exposure_ns=exposure_ns,
                            first_arrival_ns=arrival_ns,
                            deadline_ns=arrival_ns + arrival_deadline,
                            exposure_tol_ns=exposure_tol,
                            images={
                                camera: _PendingImage(
                                    camera_id=camera,
                                    exposure_ns=exposure_ns,
                                    arrival_ns=arrival_ns,
                                    source_id=f"{camera}:{exposure_ns}",
                                    frame_id=decoded.header.frame_id,
                                    image_format=str(decoded.format),
                                    data=bytes(decoded.data),
                                )
                            },
                        )
                    )

                # Close groups that are complete (all selected present) or past their arrival
                # deadline, oldest first, keeping exposure order.
                pending.sort(key=lambda g: g.anchor_exposure_ns)
                while pending:
                    oldest = pending[0]
                    complete = all(sel in oldest.images for sel in selected)
                    expired = arrival_ns > oldest.deadline_ns
                    if complete:
                        yield emit_oldest(FrameSetClosure.COMPLETE)
                    elif expired:
                        yield emit_oldest(FrameSetClosure.DEADLINE)
                    else:
                        break

        # Flush remaining groups as deadline closures, oldest first; the final one, if any,
        # is the single terminal partial EOF group.
        pending.sort(key=lambda g: g.anchor_exposure_ns)
        while pending:
            complete = all(sel in pending[0].images for sel in selected)
            if len(pending) == 1 and not complete:
                yield emit_oldest(FrameSetClosure.EOF)
            elif complete:
                yield emit_oldest(FrameSetClosure.COMPLETE)
            else:
                yield emit_oldest(FrameSetClosure.DEADLINE)

        events.append(InputEvent(kind=InputEventKind.END_OF_INPUT, log_ns=last_arrival_ns))
        self._trailing = tuple(events)


def open_frame_sets(
    bag: str | Path,
    *,
    initial_pose: InitialPose,
    episode: EpisodeBounds,
    cameras: tuple[str, ...] = (PRIMARY_CAMERA, SECONDARY_CAMERA),
    calibrations: Mapping[str, CameraCalibration] | None = None,
) -> _FrameSetSource:
    """Open a header-only ``FrameSetSource`` over the driving recording.

    ``initial_pose`` fixes the start boundary (images before its timestamp are reported as
    ``before_initial_pose`` events) and ``episode`` bounds the run (images at or after
    ``episode.end_ns`` become ``after_episode_end`` events). ``cameras`` are the selected
    streams whose headers are grouped, in order; the default is the primary ``fr`` and
    secondary ``r``. Pairing their headers keeps output timestamps identical whether or not
    the rear frontend is later enabled. When ``calibrations`` is None the calibration of all
    seven cameras is read once up front. No JPEG, ``/tf`` or live state message is decoded.
    """
    bag = Path(bag)
    episode.check_initial_pose(initial_pose)
    selected = tuple(contracts.check_camera_id(camera) for camera in cameras)
    if not selected:
        raise contracts.ContractError("open_frame_sets: at least one camera must be selected")
    if len(set(selected)) != len(selected):
        raise contracts.ContractError(f"open_frame_sets: cameras repeat in {selected}")
    if calibrations is None:
        calibrations = read_calibrations(bag)
    if set(calibrations) != set(CAMERA_IDS) or any(
        not isinstance(value, CameraCalibration) or value.camera_id != key
        for key, value in calibrations.items()
    ):
        raise contracts.ContractError("open_frame_sets: calibrations must match all seven camera ids")
    return _FrameSetSource(
        bag,
        initial_pose=initial_pose,
        episode=episode,
        cameras=selected,
        calibrations=calibrations,
    )

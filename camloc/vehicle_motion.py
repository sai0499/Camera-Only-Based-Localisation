"""Explicit PDF-permitted velocity/yaw-rate controls; never reads a pose or track state.

StateEstimation is parsed as raw CDR: pose/accuracy bytes are skipped, and parsing stops
after the velocity prefix. The default camera reader and camera-only mode are unchanged.
"""

import dataclasses
import hashlib
import math
import re
import struct
from pathlib import Path

import numpy as np

from camloc import contracts as c

TOPICS = {"state-fields": "/constructor0/state_estimation",
          "validation-twist": "/constructor0/validation/twist"}
STATE_PREFIX = ["std_msgs/Header header", "int32 se_status", "int32 se_state",
    "float64 x_m", "float64 y_m", "float64 z_m", "float64 roll_rad",
    "float64 pitch_rad", "float64 yaw_rad", "float64[3] pos_accuracy",
    "float64[3] vel_accuracy", "float64 wx_radps", "float64 wy_radps",
    "float64 wz_radps", "float64 vx_mps", "float64 vy_mps", "float64 vz_mps"]
MAX_AGE_NS = 100_000_000


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
    t_ns: int
    arrival_ns: int
    vx: float
    vy: float
    wz: float

    def __post_init__(self):
        c.as_ns(self.t_ns)
        c.as_ns(self.arrival_ns)
        if not all(math.isfinite(v) for v in (self.vx, self.vy, self.wz)):
            raise c.ContractError("vehicle motion must be finite")
        if math.hypot(self.vx, self.vy) > 150 or abs(self.wz) > 5:
            raise c.ContractError("vehicle motion outside supported physical range")


def _header(data):
    if len(data) < 16 or data[:2] not in (b"\x00\x01", b"\x00\x00"):
        raise c.ContractError("vehicle motion requires CDR1 serialization")
    endian = "<" if data[1] else ">"
    sec, ns, length = struct.unpack_from(endian + "iII", data, 4)
    if ns >= c.NS_PER_S or not 1 <= length <= 1024 or 16 + length > len(data):
        raise c.ContractError("invalid vehicle motion header")
    if data[15 + length] != 0:
        raise c.ContractError("vehicle motion frame lacks terminator")
    try:
        frame = data[16:15 + length].decode("utf-8")
    except UnicodeDecodeError as error:
        raise c.ContractError("invalid vehicle motion frame") from error
    return c.as_ns(sec * c.NS_PER_S + ns), frame, 16 + length, endian


def _align(offset, alignment):
    return 4 + ((offset - 4 + alignment - 1) // alignment) * alignment


def decode_sample(data, arrival_ns, source):
    """Decode only timestamp and body vx/vy/wz; no general ROS state decoder is used."""
    stamp, frame, offset, endian = _header(data)
    if source == "state-fields":
        if frame not in ("map", "constructor0/base_link"):
            raise c.ContractError("unsupported StateEstimation motion frame")
        offset = _align(offset, 4) + 8  # skip status integers
        offset = _align(offset, 8) + 12 * 8  # skip pose and accuracy, never unpack them
        if offset + 48 > len(data):
            raise c.ContractError("truncated StateEstimation velocity prefix")
        wz, vx, vy = struct.unpack_from(endian + "3d", data, offset + 16)
    elif source == "validation-twist":
        if frame != "constructor0/base_link":
            raise c.ContractError("validation twist must be in constructor0/base_link")
        offset = _align(offset, 8)
        if offset + 48 > len(data):
            raise c.ContractError("truncated TwistStamped")
        vx, vy = struct.unpack_from(endian + "2d", data, offset)
        wz = struct.unpack_from(endian + "d", data, offset + 40)[0]
    else:
        raise c.ContractError("unknown vehicle motion source")
    return Sample(stamp, arrival_ns, vx, vy, wz)


def _check_schema(schema, source):
    expected = ("sd_localization_msgs/msg/StateEstimation" if source == "state-fields"
                else "geometry_msgs/msg/TwistStamped")
    if schema is None or schema.name != expected or schema.encoding != "ros2msg":
        raise c.ContractError("unsupported vehicle motion schema")
    sections = re.split(r"={20,}", schema.data.decode("utf-8"))
    def fields_of(text):
        return [line for raw in text.splitlines() if (line := " ".join(raw.split("#", 1)[0].split()))]
    fields = fields_of(sections[0])
    wanted = STATE_PREFIX if source == "state-fields" else ["std_msgs/Header header", "Twist twist"]
    if fields[:len(wanted)] != wanted:
        raise c.ContractError("vehicle motion field layout differs from the supported schema")
    dependencies = {}
    for section in sections[1:]:
        rows = fields_of(section)
        if rows and rows[0].startswith("MSG: "):
            dependencies[rows[0][5:]] = rows[1:]
    required = {"std_msgs/Header": ["builtin_interfaces/Time stamp", "string frame_id"],
                "builtin_interfaces/Time": ["int32 sec", "uint32 nanosec"]}
    if source == "validation-twist":
        required.update({"geometry_msgs/Twist": ["Vector3 linear", "Vector3 angular"],
                         "geometry_msgs/Vector3": ["float64 x", "float64 y", "float64 z"]})
    if any(dependencies.get(name) != rows for name, rows in required.items()):
        raise c.ContractError("vehicle motion nested schema differs from the supported layout")


class VehicleMotion:
    """Causal controls for one recording. Availability is fixed by each image group's arrival.

    Fresh measured body rates replace the CV/decaying rate prediction. Missing/stale samples
    use the unchanged camera filter model. Controls never count as a full visual pose.
    """

    def __init__(self, samples, *, source, schema_sha256=None):
        if source not in TOPICS or not samples:
            raise c.ContractError("vehicle motion source has no supported samples")
        self.samples = tuple(samples)
        self.times = np.array([s.t_ns for s in samples], dtype=np.int64)
        self.arrivals = np.array([s.arrival_ns for s in samples], dtype=np.int64)
        self.times.setflags(write=False)
        self.arrivals.setflags(write=False)
        if np.any(np.diff(self.times) <= 0) or np.any(np.diff(self.arrivals) < 0):
            raise c.ContractError("vehicle motion stamps/arrivals must increase")
        self.source, self.schema_sha256 = source, schema_sha256
        self.arrival_cutoff = -1

    @classmethod
    def load(cls, bag, source, initial_ns, end_ns=None):
        from mcap.reader import make_reader

        samples, schemas = [], set()
        with Path(bag).open("rb") as handle:
            reader = make_reader(handle)
            for schema, channel, message in reader.iter_messages(topics=[TOPICS[source]],
                    start_time=max(0, initial_ns - 2 * c.NS_PER_S),
                    end_time=None if end_ns is None else end_ns + c.NS_PER_S):
                if channel.message_encoding != "cdr":
                    raise c.ContractError("vehicle motion is not CDR")
                digest = hashlib.sha256(schema.data).hexdigest() if schema else None
                if digest not in schemas:
                    _check_schema(schema, source)
                    schemas.add(digest)
                sample = decode_sample(message.data, message.log_time, source)
                if sample.t_ns >= initial_ns - MAX_AGE_NS and (end_ns is None or sample.t_ns < end_ns):
                    samples.append(sample)
        if len(schemas) != 1:
            raise c.ContractError("vehicle motion schema changed or topic missing")
        return cls(samples, source=source, schema_sha256=next(iter(schemas)))

    def prepare(self, frame):
        cutoff = max(image.arrival_ns for image in frame.images)
        if cutoff < self.arrival_cutoff:
            raise c.ContractError("vehicle motion image arrivals decreased")
        self.arrival_cutoff = cutoff

    def summary(self):
        return {"source": self.source, "topic": TOPICS[self.source],
                "fields": ["timestamp", "body_vx_mps", "body_vy_mps", "body_wz_radps"],
                "schema_sha256": self.schema_sha256, "samples": len(self.samples),
                "max_age_ns": MAX_AGE_NS, "pose_fields_decoded": False,
                "accuracy_status": "noise floors are conservative assumptions, not calibrated accuracy"}

    def diagnostic(self, frame):
        i = min(int(np.searchsorted(self.times, frame.t_ns, side="right")),
                int(np.searchsorted(self.arrivals, self.arrival_cutoff, side="right"))) - 1
        sample = self.samples[i] if i >= 0 else None
        return {"record": "VehicleMotionControl", "group_id": frame.group_id,
                "source": self.source, "t_ns": frame.t_ns,
                "sample_ns": None if sample is None else sample.t_ns,
                "arrival_ns": None if sample is None else sample.arrival_ns,
                "available_by_ns": self.arrival_cutoff,
                "fresh": sample is not None and frame.t_ns - sample.t_ns <= MAX_AGE_NS,
                "visual_full_pose": False}

    def fresh_at(self, stamp):
        i = min(int(np.searchsorted(self.times, stamp, side="right")),
                int(np.searchsorted(self.arrivals, self.arrival_cutoff, side="right"))) - 1
        return i >= 0 and 0 <= stamp - self.samples[i].t_ns <= MAX_AGE_NS

    def propagate(self, state, covariance, from_ns, to_ns, scale, last_full_ns, fallback):
        if to_ns < from_ns:
            raise c.ContractError("vehicle propagation cannot go backwards")
        last = min(int(np.searchsorted(self.times, to_ns, side="right")),
                   int(np.searchsorted(self.arrivals, self.arrival_cutoff, side="right"))) - 1
        if last < 0 or to_ns == from_ns:
            return fallback(state, covariance, from_ns, to_ns, scale, last_full_ns)
        index = min(int(np.searchsorted(self.times, from_ns, side="right")) - 1, last)
        state, P, stamp = state.copy(), covariance.copy(), from_ns
        while stamp < to_ns:
            next_ns = self.samples[index + 1].t_ns if index < last else to_ns
            sample = self.samples[index] if index >= 0 else None
            fresh = sample is not None and 0 <= stamp - sample.t_ns < MAX_AGE_NS
            end = min(to_ns, next_ns)
            if fresh:
                end = min(end, sample.t_ns + MAX_AGE_NS)
                state, P = integrate(state, P, sample, (end - stamp) / c.NS_PER_S)
            else:
                state, P = fallback(state, P, stamp, end, scale, last_full_ns)
            stamp = end
            if index < last and stamp >= next_ns:
                index += 1
        return state, P


def integrate(state, covariance, sample, dt):
    """Constant-body-twist SE(2) integration, with yaw/velocity uncertainty propagated."""
    theta = sample.wz * dt
    a = np.sinc(theta / math.pi)
    b = theta / 2 if abs(theta) < 1e-7 else (1 - math.cos(theta)) / theta
    local = dt * np.array([a * sample.vx - b * sample.vy, b * sample.vx + a * sample.vy])
    yaw = float(state[2])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    delta = rotation @ local
    updated = state.copy()
    updated[:2] += delta
    updated[2] = c.wrap_angle(yaw + theta)
    cy, sy = math.cos(updated[2]), math.sin(updated[2])
    velocity = np.array([cy * sample.vx - sy * sample.vy, sy * sample.vx + cy * sample.vy])
    updated[3:] = [*velocity, sample.wz]
    J = np.eye(6)
    J[:2, 2] = [-delta[1], delta[0]]
    J[3:, :] = 0
    J[3:5, 2] = [-velocity[1], velocity[0]]
    # Continuous slip/model noise plus per-output measured-rate floors; no pose certainty.
    Q = np.diag([.15**2 * dt, .15**2 * dt, math.radians(.3)**2 * dt,
                 .5**2, .5**2, math.radians(.5)**2])
    P = J @ covariance @ J.T + Q
    return updated, (P + P.T) / 2

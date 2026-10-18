"""Motion assistance is opt-in, pose-free, causal and fails closed on unsupported inputs."""
import dataclasses
import math
import struct

import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c
from camloc.filter import CameraFilter, _propagate
from camloc.vehicle_motion import Sample, VehicleMotion, decode_sample, integrate, _check_schema
from camloc import vehicle_motion as vm

T = f.EXPOSURE_NS


def packet(source="state-fields", *, endian="<", vx=20., vy=.1, wz=.2):
    frame = b"map\0" if source == "state-fields" else b"constructor0/base_link\0"
    data = bytearray(b"\0\1\0\0" if endian == "<" else b"\0\0\0\0")
    data.extend(struct.pack(endian + "iII", T // c.NS_PER_S, T % c.NS_PER_S, len(frame)))
    data.extend(frame)
    if source == "state-fields":
        data.extend(bytes(vm._align(len(data), 4) - len(data)))
        data.extend(b"\xff" * 8)
        data.extend(bytes(vm._align(len(data), 8) - len(data)))
        data.extend(b"\xff" * 96)  # deliberately invalid pose/accuracy values are never decoded
        data.extend(struct.pack(endian + "6d", 0., 0., wz, vx, vy, 0.))
    else:
        data.extend(bytes(vm._align(len(data), 8) - len(data)))
        data.extend(struct.pack(endian + "6d", vx, vy, 0., 0., 0., wz))
    return bytes(data)


@pytest.mark.parametrize("source", list(vm.TOPICS))
@pytest.mark.parametrize("endian", ["<", ">"])
def test_raw_parser_reads_only_permitted_body_motion(source, endian):
    sample = decode_sample(packet(source, endian=endian), T + 10, source)
    assert sample == Sample(T, T + 10, 20., .1, .2)


def test_mutating_pose_track_and_status_bytes_cannot_change_motion():
    a = packet()
    b = bytearray(a)
    offset = vm._align(16 + len(b"map\0"), 4)
    velocity = vm._align(offset + 8, 8) + 96
    b[offset:velocity] = bytes(velocity - offset)
    b.extend(bytes(range(255)))  # track/lane and every later field are ignored
    assert decode_sample(a, T, "state-fields") == decode_sample(bytes(b), T, "state-fields")


@pytest.mark.parametrize("payload", [b"", b"\0" * 16, packet()[:-8], b"\0\3"+packet()[2:]])
def test_malformed_serialization_rejected(payload):
    with pytest.raises(c.ContractError):
        decode_sample(payload, T, "state-fields")


def test_unknown_state_layout_rejected():
    header = "\n"+"="*80+"\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
    header += "="*80+"\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n"
    schema = type("Schema", (), {"name": "sd_localization_msgs/msg/StateEstimation",
        "encoding": "ros2msg", "data": ("\n".join(vm.STATE_PREFIX)+header).encode()})()
    _check_schema(schema, "state-fields")
    schema.data = schema.data.replace(b"float64 vx_mps", b"float64 yaw_rad")
    with pytest.raises(c.ContractError, match="layout"):
        _check_schema(schema, "state-fields")


def test_modified_nested_header_schema_rejected():
    schema = type("Schema", (), {"name": "geometry_msgs/msg/TwistStamped", "encoding": "ros2msg",
                                 "data": b"std_msgs/Header header\nTwist twist\n"})()
    with pytest.raises(c.ContractError, match="nested"):
        _check_schema(schema, "validation-twist")


@pytest.mark.parametrize("values", [(math.nan, 0., 0.), (151., 0., 0.), (20., 0., 6.)])
def test_invalid_motion_rejected(values):
    with pytest.raises(c.ContractError):
        Sample(T, T, *values)


def frame(k=0, stamp=T, arrival=None):
    return f.make_frame_set(group_id=k, selected_camera_ids=("fr",), images=(
        f.make_image_message(exposure_ns=stamp, arrival_ns=arrival or stamp + 40_000_000),))


def test_constant_body_twist_matches_circular_arc_and_covariance_stays_psd():
    state = np.zeros(6)
    sample = Sample(T, T, 20., 0., .2)
    result, P = integrate(state, np.eye(6), sample, 2.)
    np.testing.assert_allclose(result[:3], [100*math.sin(.4), 100*(1-math.cos(.4)), .4])
    assert np.linalg.eigvalsh(P).min() >= -1e-10
    assert result[5] == .2


def test_future_or_late_arriving_motion_does_not_change_current_prediction():
    base = Sample(T, T, 20., 0., 0.)
    future = Sample(T + 100_000_000, T + 200_000_000, 100., 0., 4.)
    controls = VehicleMotion([base, future], source="validation-twist")
    estimator = CameraFilter(c.InitialPose(timestamp_ns=T, x=0, y=0, yaw=0), vehicle_motion=controls)
    estimator.prepare_group(frame(stamp=T + 90_000_000, arrival=T + 140_000_000))
    np.testing.assert_allclose(estimator.snapshot(T + 90_000_000).state[:3], [1.8, 0., 0.])
    changed = VehicleMotion([base, dataclasses.replace(future, vx=1., wz=-3.)], source="validation-twist")
    changed.prepare(frame(stamp=T + 90_000_000, arrival=T + 140_000_000))
    a = controls.propagate(np.zeros(6), np.eye(6), T, T + 150_000_000, 1., T, _propagate)
    b = changed.propagate(np.zeros(6), np.eye(6), T, T + 150_000_000, 1., T, _propagate)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_no_available_motion_uses_original_prediction_exactly():
    controls = VehicleMotion([Sample(T, T + c.NS_PER_S, 20., 0., .2)], source="validation-twist")
    controls.prepare(frame())
    args = (np.zeros(6), np.eye(6), T, T + 100_000_000, 1., T)
    actual = controls.propagate(*args, _propagate)
    expected = _propagate(*args)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_motion_controls_do_not_claim_a_visual_pose_or_avoid_loss_status():
    controls = VehicleMotion([Sample(T + i*10_000_000, T + i*10_000_000, 20., 0., .2)
                              for i in range(41)], source="validation-twist")
    estimator = CameraFilter(c.InitialPose(timestamp_ns=T, x=0, y=0, yaw=0), vehicle_motion=controls)
    current = frame(stamp=T + 400_000_000)
    estimator.prepare_group(current)
    estimate = estimator.step(f.make_group_evidence(frame_set=current))
    assert estimate.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
    assert estimate.status == c.Status.LOST and estimate.max_support_exposure_ns is None
    assert estimate.state[0] > 7.9
    diagnostic = controls.diagnostic(current)
    assert diagnostic["sample_ns"] <= current.t_ns
    assert diagnostic["arrival_ns"] <= diagnostic["available_by_ns"]


def test_motion_prior_downweights_visual_outlier_without_changing_gate():
    controls = VehicleMotion([Sample(T, T, 20., 0., 0.)], source="validation-twist")
    estimator = CameraFilter(c.InitialPose(timestamp_ns=T, x=0, y=0, yaw=0), vehicle_motion=controls)
    estimator.prepare_group(frame())
    state, P = np.zeros(6), np.eye(6) * .25
    good, _ = estimator._motion_pose_update(state, P, np.array([.1, 0., 0.]),
                                          np.eye(3)*.25, T)
    bad, _ = estimator._motion_pose_update(state, P, np.array([8., 0., 0.]),
                                         np.eye(3)*.25, T, recovery=True)
    assert good[0] == pytest.approx(.05)
    assert 0 < bad[0] < .5  # no blind eight-metre reset
    log = estimator.drain_diagnostics()
    assert log[0]["weight"] == 1 and 0 < log[1]["weight"] < .1
    assert log[1]["recovery"] and estimator.drain_diagnostics() == []

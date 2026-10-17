"""Example tests of the contract records, starting with the input records.

Each test starts from a valid ``contract_factories`` builder and changes one field, so a
rejection names exactly that field: messages start with ``"<Record>.<field>: "``. Randomized
FrameSet invariants live in a separate property test. These tests write nothing.
"""

import dataclasses
import enum
import json
import math

import numpy as np
import pytest

import contract_factories as factories
from camloc import contracts as c

EXPOSURE_NS = factories.EXPOSURE_NS
ARRIVAL_NS = EXPOSURE_NS + factories.ARRIVAL_DELAY_NS
STEP_NS = factories.GROUP_EXPOSURE_STEP_NS
H, W = c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH
U_MIN, U_MAX = -0.5, W - 0.5  # canonical pixel range of the principal point, inclusive
V_MIN, V_MAX = -0.5, H - 0.5
OPTICAL_TO_BODY = np.array(factories.OPTICAL_TO_BODY)
REFLECTION = np.diag([1.0, 1.0, -1.0])

BUILDERS = {
    "ImageMessage": factories.make_image_message,
    "CameraCalibration": factories.make_calibration,
    "Observation": factories.make_observation,
}


def _with(array, index, value):
    """Return a float64 copy of ``array`` with the entry at ``index`` replaced by ``value``."""
    result = np.array(array, dtype=np.float64)
    result[index] = value
    return result


def _image(camera, exposure_offset_ns=0, arrival_offset_ns=0):
    """Return an ImageMessage of ``camera`` offset from the default exposure and arrival."""
    return factories.make_image_message(
        camera_id=camera,
        exposure_ns=EXPOSURE_NS + exposure_offset_ns,
        arrival_ns=ARRIVAL_NS + arrival_offset_ns,
    )


# ---------------------------------------------------------------------------------------
# Camera ids, stamps and identities (Requirements 1.6, 4.1)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("record", list(BUILDERS))
@pytest.mark.parametrize("camera_id", c.CAMERA_IDS)
def test_input_records_accept_every_camera_id(record, camera_id):
    assert BUILDERS[record](camera_id=camera_id).camera_id == camera_id


@pytest.mark.parametrize("record", list(BUILDERS))
@pytest.mark.parametrize(
    "camera_id",
    ["FR", "front_right", "", "fr ", None, 0],
    ids=["upper_case", "long_name", "empty", "trailing_space", "none", "int"],
)
def test_input_records_reject_unknown_camera_ids(record, camera_id):
    with pytest.raises(c.ContractError, match=rf"^{record}\.camera_id: .+ is not one of"):
        BUILDERS[record](camera_id=camera_id)


@pytest.mark.parametrize("record", ["ImageMessage", "Observation"])
def test_image_stamps_are_stored_as_exact_python_ints(record):
    assert float(EXPOSURE_NS) != EXPOSURE_NS  # a float would round this 19-digit stamp
    built = BUILDERS[record](exposure_ns=np.int64(EXPOSURE_NS), arrival_ns=np.uint64(ARRIVAL_NS))
    assert type(built.exposure_ns) is int and built.exposure_ns == EXPOSURE_NS
    assert type(built.arrival_ns) is int and built.arrival_ns == ARRIVAL_NS


@pytest.mark.parametrize("record", ["ImageMessage", "Observation"])
@pytest.mark.parametrize("field", ["exposure_ns", "arrival_ns"])
@pytest.mark.parametrize(
    "value",
    [1.0, np.float64(EXPOSURE_NS), True, str(EXPOSURE_NS), -1, 2**63, None],
    ids=["integral_float", "np_float64", "bool", "decimal_str", "negative", "two_pow_63", "none"],
)
def test_image_stamps_must_be_integer_nanoseconds(record, field, value):
    overrides = {"exposure_ns": EXPOSURE_NS, "arrival_ns": ARRIVAL_NS, field: value}
    with pytest.raises(c.ContractError, match=rf"^{record}\.{field}: "):
        BUILDERS[record](**overrides)


IDENTITY_FIELDS = [
    ("ImageMessage", "source_id"),
    ("ImageMessage", "frame_id"),
    ("ImageMessage", "image_format"),
    ("ImageMessage", "calibration_id"),
    ("CameraCalibration", "calibration_id"),
    ("CameraCalibration", "preprocessing_id"),
    ("CameraCalibration", "frame_id"),
    ("Observation", "source_id"),
    ("Observation", "calibration_id"),
    ("Observation", "preprocessing_id"),
]


@pytest.mark.parametrize(
    ("record", "field"),
    IDENTITY_FIELDS,
    ids=[f"{record}.{field}" for record, field in IDENTITY_FIELDS],
)
@pytest.mark.parametrize("value", ["", None, 7, b"id"], ids=["empty", "none", "int", "bytes"])
def test_identity_fields_must_be_non_empty_strings(record, field, value):
    message = rf"^{record}\.{field}: expected a non-empty identity string"
    with pytest.raises(c.ContractError, match=message):
        BUILDERS[record](**{field: value})


# ---------------------------------------------------------------------------------------
# ImageMessage payload and InputEvent (Requirement 4.1)
# ---------------------------------------------------------------------------------------


def test_image_message_keeps_its_payload_and_leaves_it_out_of_repr():
    message = factories.make_image_message()
    assert message.data is factories.JPEG_PAYLOAD
    assert message.frame_id == "constructor0/camera_fr"
    assert "data=" not in repr(message)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(bytearray(factories.JPEG_PAYLOAD), id="bytearray"),
        pytest.param(memoryview(factories.JPEG_PAYLOAD), id="memoryview"),
        pytest.param(np.frombuffer(factories.JPEG_PAYLOAD, dtype=np.uint8), id="ndarray"),
        pytest.param(b"", id="empty_bytes"),
        pytest.param(factories.JPEG_PAYLOAD.decode("latin-1"), id="str"),
        pytest.param(None, id="none"),
    ],
)
def test_image_message_data_must_be_non_empty_bytes(data):
    with pytest.raises(c.ContractError, match=r"^ImageMessage\.data: "):
        factories.make_image_message(data=data)


def test_input_event_optional_fields_default_to_none_and_empty_detail():
    event = c.InputEvent(kind="end_of_input", log_ns=EXPOSURE_NS)
    assert event.kind is c.InputEventKind.END_OF_INPUT
    assert (event.camera_id, event.exposure_ns, event.reference_ns, event.source_id) == (
        None,
        None,
        None,
        None,
    )
    assert event.detail == ""


def test_input_event_keeps_every_optional_field():
    event = factories.make_input_event(
        kind=c.InputEventKind.DUPLICATE_STAMP,
        camera_id="r",
        exposure_ns=EXPOSURE_NS,
        reference_ns=np.int64(EXPOSURE_NS),
        source_id=f"r@{EXPOSURE_NS}",
        detail="same header stamp as the previous r image",
    )
    assert event.kind is c.InputEventKind.DUPLICATE_STAMP
    assert (event.camera_id, event.exposure_ns) == ("r", EXPOSURE_NS)
    assert event.source_id == f"r@{EXPOSURE_NS}"
    assert type(event.reference_ns) is int and event.reference_ns == EXPOSURE_NS
    assert event.detail == "same header stamp as the previous r image"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("kind", "dropped_frame", id="unknown_kind"),
        pytest.param("kind", c.FrameSetClosure.EOF, id="other_enum_member"),
        pytest.param("kind", None, id="kind_none"),
        pytest.param("log_ns", 1.0, id="float_log_ns"),
        pytest.param("log_ns", None, id="log_ns_none"),
        pytest.param("camera_id", "FR", id="unknown_camera"),
        pytest.param("exposure_ns", -1, id="negative_exposure"),
        pytest.param("reference_ns", True, id="bool_reference"),
        pytest.param("source_id", "", id="empty_source_id"),
        pytest.param("detail", None, id="detail_none"),
        pytest.param("detail", b"late", id="detail_bytes"),
    ],
)
def test_input_event_rejects_malformed_fields(field, value):
    with pytest.raises(c.ContractError, match=rf"^InputEvent\.{field}: "):
        factories.make_input_event(**{field: value})


# ---------------------------------------------------------------------------------------
# CameraCalibration (Requirements 4.4, 4.6)
# ---------------------------------------------------------------------------------------


def test_calibration_stores_read_only_float64_copies():
    sources = {
        "K_raw": factories.raw_K(),
        "D_raw": np.zeros(5, dtype=np.int64),
        "K": factories.canonical_K().astype(np.float32),
        "T_BC": factories.camera_to_body(),
    }
    calibration = factories.make_calibration(**sources)
    for field, source in sources.items():
        stored = getattr(calibration, field)
        assert stored.dtype == np.float64, field
        assert stored.flags.writeable is False, field
        assert not np.shares_memory(stored, source), field
        np.testing.assert_array_equal(stored, source)
        assert source.flags.writeable is True, field  # the caller's array is untouched
    assert calibration.raw_width == factories.RAW_WIDTH
    assert calibration.raw_height == factories.RAW_HEIGHT


@pytest.mark.parametrize(
    "D_raw",
    [
        pytest.param(np.zeros(4), id="shape_4"),
        pytest.param(np.zeros(8), id="shape_8_rational"),
        pytest.param(np.zeros((5, 1)), id="column_5x1"),
        pytest.param(np.zeros((1, 5)), id="row_1x5"),
        pytest.param(np.float64(0.0), id="scalar"),
        pytest.param(np.zeros(0), id="empty"),
        pytest.param(_with(factories.RAW_DISTORTION, 0, math.nan), id="nan"),
        pytest.param(_with(factories.RAW_DISTORTION, 4, math.inf), id="inf"),
        pytest.param(np.zeros(5, dtype=np.bool_), id="bool"),
        pytest.param(np.zeros(5, dtype=np.complex128), id="complex"),
        pytest.param(None, id="none"),
    ],
)
def test_calibration_requires_five_finite_plumb_bob_coefficients(D_raw):
    with pytest.raises(c.ContractError, match=r"^CameraCalibration\.D_raw: "):
        factories.make_calibration(D_raw=D_raw)


def test_calibration_accepts_numpy_integer_raw_sizes_down_to_one_pixel():
    calibration = factories.make_calibration(raw_width=np.int32(1), raw_height=np.uint16(728))
    assert type(calibration.raw_width) is int and calibration.raw_width == 1
    assert type(calibration.raw_height) is int and calibration.raw_height == 728


@pytest.mark.parametrize("field", ["raw_width", "raw_height"])
@pytest.mark.parametrize(
    "value",
    [0, -1, 1506.0, True, None, "1506"],
    ids=["zero", "negative", "float", "bool", "none", "str"],
)
def test_calibration_requires_positive_integer_raw_sizes(field, value):
    with pytest.raises(c.ContractError, match=rf"^CameraCalibration\.{field}: "):
        factories.make_calibration(**{field: value})


# ---------------------------------------------------------------------------------------
# Intrinsics and camera-to-body transforms (Requirements 4.4, 4.5)
# ---------------------------------------------------------------------------------------

K_FIELDS = [
    pytest.param("Observation", "K", factories.canonical_K, id="Observation.K"),
    pytest.param("CameraCalibration", "K", factories.canonical_K, id="CameraCalibration.K"),
    pytest.param("CameraCalibration", "K_raw", factories.raw_K, id="CameraCalibration.K_raw"),
]

K_DEFECTS = [
    pytest.param(lambda K: _with(K, (0, 1), 0.5), id="skew_01"),
    pytest.param(lambda K: _with(K, (1, 0), 1e-12), id="tiny_skew_10"),
    pytest.param(lambda K: _with(K, (0, 0), -K[0, 0]), id="negative_fx"),
    pytest.param(lambda K: _with(K, (1, 1), -K[1, 1]), id="negative_fy"),
    pytest.param(lambda K: _with(K, (0, 0), 0.0), id="zero_fx"),
    pytest.param(lambda K: _with(K, (2, 2), 2.0), id="last_row_scaled"),
    pytest.param(lambda K: _with(K, (2, 0), 1e-12), id="last_row_offset"),
    pytest.param(lambda K: _with(K, (0, 2), math.nan), id="nan_cx"),
    pytest.param(lambda K: _with(K, (1, 1), math.inf), id="inf_fy"),
    pytest.param(lambda K: K[:2], id="shape_2x3"),
    pytest.param(lambda K: np.hstack([K, np.zeros((3, 1))]), id="shape_3x4_projection"),
]


@pytest.mark.parametrize(("record", "field", "valid_K"), K_FIELDS)
@pytest.mark.parametrize("defect", K_DEFECTS)
def test_intrinsics_need_positive_focal_lengths_zero_skew_and_unit_last_row(
    record, field, valid_K, defect
):
    with pytest.raises(c.ContractError, match=rf"^{record}\.{field}: "):
        BUILDERS[record](**{field: defect(valid_K())})


@pytest.mark.parametrize(
    ("cx", "cy"),
    [(U_MIN, V_MIN), (U_MAX, V_MAX), (U_MIN, V_MAX), (U_MAX, V_MIN), (479.5, 231.5)],
    ids=["top_left", "bottom_right", "bottom_left", "top_right", "centre"],
)
def test_observation_accepts_a_principal_point_on_the_canonical_pixel_bounds(cx, cy):
    observation = factories.make_observation(K=factories.canonical_K(cx=cx, cy=cy))
    assert (observation.K[0, 2], observation.K[1, 2]) == (cx, cy)


@pytest.mark.parametrize(
    ("cx", "cy"),
    [
        (math.nextafter(U_MIN, -math.inf), 231.5),
        (math.nextafter(U_MAX, math.inf), 231.5),
        (479.5, math.nextafter(V_MIN, -math.inf)),
        (479.5, math.nextafter(V_MAX, math.inf)),
    ],
    ids=["left", "right", "top", "bottom"],
)
def test_observation_rejects_a_principal_point_just_outside_the_pixel_bounds(cx, cy):
    with pytest.raises(c.ContractError, match=r"^Observation\.K: principal point"):
        factories.make_observation(K=factories.canonical_K(cx=cx, cy=cy))


T_BC_DEFECTS = [
    pytest.param(
        lambda: factories.rigid(REFLECTION @ OPTICAL_TO_BODY, factories.CAMERA_TRANSLATION_M),
        id="reflection",
    ),
    pytest.param(
        lambda: factories.rigid(1.01 * OPTICAL_TO_BODY, factories.CAMERA_TRANSLATION_M),
        id="scaled_rotation",
    ),
    pytest.param(lambda: _with(factories.camera_to_body(), (3, 2), 1e-6), id="last_row"),
    pytest.param(lambda: _with(factories.camera_to_body(), (0, 3), math.nan), id="nan"),
    pytest.param(lambda: factories.camera_to_body()[:3], id="shape_3x4"),
]


@pytest.mark.parametrize("record", ["CameraCalibration", "Observation"])
@pytest.mark.parametrize("make_T_BC", T_BC_DEFECTS)
def test_camera_to_body_transform_must_be_rigid(record, make_T_BC):
    with pytest.raises(c.ContractError, match=rf"^{record}\.T_BC: "):
        BUILDERS[record](T_BC=make_T_BC())


# ---------------------------------------------------------------------------------------
# Observation image and mask buffers (Requirements 4.2, 4.3, 4.7)
# ---------------------------------------------------------------------------------------


def test_observation_keeps_the_callers_buffers_and_marks_them_read_only():
    image = factories.canonical_image()
    mask = factories.canonical_mask()
    K = factories.canonical_K()
    T_BC = factories.camera_to_body()
    observation = factories.make_observation(image=image, valid_mask=mask, K=K, T_BC=T_BC)

    assert observation.image is image and observation.valid_mask is mask
    assert np.shares_memory(observation.image, image)
    assert np.shares_memory(observation.valid_mask, mask)
    assert image.flags.writeable is False
    assert mask.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        image[0, 0, 0] = 255
    with pytest.raises(ValueError, match="read-only"):
        mask[0, 0] = False

    # Small arrays are private read-only float64 copies; the caller's arrays stay writable.
    for stored, source in ((observation.K, K), (observation.T_BC, T_BC)):
        assert stored.dtype == np.float64 and stored.flags.writeable is False
        assert not np.shares_memory(stored, source)
        np.testing.assert_array_equal(stored, source)
        assert source.flags.writeable is True


def test_observation_checks_buffer_structure_not_content():
    image = factories.seeded_rng(22).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.bool_)  # no usable pixel: the consumer decides
    observation = factories.make_observation(image=image, valid_mask=mask)
    assert observation.image is image and not observation.valid_mask.any()

    # A buffer that is already read-only (for example np.frombuffer) is kept as it is.
    frozen = np.frombuffer(bytes(H * W * 3), dtype=np.uint8).reshape(H, W, 3)
    assert factories.make_observation(image=frozen).image is frozen


BAD_IMAGES = [
    pytest.param(lambda: np.zeros((H, W, 3), np.float32), "expected dtype uint8", id="float32"),
    pytest.param(lambda: np.zeros((H, W, 3), np.uint16), "expected dtype uint8", id="uint16"),
    pytest.param(lambda: np.zeros((H, W), np.uint8), "expected shape", id="gray"),
    pytest.param(lambda: np.zeros((W, H, 3), np.uint8), "expected shape", id="width_height_swap"),
    pytest.param(lambda: np.zeros((H, W, 4), np.uint8), "expected shape", id="rgba"),
    pytest.param(lambda: np.zeros((728, 1506, 3), np.uint8), "expected shape", id="raw_size"),
    pytest.param(
        lambda: np.asfortranarray(factories.canonical_image()),
        "expected a C-contiguous array",
        id="fortran_order",
    ),
    pytest.param(
        lambda: np.zeros((H, 2 * W, 3), np.uint8)[:, ::2],
        "expected a C-contiguous array",
        id="strided_view",
    ),
    pytest.param(
        lambda: factories.canonical_image()[..., ::-1],
        "expected a C-contiguous array",
        id="reversed_channel_view",
    ),
    pytest.param(
        lambda: np.ma.masked_array(factories.canonical_image()),
        "expected a numpy.ndarray, got MaskedArray",
        id="masked_array",
    ),
    pytest.param(lambda: None, "expected a numpy.ndarray", id="none"),
]


@pytest.mark.parametrize(("make_image", "message"), BAD_IMAGES)
def test_observation_rejects_images_of_another_dtype_shape_or_layout(make_image, message):
    image = make_image()
    with pytest.raises(c.ContractError, match=rf"^Observation\.image: {message}"):
        factories.make_observation(image=image)


BAD_MASKS = [
    pytest.param(lambda: np.ones((H, W), np.uint8), "expected dtype bool", id="uint8"),
    pytest.param(lambda: np.ones((H, W), np.float32), "expected dtype bool", id="float32"),
    pytest.param(lambda: np.ones((H, W, 1), np.bool_), "expected shape", id="extra_axis"),
    pytest.param(lambda: np.ones((H, W, 3), np.bool_), "expected shape", id="per_channel"),
    pytest.param(lambda: np.ones((W, H), np.bool_), "expected shape", id="width_height_swap"),
    pytest.param(
        lambda: np.asfortranarray(factories.canonical_mask()),
        "expected a C-contiguous array",
        id="fortran_order",
    ),
    pytest.param(
        lambda: np.ones((H, 2 * W), np.bool_)[:, ::2],
        "expected a C-contiguous array",
        id="strided_view",
    ),
    pytest.param(
        lambda: np.ma.masked_array(factories.canonical_mask()),
        "expected a numpy.ndarray, got MaskedArray",
        id="masked_array",
    ),
    pytest.param(lambda: None, "expected a numpy.ndarray", id="none"),
]


@pytest.mark.parametrize(("make_mask", "message"), BAD_MASKS)
def test_observation_rejects_masks_of_another_dtype_shape_or_layout(make_mask, message):
    mask = make_mask()
    with pytest.raises(c.ContractError, match=rf"^Observation\.valid_mask: {message}"):
        factories.make_observation(valid_mask=mask)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        pytest.param(lambda: {"valid_mask": np.ones((H, W), np.uint8)}, "valid_mask", id="mask"),
        pytest.param(lambda: {"K": factories.canonical_K(cx=W + 10.0)}, "K", id="principal_point"),
        pytest.param(
            lambda: {"T_BC": factories.rigid(REFLECTION @ OPTICAL_TO_BODY, (0.0, 0.0, 0.0))},
            "T_BC",
            id="reflected_T_BC",
        ),
    ],
)
def test_a_rejected_observation_leaves_the_callers_buffers_writable(overrides, field):
    fields = {"image": factories.canonical_image(), "valid_mask": factories.canonical_mask()}
    fields.update(overrides())
    with pytest.raises(c.ContractError, match=rf"^Observation\.{field}: "):
        factories.make_observation(**fields)
    assert fields["image"].flags.writeable is True
    assert fields["valid_mask"].flags.writeable is True


# ---------------------------------------------------------------------------------------
# FrameSet examples (Requirement 5; randomized invariants are a separate property test)
# ---------------------------------------------------------------------------------------


def test_a_complete_frame_set_derives_its_time_from_the_latest_exposure():
    frame_set = factories.make_frame_set(selected_camera_ids=("fl", "fr", "r"))
    assert [image.camera_id for image in frame_set.images] == ["fl", "fr", "r"]
    assert frame_set.t_ns == EXPOSURE_NS + 2 * STEP_NS
    assert frame_set.missing_camera_ids == ()
    assert frame_set.closure is c.FrameSetClosure.COMPLETE
    assert frame_set.terminal_partial is False
    assert frame_set.input_events == ()


def test_frame_set_time_is_the_maximum_exposure_whatever_the_camera_order():
    images = (_image("fr", 3 * STEP_NS), _image("r"))
    assert factories.make_frame_set(images=images).t_ns == EXPOSURE_NS + 3 * STEP_NS


def test_frame_set_stores_sequences_as_tuples_and_keeps_its_messages():
    images = [_image("fr"), _image("r", STEP_NS)]
    events = [factories.make_input_event(), factories.make_input_event(kind="coverage_gap")]
    frame_set = factories.make_frame_set(
        selected_camera_ids=["fr", "r"], images=images, missing_camera_ids=[], input_events=events
    )
    for stored in (
        frame_set.selected_camera_ids,
        frame_set.images,
        frame_set.missing_camera_ids,
        frame_set.input_events,
    ):
        assert type(stored) is tuple
    assert frame_set.selected_camera_ids == ("fr", "r")
    assert all(kept is given for kept, given in zip(frame_set.images, images, strict=True))
    assert all(kept is given for kept, given in zip(frame_set.input_events, events, strict=True))


@pytest.mark.parametrize(
    ("closure", "terminal_partial"),
    [("deadline", False), ("capacity", False), ("eof", True)],
)
def test_a_partial_frame_set_closes_by_deadline_capacity_or_end_of_input(closure, terminal_partial):
    frame_set = factories.make_frame_set(
        present=("r",), selected_camera_ids=("fl", "fr", "r"), closure=closure
    )
    assert [image.camera_id for image in frame_set.images] == ["r"]
    assert frame_set.missing_camera_ids == ("fl", "fr")
    assert frame_set.closure is c.FrameSetClosure(closure)
    assert frame_set.terminal_partial is terminal_partial
    assert frame_set.t_ns == frame_set.images[0].exposure_ns


@pytest.mark.parametrize(
    ("exposure_offset_ns", "arrival_offset_ns"),
    [
        (c.MAX_EXPOSURE_SPREAD_NS, 0),
        (0, c.MAX_ARRIVAL_SPREAD_NS),
        (c.MAX_EXPOSURE_SPREAD_NS, c.MAX_ARRIVAL_SPREAD_NS),
    ],
    ids=["exposure_5ms", "arrival_40ms", "both_limits"],
)
def test_frame_set_spreads_may_reach_their_limits(exposure_offset_ns, arrival_offset_ns):
    assert (c.MAX_EXPOSURE_SPREAD_NS, c.MAX_ARRIVAL_SPREAD_NS) == (5_000_000, 40_000_000)
    images = (_image("fr"), _image("r", exposure_offset_ns, arrival_offset_ns))
    assert factories.make_frame_set(images=images).t_ns == EXPOSURE_NS + exposure_offset_ns


@pytest.mark.parametrize(
    ("exposure_offset_ns", "arrival_offset_ns", "message"),
    [
        (c.MAX_EXPOSURE_SPREAD_NS + 1, 0, "exposures span 5000001 ns"),
        (0, c.MAX_ARRIVAL_SPREAD_NS + 1, "arrivals span 40000001 ns"),
    ],
    ids=["exposure_5ms_plus_1ns", "arrival_40ms_plus_1ns"],
)
def test_frame_set_spreads_beyond_their_limits_are_rejected(
    exposure_offset_ns, arrival_offset_ns, message
):
    images = (_image("fr"), _image("r", exposure_offset_ns, arrival_offset_ns))
    with pytest.raises(c.ContractError, match=rf"^FrameSet\.images: {message}"):
        factories.make_frame_set(images=images)


def test_frame_set_accepts_input_events_in_non_decreasing_log_order():
    events = (
        factories.make_input_event(kind="duplicate_stamp", camera_id="fr", log_ns=ARRIVAL_NS),
        factories.make_input_event(kind="late_support", log_ns=ARRIVAL_NS),
        factories.make_input_event(kind="coverage_gap", log_ns=ARRIVAL_NS + 1),
    )
    assert factories.make_frame_set(input_events=events).input_events == events


# Each case gives make_frame_set overrides (default group: cameras fr then r, both present).
FRAME_SET_REJECTIONS = [
    pytest.param(
        lambda: {"selected_camera_ids": (), "images": (_image("fr"),)},
        r"^FrameSet\.selected_camera_ids: at least one camera",
        id="no_selected_camera",
    ),
    pytest.param(
        lambda: {"selected_camera_ids": ("fr", "fr"), "images": (_image("fr"),)},
        r"^FrameSet\.selected_camera_ids: cameras repeat",
        id="repeated_selected_camera",
    ),
    pytest.param(
        lambda: {"selected_camera_ids": ("fr", "FR"), "images": (_image("fr"),)},
        r"^FrameSet\.selected_camera_ids\[1\]: 'FR' is not one of",
        id="unknown_selected_camera",
    ),
    pytest.param(
        lambda: {"selected_camera_ids": "fr", "images": (_image("fr"),)},
        r"^FrameSet\.selected_camera_ids: expected a tuple of camera ids, got str",
        id="selected_as_str",
    ),
    pytest.param(
        lambda: {"images": ()},
        r"^FrameSet\.images: at least one image",
        id="no_image",
    ),
    pytest.param(
        lambda: {"images": (factories.make_observation(),)},
        r"^FrameSet\.images\[0\]: expected ImageMessage, got Observation",
        id="decoded_observation",
    ),
    pytest.param(
        lambda: {"images": (_image("fr"), _image("fl"))},
        r"^FrameSet\.images\[1\]: camera 'fl' is not selected",
        id="unselected_camera",
    ),
    pytest.param(
        lambda: {"images": (_image("fr"), _image("fr", STEP_NS))},
        r"^FrameSet\.images: a camera appears more than once",
        id="repeated_camera",
    ),
    pytest.param(
        lambda: {"images": (_image("r"), _image("fr", STEP_NS))},
        r"^FrameSet\.images: cameras \('r', 'fr'\) are not in the selected order",
        id="unordered_cameras",
    ),
    pytest.param(
        lambda: {
            "present": ("fr",),
            "selected_camera_ids": ("fl", "fr", "r"),
            "missing_camera_ids": ("r", "fl"),
        },
        r"^FrameSet\.missing_camera_ids: expected \('fl', 'r'\)",
        id="missing_out_of_order",
    ),
    pytest.param(
        lambda: {"missing_camera_ids": ("r",)},
        r"^FrameSet\.missing_camera_ids: expected \(\)",
        id="present_camera_listed_missing",
    ),
    pytest.param(
        lambda: {"present": ("fr",), "missing_camera_ids": ()},
        r"^FrameSet\.missing_camera_ids: expected \('r',\)",
        id="missing_camera_omitted",
    ),
    pytest.param(
        lambda: {"present": ("fr",), "closure": "complete"},
        r"^FrameSet\.closure: 'complete' with missing cameras \('r',\)",
        id="complete_with_missing",
    ),
    pytest.param(
        lambda: {"closure": "deadline"},
        r"^FrameSet\.closure: 'deadline' with missing cameras \(\)",
        id="deadline_without_missing",
    ),
    pytest.param(
        lambda: {"closure": "eof"},
        r"^FrameSet\.closure: 'eof' with missing cameras \(\)",
        id="eof_without_missing",
    ),
    pytest.param(
        lambda: {"closure": "timeout"},
        r"^FrameSet\.closure: 'timeout' is not a FrameSetClosure",
        id="unknown_closure",
    ),
    pytest.param(
        lambda: {"present": ("fr",), "closure": "deadline", "terminal_partial": True},
        r"^FrameSet\.terminal_partial: True with closure 'deadline'",
        id="terminal_deadline",
    ),
    pytest.param(
        lambda: {"present": ("fr",), "closure": "eof", "terminal_partial": False},
        r"^FrameSet\.terminal_partial: False with closure 'eof'",
        id="eof_not_terminal",
    ),
    pytest.param(
        lambda: {"terminal_partial": 0},
        r"^FrameSet\.terminal_partial: expected a bool",
        id="int_terminal_partial",
    ),
    pytest.param(
        lambda: {"t_ns": EXPOSURE_NS + STEP_NS + 1},
        r"^FrameSet\.t_ns: \d+ must equal the maximum image exposure",
        id="t_ns_after_latest_exposure",
    ),
    pytest.param(
        lambda: {"t_ns": EXPOSURE_NS},
        r"^FrameSet\.t_ns: \d+ must equal the maximum image exposure",
        id="t_ns_earliest_exposure",
    ),
    pytest.param(
        lambda: {"images": (_image("fr", STEP_NS), _image("r")), "t_ns": EXPOSURE_NS},
        r"^FrameSet\.t_ns: \d+ must equal the maximum image exposure",
        id="t_ns_last_image_exposure",
    ),
    pytest.param(
        lambda: {"t_ns": float(EXPOSURE_NS + STEP_NS)},
        r"^FrameSet\.t_ns: expected integer nanoseconds",
        id="float_t_ns",
    ),
    pytest.param(lambda: {"group_id": -1}, r"^FrameSet\.group_id: ", id="negative_group_id"),
    pytest.param(lambda: {"group_id": True}, r"^FrameSet\.group_id: ", id="bool_group_id"),
    pytest.param(
        lambda: {
            "input_events": (
                factories.make_input_event(log_ns=ARRIVAL_NS + 1),
                factories.make_input_event(log_ns=ARRIVAL_NS),
            )
        },
        r"^FrameSet\.input_events\[1\]: log_ns \d+ precedes",
        id="events_out_of_order",
    ),
    pytest.param(
        lambda: {"input_events": ("late_support",)},
        r"^FrameSet\.input_events\[0\]: expected InputEvent, got str",
        id="event_as_str",
    ),
]


@pytest.mark.parametrize(("build", "message"), FRAME_SET_REJECTIONS)
def test_frame_set_rejects_inconsistent_groups(build, message):
    overrides = build()
    with pytest.raises(c.ContractError, match=message):
        factories.make_frame_set(**overrides)


# ---------------------------------------------------------------------------------------
# Property 6: FrameSet invariants (Requirements 5.1-5.7)
# ---------------------------------------------------------------------------------------

FRAME_SET_SEED = 20261018
FRAME_SET_ITERATIONS = 200
FRAME_SET_MIN_RUNS = 20  # coverage floor of every violation and of each exact-limit spread
MAX_EXPOSURE_NS = c.MAX_EXPOSURE_SPREAD_NS
MAX_ARRIVAL_NS = c.MAX_ARRIVAL_SPREAD_NS
PARTIAL_CLOSURES = (c.FrameSetClosure.DEADLINE, c.FrameSetClosure.CAPACITY, c.FrameSetClosure.EOF)
# (exposure, arrival) spread of iteration k: exactly the cap, a draw in [0, cap] or zero.
SPREAD_MODES = (
    ("limit", "limit"),
    ("limit", "drawn"),
    ("drawn", "limit"),
    ("zero", "zero"),
    ("drawn", "drawn"),
)
FRAME_SET_RULES = (
    "duplicate_selected_camera",
    "unselected_camera_image",
    "repeated_camera_image",
    "unordered_images",
    "wrong_missing_ids",
    "complete_with_missing",
    "partial_without_missing",
    "flipped_terminal_partial",
    "t_ns_off_by_one",
    "exposure_spread_over_limit",
    "arrival_spread_over_limit",
    "events_out_of_order",
)


def _insert(rng, items, item):
    """Return tuple ``items`` with ``item`` inserted at a random position, and the position."""
    index = int(rng.integers(len(items) + 1))
    return items[:index] + (item,) + items[index:], index


def _offsets(rng, mode, cap, count):
    """Return ``count`` offsets in [0, spread]; with two or more, both extremes occur exactly."""
    spread = 0 if mode == "zero" else cap if mode == "limit" else int(rng.integers(cap + 1))
    offsets = [int(value) for value in rng.integers(spread + 1, size=count)]
    if count >= 2:
        low, high = (int(index) for index in rng.choice(count, size=2, replace=False))
        offsets[low], offsets[high] = 0, spread
    return offsets


def _valid_frame_set_fields(rng, k):
    """Return iteration ``k``'s valid FrameSet fields; a limit spread gets two or more images."""
    exposure_mode, arrival_mode = SPREAD_MODES[k % len(SPREAD_MODES)]
    fewest = 2 if "limit" in (exposure_mode, arrival_mode) else 1
    order = [c.CAMERA_IDS[int(index)] for index in rng.permutation(len(c.CAMERA_IDS))]
    selected = tuple(order[: int(rng.integers(fewest, len(c.CAMERA_IDS) + 1))])
    count = int(rng.integers(fewest, len(selected) + 1))
    kept = sorted(int(index) for index in rng.choice(len(selected), size=count, replace=False))
    exposure_ns = EXPOSURE_NS + int(rng.integers(10**12))
    arrival_ns = exposure_ns + int(rng.integers(50_000_000, 100_000_001))  # 50-100 ms latency
    images = tuple(
        factories.make_image_message(
            camera_id=selected[index], exposure_ns=exposure_ns + de, arrival_ns=arrival_ns + da
        )
        for index, de, da in zip(
            kept,
            _offsets(rng, exposure_mode, MAX_EXPOSURE_NS, count),
            _offsets(rng, arrival_mode, MAX_ARRIVAL_NS, count),
            strict=True,
        )
    )
    missing = tuple(camera for index, camera in enumerate(selected) if index not in kept)
    closure = PARTIAL_CLOSURES[int(rng.integers(3))] if missing else c.FrameSetClosure.COMPLETE
    log_offsets = np.cumsum(rng.integers(3, size=int(rng.integers(5)))) * STEP_NS  # 0 = a tie
    return {
        "group_id": k,
        "t_ns": max(image.exposure_ns for image in images),
        "selected_camera_ids": selected,
        "images": images,
        "missing_camera_ids": missing,
        "closure": closure,
        "terminal_partial": closure is c.FrameSetClosure.EOF,
        "input_events": tuple(
            factories.make_input_event(log_ns=arrival_ns + int(offset)) for offset in log_offsets
        ),
    }


def _frame_set_violations(rng, fields):
    """Yield (rule, ``fields`` with that one invariant broken, expected message prefix)."""
    selected, images, missing = (
        fields["selected_camera_ids"],
        fields["images"],
        fields["missing_camera_ids"],
    )
    count = len(images)

    def broken(**changes):
        return {**fields, **changes}

    twice, _ = _insert(rng, selected, selected[int(rng.integers(len(selected)))])
    yield "duplicate_selected_camera", broken(selected_camera_ids=twice), (
        "FrameSet.selected_camera_ids: cameras repeat"
    )
    unselected = [camera for camera in c.CAMERA_IDS if camera not in selected]
    if unselected:
        donor = images[int(rng.integers(count))]  # in-range stamps of a present image
        extra = factories.make_image_message(
            camera_id=unselected[int(rng.integers(len(unselected)))],
            exposure_ns=donor.exposure_ns,
            arrival_ns=donor.arrival_ns,
        )
        with_extra, index = _insert(rng, images, extra)
        yield "unselected_camera_image", broken(images=with_extra), f"FrameSet.images[{index}]: "
    index = int(rng.integers(count))
    yield "repeated_camera_image", broken(images=images[: index + 1] + images[index:]), (
        "FrameSet.images: a camera appears more than once"
    )
    if count >= 2:
        order = [int(value) for value in rng.permutation(count)]
        if order == sorted(order):
            order[0], order[1] = order[1], order[0]
        unordered = tuple(images[value] for value in order)
        yield "unordered_images", broken(images=unordered), "FrameSet.images: cameras "

    wrong = [_insert(rng, missing, images[int(rng.integers(count))].camera_id)[0]]
    if missing:
        index = int(rng.integers(len(missing)))
        wrong.append(missing[:index] + missing[index + 1 :])
    if len(missing) >= 2:
        wrong.append(missing[::-1])
    yield "wrong_missing_ids", broken(missing_camera_ids=wrong[int(rng.integers(len(wrong)))]), (
        "FrameSet.missing_camera_ids: expected "
    )
    if missing:
        complete = broken(closure=c.FrameSetClosure.COMPLETE, terminal_partial=False)
        yield "complete_with_missing", complete, "FrameSet.closure: 'complete' with missing"
    else:
        closure = PARTIAL_CLOSURES[int(rng.integers(len(PARTIAL_CLOSURES)))]
        partial = broken(closure=closure, terminal_partial=closure is c.FrameSetClosure.EOF)
        yield "partial_without_missing", partial, f"FrameSet.closure: {closure.value!r} with"
    yield "flipped_terminal_partial", broken(terminal_partial=not fields["terminal_partial"]), (
        "FrameSet.terminal_partial: "
    )
    t_ns = fields["t_ns"] + int(rng.choice((-1, 1)))
    yield "t_ns_off_by_one", broken(t_ns=t_ns), "FrameSet.t_ns: "

    if count >= 2:  # move one image just past the cap, after (or before) all the others
        for rule, stamp, cap, word in (
            ("exposure_spread_over_limit", "exposure_ns", MAX_EXPOSURE_NS, "exposures"),
            ("arrival_spread_over_limit", "arrival_ns", MAX_ARRIVAL_NS, "arrivals"),
        ):
            index = int(rng.integers(count))
            others = [getattr(image, stamp) for i, image in enumerate(images) if i != index]
            old = images[index]
            stamps = {"exposure_ns": old.exposure_ns, "arrival_ns": old.arrival_ns}
            stamps[stamp] = min(others) + cap + 1 if rng.random() < 0.5 else max(others) - cap - 1
            new = factories.make_image_message(camera_id=old.camera_id, **stamps)
            shifted = images[:index] + (new,) + images[index + 1 :]
            t_ns = max(image.exposure_ns for image in shifted)  # still the maximum exposure
            yield rule, broken(images=shifted, t_ns=t_ns), (
                f"FrameSet.images: {word} span {cap + 1} ns"
            )

    events = list(fields["input_events"])
    while len(events) < 2:  # pad with ties so the broken event has a predecessor
        log_ns = events[-1].log_ns if events else ARRIVAL_NS
        events.append(factories.make_input_event(log_ns=log_ns))
    index = int(rng.integers(1, len(events)))
    earlier = events[index - 1].log_ns - int(rng.integers(1, STEP_NS + 1))
    events[index] = factories.make_input_event(log_ns=earlier)
    yield "events_out_of_order", broken(input_events=tuple(events)), (
        f"FrameSet.input_events[{index}]: log_ns "
    )


# Feature: shared-contracts, Property 6: FrameSet invariants
def test_property_6_frame_set_invariants():
    """A FrameSet is constructed exactly when its group satisfies every invariant.

    Selected cameras are unique and the images follow their order, ``t_ns`` is the maximum
    exposure, ``missing_camera_ids`` are the selected cameras without an image (in selected
    order), ``complete`` holds iff nothing is missing and ``terminal_partial`` iff ``eof``,
    exposures span at most 5 ms and arrivals at most 40 ms (inclusive), and input events
    are in non-decreasing ``log_ns`` order. Every valid group is built as given; breaking
    any one invariant raises a ContractError that names the field.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**
    """
    runs = dict.fromkeys(FRAME_SET_RULES + ("exposure_limit", "arrival_limit"), 0)
    for k in range(FRAME_SET_ITERATIONS):
        rng = factories.seeded_rng(FRAME_SET_SEED, k)
        fields = _valid_frame_set_fields(rng, k)
        where = f"seed {FRAME_SET_SEED}, iteration {k}, rule"
        try:
            frame_set = c.FrameSet(**fields)
        except c.ContractError as error:
            pytest.fail(f"{where} valid_group: rejected: {error}")
        assert frame_set.t_ns == fields["t_ns"], f"{where} valid_group: t_ns"
        assert frame_set.missing_camera_ids == fields["missing_camera_ids"], (
            f"{where} valid_group: missing_camera_ids"
        )
        exposures = [image.exposure_ns for image in frame_set.images]
        arrivals = [image.arrival_ns for image in frame_set.images]
        runs["exposure_limit"] += int(max(exposures) - min(exposures) == MAX_EXPOSURE_NS)
        runs["arrival_limit"] += int(max(arrivals) - min(arrivals) == MAX_ARRIVAL_NS)
        for rule, broken, prefix in _frame_set_violations(rng, fields):
            try:
                c.FrameSet(**broken)
            except c.ContractError as error:
                assert str(error).startswith(prefix), f"{where} {rule}: wanted {prefix!r}: {error}"
            else:
                pytest.fail(f"{where} {rule}: accepted, expected ContractError {prefix!r}")
            runs[rule] += 1
    starved = {name: total for name, total in runs.items() if total < FRAME_SET_MIN_RUNS}
    assert not starved, f"seed {FRAME_SET_SEED}: under {FRAME_SET_MIN_RUNS} runs: {starved}"

# ---------------------------------------------------------------------------------------
# State and geometry records: RoadPatch, TrackProjection, AuxiliaryGeometry,
# SearchSnapshot and Estimate (Requirements 12.5, 12.6; informally 2.4, 6.1-6.3, 12.1-12.4)
#
# These records have no builder in contract_factories (other agents own that file), so the
# small local builders below construct them directly. Each builder returns a valid record;
# a test overrides exactly one field so a rejection names that field: "<Record>.<field>: ".
# ---------------------------------------------------------------------------------------

RECORD_NS = EXPOSURE_NS  # a 19-digit stamp a float could not hold exactly
UP_NORMAL = np.array([0.0, 0.0, 1.0])


def _psd_cov(n=len(c.STATE_FIELDS)):
    """Return a symmetric positive-definite (n, n) float64 covariance."""
    return np.diag(np.arange(1.0, n + 1.0))


def _state(yaw=0.3):
    """Return a valid 6-vector estimator state in STATE_FIELDS order."""
    return np.array([1.0, 2.0, yaw, 0.5, -0.5, 0.1])


def _aux_valid(source=c.AuxSource.PNP, **overrides):
    """Return a valid AuxiliaryGeometry; ROAD_PATCH sources also name a patch id."""
    fields = {
        "source": source,
        "source_ns": RECORD_NS,
        "z_m": 0.2,
        "roll_rad": 0.01,
        "pitch_rad": -0.02,
        "road_patch_id": "patch-1" if source is c.AuxSource.ROAD_PATCH else None,
        "valid": True,
        "reason": "valid",
    }
    fields.update(overrides)
    return c.AuxiliaryGeometry(**fields)


def _road_patch(**overrides):
    """Return a valid RoadPatch on the track branch."""
    fields = {
        "patch_id": "patch-1",
        "branch": c.Branch.TRACK,
        "query_xy": np.array([1.0, 2.0]),
        "point_W": np.array([1.0, 2.0, 0.0]),
        "normal_W": UP_NORMAL,
        "support": 42,
        "residual_m": 0.05,
        "valid": True,
        "reason": "valid",
    }
    fields.update(overrides)
    return c.RoadPatch(**fields)


def _road_patch_invalid(**overrides):
    """Return an invalid RoadPatch with every optional value None."""
    fields = {
        "patch_id": "patch-1",
        "branch": c.Branch.UNKNOWN,
        "query_xy": np.array([1.0, 2.0]),
        "point_W": None,
        "normal_W": None,
        "support": 0,
        "residual_m": None,
        "valid": False,
        "reason": "off_route",
    }
    fields.update(overrides)
    return c.RoadPatch(**fields)


def _projection(**overrides):
    """Return a valid TrackProjection on the track route."""
    fields = {
        "t_ns": RECORD_NS,
        "reference_map_id": "ref-map",
        "route_id": "track",
        "s_m": 12.5,
        "n_m": -0.3,
        "epsi_rad": 0.1,
        "segment_index": 7,
        "xy_reconstruction_residual_m": 0.02,
        "valid": True,
        "reason": "valid",
    }
    fields.update(overrides)
    return c.TrackProjection(**fields)


def _snapshot(**overrides):
    """Return a valid SearchSnapshot."""
    fields = {
        "t_ns": RECORD_NS,
        "state": _state(),
        "P": _psd_cov(),
        "status": c.Status.TRACKING,
        "branch": c.Branch.TRACK,
        "auxiliary": _aux_valid(),
    }
    fields.update(overrides)
    return c.SearchSnapshot(**fields)


def _estimate(**overrides):
    """Return a valid pnp Estimate (group_id set, support stamp at most t_ns)."""
    fields = {
        "t_ns": RECORD_NS,
        "group_id": 3,
        "state": _state(),
        "P": _psd_cov(),
        "status": c.Status.TRACKING,
        "branch": c.Branch.TRACK,
        "evidence_kinds": (c.EvidenceKind.PNP,),
        "reason": "accepted",
        "auxiliary": _aux_valid(),
        "max_support_exposure_ns": RECORD_NS - 1,
    }
    fields.update(overrides)
    return c.Estimate(**fields)


# ---- RoadPatch (Requirements 12.5, 12.6) ----------------------------------------------


def test_road_patch_valid_is_accepted_and_stores_read_only_float64_copies():
    source = np.array([1.0, 2.0, 0.0])
    patch = _road_patch(point_W=source)
    assert patch.valid is True and patch.reason == "valid"
    assert patch.branch is c.Branch.TRACK
    for field in ("query_xy", "point_W", "normal_W"):
        stored = getattr(patch, field)
        assert stored.dtype == np.float64 and stored.flags.writeable is False
    assert not np.shares_memory(patch.point_W, source)
    assert source.flags.writeable is True  # the caller's array is untouched


def test_road_patch_invalid_with_everything_none_is_accepted():
    patch = _road_patch_invalid()
    assert patch.valid is False and patch.reason == "off_route"
    assert patch.point_W is None and patch.normal_W is None and patch.residual_m is None
    assert patch.branch is c.Branch.UNKNOWN


def test_road_patch_rejects_a_downward_normal():
    with pytest.raises(c.ContractError, match=r"^RoadPatch\.normal_W: .*point up"):
        _road_patch(normal_W=np.array([0.0, 0.0, -1.0]))


def test_road_patch_rejects_a_non_unit_normal():
    with pytest.raises(c.ContractError, match=r"^RoadPatch\.normal_W: "):
        _road_patch(normal_W=np.array([0.0, 0.0, 2.0]))


def test_road_patch_rejects_a_negative_residual():
    with pytest.raises(c.ContractError, match=r"^RoadPatch\.residual_m: "):
        _road_patch(residual_m=-0.01)


def test_road_patch_rejects_the_unknown_branch_when_valid():
    with pytest.raises(c.ContractError, match=r"^RoadPatch\.branch: .*valid patch"):
        _road_patch(branch=c.Branch.UNKNOWN)


@pytest.mark.parametrize("field", ["point_W", "normal_W", "residual_m"])
def test_road_patch_valid_requires_plane_values(field):
    # The valid=True completeness check is a record-level message naming the missing field.
    with pytest.raises(c.ContractError, match=rf"^RoadPatch: valid=True requires .*{field}"):
        _road_patch(**{field: None})


# ---- TrackProjection (Requirements 12.5, 12.6) ----------------------------------------


@pytest.mark.parametrize("route_id", c.ROUTE_IDS)
def test_track_projection_valid_on_each_route_id(route_id):
    projection = _projection(route_id=route_id)
    assert projection.valid is True and projection.route_id == route_id
    assert type(projection.route_id) is str


@pytest.mark.parametrize(
    "route_id",
    ["Track", "road", "", None, 0],
    ids=["mixed_case", "unknown", "empty", "none", "int"],
)
def test_track_projection_rejects_a_bad_route_id(route_id):
    with pytest.raises(c.ContractError, match=r"^TrackProjection\.route_id: "):
        _projection(route_id=route_id)


def test_track_projection_rejects_a_negative_arc_length():
    with pytest.raises(c.ContractError, match=r"^TrackProjection\.s_m: "):
        _projection(s_m=-0.1)


def test_track_projection_rejects_a_negative_segment_index():
    with pytest.raises(c.ContractError, match=r"^TrackProjection\.segment_index: "):
        _projection(segment_index=-1)


@pytest.mark.parametrize("field", list(c._PROJECTION_VALUE_FIELDS))
def test_track_projection_valid_requires_every_value(field):
    # The valid=True completeness check is a record-level message naming the missing field.
    with pytest.raises(c.ContractError, match=rf"^TrackProjection: valid=True requires .*{field}"):
        _projection(**{field: None})


def test_track_projection_stores_epsi_wrapped():
    projection = _projection(epsi_rad=3.0 * math.pi)  # 3 pi wraps to pi
    assert projection.epsi_rad == pytest.approx(math.pi)
    assert -math.pi < projection.epsi_rad <= math.pi


# ---- AuxiliaryGeometry (Requirements 12.4 informally, 2.4) ----------------------------


def test_auxiliary_geometry_unavailable_is_the_invalid_none_record():
    aux = c.AuxiliaryGeometry.unavailable("no_pnp_yet")
    assert aux.source is c.AuxSource.NONE and aux.valid is False
    assert aux.reason == "no_pnp_yet"
    assert aux.source_ns is None and aux.z_m is None
    assert aux.roll_rad is None and aux.pitch_rad is None and aux.road_patch_id is None


def test_auxiliary_geometry_valid_pnp_record():
    aux = _aux_valid(source=c.AuxSource.PNP)
    assert aux.source is c.AuxSource.PNP and aux.valid is True
    assert aux.road_patch_id is None
    assert type(aux.source_ns) is int and aux.source_ns == RECORD_NS


@pytest.mark.parametrize("field", ["source_ns", "z_m", "roll_rad", "pitch_rad"])
def test_auxiliary_geometry_valid_pnp_requires_pose_values(field):
    with pytest.raises(c.ContractError, match=r"^AuxiliaryGeometry: valid=True"):
        _aux_valid(source=c.AuxSource.PNP, **{field: None})


def test_auxiliary_geometry_valid_road_patch_record_names_the_patch():
    aux = _aux_valid(source=c.AuxSource.ROAD_PATCH)
    assert aux.source is c.AuxSource.ROAD_PATCH and aux.valid is True
    assert aux.road_patch_id == "patch-1"


def test_auxiliary_geometry_valid_road_patch_requires_the_patch_id():
    with pytest.raises(c.ContractError, match=r"^AuxiliaryGeometry: valid=True"):
        _aux_valid(source=c.AuxSource.ROAD_PATCH, road_patch_id=None)


def test_auxiliary_geometry_source_none_cannot_be_valid():
    with pytest.raises(c.ContractError, match=r"^AuxiliaryGeometry: source 'none' cannot be valid"):
        c.AuxiliaryGeometry(
            source=c.AuxSource.NONE,
            source_ns=None,
            z_m=None,
            roll_rad=None,
            pitch_rad=None,
            road_patch_id=None,
            valid=True,
            reason="valid",
        )


def test_auxiliary_geometry_stores_roll_and_pitch_wrapped():
    aux = _aux_valid(roll_rad=3.0 * math.pi, pitch_rad=-3.0 * math.pi)
    assert aux.roll_rad == pytest.approx(math.pi)
    assert aux.pitch_rad == pytest.approx(math.pi)  # -3 pi wraps to pi, not -pi


# ---- SearchSnapshot and Estimate (Requirements 6.1-6.3, 12.1-12.3 informally) ---------


def test_search_snapshot_valid_build_stores_read_only_state_and_covariance():
    snapshot = _snapshot()
    assert snapshot.t_ns == RECORD_NS
    assert snapshot.state.dtype == np.float64 and snapshot.state.flags.writeable is False
    assert snapshot.P.shape == (len(c.STATE_FIELDS), len(c.STATE_FIELDS))
    assert snapshot.P.flags.writeable is False
    assert snapshot.auxiliary.valid is True


def test_search_snapshot_rejects_a_non_psd_covariance():
    with pytest.raises(c.ContractError, match=r"^SearchSnapshot\.P: "):
        _snapshot(P=np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize("length", [5, 7], ids=["five", "seven"])
def test_search_snapshot_rejects_a_mis_sized_state(length):
    with pytest.raises(c.ContractError, match=r"^SearchSnapshot\.state: "):
        _snapshot(state=np.zeros(length))


def test_search_snapshot_wraps_yaw_into_the_stored_state():
    snapshot = _snapshot(state=_state(yaw=3.0 * math.pi))  # 3 pi wraps to pi
    assert snapshot.state[2] == pytest.approx(math.pi)
    assert -math.pi < snapshot.state[2] <= math.pi


def test_estimate_valid_build():
    estimate = _estimate()
    assert estimate.group_id == 3 and estimate.t_ns == RECORD_NS
    assert estimate.evidence_kinds == (c.EvidenceKind.PNP,)
    assert estimate.max_support_exposure_ns == RECORD_NS - 1
    assert estimate.state.flags.writeable is False


def test_estimate_rejects_a_non_psd_covariance():
    with pytest.raises(c.ContractError, match=r"^Estimate\.P: "):
        _estimate(P=np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize("length", [5, 7], ids=["five", "seven"])
def test_estimate_rejects_a_mis_sized_state(length):
    with pytest.raises(c.ContractError, match=r"^Estimate\.state: "):
        _estimate(state=np.zeros(length))


def test_estimate_wraps_yaw_into_the_stored_state():
    estimate = _estimate(state=_state(yaw=-3.0 * math.pi))  # -3 pi wraps to pi
    assert estimate.state[2] == pytest.approx(math.pi)
    assert -math.pi < estimate.state[2] <= math.pi


def test_estimate_group_id_is_none_only_for_the_initialization_estimate():
    initial = _estimate(
        group_id=None,
        evidence_kinds=(c.EvidenceKind.INITIAL_POSE,),
        reason="initialized",
        max_support_exposure_ns=None,
    )
    assert initial.group_id is None
    # A pnp estimate (not the initialization) may not drop its group id.
    with pytest.raises(c.ContractError, match=r"^Estimate\.group_id: "):
        _estimate(group_id=None)


def test_estimate_initial_pose_rejects_a_support_stamp():
    with pytest.raises(c.ContractError, match=r"^Estimate\.max_support_exposure_ns: "):
        _estimate(
            group_id=None,
            evidence_kinds=(c.EvidenceKind.INITIAL_POSE,),
            reason="initialized",
            max_support_exposure_ns=RECORD_NS,
        )


def test_estimate_prediction_only_has_no_support_stamp_and_keeps_its_group_id():
    estimate = _estimate(
        evidence_kinds=(c.EvidenceKind.PREDICTION_ONLY,),
        reason="no_evidence",
        max_support_exposure_ns=None,
    )
    assert estimate.group_id == 3 and estimate.max_support_exposure_ns is None
    # prediction_only may not carry a support stamp.
    with pytest.raises(c.ContractError, match=r"^Estimate\.max_support_exposure_ns: "):
        _estimate(
            evidence_kinds=(c.EvidenceKind.PREDICTION_ONLY,),
            reason="no_evidence",
            max_support_exposure_ns=RECORD_NS,
        )


def test_estimate_pnp_requires_a_support_stamp_at_most_t_ns():
    with pytest.raises(c.ContractError, match=r"^Estimate\.max_support_exposure_ns: "):
        _estimate(max_support_exposure_ns=None)
    with pytest.raises(c.ContractError, match=r"^Estimate\.max_support_exposure_ns: "):
        _estimate(max_support_exposure_ns=RECORD_NS + 1)  # after t_ns


def test_estimate_accepts_a_support_stamp_equal_to_t_ns():
    estimate = _estimate(max_support_exposure_ns=RECORD_NS)
    assert estimate.max_support_exposure_ns == RECORD_NS


@pytest.mark.parametrize(
    "kinds",
    [
        pytest.param((c.EvidenceKind.PNP, c.EvidenceKind.BORDER), id="pnp_border"),
        pytest.param((c.EvidenceKind.BORDER, c.EvidenceKind.GROUND_MOTION), id="reversed_pair"),
        pytest.param((), id="empty"),
        pytest.param((c.EvidenceKind.INITIAL_POSE, c.EvidenceKind.PNP), id="initial_plus_pnp"),
    ],
)
def test_estimate_rejects_a_disallowed_evidence_kinds_tuple(kinds):
    with pytest.raises(c.ContractError, match=r"^Estimate\.evidence_kinds: "):
        _estimate(evidence_kinds=kinds)


# ---- require_accepted on geometry records (Requirement 12.5, 12.6) --------------------


def test_require_accepted_passes_valid_geometry_records():
    patch = _road_patch()
    projection = _projection()
    assert c.require_accepted(patch) is patch
    assert c.require_accepted(projection) is projection


def test_require_accepted_rejects_invalid_geometry_records():
    with pytest.raises(c.RejectedEvidenceError, match=r"^RoadPatch: off_route"):
        c.require_accepted(_road_patch_invalid())
    invalid_projection = _projection(valid=False, reason="off_route", s_m=None)
    with pytest.raises(c.RejectedEvidenceError, match=r"^TrackProjection: off_route"):
        c.require_accepted(invalid_projection)

# ---------------------------------------------------------------------------------------
# Property 15: Estimate consistency (Requirements 12.1, 12.2, 12.3)
#
# For random state, covariance, evidence_kinds, group id and support stamp, Estimate
# constructs exactly when every rule holds: the state is a finite float64 (6,) with yaw
# stored wrapped; P is a symmetric PSD 6 x 6; evidence_kinds is one of the eight allowed
# tuples; group_id is None iff the kinds are (initial_pose,); and max_support_exposure_ns
# is None iff the kinds are (initial_pose,) or (prediction_only,) and otherwise an int at
# most t_ns. Each iteration draws a valid combination and checks it builds, then breaks
# exactly one rule and checks it is rejected. The expected outcome comes from the
# generator, never from Estimate.
# ---------------------------------------------------------------------------------------

ESTIMATE_SEED = 20261019
ESTIMATE_ITERATIONS = 200
ESTIMATE_MIN_RUNS = 20  # coverage floor of every violation rule

STATE_LEN = len(c.STATE_FIELDS)
YAW_INDEX = c.STATE_FIELDS.index("yaw_rad")
# The eight allowed evidence_kinds tuples, split by their group-id and support-stamp rules.
NO_GROUP_KINDS = ((c.EvidenceKind.INITIAL_POSE,),)
NO_SUPPORT_KINDS = ((c.EvidenceKind.INITIAL_POSE,), (c.EvidenceKind.PREDICTION_ONLY,))
ALLOWED_ESTIMATE_KINDS = (
    (c.EvidenceKind.INITIAL_POSE,),
    (c.EvidenceKind.PNP,),
    (c.EvidenceKind.HANDOFF_PNP,),
    (c.EvidenceKind.RECOVERY_RESET,),
    (c.EvidenceKind.GROUND_MOTION,),
    (c.EvidenceKind.BORDER,),
    (c.EvidenceKind.GROUND_MOTION, c.EvidenceKind.BORDER),
    (c.EvidenceKind.PREDICTION_ONLY,),
)
# evidence_kinds tuples that Estimate must reject outright (disallowed combinations).
DISALLOWED_ESTIMATE_KINDS = (
    (),
    (c.EvidenceKind.PNP, c.EvidenceKind.BORDER),
    (c.EvidenceKind.BORDER, c.EvidenceKind.GROUND_MOTION),
    (c.EvidenceKind.INITIAL_POSE, c.EvidenceKind.PNP),
    (c.EvidenceKind.PNP, c.EvidenceKind.PNP),
    (c.EvidenceKind.HANDOFF_PNP, c.EvidenceKind.GROUND_MOTION),
    (c.EvidenceKind.RECOVERY_RESET, c.EvidenceKind.BORDER),
)
ESTIMATE_RULES = (
    "disallowed_kinds",
    "group_id_none_mismatch",
    "group_id_set_mismatch",
    "support_present_mismatch",
    "support_absent_mismatch",
    "support_after_t_ns",
    "non_psd_P",
    "wrong_state_length",
)


def _random_state(rng):
    """Return a writable finite float64 (6,) state (yaw drawn well outside (-pi, pi])."""
    state = rng.uniform(-50.0, 50.0, size=STATE_LEN)
    state[YAW_INDEX] = rng.uniform(-5.0 * math.pi, 5.0 * math.pi)  # exercises the wrap
    return state


def _random_psd(rng, n=STATE_LEN):
    """Return a symmetric positive-definite float64 (n, n) covariance via A A^T + I."""
    a = rng.standard_normal((n, n))
    return a @ a.T + np.eye(n)


def _random_non_psd(rng, n=STATE_LEN):
    """Return a symmetric float64 (n, n) matrix with a clearly negative eigenvalue."""
    q = factories.random_rotation(rng) if n == 3 else np.linalg.qr(rng.standard_normal((n, n)))[0]
    eigenvalues = np.abs(rng.standard_normal(n)) + 1.0
    eigenvalues[int(rng.integers(n))] = -(abs(float(rng.standard_normal())) + 1.0)
    return q @ np.diag(eigenvalues) @ q.T


def _valid_estimate_fields(rng, k):
    """Return iteration ``k``'s valid Estimate fields with a drawn allowed kinds tuple."""
    kinds = ALLOWED_ESTIMATE_KINDS[k % len(ALLOWED_ESTIMATE_KINDS)]
    t_ns = RECORD_NS + int(rng.integers(0, 10**9))
    group_id = None if kinds in NO_GROUP_KINDS else int(rng.integers(0, 10**6))
    if kinds in NO_SUPPORT_KINDS:
        support_ns = None
        reason = "initialized" if kinds == (c.EvidenceKind.INITIAL_POSE,) else "no_evidence"
    else:
        support_ns = t_ns - int(rng.integers(0, t_ns - RECORD_NS + 1))  # in [RECORD_NS, t_ns]
        reason = "accepted"
    return {
        "t_ns": t_ns,
        "group_id": group_id,
        "state": _random_state(rng),
        "P": _random_psd(rng),
        "status": c.Status.TRACKING,
        "branch": c.Branch.TRACK,
        "evidence_kinds": kinds,
        "reason": reason,
        "auxiliary": _aux_valid(),
        "max_support_exposure_ns": support_ns,
    }


def _estimate_violations(rng, fields):
    """Yield (rule, ``fields`` with that one rule broken) for every applicable rule."""
    kinds = fields["evidence_kinds"]

    def broken(**changes):
        return {**fields, **changes}

    bad_kinds = DISALLOWED_ESTIMATE_KINDS[int(rng.integers(len(DISALLOWED_ESTIMATE_KINDS)))]
    yield "disallowed_kinds", broken(evidence_kinds=bad_kinds)

    # group_id: None iff kinds == (initial_pose,). Flip exactly that binding.
    if kinds in NO_GROUP_KINDS:
        yield "group_id_set_mismatch", broken(group_id=int(rng.integers(0, 10**6)))
    else:
        yield "group_id_none_mismatch", broken(group_id=None)

    # max_support_exposure_ns: None iff kinds in {(initial_pose,), (prediction_only,)}.
    if kinds in NO_SUPPORT_KINDS:
        yield "support_present_mismatch", broken(max_support_exposure_ns=fields["t_ns"])
    else:
        yield "support_absent_mismatch", broken(max_support_exposure_ns=None)
        yield "support_after_t_ns", broken(max_support_exposure_ns=fields["t_ns"] + 1)

    yield "non_psd_P", broken(P=_random_non_psd(rng))
    length = STATE_LEN + int(rng.choice((-1, 1)))
    yield "wrong_state_length", broken(state=rng.standard_normal(length))


# Feature: shared-contracts, Property 15: Estimate consistency
def test_property_15_estimate_consistency():
    """An Estimate is constructed exactly when every field rule holds.

    The generator draws an allowed evidence_kinds tuple and a consistent group id and
    support stamp, so each valid combination must build and store yaw wrapped into (-pi,
    pi]; breaking exactly one rule (a disallowed kinds tuple, a group-id presence mismatch,
    a support-stamp presence mismatch, a support stamp after t_ns, a non-PSD covariance or
    a mis-sized state) must raise a ContractError. Outcomes come from the generator.

    **Validates: Requirements 12.1, 12.2, 12.3**
    """
    runs = dict.fromkeys(ESTIMATE_RULES, 0)
    for k in range(ESTIMATE_ITERATIONS):
        rng = factories.seeded_rng(ESTIMATE_SEED, k)
        fields = _valid_estimate_fields(rng, k)
        where = f"seed {ESTIMATE_SEED}, iteration {k}, kinds {fields['evidence_kinds']}"
        try:
            estimate = c.Estimate(**fields)
        except c.ContractError as error:
            pytest.fail(f"{where} valid: rejected: {error}")
        expected_yaw = c.wrap_angle(fields["state"][YAW_INDEX])
        assert estimate.state[YAW_INDEX] == pytest.approx(expected_yaw), f"{where} valid: yaw"
        assert -math.pi < estimate.state[YAW_INDEX] <= math.pi, f"{where} valid: yaw range"
        assert estimate.state.flags.writeable is False, f"{where} valid: state writable"
        assert estimate.group_id == fields["group_id"], f"{where} valid: group_id"
        assert estimate.max_support_exposure_ns == fields["max_support_exposure_ns"], (
            f"{where} valid: support stamp"
        )
        for rule, bad in _estimate_violations(rng, fields):
            with pytest.raises(c.ContractError):
                c.Estimate(**bad)
            runs[rule] += 1
    starved = {name: total for name, total in runs.items() if total < ESTIMATE_MIN_RUNS}
    assert not starved, f"seed {ESTIMATE_SEED}: under {ESTIMATE_MIN_RUNS} runs: {starved}"

# ---------------------------------------------------------------------------------------
# Property 17: Immutability (Requirements 13.2, 13.3)
#
# For a representative record of each type, assigning to any field raises
# (FrozenInstanceError or TypeError), writing into any stored numpy array raises
# (read-only ValueError), and mutating the caller's input arrays after construction leaves
# the stored small arrays unchanged, because records copy them read-only. Observation's
# image and mask are the deliberate exception: they are the caller's own buffers marked
# read-only (not copies), so the test confirms the caller's buffer itself became read-only.
# ---------------------------------------------------------------------------------------

IMMUTABILITY_SEED = 20261020
IMMUTABILITY_ITERATIONS = 120


def _scratch_image():
    """Return a fresh writable canonical image whose pixels carry an identifiable pattern."""
    image = factories.canonical_image()
    image[0, 0, 0] = 123
    return image


def _scratch_mask():
    """Return a fresh writable canonical mask with one pixel cleared."""
    mask = factories.canonical_mask()
    mask[0, 0] = False
    return mask


def _immutability_cases(rng):
    """Return (label, record, scratch) tuples for one iteration.

    ``scratch`` maps a stored array field to the writable source array passed in, so the
    test can mutate the source and assert the stored copy did not change. Observation uses
    the caller-buffer convention for image and mask, so those are omitted from ``scratch``.
    """
    cases = []

    # Observation: image/mask are the caller's own buffers (marked read-only, not copied);
    # K and T_BC are private read-only copies.
    image, mask = _scratch_image(), _scratch_mask()
    k_source, t_source = factories.canonical_K(), factories.camera_to_body()
    observation = factories.make_observation(
        image=image, valid_mask=mask, K=k_source, T_BC=t_source
    )
    cases.append(("Observation", observation, {"K": k_source, "T_BC": t_source}))

    # FrameSet: a frozen record holding nested ImageMessages and no small array of its own.
    cases.append(("FrameSet", factories.make_frame_set(), {}))

    # PoseEvidence: id arrays and T_WB are read-only copies of the caller's arrays.
    track_ids = np.array([0, 1, 2], dtype=np.int64)
    landmark_ids = np.array([10, 11, 12], dtype=np.int64)
    t_wb = np.eye(4)
    pose = factories.make_pose_evidence(
        track_ids=track_ids, landmark_ids=landmark_ids, T_WB=t_wb
    )
    cases.append(
        ("PoseEvidence", pose, {"track_ids": track_ids, "landmark_ids": landmark_ids, "T_WB": t_wb})
    )

    # Estimate: state and P are read-only copies of the caller's arrays.
    state, cov = _state(yaw=float(rng.uniform(-math.pi, math.pi))), _psd_cov()
    estimate = _estimate(state=state, P=cov)
    cases.append(("Estimate", estimate, {"state": state, "P": cov}))

    # RoadPatch: query_xy, point_W and normal_W are read-only copies of the caller's arrays.
    query = np.array([1.0, 2.0])
    point = np.array([1.0, 2.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    patch = _road_patch(query_xy=query, point_W=point, normal_W=normal)
    cases.append(
        ("RoadPatch", patch, {"query_xy": query, "point_W": point, "normal_W": normal})
    )

    # GroupEvidence: a frozen record built from nested records, no small array of its own.
    cases.append(("GroupEvidence", factories.make_group_evidence(), {}))

    return cases


def _assert_field_assignment_raises(label, record, where):
    """Assert that assigning to each dataclass field of ``record`` is rejected."""
    for field in dataclasses.fields(record):
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            setattr(record, field.name, getattr(record, field.name))


def _assert_stored_arrays_read_only(label, record, where):
    """Assert that every stored numpy array on ``record`` refuses an in-place write."""
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        if isinstance(value, np.ndarray):
            assert value.flags.writeable is False, f"{where} {label}.{field.name}: writable"
            flat = value.reshape(-1)
            if flat.size:
                with pytest.raises(ValueError, match="read-only"):
                    flat[0] = flat[0]


# Feature: shared-contracts, Property 17: Immutability
def test_property_17_immutability():
    """Every record is frozen, its stored arrays are read-only and copied from the caller.

    For a representative record of each type, assigning any field raises, writing into any
    stored numpy array raises, and mutating the caller's source arrays after construction
    leaves the stored small arrays unchanged. Observation keeps the caller's own image and
    mask buffers (marked read-only rather than copied), so those two are checked as buffers
    that became read-only instead of as copies.

    **Validates: Requirements 13.2, 13.3**
    """
    seen = set()
    for k in range(IMMUTABILITY_ITERATIONS):
        rng = factories.seeded_rng(IMMUTABILITY_SEED, k)
        for label, record, scratch in _immutability_cases(rng):
            where = f"seed {IMMUTABILITY_SEED}, iteration {k}, record {label}"
            seen.add(label)
            _assert_field_assignment_raises(label, record, where)
            _assert_stored_arrays_read_only(label, record, where)
            # Mutating the caller's source arrays must not change the stored copies.
            for field_name, source in scratch.items():
                stored = getattr(record, field_name)
                before = np.array(stored)
                assert source.flags.writeable is True, f"{where}.{field_name}: source frozen"
                source.reshape(-1)[0] += 7.0
                np.testing.assert_array_equal(
                    getattr(record, field_name),
                    before,
                    err_msg=f"{where}.{field_name}: stored copy changed with its source",
                )
    expected = {"Observation", "FrameSet", "PoseEvidence", "Estimate", "RoadPatch", "GroupEvidence"}
    assert seen == expected, f"covered {sorted(seen)}, expected {sorted(expected)}"

# ---------------------------------------------------------------------------------------
# Property 18: JSON-safe summaries (Requirements 13.4, 13.5)
#
# For a representative record of every record type, json.dumps(record.summary(),
# allow_nan=False) succeeds; integer stamps decode back to the identical Python ints;
# enum fields appear as their string values; and no image, mask, JPEG payload or array of
# more than SUMMARY_MAX_ARRAY_ITEMS elements is listed as data (each is shape/dtype or
# {"bytes": n}). A large PoseEvidence/Correspondences exercises the big-array path and an
# ImageMessage/Observation the image and bytes paths.
# ---------------------------------------------------------------------------------------

SUMMARY_SEED = 20261021
SUMMARY_ITERATIONS = 100
MAX_ITEMS = c.SUMMARY_MAX_ARRAY_ITEMS

# Record fields that hold integer nanosecond stamps and must survive the JSON round trip.
STAMP_FIELDS = {
    "ImageMessage": ("exposure_ns", "arrival_ns"),
    "InputEvent": ("log_ns", "reference_ns"),
    "Observation": ("exposure_ns", "arrival_ns"),
    "SearchSnapshot": ("t_ns",),
    "Estimate": ("t_ns", "max_support_exposure_ns"),
    "TrackProjection": ("t_ns",),
    "Correspondences": ("exposure_ns",),
    "PoseEvidence": ("exposure_ns",),
    "AnchorRequest": ("source_exposure_ns", "delivery_not_before_ns"),
    "AnchorResult": ("source_exposure_ns", "worker_duration_ns"),
    "RecoveryReset": ("t_ns",),
}


def _large_correspondences(n):
    """Return Correspondences with ``n`` rows (ids beyond MAX_ITEMS go to the big-array path)."""
    rng = factories.seeded_rng(SUMMARY_SEED, 999, n)
    pixels = np.column_stack(
        (rng.uniform(0.0, float(W - 1), size=n), rng.uniform(0.0, float(H - 1), size=n))
    )
    return factories.make_correspondences(
        keypoint_ids=np.arange(n, dtype=np.int64),
        pixels=pixels,
        landmark_ids=np.arange(1000, 1000 + n, dtype=np.int64),
    )


def _large_pose_evidence(n):
    """Return an accepted tracking PoseEvidence with ``n`` > MAX_ITEMS inliers."""
    return factories.make_pose_evidence(
        inlier_count=n,
        track_ids=np.arange(n, dtype=np.int64),
        landmark_ids=np.arange(5000, 5000 + n, dtype=np.int64),
    )


def _representative_records():
    """Return one valid record of every contract record type, keyed by class name."""
    big = MAX_ITEMS + 5  # just over the shape/dtype threshold
    records = [
        factories.make_image_message(),
        factories.make_input_event(reference_ns=RECORD_NS),
        factories.make_calibration(),
        factories.make_observation(),
        factories.make_frame_set(),
        c.AuxiliaryGeometry.unavailable("not_estimated"),
        _aux_valid(),
        _road_patch(),
        _road_patch_invalid(),
        _snapshot(),
        _estimate(),
        _projection(),
        factories.make_correspondences(),
        _large_correspondences(big),
        factories.make_pose_evidence(),
        _large_pose_evidence(big),
        factories.make_handoff_evidence(),
        factories.make_motion_evidence(),
        factories.make_border_evidence(),
        factories.make_anchor_request(),
        factories.make_anchor_result(),
        factories.make_recovery_reset(),
        factories.make_unused_evidence(),
        factories.make_group_evidence(),
    ]
    by_type: dict[str, object] = {}
    for record in records:
        by_type.setdefault(type(record).__name__, record)
    return by_type


# The 20 record classes that Property 18 must cover with at least one representative.
EXPECTED_RECORD_TYPES = frozenset(
    {
        "ImageMessage",
        "InputEvent",
        "CameraCalibration",
        "Observation",
        "FrameSet",
        "AuxiliaryGeometry",
        "RoadPatch",
        "SearchSnapshot",
        "Estimate",
        "TrackProjection",
        "Correspondences",
        "PoseEvidence",
        "HandoffEvidence",
        "MotionEvidence",
        "BorderEvidence",
        "AnchorRequest",
        "AnchorResult",
        "RecoveryReset",
        "UnusedEvidence",
        "GroupEvidence",
    }
)


def _assert_no_raw_array_data(summary, where, path="summary"):
    """Walk a JSON-safe summary and assert no list is longer than MAX_ITEMS entries.

    Large arrays, images and masks must be reduced to {"shape": ..., "dtype": ...} and
    byte payloads to {"bytes": n}, so no raw data list of more than MAX_ITEMS items (and
    in particular no image, mask or JPEG payload) survives anywhere in the structure.
    """
    if isinstance(summary, dict):
        if "shape" in summary and "dtype" in summary:
            return  # a reduced array descriptor; its shape list is metadata, not data
        if set(summary) == {"bytes"}:
            assert isinstance(summary["bytes"], int), f"{where} {path}.bytes"
            return
        for key, value in summary.items():
            _assert_no_raw_array_data(value, where, f"{path}.{key}")
    elif isinstance(summary, list):
        assert len(summary) <= MAX_ITEMS, f"{where} {path}: raw data list of {len(summary)} items"
        for index, value in enumerate(summary):
            _assert_no_raw_array_data(value, where, f"{path}[{index}]")


def _collect_enum_values(record):
    """Return the string value expected for each top-level enum field of ``record``."""
    expected = {}
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        if isinstance(value, enum.Enum):
            expected[field.name] = value.value
    return expected


# Feature: shared-contracts, Property 18: JSON-safe summaries
def test_property_18_json_safe_summaries():
    """Every record summarizes to a value that survives strict JSON, keeping ints and enums.

    For one representative of every record type: json.dumps(summary, allow_nan=False)
    succeeds, integer stamps decode back to the identical ints, enum fields appear as their
    string values and no image, mask, JPEG payload or array longer than
    SUMMARY_MAX_ARRAY_ITEMS appears as data (each is shape/dtype or {"bytes": n}). A large
    PoseEvidence and Correspondences drive the big-array path; an ImageMessage and
    Observation drive the image and bytes paths.

    **Validates: Requirements 13.4, 13.5**
    """
    seen = set()
    for k in range(SUMMARY_ITERATIONS):
        records = _representative_records()
        for name, record in records.items():
            where = f"seed {SUMMARY_SEED}, iteration {k}, record {name}"
            seen.add(name)
            summary = record.summary()
            assert summary["record"] == name, f"{where}: record tag"
            text = json.dumps(summary, allow_nan=False)
            decoded = json.loads(text)  # Python json keeps integers exact

            for field_name in STAMP_FIELDS.get(name, ()):
                original = getattr(record, field_name)
                if original is None:
                    assert decoded[field_name] is None, f"{where}.{field_name}: lost None"
                    continue
                round_tripped = decoded[field_name]
                assert type(round_tripped) is int, f"{where}.{field_name}: not an int"
                assert round_tripped == original, f"{where}.{field_name}: {round_tripped} != "
                assert int(original) == original, f"{where}.{field_name}: exactness"

            for field_name, value in _collect_enum_values(record).items():
                assert summary[field_name] == value, f"{where}.{field_name}: enum value"
                assert isinstance(summary[field_name], str), f"{where}.{field_name}: enum type"

            _assert_no_raw_array_data(decoded, where)

    # ImageMessage keeps its payload out of data: the JPEG is a {"bytes": n} descriptor.
    message = factories.make_image_message()
    assert message.summary()["data"] == {"bytes": len(factories.JPEG_PAYLOAD)}
    # Observation keeps image and mask as shape/dtype descriptors, never as pixel lists.
    observation = factories.make_observation()
    for field_name in ("image", "valid_mask"):
        descriptor = observation.summary()[field_name]
        assert set(descriptor) == {"shape", "dtype"}, f"Observation.{field_name}: not shape/dtype"

    assert seen == set(EXPECTED_RECORD_TYPES), (
        f"covered {sorted(seen)}, expected {sorted(EXPECTED_RECORD_TYPES)}"
    )

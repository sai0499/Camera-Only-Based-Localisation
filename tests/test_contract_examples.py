"""Copyable usage examples of the shared contracts for A, B and C, plus Protocol checks.

Each example builds valid records the way an owner's module would. ``contract_factories``
(``f``) supplies valid defaults; a few records are spelled out with every keyword field so
they can be copied as templates. The last section checks the Protocol call boundaries:
parameter names, docstring-only bodies, ``isinstance`` conformance of test-only fakes
(which only ever return rejected evidence) and the module docstring's flow notes.
"""

import ast
import dataclasses
import inspect
import json
from pathlib import Path

import contract_factories as f
import numpy as np
import pytest

from camloc import contracts as c

E = f.EXPOSURE_NS  # exposure of the default fr image; the default group's r image is 0.5 ms later


def as_json(record) -> dict:
    """Diagnostics form of any record: one JSONL line, never a CSV row."""
    return json.loads(json.dumps(record.summary(), allow_nan=False))


# ---------------------------------------------------------------------------------------
# A: input, geometry, borders, track coordinates
# ---------------------------------------------------------------------------------------


def test_a_image_messages_to_frame_set():
    front = c.ImageMessage(
        camera_id="fr",
        exposure_ns=E,  # header.stamp
        arrival_ns=E + 42_000_000,  # MCAP log time
        source_id=f"fr@{E}",
        frame_id="constructor0/camera_fr",
        image_format="jpeg",
        calibration_id="calibration_fr_v1",
        data=f.JPEG_PAYLOAD,  # undecoded bytes; grouping never decodes
    )
    rear = f.make_image_message(camera_id="r", exposure_ns=E + 500_000)
    frame_set = c.FrameSet(
        group_id=0,
        t_ns=rear.exposure_ns,  # group time = max exposure = CSV timestamp
        selected_camera_ids=(c.PRIMARY_CAMERA, c.SECONDARY_CAMERA),
        images=(front, rear),
        missing_camera_ids=(),
        closure=c.FrameSetClosure.COMPLETE,
        terminal_partial=False,
        input_events=(),
    )
    assert frame_set.t_ns == E + 500_000
    assert as_json(frame_set)["images"][0]["data"] == {"bytes": len(f.JPEG_PAYLOAD)}

    partial = f.make_frame_set(present=("fr",))  # r missed the deadline
    assert partial.missing_camera_ids == ("r",)
    assert partial.closure is c.FrameSetClosure.DEADLINE


def test_a_observation():
    observation = f.make_observation()  # canonical RGB uint8 (464, 960, 3), bool mask
    assert observation.image.shape == (c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH, 3)
    assert not observation.image.flags.writeable  # buffers become read-only, never copied
    assert as_json(observation)["image"] == {"shape": [464, 960, 3], "dtype": "uint8"}


def test_a_road_patch_valid_and_invalid():
    patch = c.RoadPatch(
        patch_id="road_patch_0",
        branch=c.Branch.TRACK,
        query_xy=np.array([120.0, -35.0]),
        point_W=np.array([120.0, -35.0, 0.12]),
        normal_W=np.array([0.0, 0.0, 1.0]),  # unit, pointing up
        support=24,
        residual_m=0.02,
        valid=True,
        reason="valid",
    )
    assert c.require_accepted(patch) is patch

    no_plane = c.RoadPatch(
        patch_id="road_patch_1",
        branch=c.Branch.UNKNOWN,
        query_xy=np.array([400.0, 10.0]),
        point_W=None,
        normal_W=None,
        support=0,
        residual_m=None,
        valid=False,
        reason="no_paired_borders",  # disables ground motion and borders for the group
    )
    with pytest.raises(c.RejectedEvidenceError, match="no_paired_borders"):
        c.require_accepted(no_plane)
    assert as_json(no_plane)["valid"] is False


def test_a_border_evidence():
    border = f.make_border_evidence()  # lateral + yaw only; no along-track field
    assert border.accepted and border.R.shape == (2, 2)
    weak = f.make_border_evidence(
        accepted=False, reason="weak_contours", lateral_coordinate_m=None, yaw_rad=None, R=None
    )
    assert as_json(weak)["reason"] == "weak_contours"


def test_a_track_projection():
    on_track = c.TrackProjection(
        t_ns=E,
        reference_map_id=c.SUPPLIED_MAP_SHA256,
        route_id="track",
        s_m=812.4,
        n_m=-1.2,  # positive to the left of the segment direction
        epsi_rad=0.03,
        segment_index=402,
        xy_reconstruction_residual_m=0.004,
        valid=True,
        reason="valid",
    )
    off_pit = c.TrackProjection(
        t_ns=E,
        reference_map_id=c.SUPPLIED_MAP_SHA256,
        route_id="pit",
        s_m=None,
        n_m=None,
        epsi_rad=None,
        segment_index=None,
        xy_reconstruction_residual_m=None,
        valid=False,
        reason="outside_route",
    )
    assert (on_track.route_id, off_pit.route_id) == c.ROUTE_IDS
    assert as_json(off_pit)["s_m"] is None


# ---------------------------------------------------------------------------------------
# B: tracking, anchors, handoff, ground motion
# ---------------------------------------------------------------------------------------


def test_b_pose_evidence_rejected_and_accepted():
    rejected = c.PoseEvidence(
        camera_id="fr",
        exposure_ns=E,
        source=c.PoseSource.TRACKING,
        T_WB=None,
        inlier_count=0,
        inlier_fraction=0.0,
        median_error_px=None,
        p90_error_px=None,
        occupied_cells=0,
        track_ids=np.array([], dtype=np.int64),
        landmark_ids=np.array([], dtype=np.int64),
        map_id=f.MAP_ID,
        calibration_id="calibration_fr_v1",
        preprocessing_id=f.PREPROCESSING_ID,
        accepted=False,
        reason="too_few_inliers",  # weak evidence is a valid record, never an exception
    )
    assert as_json(rejected)["accepted"] is False

    accepted = f.make_pose_evidence()  # accepted iff reason == "accepted"
    T_WB = accepted.T_WB
    x, y, yaw = T_WB[0, 3], T_WB[1, 3], np.arctan2(T_WB[1, 0], T_WB[0, 0])
    assert (x, y, yaw) == (0.0, 0.0, 0.0)
    assert not T_WB.flags.writeable
    with pytest.raises(c.ContractError):
        f.make_pose_evidence(accepted=True, reason="too_few_inliers")


def test_b_anchor_result_has_no_pose():
    request = f.make_anchor_request()
    result = f.make_anchor_result()  # status ok: source-image correspondences only
    result.check_against(request)
    corr = result.correspondences
    assert (corr.camera_id, corr.exposure_ns) == (request.camera_id, request.source_exposure_ns)
    names = {field.name for field in dataclasses.fields(result)}
    assert not {"T_WB", "pose", "x", "y", "yaw"} & names

    failed = f.make_anchor_result(status=c.AnchorStatus.FAILED)
    assert (failed.reason, failed.correspondences) == ("insufficient_matches", None)


def test_b_handoff_evidence():
    handoff = f.make_handoff_evidence()  # PnP re-solved on the current image
    assert handoff.source_exposure_ns < handoff.measurement_ns
    assert handoff.support_ns[0] == handoff.source_exposure_ns
    assert handoff.support_ns[-1] == handoff.measurement_ns
    assert handoff.transitions == 2 <= c.MAX_HANDOFF_TRANSITIONS
    assert handoff.pose.source is c.PoseSource.HANDOFF
    assert handoff.pose.exposure_ns == handoff.measurement_ns
    assert as_json(handoff)["pose"]["record"] == "PoseEvidence"


def test_b_motion_evidence():
    motion = f.make_motion_evidence()  # planar T_Bprev_Bcurr, 3 x 3 (dx, dy, dyaw) covariance
    assert motion.previous_ns < motion.current_ns
    assert motion.T_Bprev_Bcurr[0, 3] == 1.5
    no_plane = f.make_motion_evidence(
        accepted=False,
        reason="invalid_road_patch",
        T_Bprev_Bcurr=None,
        covariance=None,
        ground_residual_m=None,
        road_patch_id=None,
        inlier_count=0,
        inlier_fraction=0.0,
    )
    assert as_json(no_plane)["reason"] == "invalid_road_patch"


# ---------------------------------------------------------------------------------------
# C: scheduler, group assembly, estimate, diagnostics
# ---------------------------------------------------------------------------------------


def test_c_anchor_request_and_delivery_group():
    request = f.make_anchor_request()
    source = request.source_exposure_ns
    assert request.delivery_not_before_ns == source + c.DELIVERY_MIN_DELAY_NS
    group_times = [source + k * 33_000_000 for k in range(6)]
    delivery = next(t for t in group_times if t >= request.delivery_not_before_ns)
    assert delivery == source + 99_000_000  # first group at or after source + 90 ms


def test_c_group_evidence():
    frame_set = f.make_frame_set()  # fr at E, r at E + 0.5 ms
    rear_rejected = f.make_pose_evidence(
        camera_id="r",
        exposure_ns=E + f.GROUP_EXPOSURE_STEP_NS,
        T_WB=None,
        accepted=False,
        reason="too_few_inliers",
    )
    group = c.GroupEvidence(
        frame_set=frame_set,
        pose=f.make_pose_evidence(),  # selected in-gate fr tracking pose
        motion=None,
        border=None,
        reset=None,
        rejected=(rear_rejected,),
        unused=(c.UnusedEvidence(reason="superseded_by_pose", evidence=f.make_border_evidence()),),
    )
    assert group.consumed_kinds() == (c.EvidenceKind.PNP,)
    assert f.make_group_evidence(pose=f.make_handoff_evidence()).consumed_kinds() == (
        c.EvidenceKind.HANDOFF_PNP,
    )
    assert f.make_group_evidence().consumed_kinds() == (c.EvidenceKind.PREDICTION_ONLY,)
    assert as_json(group)["rejected"][0]["reason"] == "too_few_inliers"


def test_c_estimate():
    pose = f.make_pose_evidence()
    estimate = c.Estimate(
        t_ns=E + f.GROUP_EXPOSURE_STEP_NS,
        group_id=0,
        state=np.array([120.0, -35.0, 0.25, 18.0, 0.1, 0.02]),  # STATE_FIELDS order
        P=np.diag([0.04, 0.04, 0.001, 0.25, 0.25, 0.01]),
        status=c.Status.TRACKING,
        branch=c.Branch.TRACK,
        evidence_kinds=(c.EvidenceKind.PNP,),
        reason="accepted",
        auxiliary=c.AuxiliaryGeometry(
            source=c.AuxSource.PNP,
            source_ns=pose.exposure_ns,
            z_m=0.31,
            roll_rad=0.0,
            pitch_rad=0.01,
            road_patch_id=None,
            valid=True,
            reason="valid",
        ),
        max_support_exposure_ns=pose.exposure_ns,
    )
    assert estimate.max_support_exposure_ns <= estimate.t_ns
    assert as_json(estimate)["evidence_kinds"] == ["pnp"]


def test_c_require_accepted_and_slot_rejection():
    rejected = f.make_pose_evidence(T_WB=None, accepted=False, reason="too_few_inliers")
    with pytest.raises(c.RejectedEvidenceError, match="too_few_inliers"):
        c.require_accepted(rejected)
    with pytest.raises(c.RejectedEvidenceError):
        f.make_group_evidence(pose=rejected)  # rejected records belong in `rejected`


def test_c_decode_failure_and_jsonl_diagnostics():
    image = f.make_image_message()
    event = c.InputEvent(
        kind=c.InputEventKind.DECODE_FAILURE,  # C's record when canonicalize raises InputError
        log_ns=image.arrival_ns,
        camera_id=image.camera_id,
        exposure_ns=image.exposure_ns,
        source_id=image.source_id,
        detail="corrupt JPEG",
    )
    records = (event, f.make_frame_set(), f.make_anchor_request(), f.make_group_evidence())
    lines = [json.dumps(record.summary(), allow_nan=False) for record in records]
    assert [json.loads(line)["record"] for line in lines] == [
        "InputEvent",
        "FrameSet",
        "AnchorRequest",
        "GroupEvidence",
    ]


# ---------------------------------------------------------------------------------------
# Protocol call boundaries
# ---------------------------------------------------------------------------------------

PROTOCOL_METHODS = {
    c.FrameSetSource: {"calibrations": (), "__iter__": (), "trailing_events": ()},
    c.Canonicalizer: {"canonicalize": ("image",)},
    c.RoadGeometry: {"road_patch": ("x", "y", "yaw", "branch")},
    c.BorderEvidenceSource: {"border_evidence": ("observation", "snapshot", "aux")},
    c.TrackProjector: {"project": ("estimate",), "reset": ("reset",)},
    c.MappedTracker: {
        "observe": ("observation",),
        "track": ("observation", "snapshot", "aux"),
        "handoff": ("result", "current"),
        "settle": ("handoff", "disposition"),
        "verify_candidate": ("observation",),
        "drop_candidate": ("reason",),
        "reset": ("reset",),
    },
    c.AnchorWorker: {"run": ("request",)},
    c.GroundMotionEstimator: {
        "estimate": ("previous", "current", "patch", "previous_snapshot", "current_snapshot")
    },
    c.Estimator: {"snapshot": ("t_ns",), "step": ("evidence",)},
}


@pytest.mark.parametrize("protocol", PROTOCOL_METHODS, ids=lambda p: p.__name__)
def test_protocol_signatures(protocol):
    for method, expected in PROTOCOL_METHODS[protocol].items():
        signature = inspect.signature(getattr(protocol, method))
        parameters = list(signature.parameters.values())
        assert tuple(p.name for p in parameters) == ("self", *expected), method
        assert all(p.annotation is not p.empty for p in parameters[1:]), method
        assert signature.return_annotation is not signature.empty, method


def test_ground_motion_requires_explicit_exposure_geometry():
    signature = inspect.signature(c.GroundMotionEstimator.estimate)
    with pytest.raises(TypeError):
        signature.bind(None, f.make_observation(), f.make_observation(), FakeRoad())
    for name in ("previous_snapshot", "current_snapshot"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def _protocol_classes() -> dict[str, ast.ClassDef]:
    tree = ast.parse(Path(c.__file__).read_text(encoding="utf-8"))
    names = {protocol.__name__ for protocol in PROTOCOL_METHODS}
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    }


def _is_constant(statement: ast.stmt, kind: type) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, kind)
    )


def test_protocol_bodies_hold_only_docstrings_and_ellipsis():
    classes = _protocol_classes()
    assert set(classes) == {protocol.__name__ for protocol in PROTOCOL_METHODS}
    for protocol, methods in PROTOCOL_METHODS.items():
        node = classes[protocol.__name__]
        assert [ast.unparse(d) for d in node.decorator_list] == ["runtime_checkable"]
        assert [ast.unparse(b) for b in node.bases] == ["Protocol"]
        assert ast.get_docstring(node), protocol.__name__
        functions = [s for s in node.body[1:] if isinstance(s, ast.FunctionDef)]
        assert len(functions) == len(node.body) - 1, f"{protocol.__name__}: only methods"
        assert [fn.name for fn in functions] == list(methods)
        for fn in functions:
            where = f"{protocol.__name__}.{fn.name}"
            assert ast.get_docstring(fn), f"{where}: needs a docstring"
            assert all(
                _is_constant(s, str) or _is_constant(s, type(Ellipsis)) for s in fn.body
            ), f"{where}: body must hold only a docstring and ..."


# Test-only fakes: they satisfy the Protocols structurally and only ever return rejected or
# invalid records (or raise), never a plausible accepted pose.


class FakeSource:
    def calibrations(self):
        return {}

    def __iter__(self):
        return iter(())

    def trailing_events(self):
        return (c.InputEvent(kind=c.InputEventKind.END_OF_INPUT, log_ns=E),)


class FakeCanonicalizer:
    def canonicalize(self, image):
        raise c.InputError(f"fake decoder cannot decode {image.source_id}")


class FakeRoad:
    def road_patch(self, x, y, yaw, branch):
        return c.RoadPatch(
            patch_id="fake_patch",
            branch=branch,
            query_xy=np.array([x, y]),
            point_W=None,
            normal_W=None,
            support=0,
            residual_m=None,
            valid=False,
            reason="not_implemented",
        )


class FakeBorders:
    def border_evidence(self, observation, snapshot, aux):
        return f.make_border_evidence(
            camera_id=observation.camera_id,
            exposure_ns=observation.exposure_ns,
            accepted=False,
            reason="not_implemented",
        )


class FakeProjector:
    def project(self, estimate):
        return tuple(
            c.TrackProjection(
                t_ns=estimate.t_ns,
                reference_map_id=c.SUPPLIED_MAP_SHA256,
                route_id=route,
                s_m=None,
                n_m=None,
                epsi_rad=None,
                segment_index=None,
                xy_reconstruction_residual_m=None,
                valid=False,
                reason="not_implemented",
            )
            for route in c.ROUTE_IDS
        )

    def reset(self, reset):
        return None


class FakeTracker:
    def observe(self, observation):
        return None

    def track(self, observation, snapshot, aux):
        return f.make_pose_evidence(
            camera_id=observation.camera_id,
            exposure_ns=observation.exposure_ns,
            T_WB=None,
            accepted=False,
            reason="not_implemented",
        )

    def handoff(self, result, current):
        return f.make_handoff_evidence(
            job_id=result.job_id,
            camera_id=current.camera_id,
            source_exposure_ns=result.source_exposure_ns,
            measurement_ns=current.exposure_ns,
            support_ns=(),
            correspondences=None,
            pose=None,
            accepted=False,
            reason="not_implemented",
        )

    def settle(self, handoff, disposition):
        return None

    def verify_candidate(self, observation):
        return None  # no candidate held

    def drop_candidate(self, reason):
        return None

    def reset(self, reset):
        return None


class FakeWorker:
    def run(self, request):
        return f.make_anchor_result(
            job_id=request.job_id,
            purpose=request.purpose,
            camera_id=request.camera_id,
            source_group_id=request.source_group_id,
            source_exposure_ns=request.source_exposure_ns,
            map_id=request.map_id,
            preprocessing_id=request.preprocessing_id,
            status=c.AnchorStatus.FAILED,
            matched_reference_ids=(),
            remaining_candidate_ids=(),
        )


class FakeMotion:
    def estimate(self, previous, current, patch, *, previous_snapshot, current_snapshot):
        return f.make_motion_evidence(
            camera_id=current.camera_id,
            previous_ns=previous.exposure_ns,
            current_ns=current.exposure_ns,
            T_Bprev_Bcurr=None,
            covariance=None,
            ground_residual_m=None,
            road_patch_id=None,
            inlier_count=0,
            inlier_fraction=0.0,
            accepted=False,
            reason=patch.reason if not patch.valid else "not_implemented",
        )


class FakeEstimator:
    def snapshot(self, t_ns):
        raise NotImplementedError("test-only fake; C's estimator lives in camloc/filter.py")

    def step(self, evidence):
        raise NotImplementedError("test-only fake; C's estimator lives in camloc/filter.py")


FAKES = {
    c.FrameSetSource: FakeSource,
    c.Canonicalizer: FakeCanonicalizer,
    c.RoadGeometry: FakeRoad,
    c.BorderEvidenceSource: FakeBorders,
    c.TrackProjector: FakeProjector,
    c.MappedTracker: FakeTracker,
    c.AnchorWorker: FakeWorker,
    c.GroundMotionEstimator: FakeMotion,
    c.Estimator: FakeEstimator,
}


def test_fakes_conform_and_partial_objects_do_not():
    for protocol, fake in FAKES.items():
        assert isinstance(fake(), protocol), protocol.__name__
        assert not isinstance(object(), protocol), protocol.__name__

    class OnlyObserves:
        def observe(self, observation):
            return None

    assert not isinstance(OnlyObserves(), c.MappedTracker)


def test_fakes_only_return_rejected_evidence():
    observation = f.make_observation()  # the default group's fr image
    previous = f.make_observation(exposure_ns=E - f.MOTION_STEP_NS)
    snapshot = f.make_search_snapshot()
    aux = snapshot.auxiliary
    tracker = FakeTracker()

    tracker.observe(observation)
    pose = tracker.track(observation, snapshot, aux)
    patch = FakeRoad().road_patch(120.0, -35.0, 0.25, c.Branch.TRACK)
    motion = FakeMotion().estimate(
        previous, observation, patch,
        previous_snapshot=f.make_search_snapshot(t_ns=previous.exposure_ns),
        current_snapshot=snapshot,
    )
    border = FakeBorders().border_evidence(observation, snapshot, aux)
    request = f.make_anchor_request(source_exposure_ns=E - c.DELIVERY_MIN_DELAY_NS)
    result = FakeWorker().run(request)
    result.check_against(request)
    handoff = tracker.handoff(result, observation)

    for record in (pose, patch, motion, border, handoff):
        with pytest.raises(c.RejectedEvidenceError):
            c.require_accepted(record)
    assert result.status is c.AnchorStatus.FAILED and result.correspondences is None
    assert tracker.verify_candidate(observation) is None
    with pytest.raises(c.InputError):
        FakeCanonicalizer().canonicalize(f.make_image_message())
    with pytest.raises(NotImplementedError):
        FakeEstimator().step(f.make_group_evidence())

    group = f.make_group_evidence(rejected=(pose, motion, border))
    assert group.consumed_kinds() == (c.EvidenceKind.PREDICTION_ONLY,)
    assert as_json(group)["rejected"][1]["reason"] == "not_implemented"


FLOW_NOTES = (
    "Call order and flows between owners",
    "Decode set",
    "Road patches",
    "Auxiliary",
    "Anchor delivery",
    "DELIVERY_MIN_DELAY_NS = 90 ms",
    "re-solves",
    "Recovery reset",
    "Input events",
    "decode_failure",
    "Diagnostics",
    "never to the CSV",
    "open_frame_sets",
    "make_canonicalizer",
    "load_road_geometry",
    "make_border_source",
    "load_track_projector",
    "load_tracker",
    "load_anchor_worker",
    "make_ground_motion",
    "SubmissionWriter",
    "tests/test_contract_examples.py",
)


def test_module_docstring_mentions_each_flow():
    doc = c.__doc__
    missing = [note for note in FLOW_NOTES if note not in doc]
    missing += [p.__name__ for p in PROTOCOL_METHODS if p.__name__ not in doc]
    assert not missing

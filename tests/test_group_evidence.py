"""Example tests of the anchor, recovery and group records.

Each test starts from a valid ``contract_factories`` builder and changes one thing, so a
rejection names that rule. The defaults line up with ``make_frame_set()``: evidence is
measured on the group's own ``fr`` image at EXPOSURE_NS (``r`` is exposed STEP_NS later).
These tests write nothing.
"""

import dataclasses
import json

import pytest

import contract_factories as factories
from camloc import contracts as c

EXPOSURE_NS = factories.EXPOSURE_NS
STEP_NS = factories.GROUP_EXPOSURE_STEP_NS
EARLIEST_DELIVERY_NS = EXPOSURE_NS + c.DELIVERY_MIN_DELAY_NS


def _rejected_pose(**overrides):
    return factories.make_pose_evidence(accepted=False, reason="too_few_inliers", **overrides)


def _rejected_motion(**overrides):
    return factories.make_motion_evidence(accepted=False, reason="insufficient_depth", **overrides)


# ---------------------------------------------------------------------------------------
# AnchorRequest
# ---------------------------------------------------------------------------------------


def test_anchor_request_default_is_valid():
    request = factories.make_anchor_request()
    assert request.delivery_not_before_ns == EARLIEST_DELIVERY_NS
    assert request.reference_ids == (3, 8)


def test_anchor_request_delivery_boundary_is_source_plus_90_ms():
    factories.make_anchor_request(delivery_not_before_ns=EARLIEST_DELIVERY_NS)
    with pytest.raises(c.ContractError, match=r"^AnchorRequest\.delivery_not_before_ns: "):
        factories.make_anchor_request(delivery_not_before_ns=EARLIEST_DELIVERY_NS - 1)


def test_anchor_request_observation_exposure_must_match_source():
    observation = factories.make_observation(exposure_ns=EXPOSURE_NS + 1)
    with pytest.raises(c.ContractError, match=r"^AnchorRequest\.observation: exposure_ns"):
        factories.make_anchor_request(observation=observation)


@pytest.mark.parametrize("reference_ids", [(1, 2, 3), (4, 4)], ids=["three", "duplicate"])
def test_anchor_request_rejects_three_or_duplicate_reference_ids(reference_ids):
    with pytest.raises(c.ContractError, match=r"^AnchorRequest\.reference_ids"):
        factories.make_anchor_request(reference_ids=reference_ids)


def test_anchor_request_empty_reference_ids_mean_retrieval_in_job():
    assert factories.make_anchor_request(reference_ids=()).reference_ids == ()


@pytest.mark.parametrize(
    ("purpose", "camera_id"),
    [(c.AnchorPurpose.FRONT_ANCHOR, c.SECONDARY_CAMERA), (c.AnchorPurpose.REAR_ANCHOR, "fr")],
)
def test_anchor_request_camera_specific_purpose_on_wrong_camera(purpose, camera_id):
    with pytest.raises(c.ContractError, match=r"^AnchorRequest\.camera_id: purpose"):
        factories.make_anchor_request(purpose=purpose, camera_id=camera_id)


def test_anchor_request_recovery_may_use_rear_camera():
    request = factories.make_anchor_request(
        purpose=c.AnchorPurpose.RECOVERY, camera_id=c.SECONDARY_CAMERA
    )
    assert request.observation.camera_id == c.SECONDARY_CAMERA


# ---------------------------------------------------------------------------------------
# AnchorResult
# ---------------------------------------------------------------------------------------


def test_anchor_result_has_no_pose_like_field():
    pose_tokens = {"pose", "x", "y", "yaw", "position", "transform"}
    names = [field.name for field in dataclasses.fields(c.AnchorResult)]
    for name in names:
        assert not name.startswith("T_"), name
        assert not pose_tokens.intersection(name.lower().split("_")), name
        assert "position" not in name and "yaw" not in name, name


def test_anchor_result_ok_requires_reason_ok():
    with pytest.raises(c.ContractError, match=r"^AnchorResult\.reason: status 'ok'"):
        factories.make_anchor_result(reason="matched")


def test_anchor_result_ok_requires_correspondences():
    with pytest.raises(c.ContractError, match=r"^AnchorResult\.correspondences: status 'ok'"):
        factories.make_anchor_result(correspondences=None)


def test_anchor_result_failure_carries_failure_reason_and_no_correspondences():
    failed = factories.make_anchor_result(status=c.AnchorStatus.FAILED)
    assert failed.correspondences is None
    with pytest.raises(c.ContractError, match=r"^AnchorResult\.reason: status 'failed'"):
        factories.make_anchor_result(status=c.AnchorStatus.FAILED, reason="ok")
    with pytest.raises(c.ContractError, match=r"^AnchorResult\.correspondences: only"):
        factories.make_anchor_result(
            status=c.AnchorStatus.FAILED, correspondences=factories.make_correspondences()
        )


def test_anchor_result_check_against_matching_and_other_job():
    request = factories.make_anchor_request()
    factories.make_anchor_result().check_against(request)
    with pytest.raises(c.ContractError, match=r"job_id 8 != 7"):
        factories.make_anchor_result(job_id=8).check_against(request)


# ---------------------------------------------------------------------------------------
# RecoveryReset and UnusedEvidence
# ---------------------------------------------------------------------------------------


def test_recovery_reset_confirmation_boundary_is_50_ms():
    confirm_ns = EXPOSURE_NS
    factories.make_recovery_reset(
        candidate=factories.make_pose_evidence(exposure_ns=confirm_ns - c.RECOVERY_CONFIRM_MIN_NS)
    )
    late_candidate = factories.make_pose_evidence(
        exposure_ns=confirm_ns - c.RECOVERY_CONFIRM_MIN_NS + 1
    )
    with pytest.raises(c.ContractError, match=r"^RecoveryReset\.confirmation: "):
        factories.make_recovery_reset(candidate=late_candidate)


def test_recovery_reset_rejected_candidate_raises_rejected_evidence():
    with pytest.raises(c.RejectedEvidenceError, match=r"^RecoveryReset\.candidate: "):
        factories.make_recovery_reset(candidate=_rejected_pose(exposure_ns=EXPOSURE_NS - 10**8))


def test_unused_evidence_rejects_producer_rejected_record():
    with pytest.raises(c.ContractError, match=r"^UnusedEvidence\.evidence: "):
        factories.make_unused_evidence(evidence=_rejected_motion())


# ---------------------------------------------------------------------------------------
# GroupEvidence: slots and hierarchy
# ---------------------------------------------------------------------------------------


def test_group_empty_is_prediction_only():
    group = factories.make_group_evidence()
    assert group.consumed_kinds() == (c.EvidenceKind.PREDICTION_ONLY,)
    assert group.t_ns == EXPOSURE_NS + STEP_NS
    assert group.group_id == 0


@pytest.mark.parametrize(
    ("slot", "build", "kinds"),
    [
        ("pose", factories.make_pose_evidence, (c.EvidenceKind.PNP,)),
        ("pose", factories.make_handoff_evidence, (c.EvidenceKind.HANDOFF_PNP,)),
        ("motion", factories.make_motion_evidence, (c.EvidenceKind.GROUND_MOTION,)),
        ("border", factories.make_border_evidence, (c.EvidenceKind.BORDER,)),
        ("reset", factories.make_recovery_reset, (c.EvidenceKind.RECOVERY_RESET,)),
    ],
    ids=["pose", "handoff", "motion", "border", "reset"],
)
def test_group_consumed_kinds_per_slot(slot, build, kinds):
    assert factories.make_group_evidence(**{slot: build()}).consumed_kinds() == kinds


def test_group_motion_with_border_is_accepted():
    group = factories.make_group_evidence(
        motion=factories.make_motion_evidence(), border=factories.make_border_evidence()
    )
    assert group.consumed_kinds() == (c.EvidenceKind.GROUND_MOTION, c.EvidenceKind.BORDER)


def test_group_rejected_pose_in_pose_slot_raises_rejected_evidence():
    with pytest.raises(c.RejectedEvidenceError, match=r"^GroupEvidence\.pose: "):
        factories.make_group_evidence(pose=_rejected_pose())


@pytest.mark.parametrize("source", [c.PoseSource.CANDIDATE, c.PoseSource.HANDOFF])
def test_group_pose_slot_requires_tracking_source(source):
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.pose: .*'tracking'"):
        factories.make_group_evidence(pose=factories.make_pose_evidence(source=source))


@pytest.mark.parametrize(
    ("slot", "build"),
    [("motion", factories.make_motion_evidence), ("border", factories.make_border_evidence)],
)
def test_group_pose_excludes_fallback_slots(slot, build):
    with pytest.raises(c.ContractError, match=r"^GroupEvidence: PnP-first"):
        factories.make_group_evidence(pose=factories.make_pose_evidence(), **{slot: build()})


def test_group_reset_with_pose_is_rejected():
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.reset: "):
        factories.make_group_evidence(
            pose=factories.make_pose_evidence(), reset=factories.make_recovery_reset()
        )


# ---------------------------------------------------------------------------------------
# GroupEvidence: current-image rule, partition and identity
# ---------------------------------------------------------------------------------------


def test_group_rejects_evidence_from_camera_absent_from_group():
    frame_set = factories.make_frame_set(present=(c.PRIMARY_CAMERA,))
    rear_pose = factories.make_pose_evidence(
        camera_id=c.SECONDARY_CAMERA, exposure_ns=EXPOSURE_NS + STEP_NS
    )
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.pose: .*has no r image"):
        factories.make_group_evidence(frame_set=frame_set, pose=rear_pose)


def test_group_rejects_stale_evidence_of_same_camera():
    stale = factories.make_motion_evidence(current_ns=EXPOSURE_NS - 1)
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.motion: .*exposed at"):
        factories.make_group_evidence(motion=stale)


def test_group_rejects_future_evidence():
    future = factories.make_border_evidence(exposure_ns=EXPOSURE_NS + c.DELIVERY_MIN_DELAY_NS)
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.border: .*exposed at"):
        factories.make_group_evidence(border=future)


def test_group_rejects_accepted_record_in_rejected():
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.rejected\[0\]: .*accepted"):
        factories.make_group_evidence(rejected=(factories.make_pose_evidence(),))


@pytest.mark.parametrize("place", ["unused", "rejected"])
def test_group_rejects_same_record_object_twice(place):
    if place == "unused":
        pose = factories.make_pose_evidence()
        unused = (factories.make_unused_evidence(evidence=pose),)
        overrides = {"pose": pose, "unused": unused}
    else:
        motion = _rejected_motion()
        overrides = {"rejected": (motion, motion)}
    with pytest.raises(c.ContractError, match=r"already appears as"):
        factories.make_group_evidence(**overrides)


def test_group_summary_is_strict_json():
    rear_pose = _rejected_pose(camera_id=c.SECONDARY_CAMERA, exposure_ns=EXPOSURE_NS + STEP_NS)
    group = factories.make_group_evidence(
        pose=factories.make_handoff_evidence(),
        rejected=(_rejected_motion(), rear_pose),
        unused=(factories.make_unused_evidence(),),
    )
    summary = json.loads(json.dumps(group.summary(), allow_nan=False))
    assert summary["record"] == "GroupEvidence"
    assert summary["pose"]["record"] == "HandoffEvidence"
    assert summary["frame_set"]["t_ns"] == EXPOSURE_NS + STEP_NS
    assert [item["record"] for item in summary["rejected"]] == ["MotionEvidence", "PoseEvidence"]

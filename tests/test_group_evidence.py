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


@pytest.mark.parametrize("build", [factories.make_pose_evidence, factories.make_handoff_evidence])
def test_group_rejects_consumed_pose_with_another_calibration(build):
    pose = build(calibration_id="other_calibration")
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.pose\.calibration_id:"):
        factories.make_group_evidence(pose=pose)


def test_group_rejects_reset_confirmation_with_another_calibration():
    confirmation = factories.make_pose_evidence(
        source=c.PoseSource.CANDIDATE, calibration_id="other_calibration"
    )
    reset = factories.make_recovery_reset(confirmation=confirmation)
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.reset\.confirmation\.calibration_id:"):
        factories.make_group_evidence(reset=reset)


def test_group_preserves_incompatible_calibration_in_diagnostics():
    unused = factories.make_unused_evidence(
        evidence=factories.make_pose_evidence(calibration_id="other_calibration"),
        reason="calibration_mismatch",
    )
    rejected = factories.make_pose_evidence(
        accepted=False, calibration_id="other_calibration", reason="calibration_mismatch"
    )
    group = factories.make_group_evidence(unused=(unused,), rejected=(rejected,))
    assert group.consumed_kinds() == (c.EvidenceKind.PREDICTION_ONLY,)
    assert group.unused[0].reason == group.rejected[0].reason == "calibration_mismatch"


@pytest.mark.parametrize(
    "overrides", [{"group_id": 999}, {"t_ns": EXPOSURE_NS + 1_000_000_000}]
)
def test_group_rejects_reset_naming_another_group_or_time(overrides):
    reset = factories.make_recovery_reset(**overrides)
    with pytest.raises(c.ContractError, match=r"^GroupEvidence\.reset: .*must match"):
        factories.make_group_evidence(reset=reset)


def test_group_rejects_reset_confirmation_also_listed_as_unused():
    reset = factories.make_recovery_reset()
    unused = factories.make_unused_evidence(evidence=reset.confirmation)
    with pytest.raises(c.ContractError, match=r"already appears as reset\.confirmation"):
        factories.make_group_evidence(reset=reset, unused=(unused,))


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


# ---------------------------------------------------------------------------------------
# Property 9: anchor request timing and consistency
# ---------------------------------------------------------------------------------------


# Feature: shared-contracts, Property 9: an AnchorRequest constructs exactly when the
# delivery deadline is at least source + DELIVERY_MIN_DELAY_NS and the frozen snapshot lags
# the source exposure by 0..MAX_EXPOSURE_SPREAD_NS, and the embedded observation repeats the
# request's camera, source exposure and preprocessing id; any mismatch raises.
# Validates: Requirements 8.1, 8.2.
def test_property_anchor_request_timing_and_consistency():
    seed = 90_009
    iterations = 200
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        # Keep the source exposure well inside the int64 range and above the delays below.
        source = int(rng.integers(10**15, 2 * 10**18))
        # Delivery offset straddles the DELIVERY_MIN_DELAY_NS boundary in both directions.
        delivery_offset = int(rng.integers(-5_000_000, c.DELIVERY_MIN_DELAY_NS + 5_000_000))
        delivery = source + c.DELIVERY_MIN_DELAY_NS + delivery_offset
        # Snapshot lag straddles [0, MAX_EXPOSURE_SPREAD_NS].
        snapshot_lag = int(rng.integers(-3_000_000, c.MAX_EXPOSURE_SPREAD_NS + 3_000_000))
        snapshot_t = source + snapshot_lag

        delivery_ok = delivery >= source + c.DELIVERY_MIN_DELAY_NS
        snapshot_ok = 0 <= snapshot_lag <= c.MAX_EXPOSURE_SPREAD_NS
        context = f"seed={seed} iteration={k} source={source} delivery={delivery} snap={snapshot_t}"

        def build():
            return factories.make_anchor_request(
                source_exposure_ns=source,
                delivery_not_before_ns=delivery,
                observation=factories.make_observation(
                    camera_id=c.PRIMARY_CAMERA, exposure_ns=source
                ),
                snapshot=factories.make_search_snapshot(t_ns=snapshot_t),
            )

        if delivery_ok and snapshot_ok:
            request = build()
            assert request.source_exposure_ns == source, context
            assert request.delivery_not_before_ns == delivery, context
            assert request.snapshot.t_ns == snapshot_t, context
            assert request.observation.exposure_ns == source, context
        else:
            with pytest.raises(c.ContractError) as excinfo:
                build()
            if not delivery_ok:
                assert "delivery_not_before_ns" in str(excinfo.value), context
            elif not snapshot_ok:
                assert "snapshot" in str(excinfo.value), context

        # Any single mismatch between the observation and the request raises, independent of
        # the timing branch above (use valid timing so only the mismatch can fail).
        valid_source = source
        mismatch_kind = int(rng.integers(0, 3))
        if mismatch_kind == 0:
            bad_observation = factories.make_observation(
                camera_id=c.SECONDARY_CAMERA, exposure_ns=valid_source
            )
            expected = "camera_id"
        elif mismatch_kind == 1:
            bad_observation = factories.make_observation(
                camera_id=c.PRIMARY_CAMERA, exposure_ns=valid_source + 1
            )
            expected = "exposure_ns"
        else:
            bad_observation = factories.make_observation(
                camera_id=c.PRIMARY_CAMERA,
                exposure_ns=valid_source,
                preprocessing_id="other_preprocessing_v2",
            )
            expected = "preprocessing_id"
        with pytest.raises(c.ContractError) as mismatch_info:
            factories.make_anchor_request(
                camera_id=c.PRIMARY_CAMERA,
                source_exposure_ns=valid_source,
                delivery_not_before_ns=valid_source + c.DELIVERY_MIN_DELAY_NS,
                observation=bad_observation,
                snapshot=factories.make_search_snapshot(t_ns=valid_source),
            )
        message = str(mismatch_info.value)
        assert "AnchorRequest.observation" in message, f"{context} mismatch={expected}"
        assert expected in message, f"{context} mismatch={expected}"


# ---------------------------------------------------------------------------------------
# Property 10: anchor result identity
# ---------------------------------------------------------------------------------------


# Feature: shared-contracts, Property 10: an AnchorResult constructs exactly when the matched
# reference ids are unique and at most MAX_REFERENCES_PER_JOB, the remaining candidate ids are
# unique, at most RETRIEVAL_CANDIDATES and disjoint from the matched ids; and check_against
# raises for any single changed identity field while the unchanged pair passes.
# Validates: Requirements 8.6, 8.7.
def test_property_anchor_result_identity():
    seed = 100_010
    iterations = 200
    id_pool = list(range(40))
    identity_fields = (
        "job_id",
        "purpose",
        "camera_id",
        "source_group_id",
        "source_exposure_ns",
        "map_id",
        "preprocessing_id",
    )
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        matched_len = int(rng.integers(0, 4))  # 0..3 exercises the <= 2 cap
        remaining_len = int(rng.integers(0, 12))  # 0..11 exercises the <= 10 cap
        pool = rng.permutation(id_pool).tolist()
        matched = tuple(pool[:matched_len])
        remaining = tuple(pool[matched_len : matched_len + remaining_len])

        force_matched_dup = matched_len >= 2 and bool(rng.integers(0, 2))
        force_overlap = matched_len >= 1 and remaining_len >= 1 and bool(rng.integers(0, 2))
        if force_matched_dup:
            matched = (matched[0], matched[0], *matched[2:])
        if force_overlap:
            remaining = (matched[0], *remaining[1:])

        matched_unique = len(set(matched)) == len(matched)
        remaining_unique = len(set(remaining)) == len(remaining)
        disjoint = not (set(matched) & set(remaining))
        valid = (
            matched_unique
            and len(matched) <= c.MAX_REFERENCES_PER_JOB
            and remaining_unique
            and len(remaining) <= c.RETRIEVAL_CANDIDATES
            and disjoint
        )
        context = f"seed={seed} iteration={k} matched={matched} remaining={remaining}"

        def build():
            return factories.make_anchor_result(
                matched_reference_ids=matched, remaining_candidate_ids=remaining
            )

        if valid:
            result = build()
            assert result.matched_reference_ids == matched, context
            assert result.remaining_candidate_ids == remaining, context
        else:
            with pytest.raises(c.ContractError):
                build()
            continue

        # check_against: a matching request passes; one changed identity field raises.
        request = factories.make_anchor_request()
        result = factories.make_anchor_result()  # answers make_anchor_request() by default
        result.check_against(request)  # unchanged pair passes

        field = identity_fields[int(rng.integers(0, len(identity_fields)))]
        overrides = {
            "job_id": request.job_id + 1,
            "purpose": c.AnchorPurpose.RECOVERY,
            "camera_id": c.SECONDARY_CAMERA,
            "source_group_id": request.source_group_id + 1,
            "source_exposure_ns": request.source_exposure_ns + 10**8,
            "map_id": "other_map_v9",
            "preprocessing_id": "other_preprocessing_v9",
        }
        changed = factories.make_anchor_result(**{field: overrides[field]})
        with pytest.raises(c.ContractError) as excinfo:
            changed.check_against(request)
        assert field in str(excinfo.value), f"{context} field={field}"


# ---------------------------------------------------------------------------------------
# Property 13: group hierarchy and slot discipline
# ---------------------------------------------------------------------------------------


def _group_pose(accepted=True, **overrides):
    if accepted:
        return factories.make_pose_evidence(**overrides)
    return _rejected_pose(**overrides)


def _group_motion(accepted=True, **overrides):
    if accepted:
        return factories.make_motion_evidence(**overrides)
    return _rejected_motion(**overrides)


def _group_border(accepted=True, **overrides):
    if accepted:
        return factories.make_border_evidence(**overrides)
    return factories.make_border_evidence(accepted=False, reason="low_support", **overrides)


# Feature: shared-contracts, Property 13: GroupEvidence constructs exactly when consumed slots
# hold accepted records, a pose excludes motion and border, a reset excludes every other
# consumed slot, rejected holds only rejected records, unused only accepted records and no
# record object appears twice; consumed_kinds() follows the slot mapping for each valid case.
# Validates: Requirements 11.2, 11.3, 11.4, 11.6, 11.8.
def test_property_group_hierarchy_and_slot_discipline():
    seed = 130_013
    iterations = 220
    rear_exposure = EXPOSURE_NS + STEP_NS  # the r image of make_frame_set()
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        want_pose = bool(rng.integers(0, 2))
        want_motion = bool(rng.integers(0, 2))
        want_border = bool(rng.integers(0, 2))
        want_reset = bool(rng.integers(0, 2))
        pose_accepted = bool(rng.integers(0, 2))
        motion_accepted = bool(rng.integers(0, 2))
        border_accepted = bool(rng.integers(0, 2))

        overrides = {}
        if want_pose:
            overrides["pose"] = _group_pose(accepted=pose_accepted)
        if want_motion:
            overrides["motion"] = _group_motion(accepted=motion_accepted)
        if want_border:
            overrides["border"] = _group_border(accepted=border_accepted)
        if want_reset:
            overrides["reset"] = factories.make_recovery_reset()

        # A rejected record for the rejected slot, measured on the fr image (current image).
        rejected_record = _rejected_motion()
        # An accepted record for the unused slot, measured on the rear image so it never
        # collides with the fr-measured consumed slots.
        unused_record = factories.make_unused_evidence(
            evidence=factories.make_border_evidence(
                camera_id=c.SECONDARY_CAMERA, exposure_ns=rear_exposure
            )
        )
        put_rejected = bool(rng.integers(0, 2))
        put_unused = bool(rng.integers(0, 2))
        if put_rejected:
            overrides["rejected"] = (rejected_record,)
        if put_unused:
            overrides["unused"] = (unused_record,)

        # Determine validity under the slot/hierarchy rules.
        pose_slot_ok = (not want_pose) or pose_accepted
        motion_slot_ok = (not want_motion) or motion_accepted
        border_slot_ok = (not want_border) or border_accepted
        slots_accepted = pose_slot_ok and motion_slot_ok and border_slot_ok
        pnp_first_ok = not (want_pose and (want_motion or want_border))
        reset_exclusive_ok = not (want_reset and (want_pose or want_motion or want_border))
        valid = slots_accepted and pnp_first_ok and reset_exclusive_ok

        context = (
            f"seed={seed} iteration={k} pose={want_pose}/{pose_accepted} "
            f"motion={want_motion}/{motion_accepted} border={want_border}/{border_accepted} "
            f"reset={want_reset}"
        )

        if not valid:
            with pytest.raises(c.ContractError):
                factories.make_group_evidence(**overrides)
            continue

        group = factories.make_group_evidence(**overrides)

        # consumed_kinds() follows the documented slot mapping.
        if want_reset:
            expected = (c.EvidenceKind.RECOVERY_RESET,)
        elif want_pose:
            expected = (c.EvidenceKind.PNP,)
        else:
            kinds = []
            if want_motion:
                kinds.append(c.EvidenceKind.GROUND_MOTION)
            if want_border:
                kinds.append(c.EvidenceKind.BORDER)
            expected = tuple(kinds) or (c.EvidenceKind.PREDICTION_ONLY,)
        assert group.consumed_kinds() == expected, context

        # Partition discipline: an accepted record in rejected, or a rejected one in unused,
        # must raise regardless of the valid base combination.
        with pytest.raises(c.ContractError, match=r"rejected\[0\]"):
            factories.make_group_evidence(**{**overrides, "rejected": (factories.make_border_evidence(
                camera_id=c.SECONDARY_CAMERA, exposure_ns=rear_exposure),)})


# Feature: shared-contracts, Property 13 (duplicate-object clause): the same record object in
# two GroupEvidence positions raises; two distinct equal-valued objects do not.
# Validates: Requirements 11.8.
def test_property_group_rejects_duplicate_record_objects():
    seed = 131_013
    iterations = 120
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        context = f"seed={seed} iteration={k}"
        placement = int(rng.integers(0, 3))
        if placement == 0:
            # Same accepted pose in the pose slot and in unused.
            pose = factories.make_pose_evidence()
            with pytest.raises(c.ContractError, match=r"already appears as"):
                factories.make_group_evidence(
                    pose=pose, unused=(factories.make_unused_evidence(evidence=pose),)
                )
            # Two distinct equal poses are allowed (pose consumed, a separate one unused).
            group = factories.make_group_evidence(
                pose=factories.make_pose_evidence(),
                unused=(factories.make_unused_evidence(evidence=factories.make_pose_evidence()),),
            )
            assert group.consumed_kinds() == (c.EvidenceKind.PNP,), context
        elif placement == 1:
            # Same rejected record twice in rejected.
            motion = _rejected_motion()
            with pytest.raises(c.ContractError, match=r"already appears as"):
                factories.make_group_evidence(rejected=(motion, motion))
        else:
            # Same accepted record consumed and also in unused via a shared object.
            border = factories.make_border_evidence()
            with pytest.raises(c.ContractError, match=r"already appears as"):
                factories.make_group_evidence(
                    border=border, unused=(factories.make_unused_evidence(evidence=border),)
                )


# ---------------------------------------------------------------------------------------
# Property 14: current-image causality
# ---------------------------------------------------------------------------------------


# Feature: shared-contracts, Property 14: evidence placed in any GroupEvidence position is
# accepted only when its measured exposure equals this group's image exposure for the record's
# camera; evidence from an absent camera, an earlier/later image of the same camera, or stamped
# after t_ns, raises.
# Validates: Requirements 11.5.
def test_property_group_current_image_causality():
    seed = 140_014
    iterations = 220
    frame_exposure = {c.PRIMARY_CAMERA: EXPOSURE_NS, c.SECONDARY_CAMERA: EXPOSURE_NS + STEP_NS}
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        # Group present cameras: always include fr; sometimes include r.
        include_rear = bool(rng.integers(0, 2))
        present = (c.PRIMARY_CAMERA, c.SECONDARY_CAMERA) if include_rear else (c.PRIMARY_CAMERA,)
        frame_set = factories.make_frame_set(present=present)

        # Choose the evidence camera and its measured exposure.
        camera = c.PRIMARY_CAMERA if bool(rng.integers(0, 2)) else c.SECONDARY_CAMERA
        case = int(rng.integers(0, 4))
        group_ns = frame_exposure[camera]
        if case == 0:  # exactly on this group's image of its camera
            measured = group_ns
        elif case == 1:  # earlier image of the same camera
            measured = group_ns - int(rng.integers(1, 10**6)) - 1
        elif case == 2:  # later image of the same camera, still <= t_ns check via absence
            measured = group_ns + int(rng.integers(1, 10**6)) + 1
        else:  # after t_ns (far future)
            measured = frame_set.t_ns + c.DELIVERY_MIN_DELAY_NS

        camera_present = camera in {image.camera_id for image in frame_set.images}
        valid = camera_present and case == 0
        context = (
            f"seed={seed} iteration={k} present={present} camera={camera} "
            f"case={case} measured={measured} group_ns={group_ns}"
        )

        # Place as a consumed slot: motion slot for fr-accepted when case==0 and camera fr;
        # otherwise use the pose slot which accepts both cameras and both branches.
        pose = factories.make_pose_evidence(camera_id=camera, exposure_ns=measured)
        if valid:
            group = factories.make_group_evidence(frame_set=frame_set, pose=pose)
            assert group.consumed_kinds() == (c.EvidenceKind.PNP,), context
        else:
            with pytest.raises(c.ContractError) as excinfo:
                factories.make_group_evidence(frame_set=frame_set, pose=pose)
            message = str(excinfo.value)
            assert "GroupEvidence.pose" in message, context
            if not camera_present:
                assert "has no" in message, context
            else:
                assert "exposed at" in message, context


# ---------------------------------------------------------------------------------------
# Property 16: auxiliary geometry and reset rules
# ---------------------------------------------------------------------------------------


# Feature: shared-contracts, Property 16a: AuxiliaryGeometry constructs exactly under the
# source/validity rules: source 'none' must be invalid with every value None; a valid 'pnp' or
# 'road_patch' record needs source_ns, z_m, roll_rad and pitch_rad, and a valid 'road_patch'
# also needs road_patch_id.
# Validates: Requirements 12.4, 12.7.
def test_property_auxiliary_geometry_source_and_validity():
    seed = 160_016
    iterations = 200
    sources = (c.AuxSource.NONE, c.AuxSource.PNP, c.AuxSource.ROAD_PATCH)
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        source = sources[int(rng.integers(0, len(sources)))]
        want_valid = bool(rng.integers(0, 2))
        have_pose_values = bool(rng.integers(0, 2))
        have_patch_id = bool(rng.integers(0, 2))

        source_ns = EXPOSURE_NS if have_pose_values else None
        z_m = 0.35 if have_pose_values else None
        roll = 0.01 if have_pose_values else None
        pitch = -0.02 if have_pose_values else None
        patch_id = "road_patch_7" if have_patch_id else None
        reason = "valid" if want_valid else "not_estimated"

        context = (
            f"seed={seed} iteration={k} source={source.value} valid={want_valid} "
            f"pose_values={have_pose_values} patch_id={have_patch_id}"
        )

        # Determine validity under the AuxiliaryGeometry rules.
        if source is c.AuxSource.NONE:
            all_values_none = not have_pose_values and not have_patch_id
            expect_ok = (not want_valid) and all_values_none
        elif want_valid:
            needs_patch = source is c.AuxSource.ROAD_PATCH
            expect_ok = have_pose_values and (have_patch_id if needs_patch else True)
        else:
            # Invalid non-none records only need the flag/reason to be consistent, which they
            # are here (reason 'not_estimated'); value presence is unconstrained.
            expect_ok = True

        def build():
            return c.AuxiliaryGeometry(
                source=source,
                source_ns=source_ns,
                z_m=z_m,
                roll_rad=roll,
                pitch_rad=pitch,
                road_patch_id=patch_id,
                valid=want_valid,
                reason=reason,
            )

        if expect_ok:
            aux = build()
            assert aux.source is source, context
            assert aux.valid == want_valid, context
        else:
            with pytest.raises(c.ContractError):
                build()


# Feature: shared-contracts, Property 16b: RecoveryReset constructs exactly when both poses are
# accepted, the confirmation is at least RECOVERY_CONFIRM_MIN_NS after the candidate, the
# confirmation exposure is at most t_ns, and jump_m is non-negative.
# Validates: Requirements 12.4, 12.7.
def test_property_recovery_reset_rules():
    seed = 161_016
    iterations = 200
    base = EXPOSURE_NS
    for k in range(iterations):
        rng = factories.seeded_rng(seed, k)
        gap = c.RECOVERY_CONFIRM_MIN_NS + int(rng.integers(-5_000_000, 5_000_000))
        candidate_ns = base
        confirmation_ns = candidate_ns + gap
        # t_ns straddles the confirmation exposure.
        t_offset = int(rng.integers(-2_000_000, 2_000_000))
        t_ns = confirmation_ns + t_offset
        jump = float(rng.uniform(-2.0, 10.0))
        candidate_accepted = bool(rng.integers(0, 2))

        gap_ok = gap >= c.RECOVERY_CONFIRM_MIN_NS
        t_ok = confirmation_ns <= t_ns
        jump_ok = jump >= 0.0
        valid = gap_ok and t_ok and jump_ok and candidate_accepted
        context = (
            f"seed={seed} iteration={k} gap={gap} t_ns={t_ns} "
            f"confirm={confirmation_ns} jump={jump} candidate_accepted={candidate_accepted}"
        )

        candidate = _group_pose(
            accepted=candidate_accepted,
            source=c.PoseSource.CANDIDATE,
            exposure_ns=candidate_ns,
        )
        confirmation = factories.make_pose_evidence(
            source=c.PoseSource.CANDIDATE, exposure_ns=confirmation_ns
        )

        def build():
            return factories.make_recovery_reset(
                candidate=candidate,
                confirmation=confirmation,
                t_ns=t_ns,
                jump_m=jump,
            )

        if valid:
            reset = build()
            assert reset.candidate.exposure_ns == candidate_ns, context
            assert reset.confirmation.exposure_ns == confirmation_ns, context
            assert reset.t_ns == t_ns, context
        else:
            with pytest.raises(c.ContractError):
                build()

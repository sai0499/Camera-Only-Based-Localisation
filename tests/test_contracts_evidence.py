"""Example tests of the evidence records in ``camloc.contracts``.

Each test builds a valid evidence record with a small local builder and changes one thing,
so the rejection names the rule that caught it. Builders live in this file only: this test
intentionally does not import ``tests/contract_factories.py``. Rejection messages all begin
``"<Record>.<field>: "``, so ``pytest.raises(c.ContractError, match=...)`` anchors on that
prefix. These tests write nothing.

Validates: Requirements 6.6, 7.4, 7.5, 9.5, 10.2, 10.4, 10.5.
"""

import dataclasses

import numpy as np
import pytest

from camloc import contracts as c

# A same-camera chain: source exposure, then the current measurement STEP_NS later.
CAMERA = "fr"
SOURCE_NS = 1_700_000_000_000_000_000
STEP_NS = 20_000_000  # 20 ms between the source and the current image
MEASUREMENT_NS = SOURCE_NS + STEP_NS


# ---------------------------------------------------------------------------------------
# Local builders. Each returns a record that is accepted (or valid) by default; callers
# override one field to probe a single rule.
# ---------------------------------------------------------------------------------------


def _identity_rigid() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _yaw_rigid(yaw: float) -> np.ndarray:
    """A planar rigid transform: rotation about z by ``yaw``, no translation."""
    transform = np.eye(4, dtype=np.float64)
    cos, sin = np.cos(yaw), np.sin(yaw)
    transform[0, 0], transform[0, 1] = cos, -sin
    transform[1, 0], transform[1, 1] = sin, cos
    return transform


def _ids(count: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.int64)


def _pixels(count: int) -> np.ndarray:
    return np.zeros((count, 2), dtype=np.float32)


def make_correspondences(*, count: int = 4, camera_id: str = CAMERA,
                         exposure_ns: int = MEASUREMENT_NS, map_id: str = "map", **overrides):
    fields = {
        "camera_id": camera_id,
        "exposure_ns": exposure_ns,
        "keypoint_ids": _ids(count),
        "pixels": _pixels(count),
        "landmark_ids": _ids(count, start=1000),
        "map_id": map_id,
    }
    fields.update(overrides)
    return c.Correspondences(**fields)


def make_pose_evidence(*, source=c.PoseSource.TRACKING, count: int = 3, accepted: bool = True,
                       reason: str | None = None, camera_id: str = CAMERA,
                       exposure_ns: int = MEASUREMENT_NS, landmark_start: int = 1000,
                       **overrides):
    accepted_defaults = {
        "T_WB": _identity_rigid(),
        "median_error_px": 1.0,
        "p90_error_px": 2.0,
        "occupied_cells": 4,
    }
    fields = {
        "camera_id": camera_id,
        "exposure_ns": exposure_ns,
        "source": source,
        "T_WB": accepted_defaults["T_WB"],
        "inlier_count": count,
        "inlier_fraction": 0.9,
        "median_error_px": accepted_defaults["median_error_px"],
        "p90_error_px": accepted_defaults["p90_error_px"],
        "occupied_cells": accepted_defaults["occupied_cells"],
        "track_ids": _ids(count),
        "landmark_ids": _ids(count, start=landmark_start),
        "map_id": "map",
        "calibration_id": "calib",
        "preprocessing_id": "prep",
        "accepted": accepted,
        "reason": reason if reason is not None else ("accepted" if accepted else "rejected_pose"),
    }
    fields.update(overrides)
    return c.PoseEvidence(**fields)


def make_handoff_pose(*, count: int = 3, landmark_start: int = 1000, **overrides):
    return make_pose_evidence(source=c.PoseSource.HANDOFF, count=count,
                              landmark_start=landmark_start, **overrides)


def make_handoff_evidence(*, accepted: bool = True, reason: str | None = None,
                          count: int = 3, **overrides):
    correspondences = make_correspondences(count=count, landmark_ids=_ids(count, start=1000))
    pose = make_handoff_pose(count=count, landmark_start=1000)
    fields = {
        "job_id": 7,
        "camera_id": CAMERA,
        "source_exposure_ns": SOURCE_NS,
        "measurement_ns": MEASUREMENT_NS,
        "support_ns": (SOURCE_NS, SOURCE_NS + STEP_NS // 2, MEASUREMENT_NS),
        "correspondences": correspondences,
        "pose": pose,
        "map_id": "map",
        "preprocessing_id": "prep",
        "calibration_id": "calib",
        "accepted": accepted,
        "reason": reason if reason is not None else ("accepted" if accepted else "stale_chain"),
    }
    fields.update(overrides)
    return c.HandoffEvidence(**fields)


def make_motion_evidence(*, accepted: bool = True, reason: str | None = None, **overrides):
    fields = {
        "camera_id": CAMERA,
        "previous_ns": SOURCE_NS,
        "current_ns": MEASUREMENT_NS,
        "T_Bprev_Bcurr": _yaw_rigid(0.01),
        "covariance": np.diag([0.1, 0.1, 0.01]).astype(np.float64),
        "inlier_count": 12,
        "inlier_fraction": 0.8,
        "ground_residual_m": 0.02,
        "road_patch_id": "patch-1",
        "accepted": accepted,
        "reason": reason if reason is not None else ("accepted" if accepted else "few_inliers"),
    }
    fields.update(overrides)
    return c.MotionEvidence(**fields)


def make_border_evidence(*, accepted: bool = True, reason: str | None = None, **overrides):
    fields = {
        "camera_id": CAMERA,
        "exposure_ns": MEASUREMENT_NS,
        "branch": c.Branch.TRACK,
        "normal_xy": np.array([0.0, 1.0], dtype=np.float64),
        "lateral_coordinate_m": 0.3,
        "yaw_rad": 0.05,
        "R": np.diag([0.05, 0.01]).astype(np.float64),
        "support_samples": 20,
        "support_contours": 3,
        "conditioning": 0.5,
        "accepted": accepted,
        "reason": reason if reason is not None else ("accepted" if accepted else "ill_conditioned"),
    }
    fields.update(overrides)
    return c.BorderEvidence(**fields)


# ---------------------------------------------------------------------------------------
# PoseEvidence  (Requirements 7.4, 7.5)
# ---------------------------------------------------------------------------------------


def test_pose_default_builders_construct():
    assert make_pose_evidence().accepted is True
    assert make_pose_evidence(source=c.PoseSource.HANDOFF).source is c.PoseSource.HANDOFF
    assert make_pose_evidence(source=c.PoseSource.CANDIDATE).source is c.PoseSource.CANDIDATE


def test_pose_accepted_without_transform_rejected():
    with pytest.raises(c.ContractError, match=r"PoseEvidence: accepted=True"):
        make_pose_evidence(T_WB=None)


def test_pose_accepted_median_above_p90_rejected():
    with pytest.raises(c.ContractError, match=r"PoseEvidence\.median_error_px:"):
        make_pose_evidence(median_error_px=5.0, p90_error_px=2.0)


def test_pose_accepted_zero_occupied_cells_rejected():
    with pytest.raises(c.ContractError, match=r"PoseEvidence: accepted=True"):
        make_pose_evidence(occupied_cells=0)


def test_pose_one_inlier_accepted_constructs():
    # Contracts set no tuning gate: a single inlier is structurally valid.
    pose = make_pose_evidence(count=1)
    assert pose.inlier_count == 1
    assert pose.accepted is True


def test_pose_tracking_inlier_count_boundary_accepted_and_rejected():
    limit = c.MAX_ACTIVE_TRACKS
    assert limit == 512
    accepted = make_pose_evidence(source=c.PoseSource.TRACKING, count=limit)
    assert accepted.inlier_count == limit
    with pytest.raises(c.ContractError, match=r"PoseEvidence\.inlier_count:"):
        make_pose_evidence(source=c.PoseSource.TRACKING, count=limit + 1)


@pytest.mark.parametrize("source", [c.PoseSource.HANDOFF, c.PoseSource.CANDIDATE])
def test_pose_handoff_candidate_inlier_count_boundary(source):
    limit = c.MAX_HANDOFF_POINTS
    assert limit == 256
    accepted = make_pose_evidence(source=source, count=limit)
    assert accepted.inlier_count == limit
    with pytest.raises(c.ContractError, match=r"PoseEvidence\.inlier_count:"):
        make_pose_evidence(source=source, count=limit + 1)


def test_pose_inlier_count_must_match_id_lengths():
    with pytest.raises(c.ContractError, match=r"PoseEvidence\.inlier_count:"):
        make_pose_evidence(inlier_count=3, track_ids=_ids(2), landmark_ids=_ids(3, start=1000))
    with pytest.raises(c.ContractError, match=r"PoseEvidence\.inlier_count:"):
        make_pose_evidence(inlier_count=3, track_ids=_ids(3), landmark_ids=_ids(2, start=1000))


# ---------------------------------------------------------------------------------------
# Correspondences  (Requirements 7.4, 7.5)
# ---------------------------------------------------------------------------------------


def test_correspondences_count_cap_boundary():
    cap = c.MAX_KEYPOINTS
    assert cap == 1024
    accepted = make_correspondences(count=cap)
    assert accepted.keypoint_ids.size == cap
    with pytest.raises(c.ContractError, match=r"keypoint_ids:"):
        make_correspondences(count=cap + 1)


def test_correspondences_empty_is_accepted():
    empty = make_correspondences(count=0)
    assert empty.keypoint_ids.size == 0
    assert empty.pixels.shape == (0, 2)


def test_correspondences_duplicate_keypoint_id_rejected():
    with pytest.raises(c.ContractError, match=r"keypoint_ids:"):
        make_correspondences(keypoint_ids=np.array([0, 0, 2, 3], dtype=np.int64))


def test_correspondences_duplicate_landmark_id_rejected():
    with pytest.raises(c.ContractError, match=r"landmark_ids:"):
        make_correspondences(landmark_ids=np.array([5, 5, 7, 8], dtype=np.int64))


def test_correspondences_pixels_outside_bounds_rejected():
    bad = _pixels(4)
    bad[1, 0] = 1000.0  # beyond the canonical u bound of 959.5
    with pytest.raises(c.ContractError, match=r"pixels:"):
        make_correspondences(pixels=bad)


def test_correspondences_mismatched_lengths_rejected():
    with pytest.raises(c.ContractError, match=r"Correspondences:"):
        make_correspondences(
            keypoint_ids=_ids(4), pixels=_pixels(4), landmark_ids=_ids(3, start=1000)
        )


# ---------------------------------------------------------------------------------------
# HandoffEvidence  (Requirement 9.5)
# ---------------------------------------------------------------------------------------


def test_handoff_default_builder_is_accepted():
    handoff = make_handoff_evidence()
    assert handoff.accepted is True
    assert handoff.support_ns[0] == SOURCE_NS
    assert handoff.support_ns[-1] == MEASUREMENT_NS
    assert 1 <= handoff.transitions <= c.MAX_HANDOFF_TRANSITIONS


def test_handoff_accepted_requires_chain_endpoints():
    # Chain that does not start at the source exposure.
    with pytest.raises(c.ContractError, match=r"HandoffEvidence\.support_ns:"):
        make_handoff_evidence(
            support_ns=(SOURCE_NS + 1, SOURCE_NS + STEP_NS // 2, MEASUREMENT_NS)
        )
    # Chain that does not end at the measurement.
    with pytest.raises(c.ContractError, match=r"HandoffEvidence\.support_ns:"):
        make_handoff_evidence(
            support_ns=(SOURCE_NS, SOURCE_NS + STEP_NS // 2, MEASUREMENT_NS - 1)
        )


def test_handoff_one_transition_accepted():
    # A direct source-to-measurement chain is one transition, within the 1..4 range.
    handoff = make_handoff_evidence(support_ns=(SOURCE_NS, MEASUREMENT_NS))
    assert handoff.transitions == 1
    assert handoff.accepted is True


def test_handoff_four_transitions_accepted_five_stamps_cap():
    # Four transitions (five stamps) is the maximum the support chain may carry.
    assert c.MAX_HANDOFF_TRANSITIONS == 4
    stamps = (
        SOURCE_NS,
        SOURCE_NS + STEP_NS // 4,
        SOURCE_NS + STEP_NS // 2,
        SOURCE_NS + 3 * STEP_NS // 4,
        MEASUREMENT_NS,
    )
    handoff = make_handoff_evidence(support_ns=stamps)
    assert handoff.transitions == 4
    # A sixth stamp exceeds MAX_HANDOFF_TRANSITIONS + 1, caught by the length cap.
    over = stamps[:-1] + (SOURCE_NS + 4 * STEP_NS // 5, MEASUREMENT_NS)
    with pytest.raises(c.ContractError, match=r"HandoffEvidence\.support_ns"):
        make_handoff_evidence(support_ns=over)


def test_handoff_age_boundary_250ms_accepted_plus_one_rejected():
    max_age = c.HISTORY_MAX_AGE_NS
    assert max_age == 250_000_000
    measurement = SOURCE_NS + max_age
    accepted = make_handoff_evidence(
        measurement_ns=measurement,
        support_ns=(SOURCE_NS, SOURCE_NS + max_age // 2, measurement),
        correspondences=make_correspondences(count=3, exposure_ns=measurement,
                                              landmark_ids=_ids(3, start=1000)),
        pose=make_handoff_pose(count=3, exposure_ns=measurement, landmark_start=1000),
    )
    assert accepted.accepted is True

    over = SOURCE_NS + max_age + 1
    with pytest.raises(c.ContractError, match=r"HandoffEvidence:"):
        make_handoff_evidence(
            measurement_ns=over,
            support_ns=(SOURCE_NS, SOURCE_NS + max_age // 2, over),
            correspondences=make_correspondences(count=3, exposure_ns=over,
                                                 landmark_ids=_ids(3, start=1000)),
            pose=make_handoff_pose(count=3, exposure_ns=over, landmark_start=1000),
        )


def test_handoff_accepted_requires_correspondences_and_accepted_pose():
    with pytest.raises(c.ContractError, match=r"HandoffEvidence: accepted=True"):
        make_handoff_evidence(correspondences=None, pose=None)


def test_handoff_pose_landmarks_not_in_correspondences_rejected():
    correspondences = make_correspondences(count=3, landmark_ids=_ids(3, start=1000))
    pose = make_handoff_pose(count=3, landmark_start=5000)  # disjoint landmark ids
    with pytest.raises(c.ContractError, match=r"HandoffEvidence\.pose:"):
        make_handoff_evidence(correspondences=correspondences, pose=pose)


# ---------------------------------------------------------------------------------------
# MotionEvidence  (Requirement 10.2)
# ---------------------------------------------------------------------------------------


def test_motion_default_builder_is_accepted():
    assert make_motion_evidence().accepted is True


@pytest.mark.parametrize(
    "field", ["T_Bprev_Bcurr", "covariance", "ground_residual_m", "road_patch_id"]
)
def test_motion_accepted_missing_field_rejected(field):
    with pytest.raises(c.ContractError, match=r"MotionEvidence: accepted=True"):
        make_motion_evidence(**{field: None})


def test_motion_non_planar_rotation_off_z_rejected():
    # A rotation about x tilts row/column 2 away from [0, 0, 1].
    tilted = np.eye(4, dtype=np.float64)
    angle = 0.2
    cos, sin = np.cos(angle), np.sin(angle)
    tilted[1, 1], tilted[1, 2] = cos, -sin
    tilted[2, 1], tilted[2, 2] = sin, cos
    with pytest.raises(c.ContractError, match=r"MotionEvidence\.T_Bprev_Bcurr:"):
        make_motion_evidence(T_Bprev_Bcurr=tilted)


def test_motion_non_planar_z_translation_rejected():
    with_z = _yaw_rigid(0.0)
    with_z[2, 3] = 0.5
    with pytest.raises(c.ContractError, match=r"MotionEvidence\.T_Bprev_Bcurr:"):
        make_motion_evidence(T_Bprev_Bcurr=with_z)


def test_motion_singular_covariance_rejected():
    singular = np.diag([0.1, 0.1, 0.0]).astype(np.float64)
    with pytest.raises(c.ContractError, match=r"MotionEvidence\.covariance:"):
        make_motion_evidence(covariance=singular)


# ---------------------------------------------------------------------------------------
# BorderEvidence  (Requirements 10.4, 10.5)
# ---------------------------------------------------------------------------------------


def test_border_default_builder_is_accepted():
    assert make_border_evidence().accepted is True


def test_border_has_no_along_track_field():
    field_names = {field.name for field in dataclasses.fields(c.BorderEvidence)}
    for forbidden in ("along_track", "along_track_m", "s", "s_m", "longitudinal"):
        assert forbidden not in field_names
    # The record is defined only by lateral coordinate and yaw.
    assert "lateral_coordinate_m" in field_names
    assert "yaw_rad" in field_names


@pytest.mark.parametrize("field", ["lateral_coordinate_m", "yaw_rad", "R"])
def test_border_accepted_missing_field_rejected(field):
    with pytest.raises(c.ContractError, match=r"BorderEvidence: accepted=True"):
        make_border_evidence(**{field: None})


def test_border_non_unit_normal_rejected():
    with pytest.raises(c.ContractError, match=r"BorderEvidence\.normal_xy:"):
        make_border_evidence(normal_xy=np.array([0.0, 2.0], dtype=np.float64))


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_border_conditioning_outside_unit_interval_rejected(value):
    with pytest.raises(c.ContractError, match=r"BorderEvidence\.conditioning:"):
        make_border_evidence(conditioning=value)


def test_border_accepted_unknown_branch_rejected():
    with pytest.raises(c.ContractError, match=r"BorderEvidence\.branch:"):
        make_border_evidence(branch=c.Branch.UNKNOWN)


# ---------------------------------------------------------------------------------------
# require_accepted on each evidence type  (Requirement 6.6)
# ---------------------------------------------------------------------------------------


def test_require_accepted_returns_accepted_record_unchanged():
    for builder in (make_pose_evidence, make_motion_evidence, make_border_evidence,
                    make_handoff_evidence):
        record = builder()
        assert c.require_accepted(record) is record


def test_require_accepted_raises_on_rejected_record():
    rejected = [
        make_pose_evidence(accepted=False, reason="too_few_inliers"),
        make_motion_evidence(accepted=False, reason="insufficient_depth"),
        make_border_evidence(accepted=False, reason="ill_conditioned"),
        make_handoff_evidence(accepted=False, reason="stale_chain"),
    ]
    for record in rejected:
        with pytest.raises(c.RejectedEvidenceError, match=type(record).__name__):
            c.require_accepted(record)


# =======================================================================================
# Property tests. Each is one seeded ``numpy.random.default_rng([seed, k])`` loop of at
# least 100 iterations; a failure names the seed, the iteration and the offending input.
# These loops reuse the example builders above and do not import ``contract_factories``.
# =======================================================================================

ITERATIONS = 128


# ---------------------------------------------------------------------------------------
# Local builders for the geometry records that carry a ``valid`` flag (Property 7 only).
# The evidence records reuse the builders defined at the top of this file.
# ---------------------------------------------------------------------------------------


def make_road_patch(*, valid: bool = True, reason: str | None = None, **overrides):
    fields = {
        "patch_id": "patch-1",
        "branch": c.Branch.TRACK,
        "query_xy": np.array([1.0, 2.0], dtype=np.float64),
        "point_W": np.array([1.0, 2.0, 0.0], dtype=np.float64),
        "normal_W": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "support": 10,
        "residual_m": 0.01,
        "valid": valid,
        "reason": reason if reason is not None else ("valid" if valid else "no_plane"),
    }
    fields.update(overrides)
    return c.RoadPatch(**fields)


def make_track_projection(*, valid: bool = True, reason: str | None = None, **overrides):
    fields = {
        "t_ns": MEASUREMENT_NS,
        "reference_map_id": "ref-map",
        "route_id": "track",
        "s_m": 12.0,
        "n_m": 0.3,
        "epsi_rad": 0.02,
        "segment_index": 4,
        "xy_reconstruction_residual_m": 0.01,
        "valid": valid,
        "reason": reason if reason is not None else ("valid" if valid else "off_route"),
    }
    fields.update(overrides)
    return c.TrackProjection(**fields)


def make_auxiliary_geometry(*, valid: bool = True, reason: str | None = None, **overrides):
    fields = {
        "source": c.AuxSource.PNP,
        "source_ns": MEASUREMENT_NS,
        "z_m": 0.4,
        "roll_rad": 0.01,
        "pitch_rad": 0.02,
        "road_patch_id": None,
        "valid": valid,
        "reason": reason if reason is not None else ("valid" if valid else "no_pnp"),
    }
    fields.update(overrides)
    return c.AuxiliaryGeometry(**fields)


# Evidence builders keyed by name, and the geometry builders with their positive word.
_EVIDENCE_BUILDERS = {
    "PoseEvidence": make_pose_evidence,
    "HandoffEvidence": make_handoff_evidence,
    "MotionEvidence": make_motion_evidence,
    "BorderEvidence": make_border_evidence,
}
_GEOMETRY_BUILDERS = {
    "RoadPatch": (make_road_patch, "valid"),
    "TrackProjection": (make_track_projection, "valid"),
    "AuxiliaryGeometry": (make_auxiliary_geometry, "valid"),
}

# Snake_case codes that are genuine rejection reasons (not the positive words).
_REJECTION_REASONS = (
    "too_few_inliers",
    "stale_chain",
    "ill_conditioned",
    "no_plane",
    "backend_fault",
    "off_route",
    "insufficient_depth",
    "r0",
    "a",
)
# Strings that are not snake_case reason codes at all.
_INVALID_REASON_FORMS = (
    "Accepted",
    "0leading",
    "has space",
    "trailing\n",
    "",
    "UPPER",
    "camelCase",
    "-dash",
)


def _random_reason(rng, *, positive: bool):
    """Draw a reason string. ``positive`` biases toward the accepted/valid words."""
    if positive:
        return str(rng.choice(["accepted", "valid"]))
    pool = rng.integers(0, 3)
    if pool == 0:
        return str(rng.choice(["accepted", "valid"]))  # positive word, wrong for a False flag
    if pool == 1:
        return str(rng.choice(_REJECTION_REASONS))  # a legitimate rejection code
    return str(rng.choice(_INVALID_REASON_FORMS))  # not a snake_case code at all


# ---------------------------------------------------------------------------------------
# Feature: shared-contracts, Property 7: acceptance flag and reason consistency.
# Validates: Requirements 6.2, 6.3, 6.4, 6.5.
# ---------------------------------------------------------------------------------------


def test_property_acceptance_flag_and_reason_consistency():
    """Construction of every evidence/geometry record depends only on the (flag, reason)
    pair, and ``require_accepted`` honours the flag: it returns the record when the flag is
    True and raises ``RejectedEvidenceError`` carrying the reason when False."""
    seed = 70
    evidence_items = [(name, builder, "accepted") for name, builder in _EVIDENCE_BUILDERS.items()]
    geometry_items = [
        (name, builder, word) for name, (builder, word) in _GEOMETRY_BUILDERS.items()
    ]
    items = evidence_items + geometry_items

    for k in range(ITERATIONS):
        rng = np.random.default_rng([seed, k])
        name, builder, ok_word = items[int(rng.integers(0, len(items)))]
        flag = bool(rng.integers(0, 2))
        reason = _random_reason(rng, positive=flag)
        flag_field = "accepted" if name in _EVIDENCE_BUILDERS else "valid"
        where = f"seed={seed} iteration={k} record={name} flag={flag} reason={reason!r}"

        # The construction rule: accepted/valid True iff reason is exactly the positive word;
        # False iff reason is another snake_case code that is neither "accepted" nor "valid".
        reason_ok = bool(c._REASON_PATTERN.fullmatch(reason))
        expect_ok = (
            reason_ok
            and flag
            and reason == ok_word
        ) or (
            reason_ok
            and not flag
            and reason not in ("accepted", "valid")
        )

        try:
            record = builder(**{flag_field: flag, "reason": reason})
        except c.ContractError:
            assert not expect_ok, f"unexpected rejection: {where}"
            continue
        assert expect_ok, f"unexpectedly constructed a record: {where}"

        # Acceptance depends only on the flag: the True record is returned unchanged, the
        # False record raises carrying its own reason.
        if flag:
            assert c.require_accepted(record) is record, where
        else:
            with pytest.raises(c.RejectedEvidenceError, match=name) as excinfo:
                c.require_accepted(record)
            assert reason in str(excinfo.value), where


# ---------------------------------------------------------------------------------------
# Feature: shared-contracts, Property 8: unique correspondence identities.
# Validates: Requirements 7.1, 7.2, 7.3.
# ---------------------------------------------------------------------------------------


def test_property_unique_correspondence_identities():
    """``Correspondences`` and ``PoseEvidence`` construct only when their id arrays are each
    unique, non-negative and of equal length (with ``inlier_count`` equal to that length for
    the pose); injecting a single duplicate into either id array always raises."""
    seed = 80
    for k in range(ITERATIONS):
        rng = np.random.default_rng([seed, k])
        count = int(rng.integers(1, 1025))  # 1..1024, within MAX_KEYPOINTS

        # Disjoint, unique, non-negative keypoint and landmark ids.
        keypoint_ids = rng.permutation(np.arange(count, dtype=np.int64))
        landmark_ids = rng.permutation(np.arange(count, dtype=np.int64)) + 1_000_000
        pixels = _pixels(count)
        where = f"seed={seed} iteration={k} count={count}"

        # The valid record and pose construct. PoseEvidence is capped at MAX_HANDOFF_POINTS
        # for the handoff/candidate sources, so stay on tracking (cap MAX_ACTIVE_TRACKS) and
        # only when the count fits, otherwise just exercise Correspondences.
        correspondences = make_correspondences(
            count=count, keypoint_ids=keypoint_ids, pixels=pixels, landmark_ids=landmark_ids
        )
        assert correspondences.keypoint_ids.size == count, where

        pose_count = count if count <= c.MAX_ACTIVE_TRACKS else None
        if pose_count is not None:
            track_ids = rng.permutation(np.arange(pose_count, dtype=np.int64))
            pose_landmarks = rng.permutation(np.arange(pose_count, dtype=np.int64)) + 2_000_000
            pose = make_pose_evidence(
                source=c.PoseSource.TRACKING, count=pose_count,
                inlier_count=pose_count, track_ids=track_ids, landmark_ids=pose_landmarks,
            )
            assert pose.inlier_count == pose_count == pose.track_ids.size, where

        # Injecting one duplicate into either id array always raises. Overwrite a random row
        # with the value of another row to force a repeat without changing the length.
        if count >= 2:
            i, j = (int(x) for x in rng.choice(count, size=2, replace=False))

            dup_keypoints = keypoint_ids.copy()
            dup_keypoints[i] = dup_keypoints[j]
            with pytest.raises(c.ContractError):
                make_correspondences(
                    count=count, keypoint_ids=dup_keypoints, pixels=pixels,
                    landmark_ids=landmark_ids,
                )

            dup_landmarks = landmark_ids.copy()
            dup_landmarks[i] = dup_landmarks[j]
            with pytest.raises(c.ContractError):
                make_correspondences(
                    count=count, keypoint_ids=keypoint_ids, pixels=pixels,
                    landmark_ids=dup_landmarks,
                )

            if pose_count is not None and pose_count >= 2:
                track_ids = rng.permutation(np.arange(pose_count, dtype=np.int64))
                pose_landmarks = rng.permutation(np.arange(pose_count, dtype=np.int64)) + 2_000_000
                dup_tracks = track_ids.copy()
                dup_tracks[i] = dup_tracks[j]
                with pytest.raises(c.ContractError):
                    make_pose_evidence(
                        source=c.PoseSource.TRACKING, count=pose_count,
                        inlier_count=pose_count, track_ids=dup_tracks,
                        landmark_ids=pose_landmarks,
                    )


# ---------------------------------------------------------------------------------------
# Feature: shared-contracts, Property 11: handoff time ordering.
# Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.6.
# ---------------------------------------------------------------------------------------


def test_property_handoff_time_ordering():
    """An accepted ``HandoffEvidence`` needs ``source < measurement``, a strictly increasing
    support chain from source to measurement with 1..4 transitions, an age within
    HISTORY_MAX_AGE_NS, current-image correspondences with the pose's landmark ids and
    matching identities. Non-increasing stamps, a chain that leaves the window or an
    over-age chain are rejected."""
    seed = 110
    accepted_seen = False
    for k in range(ITERATIONS):
        rng = np.random.default_rng([seed, k])
        source_ns = int(rng.integers(1_000_000_000_000_000_000, 1_700_000_000_000_000_000))
        # Age within the 250 ms budget so an accepted chain is reachable.
        age = int(rng.integers(1, c.HISTORY_MAX_AGE_NS + 1))
        measurement_ns = source_ns + age
        where = f"seed={seed} iteration={k} source={source_ns} age={age}"

        # Build an accepted record: chain endpoints at source/measurement, matching ids.
        # A chain of n_transitions transitions needs n_transitions + 1 stamps, so
        # n_transitions - 1 strictly-interior midpoints. Only feasible when the open window
        # (source_ns, measurement_ns) holds that many distinct integers.
        max_interior = min(c.MAX_HANDOFF_TRANSITIONS - 1, max(age - 1, 0))
        n_interior = int(rng.integers(0, max_interior + 1))
        if n_interior == 0:
            support = (source_ns, measurement_ns)
        else:
            mids = set()
            while len(mids) < n_interior:
                mids.add(int(rng.integers(source_ns + 1, measurement_ns)))
            support = (source_ns, *sorted(mids), measurement_ns)
        n_landmarks = int(rng.integers(1, 17))
        landmark_ids = rng.permutation(np.arange(n_landmarks, dtype=np.int64)) + 3000
        correspondences = make_correspondences(
            count=n_landmarks, exposure_ns=measurement_ns,
            keypoint_ids=_ids(n_landmarks), pixels=_pixels(n_landmarks),
            landmark_ids=landmark_ids,
        )
        pose = make_handoff_pose(
            count=n_landmarks, exposure_ns=measurement_ns,
            track_ids=_ids(n_landmarks), landmark_ids=landmark_ids.copy(),
            inlier_count=n_landmarks,
        )
        accepted = make_handoff_evidence(
            source_exposure_ns=source_ns, measurement_ns=measurement_ns,
            support_ns=support, correspondences=correspondences, pose=pose,
        )
        assert accepted.accepted is True, where
        assert accepted.support_ns[0] == source_ns and accepted.support_ns[-1] == measurement_ns
        assert 1 <= accepted.transitions <= c.MAX_HANDOFF_TRANSITIONS, where
        accepted_seen = True

        # Rejection: a non-increasing support chain (duplicate a stamp) always raises,
        # regardless of the flag. Only meaningful for chains of length >= 2.
        if len(support) >= 2:
            broken = list(support)
            broken[-1] = broken[-2]  # non-increasing final stamp
            with pytest.raises(c.ContractError, match=r"HandoffEvidence\.support_ns"):
                make_handoff_evidence(
                    source_exposure_ns=source_ns, measurement_ns=measurement_ns,
                    support_ns=tuple(broken), correspondences=correspondences, pose=pose,
                    accepted=False, reason="stale_chain",
                )

        # Rejection: source not before measurement is a structural error on any flag.
        with pytest.raises(c.ContractError, match=r"HandoffEvidence\.measurement_ns"):
            make_handoff_evidence(
                source_exposure_ns=measurement_ns, measurement_ns=measurement_ns,
                support_ns=(source_ns, measurement_ns),
                correspondences=correspondences, pose=pose,
                accepted=False, reason="stale_chain",
            )

        # Rejection: an accepted chain older than HISTORY_MAX_AGE_NS. Push the measurement
        # past the budget and move the chain and images with it.
        over_measurement = source_ns + c.HISTORY_MAX_AGE_NS + 1 + int(rng.integers(0, 1000))
        over_corr = make_correspondences(
            count=n_landmarks, exposure_ns=over_measurement,
            keypoint_ids=_ids(n_landmarks), pixels=_pixels(n_landmarks),
            landmark_ids=landmark_ids.copy(),
        )
        over_pose = make_handoff_pose(
            count=n_landmarks, exposure_ns=over_measurement,
            track_ids=_ids(n_landmarks), landmark_ids=landmark_ids.copy(),
            inlier_count=n_landmarks,
        )
        with pytest.raises(c.ContractError, match=r"HandoffEvidence:"):
            make_handoff_evidence(
                source_exposure_ns=source_ns, measurement_ns=over_measurement,
                support_ns=(source_ns, over_measurement),
                correspondences=over_corr, pose=over_pose,
            )

    assert accepted_seen, f"seed={seed}: no accepted handoff built across the loop"


# ---------------------------------------------------------------------------------------
# Feature: shared-contracts, Property 12: motion and border geometry.
# Validates: Requirements 10.1, 10.3.
# ---------------------------------------------------------------------------------------


def _random_spd(rng, n):
    """Draw a symmetric positive-definite ``n`` x ``n`` float64 matrix: A A^T + n I."""
    a = rng.standard_normal((n, n))
    return (a @ a.T + n * np.eye(n)).astype(np.float64)


def test_property_motion_and_border_geometry():
    """A planar ``MotionEvidence`` (rotation about z, zero z-translation, SPD covariance,
    increasing stamps) constructs; out-of-plane rotation, a z-translation, a singular
    covariance or non-increasing stamps raise. ``BorderEvidence`` accepts the lateral normal
    [-sin(yaw), cos(yaw)] with conditioning in [0, 1] and rejects a non-unit normal and
    conditioning outside the interval."""
    seed = 120
    for k in range(ITERATIONS):
        rng = np.random.default_rng([seed, k])

        # --- MotionEvidence ---
        yaw = float(rng.uniform(-np.pi, np.pi))
        transform = _yaw_rigid(yaw)
        transform[0, 3] = float(rng.uniform(-5.0, 5.0))  # planar translation only
        transform[1, 3] = float(rng.uniform(-5.0, 5.0))
        covariance = _random_spd(rng, 3)
        previous_ns = int(rng.integers(1_000_000_000_000_000_000, 1_700_000_000_000_000_000))
        current_ns = previous_ns + int(rng.integers(1, 100_000_000))
        where = f"seed={seed} iteration={k} yaw={yaw:.4f}"

        motion = make_motion_evidence(
            previous_ns=previous_ns, current_ns=current_ns,
            T_Bprev_Bcurr=transform, covariance=covariance,
        )
        assert motion.accepted is True, where

        # Out-of-plane rotation about x raises.
        tilted = _yaw_rigid(yaw)
        angle = float(rng.uniform(0.05, 0.5))
        cos, sin = np.cos(angle), np.sin(angle)
        tilted[1, 1], tilted[1, 2] = cos, -sin
        tilted[2, 1], tilted[2, 2] = sin, cos
        with pytest.raises(c.ContractError, match=r"MotionEvidence\.T_Bprev_Bcurr"):
            make_motion_evidence(
                previous_ns=previous_ns, current_ns=current_ns,
                T_Bprev_Bcurr=tilted, covariance=covariance,
            )

        # A z-translation raises.
        with_z = _yaw_rigid(yaw)
        with_z[2, 3] = float(rng.uniform(0.1, 2.0))
        with pytest.raises(c.ContractError, match=r"MotionEvidence\.T_Bprev_Bcurr"):
            make_motion_evidence(
                previous_ns=previous_ns, current_ns=current_ns,
                T_Bprev_Bcurr=with_z, covariance=covariance,
            )

        # A singular covariance (rank-deficient, one zero eigenvalue) raises: the motion
        # covariance must be positive definite. Build it from two random directions so the
        # third eigenvalue is exactly zero while staying symmetric.
        zero_axis = int(rng.integers(0, 3))
        eigenvalues = rng.uniform(0.5, 2.0, size=3)
        eigenvalues[zero_axis] = 0.0  # one eigenvalue is exactly zero -> not definite
        singular = np.diag(eigenvalues).astype(np.float64)
        with pytest.raises(c.ContractError, match=r"MotionEvidence\.covariance"):
            make_motion_evidence(
                previous_ns=previous_ns, current_ns=current_ns,
                T_Bprev_Bcurr=transform, covariance=singular,
            )

        # Non-increasing stamps raise.
        with pytest.raises(c.ContractError, match=r"MotionEvidence\.current_ns"):
            make_motion_evidence(
                previous_ns=current_ns, current_ns=current_ns,
                T_Bprev_Bcurr=transform, covariance=covariance,
            )

        # --- BorderEvidence ---
        prior_yaw = float(rng.uniform(-np.pi, np.pi))
        normal = np.array([-np.sin(prior_yaw), np.cos(prior_yaw)], dtype=np.float64)
        conditioning = float(rng.uniform(0.0, 1.0))
        R = _random_spd(rng, 2)
        border = make_border_evidence(
            normal_xy=normal, conditioning=conditioning, R=R,
        )
        assert border.accepted is True, where
        assert np.allclose(np.linalg.norm(border.normal_xy), 1.0), where

        # A non-unit normal raises (scale the unit normal away from length one).
        scale = float(rng.uniform(1.5, 3.0))
        with pytest.raises(c.ContractError, match=r"BorderEvidence\.normal_xy"):
            make_border_evidence(normal_xy=normal * scale, conditioning=conditioning, R=R)

        # Conditioning outside [0, 1] raises, on either side.
        below = float(rng.uniform(-2.0, -0.01))
        above = float(rng.uniform(1.01, 3.0))
        for bad in (below, above):
            with pytest.raises(c.ContractError, match=r"BorderEvidence\.conditioning"):
                make_border_evidence(normal_xy=normal, conditioning=bad, R=R)

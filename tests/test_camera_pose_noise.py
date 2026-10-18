"""New rescue views use conservative noise, while primary measurements remain unchanged."""
import numpy as np
import pytest

import contract_factories as f
from camloc import contracts as c
from camloc.filter import CameraFilter


def correction(camera, scales):
    stamp = f.EXPOSURE_NS
    estimator = CameraFilter(c.InitialPose(timestamp_ns=stamp-100_000_000, x=0., y=0., yaw=0.),
                             pose_variance_scale=scales)
    frame = f.make_frame_set(selected_camera_ids=(camera,), images=(f.make_image_message(camera_id=camera),))
    pose_matrix = np.eye(4)
    pose_matrix[0, 3] = 1.
    pose = f.make_pose_evidence(camera_id=camera, T_WB=pose_matrix)
    return estimator.step(f.make_group_evidence(frame_set=frame, pose=pose))


def test_conservative_auxiliary_noise_reduces_correction_without_rejecting_the_visual_pose():
    baseline = correction("cl", {})
    weaker = correction("cl", {"cl": 4.})
    assert 0 < weaker.state[0] < baseline.state[0]
    assert weaker.evidence_kinds == baseline.evidence_kinds == (c.EvidenceKind.PNP,)
    front = correction("fr", {"cl": 4.})
    np.testing.assert_array_equal(front.state, correction("fr", {}).state)


@pytest.mark.parametrize("value", [.5, float('nan'), float('inf')])
def test_invalid_camera_noise_scale_rejected(value):
    with pytest.raises(c.ContractError, match="variance scale"):
        CameraFilter(c.InitialPose(timestamp_ns=f.EXPOSURE_NS, x=0., y=0., yaw=0.),
                     pose_variance_scale={"cl": value})

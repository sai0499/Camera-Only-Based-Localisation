"""Causal camera priority from frozen spatial support and recent job outcomes."""
import math

from camloc import contracts as c
from camloc.pnp import occupied_cells

# Opt-in turn bias (owner suggestion): while turning, prefer side cameras looking toward the
# inside of the turn and de-prioritise the outside side. Uses only the causal prior's yaw rate.
TURN_RATE_RAD_S = 0.15
TURN_INSIDE_GAIN, TURN_OUTSIDE_GAIN = 2.0, 0.5
SIDE_AXIS_MIN = 0.5  # |body-y| of the optical axis that makes a camera side-looking


class CameraPolicy:
    def __init__(self, bank, cameras, *, turn_bias=False):
        self.bank = bank
        self.turn_bias = bool(turn_bias)
        # +1 looks left, -1 looks right, 0 forward/rear (frozen camera-to-body axis).
        self.side = {}
        for camera in (cameras if self.turn_bias else ()):
            axis_y = float(bank.T_BC(camera)[1, 2])
            self.side[camera] = (1 if axis_y > SIDE_AXIS_MIN else -1 if axis_y < -SIDE_AXIS_MIN else 0)
        self.cameras = tuple(cameras)
        self.failed = {}
        self.quality = {}
        for frame_id in bank.keyframe_ids:
            frame = bank.keyframe(int(frame_id))
            linked = frame.landmark_ids >= 0
            cells = occupied_cells(frame.keypoints[linked], bank.mask(frame.camera_id))
            self.quality[int(frame_id)] = min(int(linked.sum()), 256) * cells / 16

    def feedback(self, camera, accepted, stamp):
        count = self.failed.get(camera, (0, 0))[0]
        self.failed[camera] = (0 if accepted else min(count + 1, 3), stamp)

    def scores(self, snapshot, stamp, available):
        x, y, yaw = snapshot.state[:3]
        sigma = min(math.sqrt(max(snapshot.P[0, 0], snapshot.P[1, 1])), 20.)
        values = {}
        for camera in self.cameras:
            if camera not in available:
                continue
            references = self.bank.select_references(camera, x, y, yaw,
                sigma_xy_m=sigma, count=2)
            score = max((self.quality[ref] for ref in references), default=0.)
            # The primary can still request global recovery outside local mapped support.
            if camera == c.PRIMARY_CAMERA:
                score = max(score, 1.)
            if self.turn_bias and self.side.get(camera) and abs(snapshot.state[5]) >= TURN_RATE_RAD_S:
                inside = self.side[camera] * snapshot.state[5] > 0
                score *= TURN_INSIDE_GAIN if inside else TURN_OUTSIDE_GAIN
            failures, last = self.failed.get(camera, (0, 0))
            if 0 <= stamp - last < 500_000_000:
                score /= (1 + 3 * failures) ** 2
            values[camera] = score
        return values

    @staticmethod
    def rank(scores):
        return sorted((cam for cam in scores if scores[cam] > 0),
                      key=lambda cam: (-scores[cam], cam))

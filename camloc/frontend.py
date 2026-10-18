"""C's current-image assembly over A's geometry and B's visual components."""

import dataclasses
import math

import numpy as np

from camloc import contracts as c
from camloc.filter import CameraFilter
from camloc.scheduler import GroupContext


def _pose(item):
    return item.pose if isinstance(item, c.HandoffEvidence) else item


class VisualFrontend:
    def __init__(self, canonicalizer, tracker, estimator: CameraFilter, *, road, borders, motion):
        self.canonicalizer = canonicalizer
        self.tracker = tracker
        self.estimator = estimator
        self.road, self.borders, self.motion = road, borders, motion
        self.previous = {}
        self.events = []
        self._search = {}
        self._last_rear_proposal_ns = estimator.initial_pose.timestamp_ns - 750_000_000
        self._last_recovery_proposal_ns = None
        self._recovery_camera_index = 0

    def drain_diagnostics(self):
        events, self.events = self.events, []
        while self.tracker.events:
            events.append({"record": "TrackerEvent", **self.tracker.events.popleft()})
        return events

    def _references(self, obs, snapshot, supported):
        # shortcut: geometric search only; B's Step 5 adds global EigenPlaces retrieval.
        x, y, yaw = snapshot.state[:3]
        sigma = min(math.sqrt(max(snapshot.P[0, 0], snapshot.P[1, 1])), 60.0)
        refs = self.tracker.map.select_references(
            obs.camera_id, x, y, yaw, sigma_xy_m=sigma,
            max_heading_rad=min(math.pi, math.radians(30) + 3 * math.sqrt(snapshot.P[2, 2])))
        if supported or len(refs) <= 2:
            self._search.pop(obs.camera_id, None)
            return refs[:2]
        # Cycle the bounded geometric shortlist when a stationary startup prior misses.
        stamp, offset = self._search.get(obs.camera_id, (obs.exposure_ns, 0))
        if obs.exposure_ns - stamp >= 500_000_000:
            stamp, offset = obs.exposure_ns, offset + 2
        self._search[obs.camera_id] = (stamp, offset)
        return tuple(refs[(offset + i) % len(refs)] for i in range(2))

    def __call__(self, frame: c.FrameSet, prior: c.SearchSnapshot, context: GroupContext):
        observations, poses, rejected, unused = {}, [], [], []
        rear_due = ("r" in self.tracker.map.cameras
                    and frame.t_ns - self._last_rear_proposal_ns >= c.NS_PER_S)
        # Rear JPEGs are needed only for a mapped rear stream, an in-flight job or fallback.
        for image in sorted(frame.images, key=lambda item: (item.camera_id != "fr", item.exposure_ns)):
            front_supported = any(item.camera_id == "fr" and self.estimator.pose_in_gate(_pose(item))
                                  for item in poses)
            if image.camera_id == "r" and front_supported and not context.rear_in_flight and not rear_due:
                continue
            try:
                obs = self.canonicalizer.canonicalize(image)
            except c.InputError as exc:
                self.events.append({"record": "DecodeError", "group_id": frame.group_id,
                                    "camera_id": image.camera_id, "detail": str(exc)})
                continue
            # Apply the frozen fitting mask to every consumer, including border diagnostics.
            if obs.camera_id in self.tracker.map.cameras:
                obs = dataclasses.replace(obs, valid_mask=(
                    obs.valid_mask & self.tracker.map.mask(obs.camera_id)))
            observations[obs.camera_id] = obs
            self.tracker.observe(obs)
            snapshot = self.estimator.snapshot(obs.exposure_ns)
            tracked = self.tracker.track(obs, snapshot, snapshot.auxiliary)
            (poses if tracked.accepted else rejected).append(tracked)
            verification = self.tracker.verify_candidate(obs)
            if verification is not None:
                if verification.accepted:
                    unused.append(c.UnusedEvidence(reason="candidate_verification", evidence=verification))
                else:
                    rejected.append(verification)

        for result in context.results:
            current = observations.get(result.camera_id)
            if current is None:
                self.events.append({"record": "HandoffSkipped", "job_id": result.job_id,
                                    "group_id": frame.group_id, "reason": "current_image_unavailable"})
                continue
            handoff = self.tracker.handoff(result, current)
            (poses if handoff.accepted else rejected).append(handoff)

        eligible = [item for item in poses if self.estimator.pose_in_gate(_pose(item))]
        # Compare current poses at the group exposure using only the preceding causal velocity.
        common = []
        for item in eligible:
            pose = _pose(item)
            dt = (frame.t_ns - pose.exposure_ns) / c.NS_PER_S
            T = pose.T_WB
            common.append((item.camera_id, T[:2, 3] + prior.state[3:5] * dt,
                           math.atan2(T[1, 0], T[0, 0]) + prior.state[5] * dt))
        conflict = any((np.linalg.norm(a[1] - b[1]) > 2.0
                       or abs(c.wrap_angle(a[2] - b[2])) > math.radians(3))
                       for i, a in enumerate(common) for b in common[i + 1:])
        selected = None if conflict or not eligible else max(
            eligible, key=lambda item: (_pose(item).occupied_cells, _pose(item).inlier_count,
                                        -_pose(item).median_error_px))
        for item in poses:
            if item is not selected:
                compatible = False
                if selected is not None and isinstance(item, c.HandoffEvidence):
                    first, second = _pose(selected), item.pose
                    dt = (second.exposure_ns - first.exposure_ns) / c.NS_PER_S
                    delta = second.T_WB[:2, 3] - first.T_WB[:2, 3] - prior.state[3:5] * dt
                    yaw_delta = (math.atan2(second.T_WB[1, 0], second.T_WB[0, 0])
                                 - math.atan2(first.T_WB[1, 0], first.T_WB[0, 0]) - prior.state[5] * dt)
                    compatible = np.linalg.norm(delta) <= 1.0 and abs(c.wrap_angle(yaw_delta)) <= math.radians(2)
                reason = ("compatible_refresh" if compatible else
                          "innovation_gate" if conflict or not any(item is other for other in eligible)
                          else "stronger_pose_selected")
                unused.append(c.UnusedEvidence(reason=reason, evidence=item))

        motion, border = None, None
        if selected is None:
            for camera, obs in observations.items():
                snapshot = self.estimator.snapshot(obs.exposure_ns)
                if border is None:
                    value = self.borders.border_evidence(obs, snapshot, snapshot.auxiliary)
                    if value.accepted:
                        border = value
                    else:
                        rejected.append(value)
                previous = self.previous.get(camera)
                if previous is None or motion is not None:
                    continue
                patch = self.road.road_patch(*snapshot.state[:3], snapshot.branch)
                try:
                    previous_snapshot = self.estimator.exposure_snapshot(camera, previous.exposure_ns)
                except c.ContractError:
                    continue
                value = self.motion.estimate(previous, obs, patch,
                    previous_snapshot=previous_snapshot, current_snapshot=snapshot)
                if value.accepted:
                    # Real road-motion qualification is pending; retain measured support.
                    unused.append(c.UnusedEvidence(reason="ground_motion_pending_real_validation",
                                                    evidence=value))
                else:
                    rejected.append(value)

        proposals = []
        for camera, obs in observations.items():
            if camera not in self.tracker.map.cameras:
                continue
            snapshot = self.estimator.snapshot(obs.exposure_ns)
            recent_full = (prior.auxiliary.source is c.AuxSource.PNP
                           and prior.auxiliary.source_ns is not None
                           and frame.t_ns - prior.auxiliary.source_ns <= 500_000_000)
            refs = self._references(obs, snapshot, selected is not None or recent_full)
            if refs:
                proposals.append((camera, obs, refs))
        if selected is None:
            last = self._last_recovery_proposal_ns
            if proposals and (last is None or frame.t_ns - last >= 250_000_000):
                # Alternate available cameras; failed front tracks cannot starve rear rescue.
                proposal = proposals[self._recovery_camera_index % len(proposals)]
                self._recovery_camera_index += 1
                self._last_recovery_proposal_ns = frame.t_ns
                proposals = [proposal]
            else:
                proposals = []
        elif rear_due and any(camera == "r" for camera, _, _ in proposals):
            # Reserve this normal slot for the staggered 1 Hz rear anchor.
            proposals = [proposal for proposal in proposals if proposal[0] == "r"]
        else:
            proposals = [proposal for proposal in proposals if proposal[0] == "fr"]
        for camera, obs, refs in proposals:
            purpose = (c.AnchorPurpose.RECOVERY if selected is None else
                       c.AnchorPurpose.FRONT_ANCHOR if camera == "fr" else c.AnchorPurpose.REAR_ANCHOR)
            context.propose(purpose=purpose, observation=obs, reference_ids=refs,
                            track_count=max((_pose(item).inlier_count for item in poses
                                             if item.camera_id == camera), default=0))
            if camera == "r":
                self._last_rear_proposal_ns = obs.exposure_ns
        self.previous = observations
        return c.GroupEvidence(frame_set=frame, pose=selected, motion=motion, border=border,
                               reset=None, rejected=tuple(rejected), unused=tuple(unused))

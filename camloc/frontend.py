"""C's current-image assembly over A's geometry and B's visual components."""

import dataclasses
import math

import numpy as np

from camloc import contracts as c
from camloc.filter import CameraFilter
from camloc.landmark_map import planar_camera_center, planar_camera_heading
from camloc.scheduler import GroupContext


def _pose(item):
    return item.pose if isinstance(item, c.HandoffEvidence) else item


CARRY_FRESH_NS = 500_000_000  # carried candidates come from a source image <= 0.5 s old


class VisualFrontend:
    """Per-group current-image assembly and anchor proposals.

    With ``retrieval_enabled`` (the worker holds a global retrieval index) an unsupported
    group proposes one recovery job per group: up to two untried lower-ranked candidates
    carried from the newest settled retrieval result of that camera when its source image
    is at most 0.5 s older than the proposal image, else a retrieval request
    (``reference_ids=()``). Without it, the geometric sweep is unchanged.
    """

    def __init__(self, canonicalizer, tracker, estimator: CameraFilter, *, road, borders, motion,
                 retrieval_enabled=False, camera_turn_bias=False):
        self.canonicalizer = canonicalizer
        self.tracker = tracker
        self.estimator = estimator
        self.road, self.borders, self.motion = road, borders, motion
        self.retrieval_enabled = bool(retrieval_enabled)
        self.previous = {}
        self.events = []
        self._search = {}
        # camera -> {"source_ns", "ids", "tried"}: candidates of the newest retrieval result.
        self._carry = {}
        # camera -> exposures of this frontend's retrieval requests (identifies their results).
        self._retrieval_exposures = {}
        self._last_rear_proposal_ns = estimator.initial_pose.timestamp_ns - 750_000_000
        self._recovery_camera_index = 0
        self._recovery_switch_ns = None
        self.camera_policy = None
        if tracker.map.manifest.get("camera_policy") == "support-v1":
            from camloc.camera_policy import CameraPolicy

            cameras = [cam for cam in tracker.map.cameras
                       if tracker.calibration_compatibility.get(cam, "valid") == "valid"]
            self.camera_policy = CameraPolicy(tracker.map, cameras, turn_bias=camera_turn_bias)
        self._decode_until = {}
        self._last_auxiliary_ns = estimator.initial_pose.timestamp_ns - c.NS_PER_S

    def drain_diagnostics(self):
        events, self.events = self.events, []
        while self.tracker.events:
            events.append({"record": "TrackerEvent", **self.tracker.events.popleft()})
        return events

    def _update_carry(self, settled):
        """Carry the remaining candidates of settled retrieval results; mark matched ids tried."""
        for result in settled:
            camera = result.camera_id
            carry = self._carry.get(camera)
            if carry is not None:
                carry["tried"].update(result.matched_reference_ids)
            requested = self._retrieval_exposures.get(camera, set())
            if (result.source_exposure_ns in requested and result.remaining_candidate_ids
                    and (carry is None or result.source_exposure_ns > carry["source_ns"])):
                self._carry[camera] = {"source_ns": result.source_exposure_ns,
                                       "ids": tuple(result.remaining_candidate_ids),
                                       "tried": set(result.matched_reference_ids)}

    def _recovery_references(self, obs):
        """Fresh untried carried candidates (at most two), else a retrieval request ``()``."""
        carry = self._carry.get(obs.camera_id)
        if carry is not None and 0 <= obs.exposure_ns - carry["source_ns"] <= CARRY_FRESH_NS:
            untried = [ref for ref in carry["ids"] if ref not in carry["tried"]]
            if untried:
                return tuple(untried[: c.MAX_REFERENCES_PER_JOB])
        return ()

    def _proposal_refs(self, obs, snapshot, supported):
        """Reference ids to propose, ``()`` for a retrieval request, None for no proposal."""
        if self.retrieval_enabled and not supported:
            return self._recovery_references(obs)
        refs = self._references(obs, snapshot, supported)
        return refs if refs else None

    def _references(self, obs, snapshot, supported):
        # Geometric search: supported shortlist, or the unsupported sweep without an index.
        bank, camera = self.tracker.map, obs.camera_id
        x, y, yaw = snapshot.state[:3]
        sigma = math.sqrt(max(snapshot.P[0, 0], snapshot.P[1, 1]))
        heading = min(math.pi, math.radians(30) + 3 * math.sqrt(snapshot.P[2, 2]))
        if supported:
            self._search.pop(camera, None)
            return bank.select_references(camera, x, y, yaw, sigma_xy_m=min(sigma, 60.0),
                                          max_heading_rad=heading)[:2]
        # Unsupported: the drifted prediction gets no false radius cutoff, and an empty radius
        # falls back to the whole eligible camera bank under the same ranking cost.
        count = max(int(np.size(bank.keyframe_ids)), 1)
        refs = bank.select_references(camera, x, y, yaw, sigma_xy_m=sigma, count=count,
                                      max_heading_rad=heading)
        if not refs:
            refs = bank.select_references(camera, x, y, yaw, sigma_xy_m=math.inf, count=count,
                                          max_heading_rad=math.pi)
        if not refs:
            return ()
        if self.estimator._last_pnp_ns is None:
            refs = self._forward_first(camera, refs)
        # One episode per camera: sweep untried references, restart once all were tried.
        tried = self._search.setdefault(camera, set())
        untried = [ref for ref in refs if ref not in tried]
        if not untried:
            tried.clear()
            untried = list(refs)
        chosen = tuple(untried[:2])
        tried.update(chosen)
        return chosen

    def _forward_first(self, camera, refs):
        """Startup: references ahead of the initial camera (offset >= -5 m), nearest first."""
        bank, initial = self.tracker.map, self.estimator.initial_pose
        T_BC = bank.T_BC(camera)
        center = planar_camera_center(initial.x, initial.y, initial.yaw, T_BC)
        heading = planar_camera_heading(initial.yaw, T_BC)
        axis = np.array([math.cos(heading), math.sin(heading)])
        rows = np.searchsorted(bank.keyframe_ids, np.asarray(refs, np.int64))
        offsets = (bank.keyframe_T_WC[rows, :2, 3] - center) @ axis
        return tuple(sorted(refs, key=lambda ref, o=dict(zip(refs, offsets.tolist())):
                            (0, o[ref]) if o[ref] >= -5.0 else (1, 0.0)))

    def __call__(self, frame: c.FrameSet, prior: c.SearchSnapshot, context: GroupContext):
        if self.retrieval_enabled:
            self._update_carry(context.settled)
        observations, poses, rejected, unused = {}, [], [], []
        policy = self.camera_policy
        scores, auxiliary = {}, None
        auxiliary_due = frame.t_ns - self._last_auxiliary_ns >= c.NS_PER_S
        if policy is not None:
            for result in context.settled:
                if result.status is c.AnchorStatus.FAILED:
                    policy.feedback(result.camera_id, False, frame.t_ns)
            scores = policy.scores(prior, frame.t_ns, {image.camera_id for image in frame.images})
            auxiliary = next((cam for cam in policy.rank(scores) if cam != "fr"), None)
        rear_due = ("r" in self.tracker.map.cameras
                    and frame.t_ns - self._last_rear_proposal_ns >= c.NS_PER_S)
        # Rear JPEGs are needed only for a mapped rear stream, an in-flight job or fallback.
        for image in sorted(frame.images, key=lambda item: (item.camera_id != "fr", item.exposure_ns)):
            front_supported = any(item.camera_id == "fr" and self.estimator.pose_in_gate(_pose(item))
                                  for item in poses)
            if policy is not None:
                camera = image.camera_id
                due = any(result.camera_id == camera for result in context.results)
                carried = frame.t_ns <= self._decode_until.get(camera, -1)
                if camera not in policy.cameras or (camera != "fr" and not due and not carried
                        and not (camera == auxiliary and (not front_supported or auxiliary_due
                                 or self.tracker.active_count(camera) >= 30))):
                    continue
            elif image.camera_id == "r" and front_supported and not context.rear_in_flight and not rear_due:
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
            if policy is not None:
                policy.feedback(result.camera_id, handoff.accepted and self.estimator.pose_in_gate(handoff.pose), frame.t_ns)

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
            eligible, key=lambda item: (policy is not None and item.camera_id == "fr",
                                        _pose(item).occupied_cells, _pose(item).inlier_count,
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
            refs = self._proposal_refs(obs, snapshot, selected is not None or recent_full)
            if refs is not None:  # () is a retrieval request, None no proposal
                proposals.append((camera, obs, refs))
        if policy is not None:
            if selected is not None and auxiliary_due and any(value[0] == auxiliary for value in proposals):
                choices = [value for value in proposals if value[0] == auxiliary]
            elif selected is not None:
                choices = [value for value in proposals if value[0] == "fr"]
            else:
                choices = [value for value in proposals if scores.get(value[0], 0.) > 0]
            proposals = sorted(choices, key=lambda value: (-scores.get(value[0], 0.), value[0]))[:1]
            self.events.append({"record": "CameraSelection", "group_id": frame.group_id,
                "scores": scores, "decoded": list(observations),
                "proposed": [value[0] for value in proposals], "reason": "mapped_support_and_recent_failures"})
        elif selected is None:
            # Alternate available cameras so failed front tracks cannot starve rear rescue.
            # Hold each camera for one delivery delay: per-group alternation would let the
            # next camera's request always replace the pending one at the same group parity.
            # Every group still proposes; the scheduler alone enforces the ceilings.
            if proposals:
                switch = self._recovery_switch_ns
                if switch is not None and frame.t_ns - switch >= c.DELIVERY_MIN_DELAY_NS:
                    self._recovery_camera_index += 1
                if switch is None or frame.t_ns - switch >= c.DELIVERY_MIN_DELAY_NS:
                    self._recovery_switch_ns = frame.t_ns
                proposals = [proposals[self._recovery_camera_index % len(proposals)]]
        elif rear_due and any(camera == "r" for camera, _, _ in proposals):
            # Reserve this normal slot for the staggered 1 Hz rear anchor.
            proposals = [proposal for proposal in proposals if proposal[0] == "r"]
        else:
            proposals = [proposal for proposal in proposals if proposal[0] == "fr"]
        for camera, obs, refs in proposals:
            purpose = (c.AnchorPurpose.RECOVERY if selected is None else
                       c.AnchorPurpose.FRONT_ANCHOR if camera == "fr" else
                       c.AnchorPurpose.REAR_ANCHOR if camera == "r" else c.AnchorPurpose.WARM_RETRIEVAL)
            context.propose(purpose=purpose, observation=obs, reference_ids=refs,
                            track_count=max((_pose(item).inlier_count for item in poses
                                             if item.camera_id == camera), default=0))
            if camera == "r":
                self._last_rear_proposal_ns = obs.exposure_ns
            if policy is not None and camera != "fr":
                self._last_auxiliary_ns = obs.exposure_ns
                self._decode_until[camera] = obs.exposure_ns + c.HISTORY_MAX_AGE_NS
            if self.retrieval_enabled and refs == ():
                # Keep only exposures whose results could still be fresh carry sources.
                requested = self._retrieval_exposures.setdefault(camera, set())
                requested.difference_update({ns for ns in requested
                                             if obs.exposure_ns - ns > 2 * c.NS_PER_S})
                requested.add(obs.exposure_ns)
        self.previous = observations
        return c.GroupEvidence(frame_set=frame, pose=selected, motion=motion, border=border,
                               reset=None, rejected=tuple(rejected), unused=tuple(unused))

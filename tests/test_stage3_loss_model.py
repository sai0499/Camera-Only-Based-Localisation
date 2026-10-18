"""Stage 3 fix: bounded prolonged-loss extrapolation (OU rate decay after B = last full + 3 s).

Seeded ``numpy`` loops stand in for ``hypothesis`` (not a dependency). Synthetic only: no
localization accuracy or native high-rate claim follows from these tests.
"""

import math

import numpy as np
import pytest

import contract_factories as f
import test_filter as tf
from camloc import contracts as c
from camloc import filter as module
from camloc.filter import CameraFilter, loss_transition

T = f.EXPOSURE_NS
MS = 1_000_000
RATES = (10, 20, 30, 60, 120)


def _pose(stamp, x, y, yaw):
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return f.make_pose_evidence(
        exposure_ns=stamp, T_WB=f.rigid(R, [x, y, 0.3]),
        inlier_count=64, track_ids=np.arange(64), landmark_ids=np.arange(64) + 100,
        inlier_fraction=0.8, occupied_cells=8, median_error_px=0.5, p90_error_px=1,
    )


def _compose(F_a, Q_a, F_b, Q_b):
    return F_b @ F_a, F_b @ Q_a @ F_b.T + Q_b


# -- unit tests ------------------------------------------------------------------------------

def test_loss_transition_zero_dt_is_identity():
    F, Q = loss_transition(0)
    np.testing.assert_array_equal(F, np.eye(6))
    np.testing.assert_array_equal(Q, np.zeros((6, 6)))


@pytest.mark.parametrize("bad", [-1, 1.0, True])
def test_loss_transition_rejects_bad_dt(bad):
    with pytest.raises(c.ContractError):
        loss_transition(bad)


def test_loss_transition_semigroup_composition_and_psd():
    for iteration in range(50):
        rng = f.seeded_rng(320, iteration)
        a, b = (int(v) for v in rng.integers(1, 30 * c.NS_PER_S, 2))
        F, Q = _compose(*loss_transition(a), *loss_transition(b))
        F_ab, Q_ab = loss_transition(a + b)
        np.testing.assert_allclose(F, F_ab, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(Q, Q_ab, rtol=1e-9, atol=1e-9 * np.abs(Q_ab).max())
        np.testing.assert_array_equal(Q_ab, Q_ab.T)
        assert np.linalg.eigvalsh(Q_ab).min() >= -1e-9 * np.abs(Q_ab).max()


def test_loss_transition_large_tau_approaches_cv():
    tau = 1e6
    for dt_ns in (10 * MS, c.NS_PER_S, 7 * c.NS_PER_S):
        dt = dt_ns / c.NS_PER_S
        F, Q = loss_transition(dt_ns, tau_s=tau)
        F_cv, _ = module.transition(dt_ns)
        np.testing.assert_allclose(F, F_cv, rtol=1e-5, atol=1e-12)
        for (p, v), sigma in zip(((0, 3), (1, 4), (2, 5)), module.LOSS_RATE_STD):
            density = 2 * sigma**2 / tau  # white-acceleration limit of the OU rate model
            block = Q[np.ix_([p, v], [p, v])] / density
            np.testing.assert_allclose(block, [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]],
                                       rtol=1e-4)


def test_propagate_splits_exactly_at_mid_interval_boundary():
    rng = f.seeded_rng(321)
    state = np.r_[rng.uniform(-50, 50, 2), 0.3, rng.uniform(-30, 30, 2), 0.2]
    P = np.diag(module.INITIAL_STD**2)
    last_full, scale = T, 4.0
    B = last_full + module.FULL_POSE_LOSS_NS
    start, end = B - 700 * MS, B + 1_300 * MS
    got = module._propagate(state, P, start, end, scale, last_full)
    s_B, P_B = module._predict(state, P, B - start, scale)
    F, Q = loss_transition(end - B)
    expected = F @ s_B
    expected[2] = c.wrap_angle(expected[2])
    np.testing.assert_array_equal(got[0], expected)
    np.testing.assert_array_equal(got[1], F @ P_B @ F.T + Q)
    # Ending at or before B is the original single CV prediction, bit for bit.
    for to in (start, B - 1, B):
        got = module._propagate(state, P, start, to, scale, last_full)
        ref = module._predict(state, P, to - start, scale)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])
    # Starting at or after B uses the loss model only, with no status scale.
    got = module._propagate(state, P, B + MS, end, scale, last_full)
    other = module._propagate(state, P, B + MS, end, 1.0, last_full)
    np.testing.assert_array_equal(got[1], other[1])


# -- property test ---------------------------------------------------------------------------

# Feature: stage3-visual-baseline-gate-fix, Property 2: Bounded prolonged-loss extrapolation
def test_property_bounded_prolonged_loss_extrapolation():
    """Seeded v, w, last_full_ns and losses up to 300 s at 10-120 Hz: after B the mean moves at
    most tau*|v_B| (yaw tau*|w_B|), P stays symmetric PSD, the position-P lambda_max never
    decreases, chained rows equal one prediction, and status/reason follow the original rule.

    **Validates: Requirements 2.4, 3.6**
    """
    tau = module.LOSS_TAU_S
    for iteration in range(40):
        rng = f.seeded_rng(322, iteration)
        rate = RATES[iteration % len(RATES)]
        x0, y0 = rng.uniform(-100, 100, 2)
        heading = rng.uniform(-math.pi, math.pi)
        speed, yaw_rate = rng.uniform(0, 40), rng.uniform(-0.3, 0.3)
        velocity = speed * np.array([math.cos(heading), math.sin(heading)])
        estimator = CameraFilter(c.InitialPose(timestamp_ns=T, x=x0, y=y0, yaw=heading))
        t, group_id = T, 0
        for k in range(int(rng.integers(0, 8))):
            t = T + k * 100 * MS
            dt = k * 0.1
            estimator.step(f.make_group_evidence(
                frame_set=tf.fr_frame(group_id, t),
                pose=_pose(t, x0 + velocity[0] * dt, y0 + velocity[1] * dt,
                           c.wrap_angle(heading + yaw_rate * dt))))
            group_id += 1
        last_full = estimator._last_pnp_ns
        last_visual = estimator._last_visual_ns
        aux_valid = estimator.estimate.auxiliary.valid
        full_ns = T if last_full is None else last_full
        B = full_ns + module.FULL_POSE_LOSS_NS
        end = full_ns + int(rng.integers(4, 301)) * c.NS_PER_S
        start = t + (0 if group_id == 0 else c.NS_PER_S // rate)
        dense = [start + i * c.NS_PER_S // rate for i in range(rate * 6)]
        dense = [s for s in dense if s < end]
        sparse = sorted({int(s) for s in rng.integers(dense[-1] + 1, max(end, dense[-1] + 2), 4)
                         if s < end} | {end})
        at_B = estimator.snapshot(B)
        v_B = float(np.linalg.norm(at_B.state[3:5]))
        w_B = abs(float(at_B.state[5]))
        previous_lmax, expected_end = -np.inf, None
        for stamp in dense + sparse:
            if expected_end is None and estimator.estimate.t_ns >= B:
                expected_end = estimator.snapshot(end)
            result = estimator.step(f.make_group_evidence(frame_set=tf.fr_frame(group_id, stamp)))
            group_id += 1
            assert result.reason == "no_evidence"
            assert result.evidence_kinds == (c.EvidenceKind.PREDICTION_ONLY,)
            assert result.status is tf.reference_status(aux_valid, stamp, last_visual, last_full,
                                                        T, False, False)
            np.testing.assert_array_equal(result.P, result.P.T)
            eig = np.linalg.eigvalsh(result.P)
            assert eig.min() >= -1e-9 * eig.max(), (iteration, stamp, eig.min())
            lmax = float(np.linalg.eigvalsh(result.P[:2, :2])[-1])
            assert lmax >= previous_lmax * (1 - 1e-12), (iteration, stamp)
            previous_lmax = lmax
            if stamp > B:
                displacement = float(np.linalg.norm(result.state[:2] - at_B.state[:2]))
                yaw_change = abs(c.wrap_angle(result.state[2] - at_B.state[2]))
                assert displacement <= tau * v_B + 1e-6, (iteration, stamp, v_B, displacement)
                assert yaw_change <= tau * w_B + 1e-9, (iteration, stamp, yaw_change)
                assert result.status is c.Status.LOST
        assert expected_end is not None
        np.testing.assert_allclose(result.state, expected_end.state, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(result.P, expected_end.P, rtol=1e-9,
                                   atol=1e-10 * np.abs(expected_end.P).max())

import unittest

import numpy as np

import native_bridge as bridge


class NativeBridgeTests(unittest.TestCase):
    def fixture(self):
        origins = np.array([0+0j, 10+0j, 10+10j, 0+10j])
        axes = np.array([1+0j, 0+1j, -1+0j, 0-1j]) * (1+2e-8)
        idx = np.repeat(np.arange(4), 4)
        ds = np.tile([.1, 3., 7., -.25], 4)
        n = np.tile([-1., .2, 2., -.5], 4)
        w = origins[idx] + axes[idx]*(ds+1j*n)
        state = {"timestamp_ns": np.arange(len(w)), "x": w.real, "y": w.imag,
                 "yaw": np.zeros(len(w)), "track_idx": idx, "track_ds": ds, "n": n,
                 "pit_idx": idx.copy(), "pit_ds": ds.copy(), "n_pit": n.copy()}
        return state, np.arange(len(w)) % 4 != 3

    def test_bridge_recovers_coordinates_and_preserves_tiny_scale(self):
        state, mask = self.fixture()
        frames = bridge.fit(state, mask)
        recovered, delta = bridge.reconstruct({k: v for k, v in state.items() if k not in ("x", "y")}, frames)
        np.testing.assert_allclose(recovered["x"], state["x"], atol=1e-12)
        np.testing.assert_allclose(recovered["y"], state["y"], atol=1e-12)
        np.testing.assert_allclose(delta, 0, atol=1e-12)
        np.testing.assert_allclose(abs(frames["track_rotation"]), 1+2e-8, atol=1e-14)

    def test_nonfitting_positions_do_not_change_calibration(self):
        state, mask = self.fixture()
        original = bridge.fit(state, mask)
        state["x"][~mask] = 1e8
        state["y"][~mask] = np.nan
        changed = bridge.fit(state, mask)
        for key in original:
            np.testing.assert_array_equal(original[key], changed[key])

    def test_bad_coordinates_and_uncovered_indices_are_rejected(self):
        for bad in (-1, .5, np.nan, 4):
            state, mask = self.fixture()
            frames = bridge.fit(state, mask)
            state["track_idx"] = state["track_idx"].astype(float)
            state["track_idx"][0] = bad
            with self.subTest(index=bad), self.assertRaises(ValueError):
                bridge.reconstruct(state, frames)
        state, mask = self.fixture()
        frames = bridge.fit(state, mask)
        state["n_pit"][0] += .1
        with self.assertRaisesRegex(ValueError, "disagree"):
            bridge.reconstruct(state, frames)

    def test_degenerate_or_incomplete_fitting_data_is_rejected(self):
        state, mask = self.fixture()
        mask[0:4] = False
        with self.assertRaises(ValueError):
            bridge.fit(state, mask)
        state, mask = self.fixture()
        state["track_ds"][:] = 1
        state["n"][:] = 1
        with self.assertRaisesRegex(ValueError, "variation"):
            bridge.fit(state, mask)


if __name__ == "__main__":
    unittest.main()

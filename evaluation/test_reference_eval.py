import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import reference_eval as ev


class ReferenceEvaluationTests(unittest.TestCase):
    def write(self, text):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "poses.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_exact_nanoseconds_and_csv_roundtrip(self):
        t = 1791918157176982265
        path = self.write(f"timestamp_ns,x,y,yaw\n{t},1.2,3.4,0.5\n{t+1},2,4,-0.5\n")
        poses = ev.read_csv(path)
        self.assertEqual(poses["timestamp_ns"][0], t)
        self.assertEqual(np.diff(poses["timestamp_ns"])[0], 1)
        out = path.with_name("roundtrip.csv")
        ev.write_csv(out, poses)
        self.assertEqual(ev.read_csv(out)["timestamp_ns"].tolist(), [t, t+1])

    def test_invalid_csv_rejected(self):
        for body in ("timestamp_ns,x,y\n1,0,0\n", "timestamp_ns,x,y,yaw\n",
                     "timestamp_ns,x,y,yaw\n1,NaN,0,0\n", "timestamp_ns,x,y,yaw\n1,0,inf,0\n",
                     "timestamp_ns,x,y,yaw\n1.0,0,0,0\n", "timestamp_ns,x,y,yaw\n1e9,0,0,0\n",
                     "timestamp_ns,x,y,yaw\n9223372036854775808,0,0,0\n",
                     "timestamp_ns,x,y,yaw\n2,0,0,0\n1,0,0,0\n",
                     "timestamp_ns,x,y,yaw\n1,0,0,0\n1,1,1,1\n",
                     "timestamp_ns,x,y,yaw\n1,0,0\n", "timestamp_ns,x,x,y,yaw\n1,0,0,0,0\n"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                ev.read_csv(self.write(body))

    def test_header_stamp_both_endiannesses(self):
        for endian, flag in (("<", 1), (">", 0)):
            data = bytes([0, flag, 0, 0]) + struct.pack(endian + "iI", 1791918157, 176982265)
            self.assertEqual(ev.cdr_stamp(data), 1791918157176982265)
        with self.assertRaises(ValueError):
            ev.cdr_stamp(b"\x00\x03" + b"\x00" * 10)
        with self.assertRaises(ValueError):
            ev.cdr_stamp(b"\x00\x01\x00\x00" + struct.pack("<iI", 1, 10**9))

    def test_coordinate_sign_wrap_and_native_gate(self):
        line = ev.TRACK.Line([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
        lines = {"track": line, "pit": line}
        c = {"timestamp_ns": np.array([100, 200]), "x": np.array([3., 6.]),
             "y": np.array([1., 1.]), "yaw": np.array([0., 0.]), "lane": np.array([1, 0])}
        projected, audit = ev.project(c, lines)
        np.testing.assert_allclose(projected["s"], [3, 6])
        np.testing.assert_allclose(projected["n"], [1, 1])
        self.assertLess(audit["track"]["max"], 1e-12)
        self.assertTrue(ev.native_compatibility(projected, lines)["enabled"])
        projected["s"] = projected["s"] + 2
        self.assertFalse(ev.native_compatibility(projected, lines)["enabled"])
        x, y, yaw = line.sn_to_xy([line.lap + 3], [1], [2*np.pi])
        np.testing.assert_allclose([x[0], y[0], yaw[0]], [3, 1, 0], atol=1e-12)

    def test_projection_restarts_after_relocalization_jump(self):
        angles = np.linspace(0, 2*np.pi, 1201)
        line = ev.TRACK.Line(200*np.c_[np.cos(angles), np.sin(angles)])
        c = {"timestamp_ns": np.array([0, 100_000_000]), "x": np.array([200., -200.]),
             "y": np.array([0., 0.]), "yaw": np.array([np.pi/2, -np.pi/2])}
        _, safe = ev.project(c, {"track": line, "pit": line})
        self.assertLess(safe["track"]["max"], 1e-9)
        self.assertEqual(safe["track"]["recovery_resets"], 1)
        s, n, _, _ = line.xy_to_sn(c["x"], c["y"], c["yaw"], c["timestamp_ns"])
        x, y, _ = line.sn_to_xy(s, n)
        self.assertGreater(np.max(np.hypot(x-c["x"], y-c["y"])), 100)

    def test_validation_overlap_camera_membership_and_coverage(self):
        tg = np.arange(0, 3_000_000_001, 10_000_000, dtype=np.int64)
        gt = {"timestamp_ns": tg}
        ts = np.arange(100_000_000, 3_000_000_001, 100_000_000, dtype=np.int64)
        self.assertEqual(ev.validate_submission({"timestamp_ns": ts}, gt, ts)["coverage"], 1.0)
        for bad, stamps in ((ts + 1, ts), (ts[:1], ts), (ts+3_000_000_000, ts+3_000_000_000)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ev.validate_submission({"timestamp_ns": bad}, gt, stamps)
        with self.assertRaises(ValueError):
            ev.validate_submission({"timestamp_ns": np.array([0])}, gt, np.array([0]))

    def test_unchanged_scorers_agree_on_known_error_and_yaw_wrap(self):
        t = np.arange(0, 2_000_000_001, 10_000_000, dtype=np.int64)
        gt = {"timestamp_ns": t, "x": np.zeros(len(t)), "y": np.zeros(len(t)),
              "yaw": np.linspace(3.13, 3.15, len(t))}
        with tempfile.TemporaryDirectory() as temp:
            gt_path, sub_path = Path(temp)/"gt.csv", Path(temp)/"sub.csv"
            ev.write_csv(gt_path, gt)
            sub = {k: v[10::10].copy() for k, v in gt.items()}
            sub["x"] += -0.3*np.sin(sub["yaw"])
            sub["y"] += 0.3*np.cos(sub["yaw"])
            sub["yaw"] = ev.TRACK.wrap(sub["yaw"] + np.radians(1))
            ev.write_csv(sub_path, sub)
            old = ev.capture(ev.OLD.main, gt_path, sub_path)
            new = ev.capture(ev.NEW.main, gt_path, sub_path, 40., 40.)
        for report in (old, new):
            self.assertIn("position RMSE  [m]     0.300", report)
            self.assertIn("yaw RMSE     [deg]     1.000", report)
            self.assertIn("coverage (1 s bins)    100.0 %", report)

    def test_hash_change_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ref = Path(temp) / "reference"
            ref.mkdir()
            (ref / "manifest.json").write_text(json.dumps({
                "input_hashes": {"tools/evaluate.py": "bad-hash"}, "packages": {},
                "adapter_sha256": ev.sha256(Path(ev.__file__)),
                "bridge_code_sha256": ev.sha256(ev.ROOT / "evaluation/native_bridge.py"),
                "native_coordinate_bridge": {"enabled": True}}))
            args = argparse.Namespace(reference=ref, out=Path(temp)/"out")
            with self.assertRaisesRegex(ValueError, "Changed map, scorer or split"):
                ev.score(args)
            args.out.mkdir()
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                ev.score(args)

    @unittest.skipUnless((ev.DEFAULT_REFERENCE / "manifest.json").exists(), "reconcile real reference first")
    def test_prepared_full_recording_pipeline_with_gt_plumbing_fixture(self):
        reference = ev.DEFAULT_REFERENCE
        manifest = json.loads((reference / "manifest.json").read_text())
        self.assertFalse(manifest["native_coordinate_gate"]["enabled"])
        gt = ev.read_csv(reference / "gt_tf.csv")
        self.assertTrue(manifest["native_coordinate_bridge"]["enabled"])
        state = ev.read_csv(reference / "gt_state_reconciled.csv")
        with np.load(reference / "camera_stamps.npz", allow_pickle=False) as cameras:
            ts = cameras["fr"]
        # GT interpolation at real camera stamps tests the wrapper, never camera localization.
        ts = ts[(ts > max(gt["timestamp_ns"][0], state["timestamp_ns"][0])) &
                (ts <= min(gt["timestamp_ns"][-1], state["timestamp_ns"][-1]))]
        i = np.searchsorted(gt["timestamp_ns"], ts)
        a = (ts-gt["timestamp_ns"][i-1])/(gt["timestamp_ns"][i]-gt["timestamp_ns"][i-1])
        fixture = {"timestamp_ns": ts}
        for key in ("x", "y", "yaw"):
            v = np.unwrap(gt[key]) if key == "yaw" else gt[key]
            fixture[key] = v[i-1] + a*(v[i]-v[i-1])
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            splits = json.loads((ev.ROOT/"task_docs/build_split.json").read_text())
            for episode in ("full", "fit", "tune", "holdout"):
                if episode == "full":
                    mask = np.ones(len(ts), bool)
                else:
                    start, end = int(splits[episode]["start_ns"]), int(splits[episode]["end_ns"])
                    windows = [c["timestamp_ns"][(c["timestamp_ns"] >= start) & (c["timestamp_ns"] < end)]
                               for c in (gt, state)]
                    mask = (ts >= max(w[0] for w in windows)) & (ts <= min(w[-1] for w in windows))
                src, out = temp / f"{episode}_GT_PLUMBING_ONLY.csv", temp / episode
                ev.write_csv(src, {k: v[mask] for k, v in fixture.items()})
                args = argparse.Namespace(reference=reference, submission=src, split=episode, out=out)
                ev.capture(ev.score, args)
                for report in ("original_xy.txt", "updated_xy.txt"):
                    self.assertIn("position RMSE  [m]     0.000", (out/report).read_text())
                local = (out/"reconciled_supplied_map.txt").read_text()
                self.assertGreaterEqual(local.count("<- SCORE"), 2)
                summary = json.loads((out/"validation.json").read_text())
                self.assertEqual(summary["split"], episode)
                self.assertFalse(summary["unconverted_native_indices_used"])
                self.assertTrue(summary["native_coordinate_bridge_used"])
                self.assertTrue(summary["raw_native_fields_used_for_GT"])
                self.assertEqual(summary["validation"]["coverage"], 1.0)
                self.assertAlmostEqual(summary["validation"]["median_interval_ms"], 100, delta=1)

    @unittest.skipUnless((ev.DEFAULT_REFERENCE / "manifest.json").exists(), "reconcile real reference first")
    def test_native_geometry_diagnostic_on_fitting_only_pairs(self):
        c = ev.read_csv(ev.DEFAULT_REFERENCE / "source_state_native.csv")
        split = json.loads((ev.ROOT / "task_docs/build_split.json").read_text())
        fit = (c["timestamp_ns"] >= int(split["fit"]["start_ns"])) & (c["timestamp_ns"] < int(split["fit"]["end_ns"]))
        world = c["x"] + 1j*c["y"]
        for route, suffix in (("track", ""), ("pit", "_pit")):
            idx = c[route+"_idx"].astype(int)
            uv = c[route+"_ds"] + 1j*c["n"+suffix]
            self.assertEqual(len(np.unique(idx)), 2929)
            errors = np.full(len(idx), np.nan)
            for k in np.unique(idx):
                train = (idx == k) & fit
                self.assertGreaterEqual(train.sum(), 13)
                z, w = uv[train], world[train]
                rotation = np.sum((w-w.mean())*np.conj(z-z.mean()))
                self.assertGreater(abs(rotation), 1e-12)
                rotation /= abs(rotation)
                origin = np.mean(w-rotation*z)
                selected = idx == k
                errors[selected] = np.abs(origin + rotation*uv[selected] - world[selected])
            self.assertTrue(np.isfinite(errors).all())
            self.assertLess(np.max(errors[~fit]), 1e-6)

    @unittest.skipUnless((ev.DEFAULT_REFERENCE / "manifest.json").exists(), "reconcile real reference first")
    def test_frozen_bridge_uses_native_fields_without_source_xy(self):
        c = ev.read_csv(ev.DEFAULT_REFERENCE / "source_state_native.csv")
        with np.load(ev.DEFAULT_REFERENCE / "native_reference_bridge.npz", allow_pickle=False) as archive:
            frames = {k: archive[k] for k in archive.files}
        source = {k: v for k, v in c.items() if k not in ("x", "y")}
        recovered, disagreement = ev.BRIDGE.reconstruct(source, frames)
        self.assertLess(np.max(np.hypot(recovered["x"]-c["x"], recovered["y"]-c["y"])), 1e-9)
        self.assertLess(np.max(disagreement), 1e-9)

    @unittest.skipUnless((ev.DEFAULT_REFERENCE / "manifest.json").exists(), "reconcile real reference first")
    def test_reconciled_coordinates_match_unchanged_organizer_converter(self):
        source = ev.read_csv(ev.DEFAULT_REFERENCE / "source_state_native.csv")
        expected = ev.read_csv(ev.DEFAULT_REFERENCE / "gt_state_reconciled.csv")
        selection = np.arange(0, len(source["x"]), 10)
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp)/"source.csv", Path(temp)/"converted.csv"
            ev.write_csv(src, {k: source[k][selection] for k in ev.POSE})
            with patch.dict(sys.modules, {"track_frame": ev.TRACK}):
                converter = ev.module("organizer_convert", ev.UPDATE / "tools/convert.py")
            ev.capture(converter.main, "xy2sn", ev.UPDATE / "map.mcap", src, dst)
            converted = ev.read_csv(dst, required=("timestamp_ns", "s", "n", "epsi", "s_pit", "n_pit", "epsi_pit"))
        np.testing.assert_array_equal(converted["timestamp_ns"], expected["timestamp_ns"][selection])
        for key in ("s", "n", "epsi", "s_pit", "n_pit", "epsi_pit"):
            np.testing.assert_allclose(converted[key], expected[key][selection], atol=5.1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()

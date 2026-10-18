"""B diagnostic: run.py with B-side PnP/LK variants patched in-process (diagnostic replays only).

Usage: python -B scripts/b_run_variant.py --usable-only-cells N [--handoff-fb PX] -- <run.py args>
"""
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

argv = sys.argv[1:]
split = argv.index("--")
own, run_args = argv[:split], argv[split + 1:]
settings = {"usable_only_cells": 0, "handoff_fb": None}
for i in range(0, len(own), 2):
    key = own[i].lstrip("-").replace("-", "_")
    settings[key] = float(own[i + 1]) if key == "handoff_fb" else int(own[i + 1])

from camloc import pnp, vision  # noqa: E402

original_solve = pnp.solve_pnp


def solve(pixels, points_W, K, T_BC, gates=pnp.DEFAULT_GATES, *, valid_mask=None):
    gates = dataclasses.replace(gates,
                                min_usable_cells_without_full=settings["usable_only_cells"])
    return original_solve(pixels, points_W, K, T_BC, gates, valid_mask=valid_mask)


pnp.solve_pnp = solve
if settings["handoff_fb"] is not None:
    original_handoff = vision.VisionTracker.handoff
    base_lk = vision.lk_track

    def handoff(self, result, current):
        def carry_lk(previous, following, points, guess=None, max_fb_error_px=None):
            return base_lk(previous, following, points, guess, settings["handoff_fb"])

        vision.lk_track = carry_lk
        try:
            return original_handoff(self, result, current)
        finally:
            vision.lk_track = base_lk

    vision.VisionTracker.handoff = handoff

import run  # noqa: E402

out = Path(run_args[run_args.index("--output") + 1])
run.main(run_args)
(out / "b_variant.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

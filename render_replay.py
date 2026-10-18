"""Export a completed/partial replay to a private, entirely offline browser bundle."""

import argparse
import json
import shutil
from pathlib import Path

import plotly

from scripts.replay_data import ROOT, Replay, scene_geometry, wire


def export_run(run, output, gt=None):
    replay = Replay(run)
    replay.refresh()
    if not replay.frames:
        raise ValueError("The selected run has no saved poses")
    if gt is not None:
        replay.attach_gt(gt)
    if Path(output).exists():
        raise ValueError("Use a fresh viewer output directory")
    shutil.copytree(ROOT / "replay_ui", output)
    output = Path(output)
    shutil.copyfile(Path(plotly.__file__).parent / "package_data/plotly.min.js", output / "plotly.js")
    shutil.copyfile(Path(run) / "submission.csv", output / "submission.csv")
    data = {"payload": replay.payload(), "geometry": scene_geometry(), "name": Path(run).name}
    (output / "data.js").write_text("window.APEX_REPLAY=" + json.dumps(wire(data), allow_nan=False) + ";\n", encoding="utf-8")
    index = output / "index.html"
    html = index.read_text(encoding="utf-8").replace('<script defer src="app.js">',
                                                       '<script src="data.js"></script><script defer src="app.js">')
    index.write_text(html, encoding="utf-8")
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt", type=Path, help="optional offline comparison only")
    args = parser.parse_args()
    print(export_run(args.run, args.output, args.gt))


if __name__ == "__main__":
    main()

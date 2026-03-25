#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

# Avoid pygame display errors in headless environments.
if (
    not os.environ.get("SDL_VIDEODRIVER")
    and sys.platform.startswith("linux")
    and not os.environ.get("DISPLAY")
):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Allow running as a script from the repo root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
)
from zsceval.envs.overcooked_new.src.overcooked_ai_py.static import LAYOUTS_DIR
from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import (
    StateVisualizer,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render an Overcooked layout image to image/layout/<layout_name>.png",
    )
    parser.add_argument(
        "layout_name",
        help="Layout file name without extension (e.g., incentivized_medium)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "image" / "layout"),
        help="Output directory for rendered layout images",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=75,
        help="Tile size in pixels for rendering",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    layout_path = os.path.join(LAYOUTS_DIR, args.layout_name + ".layout")
    if not os.path.exists(layout_path):
        raise FileNotFoundError(f"Layout not found: {layout_path}")

    mdp = OvercookedGridworld.from_layout_name(args.layout_name)
    state = mdp.get_standard_start_state()
    grid = mdp.terrain_mtx

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.layout_name}.png"

    visualizer = StateVisualizer(tile_size=args.tile_size)
    visualizer.display_rendered_state(
        state=state,
        grid=grid,
        img_path=str(output_path),
        ipython_display=False,
        window_display=False,
    )
    print(f"Saved layout image to {output_path}")


if __name__ == "__main__":
    main()

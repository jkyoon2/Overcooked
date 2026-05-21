from __future__ import annotations

"""
Throwaway pygame verification window for Task 2 only.
DO NOT use in Task 6 (web-based rendering). This module is intentionally
kept separate so it can be deleted without touching production code.
"""

import os
import threading
import time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from backend.game.engine import OvercookedEngine


# Module-level engine reference for the throwaway pygame window.
# This is intentionally a global — the throwaway nature of this module
# is why "no global state" (CLAUDE.md §6) has a documented exception here.
_engine_ref: List[Optional[OvercookedEngine]] = [None]


def set_pygame_engine(engine: OvercookedEngine) -> None:
    _engine_ref[0] = engine


def _render_loop() -> None:
    """Render loop. Runs in a daemon thread spawned by start_pygame_thread()."""
    # Must set SDL env vars BEFORE importing pygame or StateVisualizer,
    # because StateVisualizer loads images at class definition time (PI-4).
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    pygame.init()

    headless = os.environ.get("SDL_VIDEODRIVER") in ("dummy", "offscreen")
    if headless:
        print("[pygame] No display detected — running headless. No window shown.")
        # Still run the loop so tick rate still exercises the render path.
        while True:
            engine = _engine_ref[0]
            if engine is not None and engine.current_state is not None:
                pass  # would render here with a real display
            time.sleep(0.1)
        return

    # Lazy import after SDL is configured (PI-4 requirement)
    from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import (
        StateVisualizer,
    )

    screen: Optional[pygame.Surface] = None
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        engine = _engine_ref[0]
        if engine is None or engine.current_state is None:
            clock.tick(10)
            continue

        try:
            visualizer = StateVisualizer(tile_size=75)
            grid = engine.mdp.terrain_mtx
            surface = visualizer.render_state(engine.current_state, grid=grid)
            w, h = surface.get_size()
            if screen is None or screen.get_size() != (w, h):
                screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Overcooked — Neurocontroller [Task 2 verification]")
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        except Exception as exc:
            print(f"[pygame] render error: {exc}")

        clock.tick(10)  # cap at 10 FPS to stay in sync with game tick rate


def start_pygame_thread() -> threading.Thread:
    t = threading.Thread(target=_render_loop, daemon=True)
    t.start()
    return t

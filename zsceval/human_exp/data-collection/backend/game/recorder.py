from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from .engine import HORIZON, RECIPES, TwoPlayerEngine

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "trajectories"

CHANNEL_NAMES = [
    "ego_loc", "partner_loc",
    "ego_orientation_0", "ego_orientation_1", "ego_orientation_2", "ego_orientation_3",
    "partner_orientation_0", "partner_orientation_1", "partner_orientation_2", "partner_orientation_3",
    "pot_loc", "counter_loc", "onion_disp_loc", "tomato_disp_loc", "dish_disp_loc", "serve_loc",
    "onions_in_pot", "tomatoes_in_pot", "onions_in_soup", "tomatoes_in_soup",
    "soup_cook_time_remaining", "dishes", "onions", "tomatoes", "urgency",
]


class Recorder:
    def __init__(self, participant_id: str, layout: str, recipe: str, engine: TwoPlayerEngine) -> None:
        self.participant_id = participant_id
        self.layout = layout
        self.recipe = recipe
        self.engine = engine
        self._start_ms = int(time.time() * 1000)

        self._obs_p0: List[Any] = []
        self._obs_p1: List[Any] = []
        self._actions: List[Tuple[int, int]] = []
        self._rewards: List[int] = []
        self._timestamps: List[int] = []

    def record(self, joint_action: Tuple[int, int], reward: int) -> None:
        enc = self.engine.lossless_encoding()
        self._obs_p0.append(enc[0])
        self._obs_p1.append(enc[1])
        self._actions.append(joint_action)
        self._rewards.append(reward)
        self._timestamps.append(int(time.time() * 1000))

    def save(self, total_score: int) -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{self.participant_id}_{self.layout}_{self.recipe}"
        npz_path = DATA_DIR / f"{stem}.npz"
        meta_path = DATA_DIR / f"{stem}_meta.json"

        obs_p0 = np.array(self._obs_p0, dtype=np.int8)
        obs_p1 = np.array(self._obs_p1, dtype=np.int8)
        actions = np.array(self._actions, dtype=np.int8)
        rewards = np.array(self._rewards, dtype=np.int16)
        timestamps = np.array(self._timestamps, dtype=np.int64)

        np.savez_compressed(
            npz_path,
            obs_p0=obs_p0,
            obs_p1=obs_p1,
            actions=actions,
            rewards=rewards,
            timestamps=timestamps,
        )

        r = RECIPES[self.recipe]
        meta = {
            "participant_id": self.participant_id,
            "layout": self.layout,
            "recipe": self.recipe,
            "recipe_ingredients": r,
            "horizon": HORIZON,
            "n_channels": 25,
            "channel_names": CHANNEL_NAMES,
            "total_steps": len(self._actions),
            "total_score": total_score,
            "duration_ms": int(time.time() * 1000) - self._start_ms,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "trajectory_file": str(npz_path.relative_to(DATA_DIR.parent.parent)),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return npz_path

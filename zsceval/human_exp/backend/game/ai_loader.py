from __future__ import annotations

import random
from typing import Any

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action


class RandomPolicy:
    """
    Uniformly random policy — stub for Task 3 (HSP checkpoint loading).

    Task 3 will replace this with rMAPPOPolicy loaded from a PyTorch checkpoint.
    The act() signature must remain stable.
    """

    def act(self, state: Any) -> Any:
        return random.choice(Action.ALL_ACTIONS)

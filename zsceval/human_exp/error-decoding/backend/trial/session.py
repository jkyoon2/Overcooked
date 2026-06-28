from __future__ import annotations

import hashlib
import random
from typing import List, Tuple, cast

from .condition import AICheckpoint, ModelCheckpoint, PlayerIntent, TrialCondition
from .manager import TrialManager


TRIALS_PER_CELL = 2
MEP_VARIANTS = (1, 2, 3, 4)
_CHECKPOINT_LAYOUT_PREFIX = {"tomato": "tto", "onion": "too"}


def _seed_from_session_id(session_id: str) -> int:
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _model_checkpoint_for(
    ai_checkpoint: AICheckpoint,
    variant: int,
) -> ModelCheckpoint:
    prefix = _CHECKPOINT_LAYOUT_PREFIX[ai_checkpoint]
    return cast(ModelCheckpoint, f"{prefix}_mep{variant}")


def generate_trials(session_id: str) -> List[TrialCondition]:
    """
    Deterministically generate 8 trials for the Phase 1 session.

    Design: 2x2 cells (ai_checkpoint x player_intent), TRIALS_PER_CELL trials
    per cell. Within each cell the MEP variant (1..4) is drawn independently
    so the participant sees four different MEP partners across the eight
    trials (with possible repeats between cells). Trial order is shuffled to
    avoid grouping cells back-to-back.

    The seed is derived from session_id so reruns of the same session yield
    the same trial sequence — useful for crash recovery and reproducibility.
    """
    rng = random.Random(_seed_from_session_id(session_id))
    cells: List[Tuple[AICheckpoint, PlayerIntent]] = [
        (ai, intent)
        for ai in ("tomato", "onion")
        for intent in ("tomato", "onion")
    ]
    trials: List[TrialCondition] = []
    for ai, intent in cells:
        for _ in range(TRIALS_PER_CELL):
            variant = rng.choice(MEP_VARIANTS)
            trials.append(
                TrialCondition(
                    ai_checkpoint=ai,
                    player_intent=intent,
                    model_checkpoint=_model_checkpoint_for(ai, variant),
                )
            )
    rng.shuffle(trials)
    return trials


class SessionManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.trials: List[TrialCondition] = generate_trials(session_id)
        self.current_trial_index = 0
        self.trial_manager = TrialManager()

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    def current_condition(self) -> TrialCondition:
        if self.is_complete():
            raise RuntimeError("Session is complete; no current condition exists")
        return self.trials[self.current_trial_index]

    def current_trial_id(self) -> int:
        if self.is_complete():
            raise RuntimeError("Session is complete; no current trial exists")
        return self.current_trial_index + 1

    def advance_trial(self) -> bool:
        self.current_trial_index += 1
        return not self.is_complete()

    def is_complete(self) -> bool:
        return self.current_trial_index >= len(self.trials)

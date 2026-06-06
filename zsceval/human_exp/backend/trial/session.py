from __future__ import annotations

from typing import List

from .condition import TrialCondition
from .manager import TrialManager


class SessionManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        # Temporary deterministic Phase 2 order until session-order file loading
        # is implemented. Each cell of the 2x2 design appears exactly once.
        self.trials: List[TrialCondition] = [
            TrialCondition(
                ai_checkpoint="tomato",
                player_intent="tomato",
                model_checkpoint="tto_sp_seed4",
            ),
            TrialCondition(
                ai_checkpoint="onion",
                player_intent="tomato",
                model_checkpoint="tto_sp_seed5",
            ),
            TrialCondition(
                ai_checkpoint="tomato",
                player_intent="onion",
                model_checkpoint="ttt_adaptive_seed1",
            ),
            TrialCondition(
                ai_checkpoint="onion",
                player_intent="onion",
                model_checkpoint="ttt_adaptive_seed2",
            ),
        ]
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

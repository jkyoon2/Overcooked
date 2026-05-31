from __future__ import annotations

from typing import List

from .condition import TrialCondition
from .manager import TrialManager


class SessionManager:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.trials: List[TrialCondition] = [
            TrialCondition(ai_checkpoint="tomato", player_intent="tomato"),
            TrialCondition(ai_checkpoint="onion", player_intent="tomato"),
        ]
        self.current_trial_index = 0
        self.trial_manager = TrialManager()

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

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


AICheckpoint = Literal["tomato", "onion"]
PlayerIntent = Literal["tomato", "onion"]


class TrialCondition(BaseModel):
    ai_checkpoint: AICheckpoint
    player_intent: PlayerIntent

    @property
    def is_aligned(self) -> bool:
        return self.ai_checkpoint == self.player_intent

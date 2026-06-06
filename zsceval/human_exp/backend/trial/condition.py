from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


AICheckpoint = Literal["tomato", "onion"]
PlayerIntent = Literal["tomato", "onion"]
ModelCheckpoint = Literal[
    "tto_sp_seed4",
    "tto_sp_seed5",
    "ttt_adaptive_seed1",
    "ttt_adaptive_seed2",
]


class TrialCondition(BaseModel):
    ai_checkpoint: AICheckpoint
    player_intent: PlayerIntent
    model_checkpoint: ModelCheckpoint = "tto_sp_seed4"

    @property
    def is_aligned(self) -> bool:
        return self.ai_checkpoint == self.player_intent

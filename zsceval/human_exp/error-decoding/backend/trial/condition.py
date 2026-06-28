from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


AICheckpoint = Literal["tomato", "onion"]
PlayerIntent = Literal["tomato", "onion"]
ModelCheckpoint = Literal[
    "tto_mep1",
    "tto_mep2",
    "tto_mep3",
    "tto_mep4",
    "too_mep1",
    "too_mep2",
    "too_mep3",
    "too_mep4",
]


class TrialCondition(BaseModel):
    ai_checkpoint: AICheckpoint
    player_intent: PlayerIntent
    model_checkpoint: ModelCheckpoint = "tto_mep1"

    @property
    def is_aligned(self) -> bool:
        return self.ai_checkpoint == self.player_intent

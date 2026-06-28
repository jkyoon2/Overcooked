from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel


class GameEvent(BaseModel):
    event_type: Literal[
        "player_pickup",
        "player_deliver",
        "ai_pickup",
        "ai_deliver",
        "soup_drop",
        "soup_cooked",
    ]
    step_index: int
    agent_id: Literal[0, 1]  # 0 = player, 1 = AI (mapping fixed — hat color in frontend)
    payload: Dict[str, Any]

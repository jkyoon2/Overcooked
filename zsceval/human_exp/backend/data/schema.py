from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Task 1: hello / hello_ack (kept for backward compatibility)
# ---------------------------------------------------------------------------


class HelloPayload(BaseModel):
    message: str


class HelloMessage(BaseModel):
    type: str
    payload: HelloPayload


class HelloAckPayload(BaseModel):
    echo: str
    server_timestamp_ms: int


class HelloAckMessage(BaseModel):
    type: str
    payload: HelloAckPayload


# ---------------------------------------------------------------------------
# Task 2: game state + WS message types
# ---------------------------------------------------------------------------


class PlayerSnapshot(BaseModel):
    position: List[int]      # [x, y]
    orientation: List[int]   # [dx, dy]
    held_object: Optional[str]  # "onion", "tomato", "dish", "soup", or None


class ObjectSnapshot(BaseModel):
    name: str
    position: List[int]
    soup_state: Optional[str] = None       # "idle" | "cooking" | "ready" (SoupState only)
    ingredients: List[str] = []            # ingredient names inside the soup


class GameState(BaseModel):
    players: List[PlayerSnapshot]
    objects: List[ObjectSnapshot]
    score: float
    step_index: int
    time_remaining: int  # ticks until episode end
    layout_name: str


class JointAction(BaseModel):
    player_action: str  # one of NORTH/SOUTH/EAST/WEST/STAY/INTERACT
    ai_action: str


# ---------------------------------------------------------------------------
# WebSocket message types (Task 2 game protocol)
# ---------------------------------------------------------------------------


class StartGamePayload(BaseModel):
    layout: str = "corner_onion_tomato"


class StartGameMessage(BaseModel):
    type: Literal["start_game"]
    payload: StartGamePayload


class PlayerActionPayload(BaseModel):
    action: str  # NORTH/SOUTH/EAST/WEST/STAY/INTERACT


class PlayerActionMessage(BaseModel):
    type: Literal["player_action"]
    payload: PlayerActionPayload


class EndGameMessage(BaseModel):
    type: Literal["end_game"]
    payload: Dict[str, Any] = {}

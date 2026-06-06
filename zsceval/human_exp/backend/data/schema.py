from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    held_object_ingredients: List[str] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Task 4: trial lifecycle protocol
# ---------------------------------------------------------------------------


IntentAlignment = Literal[
    "yes_clearly",
    "yes_somewhat",
    "no_somewhat",
    "no_clearly",
]
TrialPhaseName = Literal["instruction", "play", "rating", "break"]


class StartSessionPayload(BaseModel):
    session_id: str


class StartSessionMessage(BaseModel):
    type: Literal["start_session"]
    payload: StartSessionPayload


class PhaseReadyMessage(BaseModel):
    type: Literal["phase_ready"]
    payload: Dict[str, Any] = Field(default_factory=dict)


class SubmitRatingPayload(BaseModel):
    quality: int
    intent_alignment: IntentAlignment


class SubmitRatingMessage(BaseModel):
    type: Literal["submit_rating"]
    payload: SubmitRatingPayload


class TrialStartPayload(BaseModel):
    trial_id: int
    total_trials: int
    phase: Literal["instruction"]
    duration_ms: int
    player_hat: str
    ai_hat: str


class TrialStartMessage(BaseModel):
    type: Literal["trial_start"]
    payload: TrialStartPayload


class PhaseChangePayload(BaseModel):
    phase: TrialPhaseName
    duration_ms: int


class PhaseChangeMessage(BaseModel):
    type: Literal["phase_change"]
    payload: PhaseChangePayload


class RatingAckPayload(BaseModel):
    trial_id: int
    excluded: bool
    exclusion_reason: Optional[str] = None


class RatingAckMessage(BaseModel):
    type: Literal["rating_ack"]
    payload: RatingAckPayload


class SessionCompletePayload(BaseModel):
    session_id: str


class SessionCompleteMessage(BaseModel):
    type: Literal["session_complete"]
    payload: SessionCompletePayload


# ---------------------------------------------------------------------------
# Phase 3: trajectory replay and misalignment segment annotation
# ---------------------------------------------------------------------------


class PhaseThreeReplayTrial(BaseModel):
    trial_id: int
    frames: List[GameState]


class PhaseThreeStartPayload(BaseModel):
    session_id: str
    frame_duration_ms: int
    player_hat: str
    ai_hat: str
    trials: List[PhaseThreeReplayTrial]


class PhaseThreeStartMessage(BaseModel):
    type: Literal["phase3_start"]
    payload: PhaseThreeStartPayload


class PhaseThreeSegment(BaseModel):
    segment_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    created_at_ms: int

    @model_validator(mode="after")
    def validate_frame_order(self) -> "PhaseThreeSegment":
        if self.start_frame > self.end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame")
        return self


class PhaseThreeTrialSelection(BaseModel):
    trial_id: int
    segments: List[PhaseThreeSegment]


class SubmitPhaseThreePayload(BaseModel):
    trials: List[PhaseThreeTrialSelection]


class SubmitPhaseThreeMessage(BaseModel):
    type: Literal["submit_phase3"]
    payload: SubmitPhaseThreePayload


class PhaseThreeCompletePayload(BaseModel):
    session_id: str
    saved: bool


class PhaseThreeCompleteMessage(BaseModel):
    type: Literal["phase3_complete"]
    payload: PhaseThreeCompletePayload

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, Literal, Optional

from pydantic import BaseModel

from .condition import TrialCondition


IntentAlignment = Literal[
    "yes_clearly",
    "yes_somewhat",
    "no_somewhat",
    "no_clearly",
]

_ALIGNMENT_SCORES: Dict[str, int] = {
    "yes_clearly": 1,
    "yes_somewhat": 2,
    "no_somewhat": 3,
    "no_clearly": 4,
}
_ALIGNED_EXPECTED_SCORE = 1
_MISALIGNED_EXPECTED_SCORE = 4


class TrialPhase(str, Enum):
    IDLE = "idle"
    INSTRUCTION = "instruction"
    PLAY = "play"
    RATING = "rating"
    BREAK = "break"


class RatingResult(BaseModel):
    trial_id: int
    quality: int
    intent_alignment: IntentAlignment
    excluded: bool
    exclusion_reason: Optional[str] = None


class TrialManager:
    """
    Owns one trial lifecycle. Timing is driven externally by the WebSocket layer.
    """

    def __init__(self) -> None:
        self.phase = TrialPhase.IDLE
        self.condition: Optional[TrialCondition] = None
        self.trial_id: Optional[int] = None
        self.start_time_ms: Optional[int] = None

    def start_instruction(self, trial_id: int, condition: TrialCondition) -> None:
        self._require_phase(TrialPhase.IDLE)
        self.trial_id = trial_id
        self.condition = condition
        self.phase = TrialPhase.INSTRUCTION
        self._mark_phase_start()

    def start_play(self) -> None:
        self._require_phase(TrialPhase.INSTRUCTION)
        self.phase = TrialPhase.PLAY
        self._mark_phase_start()

    def end_play(self) -> None:
        self._require_phase(TrialPhase.PLAY)
        self.phase = TrialPhase.RATING
        self._mark_phase_start()

    def submit_rating(
        self,
        quality: int,
        intent_alignment: IntentAlignment,
    ) -> RatingResult:
        self._require_phase(TrialPhase.RATING)
        if self.condition is None or self.trial_id is None:
            raise RuntimeError("Cannot submit rating before a trial has started")
        if intent_alignment not in _ALIGNMENT_SCORES:
            raise ValueError(f"Unknown intent alignment rating: {intent_alignment!r}")

        participant_score = _ALIGNMENT_SCORES[intent_alignment]
        expected_score = (
            _ALIGNED_EXPECTED_SCORE
            if self.condition.is_aligned
            else _MISALIGNED_EXPECTED_SCORE
        )
        distance = abs(participant_score - expected_score)
        excluded = distance > 1
        exclusion_reason = (
            "manipulation_check_failed"
            if excluded
            else None
        )

        self.phase = TrialPhase.BREAK
        self._mark_phase_start()
        return RatingResult(
            trial_id=self.trial_id,
            quality=quality,
            intent_alignment=intent_alignment,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
        )

    def start_break(self) -> None:
        """Complete BREAK and return to IDLE for the next trial."""
        self._require_phase(TrialPhase.BREAK)
        self.phase = TrialPhase.IDLE
        self.condition = None
        self.trial_id = None
        self.start_time_ms = None

    @property
    def elapsed_ms(self) -> int:
        if self.start_time_ms is None:
            return 0
        return _now_ms() - self.start_time_ms

    def _require_phase(self, expected: TrialPhase) -> None:
        if self.phase != expected:
            raise RuntimeError(
                f"Invalid trial phase transition: expected {expected.value}, "
                f"got {self.phase.value}"
            )

    def _mark_phase_start(self) -> None:
        self.start_time_ms = _now_ms()


def _now_ms() -> int:
    return int(time.time() * 1000)

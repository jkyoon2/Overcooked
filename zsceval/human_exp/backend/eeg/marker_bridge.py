"""EEG marker bridge stub — coworker will implement LSL integration here (Task 9)."""
from __future__ import annotations

from typing import Any, Protocol


class EEGMarkerBridge(Protocol):
    def emit(self, event: Any) -> None:
        """Emit a timestamped marker to the EEG recording system."""
        ...

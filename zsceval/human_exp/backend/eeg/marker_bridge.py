"""EEG marker bridge stub — coworker will implement LSL integration here (Task 9)."""
"""EEG marker bridge interface.

Real-time fine-grained game events should be written to JSONL by EEGEventLogger.
EGI/NetStation should only receive coarse synchronization markers.
"""
from __future__ import annotations

from typing import Any, Protocol


class EEGMarkerBridge(Protocol):
    def emit(self, event_type: str, **payload: Any) -> None:
        """Emit a timestamped marker/log event."""
        ...


class NullMarkerBridge:
    def emit(self, event_type: str, **payload: Any) -> None:
        return
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

class EEGEventLogger:
    def __init__(self, session_id: str | None = None):
        log_dir = Path(
            os.environ.get(
                "EEG_EVENT_LOG_DIR",
                "data/logs/eeg_events",
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        safe_session_id = session_id or "no_session"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = log_dir / f"{safe_session_id}_{run_id}_eeg_events.jsonl"
        self.emit(
            "EEG_LOG_START",
            session_id=session_id,
            log_path=str(self.path),
        )

    def emit(self, event_type: str, **payload: Any) -> None:
        row = {
            "event_type": event_type,
            "wall_time_ns": time.time_ns(),
            "wall_time_ms": int(time.time() * 1000),
            "monotonic_ns": time.monotonic_ns(),
            "payload": payload,
        }

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
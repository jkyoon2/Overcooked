from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.data.schema import PhaseThreeTrialSelection


_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "logs" / "phase3"
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")


def save_phase_three_record(
    session_id: str,
    replay_trials: List[Dict[str, Any]],
    selections: List[PhaseThreeTrialSelection],
    output_dir: Optional[Path] = None,
) -> Path:
    destination = output_dir or _DEFAULT_OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)

    safe_session_id = _UNSAFE_FILENAME.sub("_", session_id).strip("._") or "session"
    output_path = destination / f"{safe_session_id}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    selections_by_trial = {
        selection.trial_id: selection.segments for selection in selections
    }

    trials = []
    for replay_trial in replay_trials:
        trial_id = int(replay_trial["trial_id"])
        frames = replay_trial["frames"]
        segment_records = []
        for segment in selections_by_trial.get(trial_id, []):
            start_state = frames[segment.start_frame]
            end_state = frames[segment.end_frame]
            segment_records.append(
                {
                    **segment.model_dump(),
                    "start_step": start_state["step_index"],
                    "end_step": end_state["step_index"],
                    "start_ms": segment.start_frame * 100,
                    "end_ms": segment.end_frame * 100,
                }
            )

        trials.append(
            {
                "trial_id": trial_id,
                "frame_duration_ms": 100,
                "frames": frames,
                "misalignment_segments": segment_records,
            }
        )

    payload = {
        "session_id": session_id,
        "saved_at_ms": int(time.time() * 1000),
        "phase": 3,
        "trials": trials,
    }
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path

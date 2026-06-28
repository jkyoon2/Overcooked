from __future__ import annotations

import json

from backend.data.phase2_store import save_phase_two_record
from backend.data.schema import PhaseTwoSegment, PhaseTwoTrialSelection


def test_save_phase_two_record_includes_trajectory_and_segments(tmp_path) -> None:
    replay_trials = [
        {
            "trial_id": 1,
            "frames": [
                {"step_index": 0, "score": 0},
                {"step_index": 1, "score": 5},
                {"step_index": 2, "score": 5},
            ],
        }
    ]
    selections = [
        PhaseTwoTrialSelection(
            trial_id=1,
            segments=[
                PhaseTwoSegment(
                    segment_id="1-segment-1",
                    start_frame=1,
                    end_frame=2,
                    created_at_ms=1_700_000_000_000,
                )
            ],
        )
    ]

    output_path = save_phase_two_record(
        session_id="../unsafe session",
        replay_trials=replay_trials,
        selections=selections,
        output_dir=tmp_path,
    )

    assert output_path.parent == tmp_path
    assert output_path.name == "unsafe_session.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "../unsafe session"
    assert payload["phase"] == 2
    assert payload["trials"][0]["frames"] == replay_trials[0]["frames"]
    assert payload["trials"][0]["misalignment_segments"] == [
        {
            "segment_id": "1-segment-1",
            "start_frame": 1,
            "end_frame": 2,
            "created_at_ms": 1_700_000_000_000,
            "start_step": 1,
            "end_step": 2,
            "start_ms": 100,
            "end_ms": 200,
        }
    ]

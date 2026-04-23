import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List

import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger
from markdown import markdown

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from zsceval.human_exp.agent_pool import CheckpointAgentPool

app = Flask(__name__)
cors = CORS()
cors.init_app(app, resources={r"/*": {"origins": "*"}})

ARGS = None
AGENT_POOL = None
USER_AGENTS: Dict[str, Dict[str, Dict[int, Any]]] = {}
USER_TRIAL_RUNTIMES: Dict[str, Dict[str, Dict[str, Any]]] = {}

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[2]
SESSION_LAYOUT = "ttt"
SESSION_LAYOUT_ALIAS = SESSION_LAYOUT
SESSION_GRID = [
    "XXXXXXXXXXXXX",
    "O   DTXTD   O",
    "XX    P    XX",
    "S   2 P 1   S",
    "XXXXXTXTXXXXX",
]
SESSION_START_ORDERS = [{"ingredients": ["tomato", "tomato", "tomato"]}]
SESSION_COOK_TIME = 20
PROBE_TIME_FRACTIONS = [1.0 / 3.0, 0.5, 2.0 / 3.0]
SESSION_TITLE = "Cook with AI chef"
SESSION_SUBTITLE = "EEG data for collaborative AI"
RECIPE_OPTIONS = [
    {"id": "ttt", "label": "Tomato x3", "short_label": "T T T", "ingredients": ["tomato", "tomato", "tomato"], "points": 5},
    {"id": "tto", "label": "Tomato x2 + Onion", "short_label": "T T O", "ingredients": ["tomato", "tomato", "onion"], "points": 10},
    {"id": "too", "label": "Tomato + Onion x2", "short_label": "T O O", "ingredients": ["tomato", "onion", "onion"], "points": 15},
    {"id": "ooo", "label": "Onion x3", "short_label": "O O O", "ingredients": ["onion", "onion", "onion"], "points": 20},
]
RECIPE_LOOKUP = {recipe["id"]: recipe for recipe in RECIPE_OPTIONS}
TRIAL_RATING_QUESTIONS = [
    {
        "id": "strategy_satisfaction",
        "label": "How satisfied were you with the AI chef's strategy in this trial?",
        "type": "scale_bar",
        "min": 1,
        "max": 7,
        "left_label": "Not at all",
        "right_label": "Very much",
    },
    {
        "id": "behavior_predictability",
        "label": "How predictable was the AI chef's behavior in this trial?",
        "type": "scale_bar",
        "min": 1,
        "max": 7,
        "left_label": "Not predictable",
        "right_label": "Very predictable",
    },
]
SUBTASK_OPTIONS = [
    {"id": "collect_tomato", "label": "Collect Tomato", "description": "Move to a tomato source or secure a tomato."},
    {"id": "collect_onion", "label": "Collect Onion", "description": "Move to an onion source or secure an onion."},
    {"id": "collect_dish", "label": "Collect Dish", "description": "Pick up a clean dish for serving."},
    {"id": "load_pot", "label": "Load Pot", "description": "Bring an ingredient to the pot."},
    {"id": "manage_pot", "label": "Manage Pot", "description": "Start cooking or check the pot state."},
    {"id": "pickup_soup", "label": "Pick Up Soup", "description": "Plate or grab ready soup."},
    {"id": "serve_soup", "label": "Serve Soup", "description": "Bring soup to the serving window."},
    {"id": "counter_transfer", "label": "Counter Handoff", "description": "Use a counter to place or retrieve an item."},
    {"id": "reposition", "label": "Reposition", "description": "Move into place for the next goal."},
    {"id": "wait", "label": "Wait", "description": "Hold position or pause briefly."},
]
SUBTASK_OPTION_LOOKUP = {option["id"]: option for option in SUBTASK_OPTIONS}
ACTION_OPTIONS = [
    {"value": 0, "label": "Move Up"},
    {"value": 1, "label": "Move Down"},
    {"value": 2, "label": "Move Right"},
    {"value": 3, "label": "Move Left"},
    {"value": 4, "label": "Stay"},
    {"value": 5, "label": "Interact"},
]
ACTION_INDEX_TO_VALUE = {
    0: Direction.NORTH,
    1: Direction.SOUTH,
    2: Direction.EAST,
    3: Direction.WEST,
    4: Action.STAY,
    5: Action.INTERACT,
}
DEFAULT_ACTION_IDX = 4


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port", type=int, default=int(os.getenv("FLASK_PORT", 8088)), help="port to run flask")
    parser.add_argument("--ip", type=str, default=os.getenv("FLASK_HOST", "localhost"), help="bind host")
    parser.add_argument(
        "--access_ip",
        type=str,
        default=os.getenv("FLASK_ACCESS_HOST", "localhost"),
        help="public host exposed to the client",
    )
    parser.add_argument(
        "--trajs_save_path",
        type=str,
        default=os.getenv("TRAJS_SAVE_PATH", os.getenv("TRAJs_SAVE_PATH", "./zsceval/human_exp/data/trajs")),
        help="trajectory save path",
    )
    parser.add_argument(
        "--progress_save_path",
        type=str,
        default=os.getenv("PROGRESS_SAVE_PATH", "./zsceval/human_exp/data/progress.json"),
        help="session progress save path",
    )
    parser.add_argument(
        "--questionnaire_save_path",
        type=str,
        default=os.getenv("QUESTIONNAIRE_SAVE_PATH", "./zsceval/human_exp/data/questionnaires"),
        help="participant/session record path",
    )
    return parser


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_user_id(user_info: Dict[str, Any]) -> str:
    raw = f"{user_info.get('name', 'anon')}_{user_info.get('phone', user_info.get('participant_id', 'session'))}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "anonymous_session"


def participant_record_path(user_id: str) -> Path:
    return Path(ARGS.questionnaire_save_path) / f"{user_id}.json"


def save_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return deepcopy(default)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def normalize_yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def normalize_intake_form(intake_form: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(intake_form)
    for section in normalized.get("sections", []):
        for field in section.get("fields", []):
            if field.get("type") != "radio":
                continue
            for option in field.get("options", []):
                label = option.get("label")
                value = option.get("value")
                if isinstance(label, bool):
                    option["label"] = "Yes" if label else "No"
                elif label is not None:
                    option["label"] = str(label)
                if isinstance(value, bool):
                    option["value"] = normalize_yes_no(value)
                elif value is not None:
                    option["value"] = str(value)
    return normalized


def default_participant_record(user_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "participant_info": {},
        "session_progress": {
            "current_stage_id": "welcome_intro",
            "current_block_trial_index": 0,
            "completed_stage_ids": [],
        },
        "session_sections": {},
        "trial_data": {},
    }


def load_participant_record(user_id: str) -> Dict[str, Any]:
    return load_json(participant_record_path(user_id), default_participant_record(user_id))


def store_participant_record(user_id: str, record: Dict[str, Any]):
    record["updated_at"] = iso_now()
    save_json(participant_record_path(user_id), record)


def build_checkpoint_specs() -> List[Dict[str, Any]]:
    checkpoint_root = REPO_ROOT / "results" / "Overcooked" / SESSION_LAYOUT / "shared" / "adaptive" / "hsp-S2-s12"
    actual_policy_config = (
        REPO_ROOT / "zsceval" / "scripts" / "policy_pool" / SESSION_LAYOUT / "policy_config" / "rnn_policy_config.pkl"
    )
    specs = []
    for seed in range(1, 6):
        for step in (49040000, 50000000):
            specs.append(
                {
                    "id": f"seed{seed}_step{step}",
                    "seed": seed,
                    "step": step,
                    "label": f"Seed {seed} / step {step}",
                    "policy_config_path": str(actual_policy_config),
                    "model_path": {
                        "actor": str(
                            checkpoint_root
                            / f"seed{seed}"
                            / "models"
                            / "hsp_adaptive"
                            / f"actor_periodic_{step}.pt"
                        )
                    },
                    "featurize_type": "ppo",
                }
            )
    return specs


CHECKPOINT_SPECS = build_checkpoint_specs()
USED_CHECKPOINT_SPECS = CHECKPOINT_SPECS[:8]
CHECKPOINT_LOOKUP = {spec["id"]: spec for spec in CHECKPOINT_SPECS}


def build_probe_schedule(max_time: int, count: int = None) -> List[Dict[str, Any]]:
    if max_time <= 0:
        return []
    fractions = PROBE_TIME_FRACTIONS if count is None else PROBE_TIME_FRACTIONS[:count]
    schedule = []
    for index, fraction in enumerate(fractions):
        schedule.append(
            {
                "index": index + 1,
                "min_time_fraction": fraction,
                "min_elapsed_s": round(max_time * fraction, 2),
            }
        )
    return schedule


def make_trial_spec(
    trial_id: str,
    block_id: str,
    mode: str,
    title: str,
    checkpoint_id: str = None,
    max_time: int = 60,
    source_trial_id: str = None,
    practice: bool = False,
    probe_count: int = 3,
    start_orders: List[Dict[str, Any]] = None,
    instruction: str = None,
    ai_player_indices: List[int] = None,
    human_player_index: int = None,
) -> Dict[str, Any]:
    start_orders = deepcopy(start_orders or SESSION_START_ORDERS)
    probe_schedule = build_probe_schedule(max_time, count=probe_count)
    spec = {
        "id": trial_id,
        "block_id": block_id,
        "mode": mode,
        "title": title,
        "layout": SESSION_LAYOUT,
        "layout_alias": SESSION_LAYOUT_ALIAS,
        "layout_grid": SESSION_GRID,
        "start_orders": start_orders,
        "practice": practice,
        "max_time": max_time,
        "source_trial_id": source_trial_id,
        "environment_backend": "overcooked_new",
        "post_trial_questions": deepcopy(TRIAL_RATING_QUESTIONS),
        "probe": {
            "enabled": probe_count > 0 and mode in {"observe", "collaborate", "replay"},
            "type": "timed",
            "count": probe_count,
            "schedule": probe_schedule,
            "target_agent_index": 0,
            "prompt": "What subtask is the highlighted AI most likely trying to complete next?",
            "sketch_prompt": "Sketch the path you expected the AI to take next.",
            "confidence_prompt": "How confident are you in this expectation?",
        },
    }
    if mode == "observe":
        spec |= {
            "checkpoint_id": checkpoint_id,
            "human_player_index": human_player_index,
            "ai_player_indices": ai_player_indices or [0, 1],
            "target_agent_index": 0,
            "instruction": instruction or "Focus on the highlighted AI chef. Watch the scene, then predict that AI chef's next subtask when the probe opens.",
        }
    elif mode == "collaborate":
        spec |= {
            "checkpoint_id": checkpoint_id,
            "human_player_index": 1 if human_player_index is None else human_player_index,
            "ai_player_indices": [0] if ai_player_indices is None else ai_player_indices,
            "target_agent_index": 0,
            "instruction": instruction or "Cook with the AI chef. When the probe opens, choose the subtask you expect from the AI chef, draw the expected route, and rate your confidence.",
        }
    else:
        checkpoint_meta = CHECKPOINT_LOOKUP[checkpoint_id] if checkpoint_id else None
        spec |= {
            "checkpoint_id": checkpoint_id,
            "human_player_index": None,
            "ai_player_indices": [],
            "target_agent_index": 0,
            "instruction": instruction or "Replay the earlier collaborate scene. The replay pauses at key moments so you can say what you expected the AI chef to do next before watching the actual continuation.",
        }
        if checkpoint_meta:
            spec["checkpoint_label"] = checkpoint_meta["label"]
    return spec


def build_session_config() -> Dict[str, Any]:
    checkpoint_ids = [spec["id"] for spec in USED_CHECKPOINT_SPECS]
    tutorial_solo = make_trial_spec(
        "tutorial_solo",
        "tutorial",
        "collaborate",
        "Solo Practice",
        checkpoint_id=None,
        max_time=50,
        probe_count=0,
        instruction="Warm up by cooking alone first. Practice collecting ingredients, filling the pot, plating the soup, and serving it without AI help.",
        ai_player_indices=[],
    )
    tutorial_team = make_trial_spec(
        "tutorial_team",
        "tutorial",
        "collaborate",
        "Practice With AI Chef",
        checkpoint_id=checkpoint_ids[0],
        max_time=55,
        probe_count=1,
        instruction="Now cook with the AI chef. One practice probe will appear so you can try the expectation workflow before the main session.",
    )
    practice_trials = [
        make_trial_spec(
            "practice_1",
            "practice_block",
            "collaborate",
            "Practice Trial",
            checkpoint_id=checkpoint_ids[1],
            max_time=40,
            practice=True,
            probe_count=1,
            instruction="This is the last short practice before the main session. One probe will appear during the trial.",
        ),
    ]
    observe_trials = [
        make_trial_spec(
            f"observe_{i + 1}",
            "main_observe",
            "observe",
            f"Observe Trial {i + 1}",
            checkpoint_id=checkpoint_ids[2 + i],
            max_time=45,
            probe_count=3,
        )
        for i in range(3)
    ]
    collaborate_trials = [
        make_trial_spec(
            f"collaborate_{i + 1}",
            "main_collaborate",
            "collaborate",
            f"Collaborate Trial {i + 1}",
            checkpoint_id=checkpoint_ids[5 + i],
            max_time=60,
            probe_count=3,
        )
        for i in range(3)
    ]
    replay_trials = [
        make_trial_spec(
            f"replay_{i + 1}",
            "main_replay",
            "replay",
            f"Replay + Annotation {i + 1}",
            checkpoint_id=collaborate_trials[i]["checkpoint_id"],
            source_trial_id=collaborate_trials[i]["id"],
            max_time=0,
            probe_count=3,
        )
        for i in range(3)
    ]

    trials = [tutorial_solo, tutorial_team] + practice_trials + observe_trials + collaborate_trials + replay_trials
    trial_map = {trial["id"]: trial for trial in trials}

    return {
        "title": SESSION_TITLE,
        "subtitle": SESSION_SUBTITLE,
        "layout": SESSION_LAYOUT,
        "layout_alias": SESSION_LAYOUT_ALIAS,
        "layout_grid": SESSION_GRID,
        "start_orders": SESSION_START_ORDERS,
        "environment_backend": "overcooked_new",
        "subtask_options": SUBTASK_OPTIONS,
        "recipe_options": RECIPE_OPTIONS,
        "control_legend": {
            "move": "Arrow keys",
            "interact": "Spacebar",
        },
        "checkpoint_summary": [
            {"id": spec["id"], "seed": spec["seed"], "step": spec["step"], "label": spec["label"]}
            for spec in USED_CHECKPOINT_SPECS
        ],
        "unused_checkpoints": [
            {"id": spec["id"], "seed": spec["seed"], "step": spec["step"], "label": spec["label"]}
            for spec in CHECKPOINT_SPECS[len(USED_CHECKPOINT_SPECS) :]
        ],
        "stages": [
            {
                "id": "welcome_intro",
                "type": "welcome",
                "title": "Welcome",
                "duration": "1-2 min",
                "body": [
                    "Cook onion/tomato soup and submit your expectation for the AI chef.",
                ],
            },
            {
                "id": "mode_overview",
                "type": "mode_overview",
                "title": "Session Flow",
                "duration": "2-3 min",
                "body": [
                    "Review the two task modes before the session starts.",
                ],
            },
            {
                "id": "consent_screening",
                "type": "intake_form",
                "title": "Consent",
                "duration": "3-5 min",
                "body": [
                    "Please confirm consent and complete the short screening form.",
                ],
            },
            {
                "id": "onboarding",
                "type": "onboarding",
                "title": "Tutorial",
                "duration": "5-7 min",
                "body": [
                    "Watch a short cooking demo and learn the controls before practice.",
                ],
            },
            {
                "id": "tutorial",
                "type": "tutorial_lab",
                "title": "Tutorial",
                "duration": "10-15 min",
                "body": [
                    "Choose a team goal, practice cooking alone, then practice with the AI chef before EEG setup.",
                ],
                "trial_ids": [tutorial_solo["id"], tutorial_team["id"]],
            },
            {
                "id": "setup_baseline",
                "type": "static",
                "title": "EEG Setup",
                "duration": "10-20 min",
                "body": [
                    "The researcher will fit the EEG cap, check signal quality, and record a short baseline before the main session starts.",
                ],
            },
            {
                "id": "practice_block",
                "type": "trial_block",
                "title": "Practice",
                "duration": "5 min",
                "body": [
                    "Try one short practice trial with one probe before the main session begins.",
                ],
                "trial_ids": [trial["id"] for trial in practice_trials],
            },
            {
                "id": "main_observe",
                "type": "trial_block",
                "title": "Main Block 1: Observe",
                "duration": "10-15 min",
                "body": [
                    "Watch the highlighted AI chef and answer the probes during the trial.",
                ],
                "trial_ids": [trial["id"] for trial in observe_trials],
            },
            {
                "id": "observe_recovery",
                "type": "survey",
                "title": "Short Survey + Break",
                "duration": "3-5 min",
                "questions": [
                    {
                        "id": "mental_demand",
                        "label": "Mental demand",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                    {
                        "id": "effort",
                        "label": "Effort",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                    {
                        "id": "attentiveness",
                        "label": "Attentiveness",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                ],
            },
            {
                "id": "main_collaborate",
                "type": "trial_block",
                "title": "Main Block 2: Collaborate",
                "duration": "10-15 min",
                "body": [
                    "Cook with the AI chef and answer the probes during the trial.",
                ],
                "trial_ids": [trial["id"] for trial in collaborate_trials],
            },
            {
                "id": "collaborate_recovery",
                "type": "survey",
                "title": "Short Survey + Break",
                "duration": "3-5 min",
                "questions": [
                    {
                        "id": "mental_demand",
                        "label": "Mental demand",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                    {
                        "id": "effort",
                        "label": "Effort",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                    {
                        "id": "attentiveness",
                        "label": "Attentiveness",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                ],
            },
            {
                "id": "main_replay",
                "type": "trial_block",
                "title": "Main Block 3: Replay + Annotation",
                "duration": "10-15 min",
                "body": [
                    "Replay the recent collaborate scenes. The replay pauses at key moments so you can report what you expected.",
                ],
                "trial_ids": [trial["id"] for trial in replay_trials],
            },
            {
                "id": "replay_recovery",
                "type": "survey",
                "title": "Short Survey + Break",
                "duration": "3-5 min",
                "questions": [
                    {
                        "id": "mental_demand",
                        "label": "Mental demand",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                    {
                        "id": "effort",
                        "label": "Effort",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                    {
                        "id": "attentiveness",
                        "label": "Attentiveness",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Low",
                        "right_label": "High",
                    },
                ],
            },
            {
                "id": "final_posthoc_survey",
                "type": "survey",
                "title": "Final Post Hoc Survey",
                "duration": "10 min",
                "questions": [
                    {
                        "id": "anticipate_action",
                        "label": "How often did you feel able to anticipate the AI chef's next action?",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Never",
                        "right_label": "Very often",
                    },
                    {
                        "id": "unexpected_behavior",
                        "label": "How often did the AI chef behave differently from what you expected?",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Never",
                        "right_label": "Very often",
                    },
                    {
                        "id": "eeg_comfort",
                        "label": "How comfortable was the EEG setup during the session?",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Very uncomfortable",
                        "right_label": "Very comfortable",
                    },
                    {
                        "id": "final_comments",
                        "label": "Any final comments about the AI chef or the task?",
                        "type": "textarea",
                    },
                ],
            },
            {
                "id": "debrief_interview",
                "type": "survey",
                "title": "Debrief / Interview",
                "duration": "5 min",
                "questions": [
                    {
                        "id": "satisfied_with_ai",
                        "label": "How satisfied were you with the AI behavior?",
                        "type": "scale_bar",
                        "min": 1,
                        "max": 7,
                        "left_label": "Not at all",
                        "right_label": "Very satisfied",
                    },
                    {
                        "id": "anticipation_possible",
                        "label": "Was the AI chef's behavior something you could anticipate?",
                        "type": "radio",
                        "options": [
                            {"value": "yes", "label": "Yes"},
                            {"value": "sometimes", "label": "Sometimes"},
                            {"value": "no", "label": "No"},
                        ],
                    },
                    {
                        "id": "anticipation_prediction_note",
                        "label": "Do anticipation and prediction feel the same to you? Tell us why.",
                        "type": "textarea",
                    },
                ],
            },
        ],
        "trials": trial_map,
    }


SESSION_CONFIG = build_session_config()
TRIAL_SPECS = SESSION_CONFIG["trials"]


def public_trial_spec(trial_id: str) -> Dict[str, Any]:
    spec = deepcopy(TRIAL_SPECS[trial_id])
    checkpoint_id = spec.get("checkpoint_id")
    if checkpoint_id:
        checkpoint_meta = CHECKPOINT_LOOKUP[checkpoint_id]
        spec["checkpoint"] = {
            "id": checkpoint_meta["id"],
            "seed": checkpoint_meta["seed"],
            "step": checkpoint_meta["step"],
            "label": checkpoint_meta["label"],
        }
    return spec


def update_progress_file(user_id: str, stage_id: str = None, trial_id: str = None):
    progress_path = Path(ARGS.progress_save_path)
    progress = load_json(progress_path, {"participants": {}})
    participant_progress = progress["participants"].setdefault(
        user_id,
        {
            "created_at": iso_now(),
            "current_stage_id": None,
            "last_trial_id": None,
            "updated_at": iso_now(),
        },
    )
    if stage_id is not None:
        participant_progress["current_stage_id"] = stage_id
    if trial_id is not None:
        participant_progress["last_trial_id"] = trial_id
    participant_progress["updated_at"] = iso_now()
    save_json(progress_path, progress)


def merge_trial_record(record: Dict[str, Any], trial_id: str, updates: Dict[str, Any]):
    trial_record = record["trial_data"].setdefault(trial_id, {})
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(trial_record.get(key), dict):
            trial_record[key] |= value
        else:
            trial_record[key] = value
    return trial_record


def resolve_trial_start_orders(user_id: str, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    if spec["block_id"] not in {"tutorial", "tutorial_block"}:
        return deepcopy(spec["start_orders"])
    record = load_participant_record(user_id)
    tutorial_setup = record.get("session_sections", {}).get("tutorial_setup", {})
    recipe_id = tutorial_setup.get("team_goal_recipe_id")
    if recipe_id and recipe_id in RECIPE_LOOKUP:
        return [{"ingredients": deepcopy(RECIPE_LOOKUP[recipe_id]["ingredients"])}]
    return deepcopy(spec["start_orders"])


def make_runtime_mdp(layout_name: str, start_orders: List[Dict[str, Any]]) -> OvercookedGridworld:
    return OvercookedGridworld.from_layout_name(layout_name, start_all_orders=start_orders, old_dynamics=True)


def compute_runtime_elapsed_seconds(runtime: Dict[str, Any]) -> float:
    elapsed = time.time() - runtime["started_ts"] - runtime["paused_duration_s"]
    if runtime["pause_started_at"] is not None:
        elapsed -= time.time() - runtime["pause_started_at"]
    return max(elapsed, 0.0)


def compute_runtime_time_left(runtime: Dict[str, Any], trial_id: str) -> float:
    max_time = TRIAL_SPECS[trial_id]["max_time"]
    if max_time <= 0:
        return 0.0
    return max(max_time - compute_runtime_elapsed_seconds(runtime), 0.0)


def build_runtime_payload(runtime: Dict[str, Any], trial_id: str) -> Dict[str, Any]:
    return {
        "state": runtime["state"].to_dict(),
        "score": runtime["score"],
        "step_count": runtime["cur_gameloop"],
        "time_left": round(compute_runtime_time_left(runtime, trial_id), 2),
        "probe_records": deepcopy(runtime["probe_records"]),
        "environment_backend": "overcooked_new",
    }


def build_trial_summary_from_runtime(runtime: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trial_id": runtime["trial_id"],
        "mode": runtime["mode"],
        "score": runtime["score"],
        "total_steps": runtime["cur_gameloop"],
        "probes": deepcopy(runtime["probe_records"]),
        "primary_probe": deepcopy(runtime["probe_records"][0]) if runtime["probe_records"] else None,
        "secondary_probe": deepcopy(runtime["probe_records"][1]) if len(runtime["probe_records"]) > 1 else None,
    }


def ensure_trial_runtime(user_id: str, trial_id: str) -> Dict[str, Any]:
    spec = TRIAL_SPECS[trial_id]
    if spec["mode"] == "replay":
        raise ValueError(f"Replay trial {trial_id} does not use an interactive runtime")

    trial_runtimes = USER_TRIAL_RUNTIMES.setdefault(user_id, {})
    if trial_id in trial_runtimes:
        return trial_runtimes[trial_id]

    ai_agents = {}
    checkpoint_id = spec.get("checkpoint_id")
    if checkpoint_id:
        for agent_index in spec["ai_player_indices"]:
            ai_agents[agent_index] = AGENT_POOL.get_agent(checkpoint_id)

    USER_AGENTS.setdefault(user_id, {})[trial_id] = ai_agents

    start_orders = resolve_trial_start_orders(user_id, spec)
    mdp = make_runtime_mdp(spec["layout"], start_orders=start_orders)
    state = mdp.get_standard_start_state()
    mdp.reset_subgoal_tracking(state)

    runtime = {
        "trial_id": trial_id,
        "block_id": spec["block_id"],
        "mode": spec["mode"],
        "mdp": mdp,
        "state": state,
        "score": 0,
        "cur_gameloop": 0,
        "probe_records": [],
        "pending_transition": None,
        "started_ts": time.time(),
        "paused_duration_s": 0.0,
        "pause_started_at": None,
        "ai_agents": ai_agents,
        "trajectory": {
            "ep_states": [[]],
            "ep_actions": [[]],
            "ep_rewards": [[]],
            "mdp_params": [
                {
                    "layout_name": spec["layout"],
                    "cook_time": SESSION_COOK_TIME,
                    "start_order_list": deepcopy(start_orders),
                }
            ],
        },
    }
    trial_runtimes[trial_id] = runtime
    return runtime


def cleanup_trial_runtime(user_id: str, trial_id: str):
    if user_id in USER_TRIAL_RUNTIMES:
        USER_TRIAL_RUNTIMES[user_id].pop(trial_id, None)
        if not USER_TRIAL_RUNTIMES[user_id]:
            del USER_TRIAL_RUNTIMES[user_id]
    if user_id in USER_AGENTS:
        USER_AGENTS[user_id].pop(trial_id, None)
        if not USER_AGENTS[user_id]:
            del USER_AGENTS[user_id]


def get_action(state, agent, pos: int) -> int:
    action = agent(state, pos)
    return int(action.item() if hasattr(action, "item") else action)


def next_due_probe(runtime: Dict[str, Any], trial_spec: Dict[str, Any]) -> Dict[str, Any]:
    probe = trial_spec["probe"]
    if not probe.get("enabled"):
        return None
    schedule = probe.get("schedule", [])
    next_index = len(runtime["probe_records"])
    if next_index >= len(schedule):
        return None
    candidate = schedule[next_index]
    if compute_runtime_elapsed_seconds(runtime) < candidate["min_elapsed_s"]:
        return None
    return candidate


def record_transition(runtime: Dict[str, Any], joint_action_idx: List[int], reward: int, next_state):
    runtime["trajectory"]["ep_states"][0].append(runtime["state"].to_dict())
    runtime["trajectory"]["ep_actions"][0].append(list(joint_action_idx))
    runtime["trajectory"]["ep_rewards"][0].append(reward)
    runtime["state"] = next_state
    runtime["score"] += reward
    runtime["cur_gameloop"] += 1


def action_idx_or_default(value) -> int:
    if value is None:
        return DEFAULT_ACTION_IDX
    try:
        action_idx = int(value)
    except (TypeError, ValueError):
        return DEFAULT_ACTION_IDX
    return action_idx if action_idx in ACTION_INDEX_TO_VALUE else DEFAULT_ACTION_IDX


def normalize_selected_subtask_id(value: Any) -> str:
    if isinstance(value, str) and value in SUBTASK_OPTION_LOOKUP:
        return value
    return "reposition"


def infer_soup_profile(object_state) -> str:
    ingredients = getattr(object_state, "_ingredients", []) or []
    names = [ingredient.name for ingredient in ingredients if hasattr(ingredient, "name")]
    tomato_count = sum(1 for name in names if name == "tomato")
    onion_count = sum(1 for name in names if name == "onion")
    return "tomato" if tomato_count >= onion_count else "onion"


def infer_subtask_id(mdp: OvercookedGridworld, state, next_state, agent_idx: int, action_idx: int) -> str:
    player = state.players[agent_idx]
    next_player = next_state.players[agent_idx]
    action = ACTION_INDEX_TO_VALUE[action_idx]
    held_name = player.get_object().name if player.has_object() else None
    next_held_name = next_player.get_object().name if next_player.has_object() else None

    if action == Action.INTERACT:
        interact_pos = Action.move_in_direction(player.position, player.orientation)
        terrain_type = mdp.get_terrain_type_at_pos(interact_pos)
        if terrain_type == "T":
            return "collect_tomato"
        if terrain_type == "O":
            return "collect_onion"
        if terrain_type == "D":
            return "collect_dish"
        if terrain_type == "S":
            return "serve_soup"
        if terrain_type == "X":
            return "counter_transfer"
        if terrain_type == "P":
            if held_name in {"tomato", "onion"}:
                return "load_pot"
            if held_name == "dish" and next_held_name == "soup":
                return "pickup_soup"
            return "manage_pot"

    if held_name == "soup":
        return "serve_soup"
    if held_name in {"tomato", "onion"}:
        return "load_pot"
    if held_name == "dish":
        return "pickup_soup"
    if action == Action.STAY:
        return "wait"
    if next_held_name == "tomato":
        return "collect_tomato"
    if next_held_name == "onion":
        return "collect_onion"
    if next_held_name == "dish":
        return "collect_dish"
    return "reposition"


def public_probe_payload(pending_transition: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "joint_action_idx": list(pending_transition["joint_action_idx"]),
        "reward": pending_transition["reward"],
        "target_action_idx": pending_transition["target_action_idx"],
        "target_subtask_id": pending_transition["target_subtask_id"],
        "target_subtask_label": pending_transition["target_subtask_label"],
        "probe_index": pending_transition["probe_index"],
        "probe_total": pending_transition["probe_total"],
        "probe_game_loop": pending_transition["probe_game_loop"],
        "time_left": pending_transition["time_left"],
        "prompt_timestamp": pending_transition["prompt_timestamp"],
    }


@app.route("/")
def root():
    return app.send_static_file("index_1.html")


@app.route("/html/<page>")
def return_html(page):
    return app.send_static_file(f"{page}.html")


@app.route("/beforegame", methods=["POST"])
def beforegame():
    config_path = APP_ROOT.parent / "configs" / "before_game.yaml"
    with open(config_path, encoding="utf-8") as handle:
        return normalize_intake_form(yaml.load(handle, Loader=yaml.FullLoader))


@app.route("/statement", methods=["POST"])
def statement():
    statement_path = APP_ROOT.parent / "configs" / "statement.md"
    html = markdown(statement_path.read_text(encoding="utf-8"))
    return html


@app.route("/session_config", methods=["POST"])
def session_config():
    payload = json.loads(request.data) if request.data else {}
    user_info = payload.get("user_info")
    progress = None
    saved_trials = {}
    saved_sections = {}
    config_path = APP_ROOT.parent / "configs" / "before_game.yaml"
    with open(config_path, encoding="utf-8") as handle:
        intake_form = normalize_intake_form(yaml.load(handle, Loader=yaml.FullLoader))
    statement_path = APP_ROOT.parent / "configs" / "statement.md"
    statement_html = markdown(statement_path.read_text(encoding="utf-8"))
    if user_info:
        user_id = make_user_id(user_info)
        record = load_participant_record(user_id)
        progress = record.get("session_progress", {})
        saved_trials = record.get("trial_data", {})
        saved_sections = record.get("session_sections", {})
    return jsonify(
        {
            "session": SESSION_CONFIG,
            "progress": progress,
            "saved_trials": saved_trials,
            "saved_sections": saved_sections,
            "intake_form": intake_form,
            "statement_html": statement_html,
        }
    )


@app.route("/create_questionnaire_before_game", methods=["POST"])
def create_questionnaire_before_game():
    os.makedirs(ARGS.questionnaire_save_path, exist_ok=True)
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json)
    record = load_participant_record(user_id)
    record["participant_info"] |= data_json
    store_participant_record(user_id, record)
    update_progress_file(user_id, stage_id=record["session_progress"].get("current_stage_id"))
    return jsonify(record)


@app.route("/save_session_section", methods=["POST"])
def save_session_section():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    section_id = data_json["section_id"]
    append = data_json.get("append", False)
    section_data = data_json.get("data", {})

    record = load_participant_record(user_id)
    if append:
        record["session_sections"].setdefault(section_id, [])
        record["session_sections"][section_id].append(section_data)
    else:
        record["session_sections"][section_id] = section_data

    progress = data_json.get("progress")
    if progress:
        record["session_progress"] |= progress
        update_progress_file(user_id, stage_id=progress.get("current_stage_id"))

    store_participant_record(user_id, record)
    return jsonify({"status": True})


@app.route("/start_trial", methods=["POST"])
def start_trial():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]
    if trial_id not in TRIAL_SPECS:
        return jsonify({"status": False, "error": f"Unknown trial_id: {trial_id}"}), 404

    spec = TRIAL_SPECS[trial_id]
    runtime_payload = None
    if spec["mode"] != "replay":
        runtime = ensure_trial_runtime(user_id, trial_id)
        runtime_payload = build_runtime_payload(runtime, trial_id)

    record = load_participant_record(user_id)
    trial_record = merge_trial_record(
        record,
        trial_id,
        {
            "trial_spec": public_trial_spec(trial_id),
            "started_at": iso_now(),
        },
    )
    record["session_progress"] |= {
        "current_stage_id": spec["block_id"],
        "current_trial_id": trial_id,
    }
    store_participant_record(user_id, record)
    update_progress_file(user_id, stage_id=spec["block_id"], trial_id=trial_id)
    return jsonify(
        {
            "status": True,
            "trial": trial_record["trial_spec"],
            "runtime": runtime_payload,
            "environment_backend": "overcooked_new",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]
    pos = int(data_json["npc_index"])

    if trial_id not in TRIAL_SPECS:
        return jsonify({"status": False, "error": f"Unknown trial_id: {trial_id}"}), 404

    spec = TRIAL_SPECS[trial_id]
    if spec["mode"] == "replay":
        return jsonify({"status": False, "error": f"Replay trial {trial_id} has no AI actions"}), 400

    runtime = ensure_trial_runtime(user_id, trial_id)
    if pos not in runtime["ai_agents"]:
        return jsonify({"status": False, "error": f"No initialized AI for slot {pos} in trial {trial_id}"}), 400

    action = get_action(runtime["state"], runtime["ai_agents"][pos], pos)
    return jsonify({"status": True, "action": action, "environment_backend": "overcooked_new"})


@app.route("/step_trial", methods=["POST"])
def step_trial():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]

    if trial_id not in TRIAL_SPECS:
        return jsonify({"status": False, "error": f"Unknown trial_id: {trial_id}"}), 404

    spec = TRIAL_SPECS[trial_id]
    if spec["mode"] == "replay":
        return jsonify({"status": False, "error": f"Replay trial {trial_id} cannot be stepped"}), 400

    runtime = ensure_trial_runtime(user_id, trial_id)
    if runtime["pending_transition"] is not None:
        return jsonify({"status": False, "error": "A probe is already pending for this trial"}), 409

    joint_action_idx = [DEFAULT_ACTION_IDX, DEFAULT_ACTION_IDX]
    human_index = spec["human_player_index"]
    if human_index is not None:
        joint_action_idx[human_index] = action_idx_or_default(data_json.get("human_action_idx"))

    for agent_index, agent in runtime["ai_agents"].items():
        joint_action_idx[agent_index] = get_action(runtime["state"], agent, agent_index)

    joint_action = [ACTION_INDEX_TO_VALUE[idx] for idx in joint_action_idx]
    next_state, infos = runtime["mdp"].get_state_transition(runtime["state"], joint_action)
    reward = int(sum(infos["sparse_reward_by_agent"]))
    time_left = round(compute_runtime_time_left(runtime, trial_id), 2)
    target_subtask_id = infer_subtask_id(
        runtime["mdp"],
        runtime["state"],
        next_state,
        spec["target_agent_index"],
        joint_action_idx[spec["target_agent_index"]],
    )

    probe_slot = next_due_probe(runtime, spec)
    if probe_slot is not None:
        runtime["pending_transition"] = {
            "joint_action_idx": list(joint_action_idx),
            "next_state": next_state,
            "reward": reward,
            "target_action_idx": joint_action_idx[spec["target_agent_index"]],
            "target_subtask_id": target_subtask_id,
            "target_subtask_label": SUBTASK_OPTION_LOOKUP[target_subtask_id]["label"],
            "probe_index": probe_slot["index"],
            "probe_total": spec["probe"]["count"],
            "probe_game_loop": runtime["cur_gameloop"],
            "time_left": time_left,
            "prompt_timestamp": int(time.time() * 1000),
        }
        runtime["pause_started_at"] = time.time()
        return jsonify(
            {
                "status": True,
                "probe_pending": True,
                "probe": public_probe_payload(runtime["pending_transition"]),
                "runtime": build_runtime_payload(runtime, trial_id),
                "environment_backend": "overcooked_new",
            }
        )

    record_transition(runtime, joint_action_idx, reward, next_state)
    return jsonify(
        {
            "status": True,
            "probe_pending": False,
            "reward": reward,
            "joint_action_idx": joint_action_idx,
            "runtime": build_runtime_payload(runtime, trial_id),
            "done": compute_runtime_time_left(runtime, trial_id) <= 0,
            "environment_backend": "overcooked_new",
        }
    )


@app.route("/submit_probe", methods=["POST"])
def submit_probe():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]

    if trial_id not in TRIAL_SPECS:
        return jsonify({"status": False, "error": f"Unknown trial_id: {trial_id}"}), 404

    runtime = ensure_trial_runtime(user_id, trial_id)
    pending = runtime["pending_transition"]
    if pending is None:
        return jsonify({"status": False, "error": "No pending probe for this trial"}), 409

    selected_subtask_id = normalize_selected_subtask_id(data_json.get("selected_subtask_id"))
    actual_subtask_id = pending["target_subtask_id"]
    record = {
        "probe_index": pending["probe_index"],
        "probe_total": pending["probe_total"],
        "probe_game_loop": pending["probe_game_loop"],
        "prompt_timestamp": pending["prompt_timestamp"],
        "response_timestamp": int(time.time() * 1000),
        "selected_subtask_id": selected_subtask_id,
        "selected_subtask_label": SUBTASK_OPTION_LOOKUP[selected_subtask_id]["label"],
        "actual_subtask_id": actual_subtask_id,
        "actual_subtask_label": SUBTASK_OPTION_LOOKUP[actual_subtask_id]["label"],
        "correct": selected_subtask_id == actual_subtask_id,
        "target_agent_index": TRIAL_SPECS[trial_id]["target_agent_index"],
        "mode": TRIAL_SPECS[trial_id]["mode"],
    }
    extra_data = data_json.get("extra")
    if extra_data is not None:
        record["extra"] = extra_data

    runtime["probe_records"].append(record)
    pending["probe_record"] = deepcopy(record)

    return jsonify({"status": True, "probe_record": record, "environment_backend": "overcooked_new"})


@app.route("/resume_trial_after_probe", methods=["POST"])
def resume_trial_after_probe():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]

    if trial_id not in TRIAL_SPECS:
        return jsonify({"status": False, "error": f"Unknown trial_id: {trial_id}"}), 404

    runtime = ensure_trial_runtime(user_id, trial_id)
    pending = runtime["pending_transition"]
    if pending is None:
        return jsonify({"status": False, "error": "No pending probe for this trial"}), 409

    if runtime["pause_started_at"] is not None:
        runtime["paused_duration_s"] += time.time() - runtime["pause_started_at"]
        runtime["pause_started_at"] = None

    record_transition(runtime, pending["joint_action_idx"], pending["reward"], pending["next_state"])
    runtime["pending_transition"] = None

    return jsonify(
        {
            "status": True,
            "reward": pending["reward"],
            "joint_action_idx": pending["joint_action_idx"],
            "runtime": build_runtime_payload(runtime, trial_id),
            "done": compute_runtime_time_left(runtime, trial_id) <= 0,
            "environment_backend": "overcooked_new",
        }
    )


@app.route("/save_trial_data", methods=["POST"])
def save_trial_data():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]
    updates = data_json.get("updates", {})

    record = load_participant_record(user_id)
    merge_trial_record(record, trial_id, updates)
    progress = data_json.get("progress")
    if progress:
        record["session_progress"] |= progress
        update_progress_file(user_id, stage_id=progress.get("current_stage_id"), trial_id=trial_id)
    store_participant_record(user_id, record)
    return jsonify({"status": True})


@app.route("/finish_episode", methods=["POST"])
def finish_episode():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]
    traj_id = data_json["traj_id"].replace(":", "_")

    runtime = USER_TRIAL_RUNTIMES.get(user_id, {}).get(trial_id)
    runtime_summary = build_trial_summary_from_runtime(runtime) if runtime is not None else {}
    trial_summary = runtime_summary | data_json.get("summary", {})
    traj_dict = deepcopy(runtime["trajectory"]) if runtime is not None else data_json.get("traj")

    trajectory_path = None
    if traj_dict:
        save_dir = Path(ARGS.trajs_save_path) / user_id
        save_dir.mkdir(parents=True, exist_ok=True)
        traj_path = save_dir / f"{trial_id}_{traj_id}.json"
        with open(traj_path, "w", encoding="utf-8") as handle:
            json.dump(traj_dict, handle)
        trajectory_path = str(traj_path)

    record = load_participant_record(user_id)
    merge_trial_record(
        record,
        trial_id,
        {
            "finished_at": iso_now(),
            "trajectory_path": trajectory_path,
            "mode": TRIAL_SPECS[trial_id]["mode"],
            "traj_id": traj_id,
            "trial_summary": trial_summary,
        },
    )
    store_participant_record(user_id, record)
    cleanup_trial_runtime(user_id, trial_id)

    return jsonify(
        {
            "status": True,
            "trajectory_path": trajectory_path,
            "trajectory": traj_dict,
            "trial_summary": trial_summary,
            "environment_backend": "overcooked_new",
        }
    )


@app.route("/load_trial_trajectory", methods=["POST"])
def load_trial_trajectory():
    data_json = json.loads(request.data)
    user_id = make_user_id(data_json["user_info"])
    trial_id = data_json["trial_id"]
    record = load_participant_record(user_id)
    trial_record = record.get("trial_data", {}).get(trial_id, {})
    trajectory_path = trial_record.get("trajectory_path")
    if not trajectory_path or not Path(trajectory_path).exists():
        return jsonify({"status": False, "error": f"No saved trajectory for {trial_id}"}), 404

    with open(trajectory_path, encoding="utf-8") as handle:
        trajectory = json.load(handle)
    return jsonify({"status": True, "trajectory": trajectory, "trial_record": trial_record})


def main(args: argparse.Namespace):
    global ARGS, AGENT_POOL

    ARGS = args

    debug = os.getenv("FLASK_ENV", "development") == "development"
    if not debug:
        import logging
        import sys

        logger.info("Production Mode")
        logging.getLogger("werkzeug").disabled = True
        app.logger.disabled = True
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.add(os.path.join(os.path.dirname(ARGS.progress_save_path), "loguru.log"), level="INFO")

    logger.info("Args:\n" + pformat(ARGS.__dict__))
    logger.info("Session config summary:\n" + pformat(SESSION_CONFIG["checkpoint_summary"]))

    Path(ARGS.questionnaire_save_path).mkdir(parents=True, exist_ok=True)
    Path(ARGS.trajs_save_path).mkdir(parents=True, exist_ok=True)
    Path(ARGS.progress_save_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(ARGS.progress_save_path).exists():
        save_json(Path(ARGS.progress_save_path), {"participants": {}})

    t0 = time.time()
    AGENT_POOL = CheckpointAgentPool(
        SESSION_LAYOUT,
        CHECKPOINT_SPECS,
        deterministic=True,
        epsilon=0.0,
    )
    logger.info(f"Loaded {len(CHECKPOINT_SPECS)} EEG checkpoints in {time.time() - t0:.2f}s")


app_args = get_args().parse_args()
main(app_args)

if __name__ == "__main__":
    host = ARGS.ip
    port = ARGS.port
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(debug=debug, host=host, port=port, threaded=True)

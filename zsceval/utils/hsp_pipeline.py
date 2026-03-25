from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from zsceval.overcooked_config import OLD_LAYOUTS

DEFAULT_LAYOUTS: List[str] = [
    "random0",
    "random0_medium",
    "random1",
    "random3",
    "small_corridor",
    "unident_s",
    "random0_m",
    "random1_m",
    "random3_m",
    "academy_3_vs_1_with_keeper",
    "ttt",
    "tto",
    "too",
    "ooo",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scripts_root() -> Path:
    return repo_root() / "zsceval" / "scripts"


def resolve_policy_pool_root(policy_pool_path: str | None = None) -> Path:
    if policy_pool_path:
        return Path(policy_pool_path).expanduser().resolve()
    return (scripts_root() / "policy_pool").resolve()


def resolve_results_root(results_root: str | None = None) -> Path:
    if results_root:
        return Path(results_root).expanduser().resolve()
    return (repo_root() / "results").resolve()


def resolve_overcooked_eval_results_root(eval_result_root: str | None = None) -> Path:
    if eval_result_root:
        return Path(eval_result_root).expanduser().resolve()
    return (scripts_root() / "overcooked" / "eval" / "results").resolve()


def normalize_layout_argument(layout: str) -> list[str]:
    if layout == "all":
        return list(DEFAULT_LAYOUTS)
    return [layout]


def overcooked_version_for_layout(layout: str) -> str:
    return "old" if layout in OLD_LAYOUTS else "new"


def wandb_projects_for_layout(env: str, layout: str) -> list[str]:
    if env.lower() != "overcooked":
        return [env]
    if overcooked_version_for_layout(layout) == "new":
        return [f"{env}-new", env]
    return [env, f"{env}-new"]


def overcooked_event_types(layout: str) -> list[str]:
    if overcooked_version_for_layout(layout) == "old":
        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
    else:
        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS

    return list(SHAPED_INFOS) + ["sparse_r"]


def default_wandb_name() -> str:
    return os.environ.get("WANDB_NAME", "juliejung98")

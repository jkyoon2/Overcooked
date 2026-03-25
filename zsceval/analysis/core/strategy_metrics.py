from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from zsceval.analysis.core.strategy_laber import decode_w_id
from zsceval.analysis.core.subtask_labeler import SUBTASK_ID


@dataclass(frozen=True)
class StrategyDataset:
    transitions: pd.DataFrame
    episodes: pd.DataFrame
    manifest: Dict


def load_strategy_dataset(dataset_dir: str | Path) -> StrategyDataset:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    transitions_path = dataset_dir / "transitions.pkl"
    episodes_path = dataset_dir / "episodes.pkl"
    manifest_path = dataset_dir / "manifest.json"

    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing {transitions_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    with transitions_path.open("rb") as f:
        transitions = pickle.load(f)
    with episodes_path.open("rb") as f:
        episodes = pickle.load(f)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    return StrategyDataset(
        transitions=pd.DataFrame(transitions),
        episodes=pd.DataFrame(episodes),
        manifest=manifest,
    )


def aggregate_pair_rewards(
    dataset: StrategyDataset,
    reward_key: str = "episode_return",
    agent_id: Optional[int] = None,
) -> pd.DataFrame:
    if reward_key not in {"episode_return", "sparse_return", "shaped_return"}:
        raise ValueError(f"Unsupported reward_key: {reward_key}")

    df = dataset.episodes.copy()
    if agent_id is not None:
        df = df[df["agent_id"] == agent_id]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ego_policy_id",
                "partner_policy_id",
                "reward_mean",
                "reward_std",
                "success_rate",
                "num_agent_episodes",
            ]
        )

    grouped = (
        df.groupby(["ego_policy_id", "partner_policy_id"], as_index=False)
        .agg(
            reward_mean=(reward_key, "mean"),
            reward_std=(reward_key, "std"),
            success_rate=("success", "mean"),
            num_agent_episodes=("episode_uid", "count"),
        )
        .fillna({"reward_std": 0.0})
    )
    return grouped


def make_reward_heatmap_matrix(
    pair_rewards: pd.DataFrame,
    value_col: str = "reward_mean",
) -> pd.DataFrame:
    if pair_rewards.empty:
        return pd.DataFrame()
    if value_col not in pair_rewards.columns:
        raise ValueError(f"Unknown value_col: {value_col}")

    matrix = pair_rewards.pivot(index="ego_policy_id", columns="partner_policy_id", values=value_col)
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix


def pair_distributions(
    dataset: StrategyDataset,
    ego_policy_id: str,
    partner_policy_id: str,
    agent_id: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    trans = dataset.transitions
    subset = trans[(trans["ego_policy_id"] == ego_policy_id) & (trans["partner_policy_id"] == partner_policy_id)]
    if agent_id is not None:
        subset = subset[subset["agent_id"] == agent_id]

    z_id_to_name = {v: k for k, v in SUBTASK_ID.items()}

    def _dist(col: str, normalize: bool = True) -> pd.DataFrame:
        if subset.empty or col not in subset.columns:
            return pd.DataFrame(columns=[col, "label", "count", "prob"])
        vc = subset[col].value_counts(dropna=False)
        dist = vc.rename_axis(col).reset_index(name="count")
        total = max(int(dist["count"].sum()), 1)
        dist["prob"] = dist["count"] / total if normalize else dist["count"]
        if col in {"w_id", "w_fixed_id", "w_posthoc_id"}:
            dist["label"] = dist[col].map(_safe_decode_w_id)
        elif col == "z_id":
            dist["label"] = dist[col].map(lambda x: z_id_to_name.get(int(x), f"unknown_{x}"))
        else:
            dist["label"] = dist[col].astype(str)
        return dist

    return {
        "w_id": _dist("w_id"),
        "w_fixed_id": _dist("w_fixed_id"),
        "w_posthoc_id": _dist("w_posthoc_id"),
        "z_id": _dist("z_id"),
    }


def _safe_decode_w_id(v) -> str:
    try:
        recipe, role = decode_w_id(int(v))
        return f"{recipe}+{role}"
    except Exception:
        return f"unknown_{v}"

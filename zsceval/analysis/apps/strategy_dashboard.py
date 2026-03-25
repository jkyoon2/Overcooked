from __future__ import annotations

"""
Strategy Rollout Dashboard (minimal)

Minimal implementation spec
1) Input:
   - dataset directory containing transitions.pkl, episodes.pkl, manifest.json
2) Main views:
   - Pair reward heatmap (ego_policy_id x partner_policy_id)
   - Pair detail panel:
     - episode-level summary (reward/success)
     - per-agent w/z distributions
3) Pair selection:
   - preferred: click heatmap cell (if streamlit-plotly-events installed)
   - fallback: selectboxes (ego_policy_id, partner_policy_id)
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from zsceval.analysis.core.strategy_metrics import (
    aggregate_pair_rewards,
    load_strategy_dataset,
    make_reward_heatmap_matrix,
    pair_distributions,
)


def main() -> None:
    st.set_page_config(page_title="Strategy Dataset Dashboard", layout="wide")
    st.title("Strategy Dataset Dashboard")

    default_dataset_dir = str((Path(__file__).resolve().parents[1] / "data" / "strategy_datasets").resolve())
    dataset_dir = st.sidebar.text_input("Dataset Dir", value=default_dataset_dir)
    reward_key = st.sidebar.selectbox("Reward Metric", ["episode_return", "sparse_return", "shaped_return"], index=0)
    heatmap_value = st.sidebar.selectbox("Heatmap Value", ["reward_mean", "success_rate"], index=0)
    use_click_select = st.sidebar.checkbox("Use Click Selection (Experimental)", value=False)
    selected_agent_id = st.sidebar.selectbox("Agent Filter", ["both", "agent0", "agent1"], index=0)

    if selected_agent_id == "both":
        agent_id: Optional[int] = None
    elif selected_agent_id == "agent0":
        agent_id = 0
    else:
        agent_id = 1

    resolved_dir = resolve_dataset_dir(dataset_dir)
    if resolved_dir is None:
        st.error(
            "No dataset found. Expected a directory containing "
            "`transitions.pkl`, `episodes.pkl`, `manifest.json`."
        )
        st.stop()

    st.sidebar.text(f"Resolved Dataset: {resolved_dir}")

    try:
        dataset = load_strategy_dataset(resolved_dir)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    st.sidebar.markdown("### Manifest")
    st.sidebar.json(dataset.manifest)
    with st.sidebar.expander("Column Definitions", expanded=False):
        st.markdown("**Transitions**")
        st.json(dataset.manifest.get("transition_column_desc", {}))
        st.markdown("**Episodes**")
        st.json(dataset.manifest.get("episode_column_desc", {}))
        st.markdown("**w-id**")
        st.write(dataset.manifest.get("w_id_definition", "w_id mapping not specified"))
        st.write(dataset.manifest.get("w_fixed_id_definition", ""))
        st.write(dataset.manifest.get("w_posthoc_id_definition", ""))

    pair_rewards = aggregate_pair_rewards(dataset, reward_key=reward_key, agent_id=agent_id)
    if pair_rewards.empty:
        st.warning("No pair-level episode records found.")
        st.stop()

    matrix = make_reward_heatmap_matrix(pair_rewards, value_col=heatmap_value)
    st.subheader("Pair Reward Heatmap")
    st.caption("Rows: ego policy, Columns: partner policy")
    fig = px.imshow(
        matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "partner_policy_id", "y": "ego_policy_id", "color": heatmap_value},
    )
    fig.update_xaxes(side="top")
    pair_from_click = render_heatmap(fig, use_click_select=use_click_select)

    all_ego = sorted(pair_rewards["ego_policy_id"].unique().tolist())
    all_partner = sorted(pair_rewards["partner_policy_id"].unique().tolist())

    st.subheader("Pair Detail")
    default_ego, default_partner = infer_default_pair(pair_rewards)
    if pair_from_click is not None:
        default_ego, default_partner = pair_from_click

    col1, col2 = st.columns(2)
    with col1:
        ego_policy_id = st.selectbox("Ego Policy", all_ego, index=max(all_ego.index(default_ego), 0))
    with col2:
        partner_policy_id = st.selectbox("Partner Policy", all_partner, index=max(all_partner.index(default_partner), 0))

    render_pair_detail(dataset.episodes, dataset, ego_policy_id, partner_policy_id)


def render_heatmap(fig, use_click_select: bool) -> Optional[Tuple[str, str]]:
    if not use_click_select:
        st.plotly_chart(fig, use_container_width=True)
        return None

    """
    Returns (ego_policy_id, partner_policy_id) if click event is available.
    Falls back to plain chart when click plugin is missing.
    """
    try:
        from streamlit_plotly_events import plotly_events

        selected = plotly_events(fig, click_event=True, select_event=False, hover_event=False)
        if selected:
            point = selected[0]
            # In px.imshow, x=column label(partner), y=row label(ego)
            partner = str(point["x"])
            ego = str(point["y"])
            return ego, partner
        return None
    except Exception:
        st.info("Click selection unavailable. Using selectboxes fallback.")
        st.plotly_chart(fig, use_container_width=True)
        return None


def resolve_dataset_dir(dataset_dir: str) -> Optional[str]:
    """
    Accept either:
    - exact dataset dir (.../foo/transitions.pkl)
    - root dir containing one or more dataset subdirs
    """
    p = Path(dataset_dir).expanduser().resolve()
    if _has_dataset_files(p):
        return str(p)

    if not p.exists() or not p.is_dir():
        return None

    candidates = sorted([d for d in p.iterdir() if d.is_dir() and _has_dataset_files(d)])
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        # Deterministic fallback: choose most recently modified dataset dir.
        latest = max(candidates, key=lambda d: d.stat().st_mtime)
        return str(latest)
    return None


def _has_dataset_files(path: Path) -> bool:
    return (path / "transitions.pkl").exists() and (path / "episodes.pkl").exists() and (path / "manifest.json").exists()


def infer_default_pair(pair_rewards: pd.DataFrame) -> Tuple[str, str]:
    top = pair_rewards.sort_values("reward_mean", ascending=False).iloc[0]
    return str(top["ego_policy_id"]), str(top["partner_policy_id"])


def render_pair_detail(
    episodes_df: pd.DataFrame,
    dataset,
    ego_policy_id: str,
    partner_policy_id: str,
) -> None:
    subset = episodes_df[
        (episodes_df["ego_policy_id"] == ego_policy_id) & (episodes_df["partner_policy_id"] == partner_policy_id)
    ]
    if subset.empty:
        st.warning("No episodes for selected pair.")
        return

    summary = {
        "num_agent_episodes": int(len(subset)),
        "reward_mean": float(subset["episode_return"].mean()),
        "reward_std": float(subset["episode_return"].std(ddof=0)),
        "success_rate": float(subset["success"].mean()),
        "sparse_mean": float(subset["sparse_return"].mean()),
        "shaped_mean": float(subset["shaped_return"].mean()),
    }
    st.json(summary)

    ep_horizon = dataset.manifest.get("episode_length", "unknown")
    st.caption(
        f"Tip: count equals total collected timesteps for selected pair/agent. "
        f"If you expected 400 but see 200, check `episode_length` (current: {ep_horizon})."
    )

    tabs = st.tabs(["Agent 0", "Agent 1"])
    for agent_id, tab in enumerate(tabs):
        with tab:
            dist = pair_distributions(dataset, ego_policy_id, partner_policy_id, agent_id=agent_id)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**w_id distribution**")
                st.plotly_chart(
                    px.bar(dist["w_id"], x="label", y="prob", text="count", title=f"Agent {agent_id} w_id"),
                    use_container_width=True,
                )
                st.dataframe(dist["w_id"], use_container_width=True)
            with c2:
                st.markdown("**z_id distribution**")
                st.plotly_chart(
                    px.bar(dist["z_id"], x="label", y="prob", text="count", title=f"Agent {agent_id} z_id"),
                    use_container_width=True,
                )
                st.dataframe(dist["z_id"], use_container_width=True)


if __name__ == "__main__":
    main()

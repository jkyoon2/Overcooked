from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wandb
from loguru import logger

from zsceval.utils.hsp_pipeline import (
    default_wandb_name,
    normalize_layout_argument,
    resolve_policy_pool_root,
    resolve_results_root,
    wandb_projects_for_layout,
)


@dataclass
class BiasRunCandidate:
    run: object
    history_steps: list[int]
    history_sparse: list[float]
    actor_versions: list[int]
    model_dir: Path
    seed_dir: Path

    @property
    def seed(self) -> int:
        return int(self.run.config["seed"])

    @property
    def num_agents(self) -> int:
        return int(self.run.config["num_agents"])

    @property
    def algorithm_name(self) -> str:
        return str(self.run.config["algorithm_name"])

    @property
    def max_step(self) -> int:
        return max(self.history_steps)

    @property
    def final_sparse(self) -> float:
        return float(np.mean(self.history_sparse[-5:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HSP S1 bias checkpoints into policy_pool.")
    parser.add_argument("layout", type=str, help="Layout name or 'all'.")
    parser.add_argument("env", type=str, help="Environment name, e.g. Overcooked.")
    parser.add_argument("--experiment_name", type=str, default="hsp-S1")
    parser.add_argument("--wandb_name", type=str, default=default_wandb_name())
    parser.add_argument("--policy_pool_path", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    return parser.parse_args()


def fetch_runs(api: wandb.Api, wandb_name: str, env: str, layout: str, exp: str) -> list[object]:
    if "overcooked" in env.lower():
        layout_config = "config.layout_name"
    else:
        layout_config = "config.scenario_name"

    filters = {
        "$and": [
            {"config.experiment_name": exp},
            {layout_config: layout},
            {"state": "finished"},
            {"tags": {"$nin": ["hidden", "unused"]}},
        ]
    }

    runs = []
    seen_ids = set()
    for project in wandb_projects_for_layout(env, layout):
        try:
            project_runs = api.runs(f"{wandb_name}/{project}", filters=filters, order="-created_at")
        except wandb.errors.CommError as exc:
            logger.warning(f"Skip project {project}: {exc}")
            continue
        for run in project_runs:
            if run.id not in seen_ids:
                seen_ids.add(run.id)
                runs.append(run)
    return runs


def compute_local_run_paths(results_root: Path, run: object) -> tuple[Path, Path]:
    policy_dir = "shared" if run.config.get("share_policy", True) else "separated"
    seed_dir = (
        results_root
        / run.config["env_name"]
        / run.config["layout_name"]
        / policy_dir
        / run.config["algorithm_name"]
        / run.config["experiment_name"]
        / f"seed{run.config['seed']}"
    )
    return seed_dir, seed_dir / "models"


def collect_actor_versions(model_dir: Path) -> list[int]:
    shared_versions = sorted(
        int(path.stem.split("_")[-1]) for path in model_dir.glob("actor_periodic_*.pt")
    )
    if shared_versions:
        return shared_versions

    agent_versions = sorted(
        int(path.stem.split("_")[-1]) for path in model_dir.glob("actor_agent0_periodic_*.pt")
    )
    if agent_versions:
        return agent_versions

    return []


def build_candidate(run: object, results_root: Path) -> BiasRunCandidate | None:
    history = run.history()
    if history.empty:
        logger.warning(f"Skip run {run.id}: empty history")
        return None

    if "_step" not in history.columns or "ep_sparse_r" not in history.columns:
        logger.warning(f"Skip run {run.id}: missing _step/ep_sparse_r columns")
        return None
    history = history[["_step", "ep_sparse_r"]].dropna()
    if history.empty:
        logger.warning(f"Skip run {run.id}: no sparse reward history")
        return None

    seed_dir, model_dir = compute_local_run_paths(results_root, run)
    if not model_dir.exists():
        logger.warning(f"Skip run {run.id}: missing model dir {model_dir}")
        return None

    actor_versions = collect_actor_versions(model_dir)
    if not actor_versions:
        logger.warning(f"Skip run {run.id}: no actor_periodic checkpoints in {model_dir}")
        return None

    history_steps = history["_step"].astype(int).tolist()
    history_sparse = history["ep_sparse_r"].astype(float).tolist()
    return BiasRunCandidate(
        run=run,
        history_steps=history_steps,
        history_sparse=history_sparse,
        actor_versions=actor_versions,
        model_dir=model_dir,
        seed_dir=seed_dir,
    )


def select_best_candidates(runs: list[object], results_root: Path) -> list[BiasRunCandidate]:
    grouped = defaultdict(list)
    for run in runs:
        candidate = build_candidate(run, results_root)
        if candidate is not None:
            grouped[candidate.seed].append(candidate)

    selected = []
    for seed, candidates in sorted(grouped.items()):
        best = max(candidates, key=lambda c: (c.max_step, c.final_sparse, len(c.actor_versions)))
        if len(candidates) > 1:
            logger.info(
                f"Seed {seed}: selected run {best.run.id} among {[c.run.id for c in candidates]}"
            )
        selected.append(best)
    return selected


def choose_versions(candidate: BiasRunCandidate) -> dict[str, int]:
    target_steps = {"mid": candidate.history_steps[0], "final": candidate.max_step}
    target_mid_sparse = candidate.final_sparse / 2.0
    best_delta = float("inf")
    for step, score in zip(candidate.history_steps, candidate.history_sparse):
        delta = abs(target_mid_sparse - score)
        if delta < best_delta:
            best_delta = delta
            target_steps["mid"] = step

    return {
        tag: min(candidate.actor_versions, key=lambda version: abs(version - target_step))
        for tag, target_step in target_steps.items()
    }


def resolve_checkpoint_sources(model_dir: Path, version: int, num_agents: int) -> list[Path]:
    shared_path = model_dir / f"actor_periodic_{version}.pt"
    if shared_path.exists():
        return [shared_path] * num_agents

    agent_paths = [model_dir / f"actor_agent{agent_id}_periodic_{version}.pt" for agent_id in range(num_agents)]
    if all(path.exists() for path in agent_paths):
        return agent_paths

    raise FileNotFoundError(f"Cannot resolve actor checkpoints for version {version} in {model_dir}")


def copy_policy_config(seed_dir: Path, policy_pool_root: Path, layout: str, algorithm_name: str) -> None:
    source = seed_dir / "policy_config.pkl"
    if not source.exists():
        raise FileNotFoundError(f"Missing policy config: {source}")

    config_name = "rnn_policy_config.pkl" if algorithm_name.startswith("r") else "mlp_policy_config.pkl"
    target = policy_pool_root / layout / "policy_config" / config_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    logger.info(f"Policy config stored in {target}")


def extract_layout(layout: str, env: str, exp: str, wandb_name: str, policy_pool_root: Path, results_root: Path) -> None:
    api = wandb.Api()
    runs = fetch_runs(api, wandb_name, env, layout, exp)
    logger.info(f"{layout}: fetched {len(runs)} finished runs")
    candidates = select_best_candidates(runs, results_root)
    if not candidates:
        raise RuntimeError(f"No usable finished runs found for layout={layout}, exp={exp}")

    hsp_s1_dir = policy_pool_root / layout / "hsp" / "s1" / exp.replace("-S1", "")
    hsp_s1_dir.mkdir(parents=True, exist_ok=True)

    copied_policy_config = False
    for candidate in candidates:
        versions = choose_versions(candidate)
        logger.info(
            f"hsp{candidate.seed}: run={candidate.run.id}, final_sparse={candidate.final_sparse:.3f}, versions={versions}"
        )
        if not copied_policy_config:
            copy_policy_config(candidate.seed_dir, policy_pool_root, layout, candidate.algorithm_name)
            copied_policy_config = True

        for tag, version in versions.items():
            sources = resolve_checkpoint_sources(candidate.model_dir, version, candidate.num_agents)
            for agent_id, source in enumerate(sources):
                target = hsp_s1_dir / f"hsp{candidate.seed}_{tag}_w{agent_id}_actor.pt"
                shutil.copy2(source, target)
                logger.info(f"Stored {target}")

    logger.success(f"Extracted {len(candidates)} bias policies for {layout}")


def main() -> None:
    args = parse_args()
    policy_pool_root = resolve_policy_pool_root(args.policy_pool_path)
    results_root = resolve_results_root(args.results_root)

    for layout in normalize_layout_argument(args.layout):
        extract_layout(
            layout=layout,
            env=args.env,
            exp=args.experiment_name,
            wandb_name=args.wandb_name,
            policy_pool_root=policy_pool_root,
            results_root=results_root,
        )


if __name__ == "__main__":
    main()

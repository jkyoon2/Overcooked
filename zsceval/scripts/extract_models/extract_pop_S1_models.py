from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from zsceval.utils.hsp_pipeline import normalize_layout_argument, resolve_policy_pool_root, resolve_results_root


@dataclass(frozen=True)
class PolicyCheckpointSelection:
    init: int
    mid: int
    final: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MEP/TRAJ stage-1 population checkpoints into policy_pool.")
    parser.add_argument("layout", type=str, help="Layout name or 'all'.")
    parser.add_argument("env", type=str, help="Environment name, e.g. Overcooked.")
    parser.add_argument("--algo", type=str, default="mep", choices=["mep", "traj"])
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--policy_mode", type=str, default="shared", choices=["shared", "separated"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy_pool_path", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    return parser.parse_args()


def collect_actor_versions(model_dir: Path) -> list[int]:
    versions = []
    for actor_path in model_dir.glob("actor_periodic_*.pt"):
        try:
            versions.append(int(actor_path.stem.split("_")[-1]))
        except ValueError:
            logger.warning(f"Skip malformed checkpoint name: {actor_path.name}")
    return sorted(set(versions))


def choose_versions(actor_versions: list[int]) -> PolicyCheckpointSelection:
    if not actor_versions:
        raise ValueError("actor_versions must not be empty")
    init = actor_versions[0]
    final = actor_versions[-1]
    mid_target = final / 2.0
    mid = min(actor_versions, key=lambda version: abs(version - mid_target))
    return PolicyCheckpointSelection(init=init, mid=mid, final=final)


def discover_policy_ids(model_root: Path, algo: str, population_size: int | None) -> list[int]:
    matched_ids = []
    pattern = re.compile(rf"^{re.escape(algo)}(\d+)$")
    for child in model_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            matched_ids.append(int(match.group(1)))
    if matched_ids:
        return sorted(set(matched_ids))
    if population_size is not None:
        return list(range(1, population_size + 1))
    raise RuntimeError(f"Could not infer policy ids under {model_root}; pass --population_size.")


def copy_selected_checkpoints(
    *,
    layout: str,
    algo: str,
    experiment_name: str,
    model_root: Path,
    output_root: Path,
    policy_ids: list[int],
) -> None:
    target_root = output_root / layout / algo / "s1" / experiment_name
    target_root.mkdir(parents=True, exist_ok=True)

    for policy_id in policy_ids:
        policy_name = f"{algo}{policy_id}"
        policy_dir = model_root / policy_name
        if not policy_dir.exists():
            raise FileNotFoundError(f"Missing policy directory: {policy_dir}")

        actor_versions = collect_actor_versions(policy_dir)
        if not actor_versions:
            raise FileNotFoundError(f"No actor_periodic checkpoints found in {policy_dir}")

        selected = choose_versions(actor_versions)
        for tag, version in selected.__dict__.items():
            source = policy_dir / f"actor_periodic_{version}.pt"
            target = target_root / f"{policy_name}_{tag}_actor.pt"
            shutil.copy2(source, target)
            logger.info(f"{policy_name}: {tag} -> {source.name} -> {target}")


def extract_layout(
    *,
    layout: str,
    env: str,
    algo: str,
    experiment_name: str,
    policy_mode: str,
    seed: int,
    population_size: int | None,
    results_root: Path,
    policy_pool_root: Path,
) -> None:
    run_dir = results_root / env / layout / policy_mode / algo / experiment_name / f"seed{seed}"
    model_root = run_dir / "models"
    if not model_root.exists():
        raise FileNotFoundError(f"Missing model directory: {model_root}")

    policy_ids = discover_policy_ids(model_root, algo, population_size)
    logger.info(f"{layout}: extracting {algo} S1 policies {policy_ids} from {model_root}")
    copy_selected_checkpoints(
        layout=layout,
        algo=algo,
        experiment_name=experiment_name,
        model_root=model_root,
        output_root=policy_pool_root,
        policy_ids=policy_ids,
    )
    logger.success(f"Extracted {len(policy_ids)} {algo.upper()} stage-1 policies for {layout}")


def main() -> None:
    args = parse_args()
    policy_pool_root = resolve_policy_pool_root(args.policy_pool_path)
    results_root = resolve_results_root(args.results_root)
    if args.experiment_name is None and args.population_size is None:
        raise ValueError("Provide either --experiment_name or --population_size.")
    experiment_name = args.experiment_name or f"{args.algo}-S1-s{args.population_size}"

    for layout in normalize_layout_argument(args.layout):
        extract_layout(
            layout=layout,
            env=args.env,
            algo=args.algo,
            experiment_name=experiment_name,
            policy_mode=args.policy_mode,
            seed=args.seed,
            population_size=args.population_size,
            results_root=results_root,
            policy_pool_root=policy_pool_root,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml

from zsceval.runner.shared.base_runner import make_trainer_policy_cls
from zsceval.utils.hsp_pipeline import resolve_policy_pool_root
from zsceval.utils.train_util import get_base_run_dir


POPULATION_ALGOS = {"population", "mep", "adaptive", "cole", "traj"}
POLICY_ARG_OVERRIDE_FIELDS = (
    "use_agent_policy_id",
    "num_v_out",
    "use_task_v_out",
    "use_policy_vhead",
    "use_proper_time_limits",
    "entropy_coefs",
    "entropy_coef_horizons",
    "use_peb",
    "data_parallel",
)


@dataclass(frozen=True)
class CheckpointPaths:
    actor: Optional[Path] = None
    critic: Optional[Path] = None
    model: Optional[Path] = None
    step: Optional[int] = None


@dataclass(frozen=True)
class PolicyLoadContext:
    policy_args: object
    actual_algorithm_name: str
    models_dir: Path


def build_models_dir(
    layout: str,
    algo: str,
    experiment: str,
    seed: int,
    env_name: str = "Overcooked",
    results_root: Optional[str] = None,
    policy_type: Optional[str] = None,
    role: Optional[str] = None,
) -> Path:
    base_dir = Path(results_root) if results_root else Path(get_base_run_dir())
    candidates = _candidate_models_dirs(
        base_dir=base_dir,
        env_name=env_name,
        layout=layout,
        algo=algo,
        experiment=experiment,
        seed=seed,
        policy_type=policy_type,
    )

    for models_dir in candidates:
        if not models_dir.exists():
            continue
        if role:
            role_dir = models_dir / role
            if role_dir.exists():
                return role_dir
        return models_dir

    fallback_dir = candidates[0]
    return fallback_dir / role if role else fallback_dir


def load_agent(
    all_args,
    envs,
    layout: str,
    algo: str,
    experiment: str,
    seed: int,
    step: Optional[int] = None,
    agent_id: Optional[int] = None,
    share_policy: bool = True,
    results_root: Optional[str] = None,
    device: Optional[torch.device] = None,
    policy_type: Optional[str] = None,
    role: Optional[str] = None,
    load_critic: bool = True,
):
    models_dir = build_models_dir(
        layout=layout,
        algo=algo,
        experiment=experiment,
        seed=seed,
        env_name=all_args.env_name,
        results_root=results_root,
        policy_type=policy_type,
        role=role,
    )
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")

    context = _resolve_policy_load_context(
        all_args=all_args,
        envs=envs,
        layout=layout,
        algo=algo,
        experiment=experiment,
        seed=seed,
        models_dir=models_dir,
    )
    policy = _build_policy(context.policy_args, envs, agent_id=agent_id, share_policy=share_policy, device=device)
    ckpt = _resolve_checkpoint_paths(
        models_dir=context.models_dir,
        step=step,
        agent_id=agent_id,
        share_policy=share_policy,
        use_single_network=getattr(context.policy_args, "use_single_network", False),
        require_critic=load_critic,
    )
    _load_weights(policy, ckpt, load_critic=load_critic)
    return policy, ckpt.step


def list_checkpoints(
    layout: str,
    algo: str,
    experiment: str,
    seed: int,
    agent_id: Optional[int] = None,
    share_policy: bool = True,
    env_name: str = "Overcooked",
    results_root: Optional[str] = None,
    policy_type: Optional[str] = None,
    role: Optional[str] = None,
) -> list[int]:
    models_dir = build_models_dir(
        layout=layout,
        algo=algo,
        experiment=experiment,
        seed=seed,
        env_name=env_name,
        results_root=results_root,
        policy_type=policy_type,
        role=role,
    )
    steps = []
    for prefix in _checkpoint_prefixes("actor", agent_id=agent_id, share_policy=share_policy):
        for path in models_dir.glob(f"{prefix}_periodic_*.pt"):
            step = _parse_step(path, prefix)
            if step is not None:
                steps.append(step)
    return sorted(set(steps))


def _build_policy(all_args, envs, agent_id: Optional[int], share_policy: bool, device: Optional[torch.device]):
    device = device or torch.device("cpu")
    if share_policy:
        obs_space = envs.observation_space[0]
        share_obs_space = envs.share_observation_space[0]
        act_space = envs.action_space[0]
    else:
        if agent_id is None:
            raise ValueError("agent_id is required when share_policy is False.")
        obs_space = envs.observation_space[agent_id]
        share_obs_space = envs.share_observation_space[agent_id]
        act_space = envs.action_space[agent_id]

    try:
        _, Policy = make_trainer_policy_cls(
            all_args.algorithm_name,
            use_single_network=getattr(all_args, "use_single_network", False),
        )
    except NotImplementedError:
        from zsceval.runner.separated.base_runner import make_trainer_policy_cls as make_trainer_policy_cls_sep

        _, Policy = make_trainer_policy_cls_sep(
            all_args.algorithm_name,
            use_single_network=getattr(all_args, "use_single_network", False),
        )
    policy = Policy(all_args, obs_space, share_obs_space, act_space, device=device)
    policy.to(device)
    return policy


def _resolve_checkpoint_paths(
    models_dir: Path,
    step: Optional[int],
    agent_id: Optional[int],
    share_policy: bool,
    use_single_network: bool,
    require_critic: bool,
) -> CheckpointPaths:
    if use_single_network:
        prefixes = _checkpoint_prefixes("model", agent_id=agent_id, share_policy=share_policy)
        model_path, resolved_step, _ = _find_checkpoint(models_dir, prefixes, step)
        return CheckpointPaths(model=model_path, step=resolved_step)

    actor_prefixes = _checkpoint_prefixes("actor", agent_id=agent_id, share_policy=share_policy)
    actor_path, resolved_step, resolved_actor_prefix = _find_checkpoint(models_dir, actor_prefixes, step)

    critic_prefixes = _checkpoint_prefixes("critic", agent_id=agent_id, share_policy=share_policy)
    if resolved_actor_prefix is not None:
        preferred = resolved_actor_prefix.replace("actor", "critic", 1)
        if preferred in critic_prefixes:
            critic_prefixes = [preferred] + [p for p in critic_prefixes if p != preferred]
    critic_path = None
    if require_critic:
        critic_path, _, _ = _find_checkpoint(models_dir, critic_prefixes, resolved_step)
    return CheckpointPaths(actor=actor_path, critic=critic_path, step=resolved_step)


def _checkpoint_prefixes(base: str, agent_id: Optional[int], share_policy: bool) -> List[str]:
    if share_policy or agent_id is None:
        return [base]
    # 1) Legacy separated ckpt naming: actor_agent0_periodic_*
    # 2) Role-specific separated ckpt naming: models/<role>/actor_periodic_*
    return [f"{base}_agent{agent_id}", base]


def _find_checkpoint(models_dir: Path, prefixes: List[str], step: Optional[int]) -> Tuple[Path, Optional[int], Optional[str]]:
    if step is not None:
        for prefix in prefixes:
            explicit = models_dir / f"{prefix}_periodic_{step}.pt"
            if explicit.exists():
                return explicit, step, prefix
            fallback = models_dir / f"{prefix}.pt"
            if fallback.exists():
                return fallback, step, prefix
        tried = ", ".join(str(models_dir / f"{p}_periodic_{step}.pt") for p in prefixes)
        raise FileNotFoundError(f"Checkpoint not found. Tried: {tried}")

    scored_candidates = []
    for prefix in prefixes:
        candidates = list(models_dir.glob(f"{prefix}_periodic_*.pt"))
        if not candidates:
            continue
        steps = [(c, _parse_step(c, prefix), prefix) for c in candidates]
        steps = [(c, s, p) for c, s, p in steps if s is not None]
        scored_candidates.extend(steps)
    if scored_candidates:
        best_path, best_step, best_prefix = max(scored_candidates, key=lambda item: item[1])
        return best_path, best_step, best_prefix

    for prefix in prefixes:
        fallback = models_dir / f"{prefix}.pt"
        if fallback.exists():
            return fallback, None, prefix
    raise FileNotFoundError(f"No checkpoints found in {models_dir} for prefixes {prefixes}")


def _parse_step(path: Path, prefix: str) -> Optional[int]:
    stem = path.stem
    if not stem.startswith(prefix + "_periodic_"):
        return None
    step_str = stem.replace(prefix + "_periodic_", "", 1)
    try:
        return int(step_str)
    except ValueError:
        return None


def _load_weights(policy, ckpt: CheckpointPaths, load_critic: bool = True) -> None:
    if ckpt.model is not None:
        if not hasattr(policy, "model"):
            raise ValueError("Policy has no 'model' attribute for single-network checkpoint.")
        policy.model.load_state_dict(torch.load(ckpt.model, map_location=policy.device))
        return
    if ckpt.actor is None:
        raise ValueError("Actor checkpoint is required.")
    policy.actor.load_state_dict(torch.load(ckpt.actor, map_location=policy.device))
    if load_critic and ckpt.critic is not None and hasattr(policy, "critic"):
        policy.critic.load_state_dict(torch.load(ckpt.critic, map_location=policy.device))


def _resolve_policy_load_context(
    all_args,
    envs,
    layout: str,
    algo: str,
    experiment: str,
    seed: int,
    models_dir: Path,
) -> PolicyLoadContext:
    if algo not in POPULATION_ALGOS:
        return PolicyLoadContext(
            policy_args=all_args,
            actual_algorithm_name=all_args.algorithm_name,
            models_dir=_resolve_checkpoint_dir(models_dir, preferred_subdir=None),
        )

    population_yaml_path = _infer_population_yaml_path(all_args, layout, experiment, seed)
    trainer_name = None
    policy_config_path = None
    if population_yaml_path is not None:
        trainer_name, policy_config_path = _read_population_entry(population_yaml_path)

    if policy_config_path is None:
        policy_config_path = _fallback_policy_config_path(all_args, layout)
    if policy_config_path is None:
        raise FileNotFoundError(
            f"Could not resolve underlying policy config for population algorithm '{algo}' "
            f"(layout={layout}, experiment={experiment}, seed={seed})."
        )

    policy_args = _load_actual_policy_args(policy_config_path, runtime_args=all_args)
    resolved_models_dir = _resolve_checkpoint_dir(models_dir, preferred_subdir=trainer_name)
    return PolicyLoadContext(
        policy_args=policy_args,
        actual_algorithm_name=policy_args.algorithm_name,
        models_dir=resolved_models_dir,
    )


def _infer_population_yaml_path(all_args, layout: str, experiment: str, seed: int) -> Optional[Path]:
    explicit = getattr(all_args, "population_yaml_path", None)
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.exists():
            return explicit_path

    policy_pool_root = _resolve_policy_pool_root(all_args)
    layout_root = policy_pool_root / layout
    if not layout_root.exists():
        return None

    family = experiment.split("-", 1)[0].lower()
    population_size = _extract_population_size(experiment)

    candidates = []
    if population_size is not None:
        candidates.extend(layout_root.glob(f"**/s2/train-s{population_size}-*-{seed}.yml"))
    if family:
        family_root = layout_root / family
        if family_root.exists():
            candidates.extend(family_root.glob(f"**/s2/*-{seed}.yml"))
    candidates.extend(layout_root.glob(f"**/s2/*-{seed}.yml"))

    deduped = sorted({path.resolve() for path in candidates})
    return deduped[0] if deduped else None


def _read_population_entry(population_yaml_path: Path) -> Tuple[Optional[str], Optional[Path]]:
    population_config = yaml.load(population_yaml_path.open("r", encoding="utf-8"), Loader=yaml.Loader) or {}
    if not population_config:
        return None, None

    trainer_name = next((name for name, cfg in population_config.items() if cfg.get("train")), None)
    if trainer_name is None:
        trainer_name = next(iter(population_config))

    policy_config_rel = population_config[trainer_name].get("policy_config_path")
    if not policy_config_rel:
        return trainer_name, None

    policy_pool_root = _resolve_policy_pool_root_from_yaml(population_yaml_path)
    policy_config_path = (policy_pool_root / policy_config_rel).resolve()
    return trainer_name, policy_config_path


def _resolve_policy_pool_root(all_args) -> Path:
    policy_pool_path = getattr(all_args, "policy_pool_path", None) or os.environ.get("POLICY_POOL")
    return resolve_policy_pool_root(policy_pool_path)


def _resolve_policy_pool_root_from_yaml(population_yaml_path: Path) -> Path:
    parts = population_yaml_path.resolve().parts
    if "policy_pool" not in parts:
        return population_yaml_path.parent
    idx = parts.index("policy_pool")
    return Path(*parts[: idx + 1])


def _fallback_policy_config_path(all_args, layout: str) -> Optional[Path]:
    policy_pool_root = _resolve_policy_pool_root(all_args)
    config_dir = policy_pool_root / layout / "policy_config"
    if not config_dir.exists():
        return None

    recurrent_flag = bool(getattr(all_args, "use_recurrent_policy", False) or getattr(all_args, "use_naive_recurrent_policy", False))
    preferred = ["rnn_policy_config.pkl", "mlp_policy_config.pkl"] if recurrent_flag else ["mlp_policy_config.pkl", "rnn_policy_config.pkl"]
    for name in preferred:
        path = config_dir / name
        if path.exists():
            return path.resolve()

    configs = sorted(config_dir.glob("*.pkl"))
    return configs[0].resolve() if configs else None


def _load_actual_policy_args(policy_config_path: Path, runtime_args):
    policy_args, _, _, _ = pickle.load(policy_config_path.open("rb"))
    merged_args = copy.deepcopy(policy_args)
    actual_algorithm_name = policy_args.algorithm_name
    for field in POLICY_ARG_OVERRIDE_FIELDS:
        if hasattr(runtime_args, field):
            setattr(merged_args, field, getattr(runtime_args, field))
    merged_args.algorithm_name = actual_algorithm_name
    return merged_args


def _resolve_checkpoint_dir(models_dir: Path, preferred_subdir: Optional[str]) -> Path:
    if preferred_subdir:
        preferred_dir = models_dir / preferred_subdir
        if preferred_dir.exists():
            return preferred_dir

    if _dir_has_checkpoint(models_dir):
        return models_dir

    child_dirs = sorted(path for path in models_dir.iterdir() if path.is_dir())
    if len(child_dirs) == 1 and _dir_has_checkpoint(child_dirs[0]):
        return child_dirs[0]

    return models_dir


def _dir_has_checkpoint(path: Path) -> bool:
    return any(path.glob("actor*.pt")) or any(path.glob("model*.pt"))


def _extract_population_size(experiment: str) -> Optional[int]:
    match = re.search(r"-s(\d+)$", experiment)
    if match is None:
        return None
    return int(match.group(1))


def _candidate_models_dirs(
    base_dir: Path,
    env_name: str,
    layout: str,
    algo: str,
    experiment: str,
    seed: int,
    policy_type: Optional[str],
) -> List[Path]:
    seed_dir = f"seed{seed}"
    legacy = base_dir / env_name / layout / algo / experiment / seed_dir / "models"
    shared = base_dir / env_name / layout / "shared" / algo / experiment / seed_dir / "models"
    separated = base_dir / env_name / layout / "separated" / algo / experiment / seed_dir / "models"

    if policy_type is None:
        return [legacy, shared, separated]

    policy = policy_type.lower()
    if policy == "shared":
        return [shared, legacy]
    if policy == "separated":
        return [separated, legacy]
    raise ValueError(f"Unknown policy_type: {policy_type}")

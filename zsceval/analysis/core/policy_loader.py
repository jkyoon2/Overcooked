from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from zsceval.runner.shared.base_runner import make_trainer_policy_cls
from zsceval.utils.train_util import get_base_run_dir


@dataclass(frozen=True)
class CheckpointPaths:
    actor: Optional[Path] = None
    critic: Optional[Path] = None
    model: Optional[Path] = None
    step: Optional[int] = None


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

    policy = _build_policy(all_args, envs, agent_id=agent_id, share_policy=share_policy, device=device)
    ckpt = _resolve_checkpoint_paths(
        models_dir=models_dir,
        step=step,
        agent_id=agent_id,
        share_policy=share_policy,
        use_single_network=getattr(all_args, "use_single_network", False),
    )
    _load_weights(policy, ckpt)
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


def _load_weights(policy, ckpt: CheckpointPaths) -> None:
    if ckpt.model is not None:
        if not hasattr(policy, "model"):
            raise ValueError("Policy has no 'model' attribute for single-network checkpoint.")
        policy.model.load_state_dict(torch.load(ckpt.model, map_location=policy.device))
        return
    if ckpt.actor is None:
        raise ValueError("Actor checkpoint is required.")
    policy.actor.load_state_dict(torch.load(ckpt.actor, map_location=policy.device))
    if ckpt.critic is not None and hasattr(policy, "critic"):
        policy.critic.load_state_dict(torch.load(ckpt.critic, map_location=policy.device))


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

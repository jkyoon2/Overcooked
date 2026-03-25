#!/usr/bin/env python
import os
import pprint
import re
import socket
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import setproctitle
import torch
import wandb
from loguru import logger

from train_sp import make_eval_env, make_train_env, parse_args
from zsceval.config import get_config
from zsceval.utils.train_util import get_base_run_dir, setup_seed

os.environ["WANDB_DIR"] = os.getcwd() + "/wandb/"
os.environ["WANDB_CACHE_DIR"] = os.getcwd() + "/wandb/.cache/"
os.environ["WANDB_CONFIG_DIR"] = os.getcwd() + "/wandb/.config/"

_ACTOR_RE = re.compile(r"^actor_periodic_(\d+)\.pt$")
_CRITIC_RE = re.compile(r"^critic_periodic_(\d+)\.pt$")


def _build_run_dir(all_args) -> Path:
    base_run_dir = Path(get_base_run_dir())
    policy_dir = "shared" if all_args.share_policy else "separated"
    run_dir = (
        base_run_dir
        / all_args.env_name
        / all_args.layout_name
        / policy_dir
        / all_args.algorithm_name
        / all_args.experiment_name
        / f"seed{all_args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _collect_periodic_pairs(model_dir: Path) -> Dict[int, Tuple[Path, Path]]:
    if not model_dir.exists():
        return {}
    actor_by_step: Dict[int, Path] = {}
    critic_by_step: Dict[int, Path] = {}
    for p in model_dir.glob("actor_periodic_*.pt"):
        m = _ACTOR_RE.match(p.name)
        if m:
            actor_by_step[int(m.group(1))] = p
    for p in model_dir.glob("critic_periodic_*.pt"):
        m = _CRITIC_RE.match(p.name)
        if m:
            critic_by_step[int(m.group(1))] = p
    common_steps = sorted(set(actor_by_step.keys()) & set(critic_by_step.keys()))
    return {step: (actor_by_step[step], critic_by_step[step]) for step in common_steps}


def _resolve_separated_role_dir(model_root: Path, role_name: Optional[str], agent_id: int) -> Path:
    if role_name:
        role_dir = model_root / role_name
        if role_dir.exists():
            return role_dir
    return model_root / f"agent{agent_id}"


def _configure_resume(all_args, run_dir: Path) -> None:
    all_args.resume_step = 0
    all_args.model_critic_path = None
    model_root = run_dir / "models"
    if not model_root.exists():
        logger.info(f"Resume: no model directory at {model_root}, training from scratch.")
        return

    if all_args.share_policy:
        step_to_paths = _collect_periodic_pairs(model_root)
        if not step_to_paths:
            logger.info(f"Resume: no shared periodic checkpoints found in {model_root}.")
            return
        resume_step = max(step_to_paths.keys())
        actor_path, critic_path = step_to_paths[resume_step]
        all_args.model_dir = str(actor_path)
        all_args.model_critic_path = str(critic_path)
        all_args.resume_step = resume_step
        logger.info(f"Resume(shared): step={resume_step}, actor={actor_path.name}, critic={critic_path.name}")
        return

    role_labels = getattr(all_args, "reward_shaping_roles_list", None)
    per_agent_steps = []
    per_agent_paths = []
    for agent_id in range(all_args.num_agents):
        role_name = role_labels[agent_id] if isinstance(role_labels, list) and len(role_labels) > agent_id else None
        model_dir = _resolve_separated_role_dir(model_root, role_name, agent_id)
        step_to_paths = _collect_periodic_pairs(model_dir)
        if not step_to_paths:
            logger.info(
                f"Resume(separated): no periodic checkpoints for agent {agent_id} in {model_dir}, training from scratch."
            )
            return
        per_agent_steps.append(set(step_to_paths.keys()))
        per_agent_paths.append(step_to_paths)

    common_steps = sorted(set.intersection(*per_agent_steps))
    if not common_steps:
        logger.info("Resume(separated): no common periodic step across all agents, training from scratch.")
        return

    resume_step = common_steps[-1]
    for agent_id in range(all_args.num_agents):
        actor_path, critic_path = per_agent_paths[agent_id][resume_step]
        setattr(all_args, f"model_dir_agent{agent_id}", str(actor_path))
        setattr(all_args, f"model_critic_agent{agent_id}", str(critic_path))
    all_args.model_dir = None
    all_args.resume_step = resume_step
    logger.info(f"Resume(separated): common_step={resume_step}, agents={all_args.num_agents}")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy is False and all_args.use_naive_recurrent_policy is False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = _build_run_dir(all_args)
    all_args.run_dir = run_dir
    _configure_resume(all_args, run_dir)

    if all_args.overcooked_version == "new":
        project_name = all_args.env_name + "-new"
    else:
        project_name = all_args.env_name
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=project_name,
            entity=all_args.wandb_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
            group=all_args.layout_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
            tags=all_args.wandb_tags,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "_"
        + str(all_args.layout_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    setup_seed(all_args.seed)

    envs = make_train_env(all_args, run_dir)
    eval_envs = make_eval_env(all_args, run_dir) if all_args.use_eval else None
    num_agents = all_args.num_agents

    logger.info(pprint.pformat(all_args.__dict__, compact=True))
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    if all_args.share_policy:
        from zsceval.runner.shared.overcooked_runner import OvercookedRunner as Runner
    else:
        from zsceval.runner.separated.overcooked_runner import OvercookedRunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish(quiet=True)
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    main(sys.argv[1:])

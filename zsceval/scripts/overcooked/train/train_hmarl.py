#!/usr/bin/env python

import argparse
import copy
import os
import pprint
import re
import socket
import sys
from pathlib import Path

import setproctitle
import torch
import wandb
from loguru import logger

from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.train_util import get_base_run_dir, setup_seed

os.environ["WANDB_DIR"] = os.getcwd() + "/wandb/"
os.environ["WANDB_CACHE_DIR"] = os.getcwd() + "/wandb/.cache/"
os.environ["WANDB_CONFIG_DIR"] = os.getcwd() + "/wandb/.config/"

try:
    import optuna
except Exception:  # pragma: no cover - optuna is optional
    optuna = None


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, rank=rank)
                else:
                    env = Overcooked_new(all_args, run_dir, rank=rank)
            else:
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
            all_args.dummy_batch_size,
        )


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, evaluation=True, rank=rank)
                else:
                    env = Overcooked_new(all_args, run_dir, evaluation=True, rank=rank)
            else:
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocDummyBatchVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
            all_args.dummy_batch_size,
        )


def parse_args(args, parser: argparse.ArgumentParser):
    parser = get_overcooked_args(parser)
    parser.add_argument("--skill_dim", type=int, default=16)
    parser.add_argument("--skill_embed_dim", type=int, default=16)
    parser.add_argument("--num_skills", type=int, default=128)
    parser.add_argument("--gumbel_tau", type=float, default=1.0)
    parser.add_argument("--gumbel_tau_start", type=float, default=1.5)
    parser.add_argument("--gumbel_tau_end", type=float, default=0.3)
    parser.add_argument("--gumbel_tau_anneal_steps", type=int, default=0)
    parser.add_argument("--t_seg", type=int, default=5)
    parser.add_argument("--intrinsic_alpha", type=float, default=0.3)
    parser.add_argument("--intrinsic_temp_start", type=float, default=1.0)
    parser.add_argument("--intrinsic_temp_end", type=float, default=None)
    parser.add_argument("--intrinsic_temp_anneal_steps", type=int, default=0)
    parser.add_argument("--vae_checkpoint_path", type=str, default="/checkpoints/vae_epoch_002_1t_small_corridor_4d.pt")
    parser.add_argument("--high_buffer_size", type=int, default=50000)
    parser.add_argument("--high_batch_size", type=int, default=64)
    parser.add_argument("--intrinsic_scale", type=float, default=5.0)
    parser.add_argument("--skill_range", type=float, default=4.0)
    parser.add_argument("--use_constraint", action="store_true", default=False)
    parser.add_argument("--bad_traj_path", type=str, default="")
    parser.add_argument("--constraint_threshold", type=float, default=0.5)
    parser.add_argument("--constraint_penalty", type=float, default=1.0)
    parser.add_argument("--train_mode", type=str, choices=["online", "adhoc"], default="online")
    parser.add_argument("--load_seed", type=int, default=None, help="Checkpoint seed to finetune from.")
    parser.add_argument("--load_step", type=int, default=None, help="Checkpoint step to finetune from.")
    parser.add_argument("--use_optuna", action="store_true", default=False, help="Enable Optuna HPO.")
    parser.add_argument("--optuna_trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--optuna_timeout", type=int, default=None, help="Optuna timeout (seconds).")
    parser.add_argument(
        "--optuna_direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction.",
    )
    parser.add_argument(
        "--optuna_metric",
        type=str,
        default="eval/high_mean",
        help="Metric key to optimize (e.g., eval/high_mean, train/low_mean).",
    )
    parser.add_argument("--optuna_seed", type=int, default=None, help="Seed for Optuna sampler.")
    parser.add_argument("--optuna_storage", type=str, default=None, help="Optuna storage URL.")
    parser.add_argument("--optuna_study_name", type=str, default=None, help="Optuna study name.")
    parser.add_argument(
        "--optuna_sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler type.",
    )
    parser.add_argument(
        "--optuna_pruner",
        type=str,
        default="median",
        choices=["median", "none"],
        help="Optuna pruner type.",
    )
    all_args = parser.parse_args(args)
    if all_args.skill_embed_dim is None:
        all_args.skill_embed_dim = all_args.skill_dim
    all_args.skill_dim = all_args.skill_embed_dim
    if all_args.gumbel_tau_start is None:
        all_args.gumbel_tau_start = all_args.gumbel_tau
    if all_args.gumbel_tau_end is None:
        all_args.gumbel_tau_end = all_args.gumbel_tau_start
    all_args.gumbel_tau = all_args.gumbel_tau_start
    if all_args.intrinsic_temp_end is None:
        all_args.intrinsic_temp_end = all_args.intrinsic_temp_start

    from zsceval.overcooked_config import OLD_LAYOUTS

    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


def _apply_optuna_suggestions(trial, all_args):
    all_args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    all_args.critic_lr = trial.suggest_float("critic_lr", 1e-5, 5e-4, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 0.2, log=True)
    all_args.entropy_coefs = [entropy_coef]
    all_args.entropy_coef_horizons = [0]
    all_args.ppo_epoch = trial.suggest_int("ppo_epoch", 5, 20)
    all_args.gumbel_tau = trial.suggest_float("gumbel_tau", 0.5, 2.0)
    all_args.gumbel_tau_start = all_args.gumbel_tau
    if getattr(all_args, "gumbel_tau_end", None) is None:
        all_args.gumbel_tau_end = all_args.gumbel_tau
    # all_args.t_seg = trial.suggest_int("t_seg", 2, 10)
    all_args.intrinsic_alpha = trial.suggest_float("intrinsic_alpha", 0.0, 0.95)
    all_args.intrinsic_scale = trial.suggest_float("intrinsic_scale", 1.0, 100.0, log=True)


def _resolve_metric(metrics, metric_key):
    if metrics is None:
        return None
    return metrics.get(metric_key)


def _find_latest_step(model_dir: Path, agent_id: int) -> int:
    """Pick the latest step for a given agent (falls back to any agent0 files)."""
    patterns = [
        f"*_agent{agent_id}_*.pt",
        f"*_agent0_*.pt",
        "hl_critic_*.pt",
        "ll_actor_*.pt",
        "ll_critic_*.pt",
        "hl_actor_*.pt",
    ]
    steps = []
    for pattern in patterns:
        for path in model_dir.glob(pattern):
            match = re.search(r"_([0-9]+)\.pt$", path.name)
            if match:
                steps.append(int(match.group(1)))
    return max(steps) if steps else None


def _maybe_load_pretrained_policy(runner, all_args):
    """Load a single-seed/step checkpoint (mirrors bad_skill_generator logic)."""
    load_seed = getattr(all_args, "load_seed", None)
    if load_seed is None:
        return

    exp_name = getattr(all_args, "load_experiment_name", all_args.experiment_name)
    base_dir = (
        Path(get_base_run_dir())
        / all_args.env_name
        / all_args.layout_name
        / all_args.algorithm_name
        / exp_name
        / f"seed{load_seed}"
        / "models"
    )
    if not base_dir.exists():
        raise FileNotFoundError(f"[HMARL] checkpoint dir not found: {base_dir}")

    load_step = getattr(all_args, "load_step", None)
    if load_step is None:
        load_step = _find_latest_step(base_dir, 0 if all_args.share_policy else 0)
    if load_step is None:
        raise FileNotFoundError(f"[HMARL] no checkpoint step found under {base_dir}")

    if all_args.share_policy:
        shared_ll_actor = base_dir / f"ll_actor_{load_step}.pt"
        shared_ll_critic = base_dir / f"ll_critic_{load_step}.pt"
        if shared_ll_actor.exists() and shared_ll_critic.exists():
            low_ckpts = {"actor": shared_ll_actor, "critic": shared_ll_critic}
        else:
            actor_path = base_dir / f"ll_actor_agent0_{load_step}.pt"
            critic_path = base_dir / f"ll_critic_agent0_{load_step}.pt"
            if not actor_path.exists() or not critic_path.exists():
                raise FileNotFoundError(f"[HMARL] shared low-level checkpoint missing at step {load_step}")
            low_ckpts = {"actor": actor_path, "critic": critic_path}

        hl_critic = base_dir / f"hl_critic_{load_step}.pt"
        if not hl_critic.exists():
            raise FileNotFoundError(f"[HMARL] high-level critic checkpoint missing at step {load_step}")

        shared_hl_actor = base_dir / f"hl_actor_{load_step}.pt"
        if shared_hl_actor.exists():
            hl_actors = [shared_hl_actor]
        else:
            fallback = base_dir / f"hl_actor_agent0_{load_step}.pt"
            if not fallback.exists():
                raise FileNotFoundError(f"[HMARL] shared high-level actor checkpoint missing at step {load_step}")
            hl_actors = [fallback]

        runner.policy.load_checkpoint({"low_level": low_ckpts, "high_level": {"critic": hl_critic, "actors": hl_actors}})
        logger.info(f"[HMARL] loaded checkpoint seed{load_seed}, step{load_step} for finetuning.")
        return

    def _low_paths(agent_id: int):
        actor_path = base_dir / f"ll_actor_agent{agent_id}_{load_step}.pt"
        critic_path = base_dir / f"ll_critic_agent{agent_id}_{load_step}.pt"
        if all_args.share_policy:
            actor_fallback = base_dir / f"ll_actor_agent0_{load_step}.pt"
            critic_fallback = base_dir / f"ll_critic_agent0_{load_step}.pt"
            if not actor_path.exists() and actor_fallback.exists():
                actor_path = actor_fallback
            if not critic_path.exists() and critic_fallback.exists():
                critic_path = critic_fallback
        return actor_path, critic_path

    low_ckpts = []
    for aid in range(all_args.num_agents):
        actor_path, critic_path = _low_paths(aid)
        if not actor_path.exists() or not critic_path.exists():
            raise FileNotFoundError(f"[HMARL] low-level checkpoint missing for agent {aid} at step {load_step}")
        low_ckpts.append({"actor": actor_path, "critic": critic_path})

    hl_critic = base_dir / f"hl_critic_{load_step}.pt"
    if not hl_critic.exists():
        raise FileNotFoundError(f"[HMARL] high-level critic checkpoint missing at step {load_step}")
    hl_actors = []
    for aid in range(all_args.num_agents):
        path = base_dir / f"hl_actor_agent{aid}_{load_step}.pt"
        if not path.exists():
            raise FileNotFoundError(f"[HMARL] high-level actor checkpoint missing for agent {aid} at step {load_step}")
        hl_actors.append(path)

    runner.policy.load_checkpoint({"low_level": low_ckpts, "high_level": {"critic": hl_critic, "actors": hl_actors}})
    logger.info(f"[HMARL] loaded checkpoint seed{load_seed}, step{load_step} for finetuning.")


def _run_training(all_args, trial=None):
    if all_args.train_mode == "adhoc":
        base_experiment_name = all_args.experiment_name
        all_args.use_constraint = True
        all_args.load_experiment_name = base_experiment_name
        if not all_args.bad_traj_path:
            all_args.bad_traj_path = "/home/juliecandoit98/neurocontroller/zsceval/scripts/overcooked/results/bad_skills/small_corridor_bad_skills_onion_counter_regrab_seed9_step3020000.npy"
        if all_args.load_seed is None:
            all_args.load_seed = all_args.seed
        if "adhoc" not in all_args.experiment_name:
            all_args.experiment_name = f"{all_args.experiment_name}_adhoc"

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    base_run_dir = Path(get_base_run_dir())
    # Use sp/hmarl path (HMARL built on top of SP) and seed-specific subdirectory.
    run_dir = base_run_dir / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    seed_run_dir = run_dir / f"seed{all_args.seed}"
    seed_run_dir.mkdir(parents=True, exist_ok=True)
    (seed_run_dir / "models").mkdir(exist_ok=True)
    (seed_run_dir / "gifs").mkdir(exist_ok=True)
    # Ensure downstream components see a plain string path
    all_args.run_dir = str(seed_run_dir)

    # sync render/gif flags: either one enables both
    wants_gif = getattr(all_args, "save_gifs", False)
    wants_render = getattr(all_args, "use_render", False)
    all_args.use_render = wants_render or wants_gif
    all_args.save_gifs = wants_render or wants_gif

    project_name = all_args.env_name if all_args.overcooked_version == "old" else all_args.env_name + "-new"
    run_dir = all_args.run_dir
    run = None
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
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}_{all_args.layout_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    setup_seed(all_args.seed)

    envs = None
    eval_envs = None
    runner = None
    try:
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
            from zsceval.runner.shared.hmarl_runner import HMARLRunner as Runner
        else:
            from zsceval.runner.separated.hmarl_runner import HMARLRunner as Runner

        runner = Runner(config)
        _maybe_load_pretrained_policy(runner, all_args)
        if trial is not None:
            runner.trial = trial
        metrics = runner.run()
        return metrics
    finally:
        if envs is not None:
            envs.close()
        if all_args.use_eval and eval_envs is not None and eval_envs is not envs:
            eval_envs.close()
        if all_args.use_wandb and run is not None:
            run.finish(quiet=True)
        elif runner is not None and hasattr(runner, "log_dir"):
            runner.writter.export_scalars_to_json(str(runner.log_dir / "summary.json"))
            runner.writter.close()


def _run_optuna(all_args):
    if optuna is None:
        raise ImportError("Optuna is not installed.")

    base_args = copy.deepcopy(all_args)
    base_experiment = base_args.experiment_name

    def objective(trial):
        trial_args = copy.deepcopy(base_args)
        trial_args.use_optuna = False
        trial_args.experiment_name = f"{base_experiment}_trial{trial.number}"
        
        # 하이퍼파라미터 적용
        _apply_optuna_suggestions(trial, trial_args)
        
        # 학습 실행
        metrics = _run_training(trial_args, trial=trial)
        
        # [핵심 변경 1] 두 가지 Metric 가져오기
        # 예: 'eval/high_mean' (Extrinsic) 과 'eval/intrinsic_mean' (Intrinsic)
        # 실제 metrics 딕셔너리에 있는 키 이름을 정확히 적어야 합니다.
        extrinsic = _resolve_metric(metrics, "eval/high_mean")
        intrinsic = _resolve_metric(metrics, "eval/intrinsic_mean")
        if intrinsic is None:
            intrinsic = _resolve_metric(metrics, "train/intrinsic_mean")
        
        if extrinsic is None or intrinsic is None:
            raise optuna.exceptions.TrialPruned("One of the metrics is missing.")
             
        # [핵심 변경 2] 튜플 형태로 두 값을 동시에 반환
        return float(extrinsic), float(intrinsic)

    # [핵심 변경 3] direction 대신 directions 사용 (둘 다 maximize)
    study = optuna.create_study(
        directions=["maximize", "maximize"],  # [Extrinsic 최대화, Intrinsic 최대화]
        sampler=optuna.samplers.TPESampler(seed=all_args.optuna_seed), # NSGA-II 알고리즘 자동 적용됨
        pruner=optuna.pruners.MedianPruner() if all_args.optuna_pruner == "median" else optuna.pruners.NopPruner(),
        storage=all_args.optuna_storage,
        study_name=all_args.optuna_study_name,
        load_if_exists=True,
    )
    
    study.optimize(objective, n_trials=all_args.optuna_trials, timeout=all_args.optuna_timeout)

    # =========================================================
    # 다목적 최적화 결과 출력 (Best Params가 하나가 아님!)
    # =========================================================
    from loguru import logger
    import json
    
    logger.info("🎉 Multi-Objective Optimization Finished!")
    
    # Pareto Front에 있는 최고의 시도들만 뽑기
    best_trials = study.best_trials

    logger.info(f"🏆 Number of Best Trials (Pareto Front): {len(best_trials)}")
    
    results_list = []
    for i, trial in enumerate(best_trials):
        logger.info(f"\n[Pareto Solution #{i+1}]")
        logger.info(f"  Values (Ext, Int): {trial.values}")
        logger.info(f"  Params: {trial.params}")
        
        results_list.append({
            "trial_number": trial.number,
            "values": trial.values,
            "params": trial.params
        })
    
    # 결과 저장
    with open(f"optuna_pareto_{all_args.experiment_name}.json", "w") as f:
        json.dump(results_list, f, indent=4)
        
    return study


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.use_optuna:
        _run_optuna(all_args)
        return

    _run_training(all_args)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    main(sys.argv[1:])

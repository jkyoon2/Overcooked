#!/usr/bin/env python
import hashlib
import os
import pprint
import socket
import sys
from itertools import product
from pathlib import Path

import numpy as np
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


HSP_GUARDRAIL_POSITIVE_FEATURES = {
    "PLACEMENT_IN_POT",
    "USEFUL_SOUP_PICKUP",
    "potting_onion",
    "potting_tomato",
    "useful_cook",
}

HSP_GUARDRAIL_NEGATIVE_FEATURES = {
    "USELESS_SOUP_PICKUP",
    "deliver_useless_order",
}

HSP_GUARDRAIL_BASELINE_CAPPED_FEATURES = {
    "PLACEMENT_IN_POT",
    "USEFUL_SOUP_PICKUP",
    "USELESS_SOUP_PICKUP",
    "potting_onion",
    "potting_tomato",
    "useful_cook",
    "deliver_size_two_order",
    "deliver_size_three_order",
    "deliver_useless_order",
}


def _parse_hsp_weight_values(spec):
    def parse_value(raw):
        if raw.startswith("r") and "[" in raw:
            raw = raw[2:-1]
            left, right, num = raw.split(":")
            left, right, num = float(left), float(right), int(num)
            return np.linspace(left, right, num).tolist()
        if raw.startswith("["):
            raw = raw[1:-1]
            return list(map(float, raw.split(":")))
        return [float(raw)]

    return [parse_value(token) for token in spec.split(",")]


def _ordered_unique(values):
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _get_hsp_feature_names(overcooked_version):
    if overcooked_version == "old":
        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
    else:
        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS

    return list(SHAPED_INFOS) + ["sparse_reward"]


def _apply_hsp_candidate_guardrails(feature_names, w0_options, w1_options):
    guarded_options = []
    guardrail_changes = []

    for feature_name, candidate_values, baseline_values in zip(feature_names, w0_options, w1_options):
        baseline_value = baseline_values[0] if baseline_values else 0.0
        filtered_values = list(candidate_values)

        if feature_name in HSP_GUARDRAIL_POSITIVE_FEATURES and baseline_value > 0:
            filtered_values = [value for value in filtered_values if value > 0]

        if feature_name in HSP_GUARDRAIL_NEGATIVE_FEATURES and baseline_value < 0:
            filtered_values = [value for value in filtered_values if value < 0]

        if feature_name in HSP_GUARDRAIL_BASELINE_CAPPED_FEATURES and baseline_value != 0:
            max_abs_value = abs(baseline_value)
            filtered_values = [
                value for value in filtered_values if abs(value) <= max_abs_value + 1e-8
            ]

        filtered_values = _ordered_unique(filtered_values)
        if not filtered_values:
            filtered_values = [baseline_value]

        if list(candidate_values) != filtered_values:
            guardrail_changes.append((feature_name, list(candidate_values), filtered_values))

        guarded_options.append(filtered_values)

    return guarded_options, guardrail_changes


def _deterministic_candidate_order(candidates, context):
    def sort_key(candidate):
        candidate_str = ",".join(f"{float(value):.12g}" for value in candidate)
        digest = hashlib.sha256(f"{context}|{candidate_str}".encode("utf-8")).hexdigest()
        return digest, tuple(float(value) for value in candidate)

    return sorted(candidates, key=sort_key)


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, rank=rank)
                else:
                    env = Overcooked_new(all_args, run_dir, rank=rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        # return ShareSubprocVecEnv(
        #     [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
        # )
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
                print("Can not support the " + all_args.env_name + "environment.")
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
        # return ShareSubprocVecEnv(
        #     [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        # )


def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument("--use_task_v_out", default=False, action="store_true")
    parser.add_argument(
        "--policy_mode",
        type=str,
        choices=["shared", "separated"],
        default="shared",
        help="shared: one shared policy; separated: one policy per agent.",
    )
    # all_args = parser.parse_known_args(args)[0]
    all_args = parser.parse_args(args)

    all_args.policy_mode = all_args.policy_mode.lower()
    all_args.share_policy = all_args.policy_mode == "shared"
    if not all_args.share_policy and all_args.random_index:
        logger.warning(
            "policy_mode=separated forces --random_index off so HSP w0/w1 assignment stays fixed per agent."
        )
        all_args.random_index = False
    if all_args.agent_policy_names is None:
        all_args.agent_policy_names = ["ppo"] * all_args.num_agents

    if not hasattr(all_args, "reward_shaping_role"):
        all_args.reward_shaping_role = "individual"
    all_args.reward_shaping_role = all_args.reward_shaping_role.lower()
    raw_roles = getattr(all_args, "reward_shaping_roles", None)
    if raw_roles:
        roles = [role.strip().lower() for role in raw_roles.split(",") if role.strip()]
        if len(roles) == 1:
            roles = roles * all_args.num_agents
        elif len(roles) != all_args.num_agents:
            raise ValueError(
                f"--reward_shaping_roles expects 1 or {all_args.num_agents} roles, got {len(roles)} ({roles})"
            )
    else:
        roles = [all_args.reward_shaping_role] * all_args.num_agents
    all_args.reward_shaping_roles_list = roles
    all_args.reward_shaping_roles = ",".join(roles)

    from zsceval.overcooked_config import OLD_LAYOUTS

    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    wants_gif = getattr(all_args, "save_gifs", False)
    wants_render = getattr(all_args, "use_render", False)
    all_args.use_render = wants_render or wants_gif
    all_args.save_gifs = wants_render or wants_gif

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    # rpg
    if all_args.use_hsp:
        w0 = _parse_hsp_weight_values(all_args.w0)
        w1 = _parse_hsp_weight_values(all_args.w1)
        feature_names = _get_hsp_feature_names(all_args.overcooked_version)

        if len(feature_names) == len(w0) == len(w1):
            w0, guardrail_changes = _apply_hsp_candidate_guardrails(feature_names, w0, w1)
            for feature_name, before, after in guardrail_changes:
                logger.info(f"HSP guardrail {feature_name}: {before} -> {after}")
        else:
            logger.warning(
                "Skip HSP guardrails because feature count does not match weight specs: "
                f"features={len(feature_names)}, w0={len(w0)}, w1={len(w1)}"
            )

        bias_index = np.array([idx for idx, values in enumerate(w0) if len(values) > 1], dtype=np.int64)
        w0_candidates = list(map(list, product(*w0)))
        if bias_index.size > 0:
            w0_candidates = [
                cand for cand in w0_candidates if sum(np.array(cand)[bias_index] != 0) <= 3
            ]
        w0_candidates = _deterministic_candidate_order(
            w0_candidates,
            context=f"{all_args.layout_name}|{all_args.overcooked_version}|w0",
        )
        logger.info(f"bias index {bias_index}")
        logger.info(f"num w0_candidates {len(w0_candidates)}")
        candidates_str = ""
        for c_i in range(len(w0_candidates)):
            candidates_str += f"{c_i+1}: {w0_candidates[c_i]}\n"
        # logger.info(
        #     f"w0_candidates:\n {pprint.pformat(w0_candidates, width=150, compact=True)}"
        # )
        logger.info(f"w0_candidates:\n{candidates_str}")
        if len(w0_candidates) == 0:
            raise RuntimeError("No valid w0 candidates remain after applying HSP guardrails.")
        w0 = w0_candidates[(all_args.seed + all_args.w0_offset) % len(w0_candidates)]
        logger.info(
            "selected w0 candidate index {} / {} using deterministic candidate order".format(
                (all_args.seed + all_args.w0_offset) % len(w0_candidates),
                len(w0_candidates),
            )
        )
        all_args.w0 = ""
        for s in w0:
            all_args.w0 += str(s) + ","
        all_args.w0 = all_args.w0[:-1]

        w1_candidates = list(map(list, product(*w1)))
        logger.debug(f"w1_candidates:\n {pprint.pformat(w1_candidates, compact=True, width=200)}")
        w1 = w1_candidates[(all_args.seed) % len(w1_candidates)]
        all_args.w1 = ""
        for s in w1:
            all_args.w1 += str(s) + ","
        all_args.w1 = all_args.w1[:-1]

    # cuda
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

    # run dir
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
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    all_args.run_dir = run_dir

    # wandb
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
        if not run_dir.exists():
            os.makedirs(str(run_dir))

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

    # seed
    # torch.manual_seed(all_args.seed)
    # torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)
    setup_seed(all_args.seed)

    # env init
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

    # run experiments
    if all_args.share_policy:
        from zsceval.runner.shared.overcooked_runner import OvercookedRunner as Runner
    else:
        from zsceval.runner.separated.overcooked_runner import (
            OvercookedRunner as Runner,
        )

    runner = Runner(config)
    runner.run()

    # post process
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
    # logger.add(sys.stdout, level="INFO")
    main(sys.argv[1:])

#!/usr/bin/env python
import re
import sys
from pathlib import Path

import torch
import yaml

# 프로젝트 루트 경로 설정
sys.path.append(str(Path(__file__).resolve().parents[3]))

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.bad_skill_generator import BadSkillGenerator
from zsceval.utils.train_util import get_base_run_dir


def load_yaml_config(yaml_path: Path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _build_model_dir(args, seed):
    base_dir = Path(get_base_run_dir())
    return (
        base_dir
        / args.env_name
        / args.layout_name
        / args.algorithm_name
        / args.experiment_name
        / f"seed{seed}"
        / "models"
    )


def _collect_checkpoint_steps(model_dir: Path, step_start: int, step_end: int):
    if not model_dir.exists():
        return []
    steps = set()
    for pattern in ("*_agent0_*.pt", "hl_critic_*.pt", "ll_actor_*.pt", "ll_critic_*.pt", "hl_actor_*.pt"):
        for path in model_dir.glob(pattern):
            match = re.search(r"_([0-9]+)\.pt$", path.name)
            if match:
                step = int(match.group(1))
                if step_start <= step <= step_end:
                    steps.add(step)
    return sorted(steps)


def _format_output_path(base_output, seed, step):
    if base_output is None:
        return None
    out_path = Path(base_output)
    suffix = f"_seed{seed}_step{step}"
    if out_path.suffix:
        return str(out_path.with_name(out_path.stem + suffix + out_path.suffix))
    return str(out_path.with_name(out_path.name + suffix))


def _run_generation(args, device, num_episodes, output_path):
    run_dir = Path(output_path).parent if output_path else Path("results")
    config = {
        "all_args": args,
        "device": device,
        "run_dir": run_dir,
        "num_agents": args.num_agents,
    }

    print("Initializing BadSkillGenerator...")
    generator = BadSkillGenerator(config)

    print(f"Start generation for {num_episodes} episodes...")
    save_path = generator.generate_bad_skills(num_episodes=num_episodes, output_path=output_path)
    if save_path:
        print(f"Successfully saved bad skills to: {save_path}")
    else:
        print("No bad skills collected.")
    return save_path


def main():
    parser = get_config()
    parser = get_overcooked_args(parser)

    default_config_path = Path(__file__).resolve().parents[1] / "overcooked" / "config" / "gen_bad_skills.yaml"
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(default_config_path),
        help="Path to the YAML config file for bad skill generation",
    )
    parser.add_argument(
        "--target_error_types",
        nargs="+",
        type=str,
        default=None,
        help="List of error types to collect (e.g., COLLISION ONION_COUNTER_REGRAB). Defaults to all.",
    )

    args = parser.parse_args()

    if args.config_path:
        yaml_config = load_yaml_config(Path(args.config_path))
        for key, value in yaml_config.items():
            setattr(args, key, value)
            print(f"[Config] {key}: {value}")

    # 데이터 생성에선 로깅 비활성화
    args.use_wandb = False

    if args.cuda and torch.cuda.is_available():
        print("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    num_episodes = getattr(args, "num_episodes", 10)
    output_path = getattr(args, "bad_skill_output", None)
    step_start = getattr(args, "load_step_start", None)
    step_end = getattr(args, "load_step_end", None)

    if step_start is not None or step_end is not None:
        if step_start is None or step_end is None:
            raise ValueError("Both load_step_start and load_step_end must be set.")
        step_start = int(step_start)
        step_end = int(step_end)
        if step_start > step_end:
            raise ValueError("load_step_start must be <= load_step_end.")

        base_output = output_path or str(Path("results") / "bad_skills.npy")
        seed_list = list(getattr(args, "load_seeds", None) or [getattr(args, "seed", 1)])
        for seed in seed_list:
            model_dir = _build_model_dir(args, seed)
            steps = _collect_checkpoint_steps(model_dir, step_start, step_end)
            if not steps:
                print(f"No checkpoints found for seed {seed} in range [{step_start}, {step_end}].")
                continue
            for step in steps:
                args.load_seeds = [seed]
                args.load_steps = [step]
                run_output = _format_output_path(base_output, seed, step)
                print(f"Run seed {seed}, step {step}...")
                _run_generation(args, device, num_episodes, run_output)
    else:
        _run_generation(args, device, num_episodes, output_path)


if __name__ == "__main__":
    main()

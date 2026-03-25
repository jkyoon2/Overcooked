from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from zsceval.utils.hsp_pipeline import normalize_layout_argument, resolve_policy_pool_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-1 population yaml files.")
    parser.add_argument("layout", type=str, help="Layout name or 'all'.")
    parser.add_argument("algo", type=str, choices=["traj", "mep"])
    parser.add_argument("-s", "--population_size", type=int, default=15)
    parser.add_argument("--policy_pool_path", type=str, default=None)
    parser.add_argument("--adaptive_policy_config", type=str, default="rnn_policy_config.pkl")
    parser.add_argument("--population_policy_config", type=str, default="rnn_policy_config.pkl")
    return parser.parse_args()


def warn_missing_policy_config(policy_pool_root: Path, layout: str, config_name: str) -> None:
    config_path = policy_pool_root / layout / "policy_config" / config_name
    if not config_path.exists():
        logger.warning(f"Missing policy config referenced by yaml: {config_path}")


def write_population_yaml(
    *,
    policy_pool_root: Path,
    layout: str,
    algo: str,
    population_size: int,
    adaptive_policy_config: str,
    population_policy_config: str,
) -> Path:
    source_dir = policy_pool_root / layout / algo / "s1"
    source_dir.mkdir(parents=True, exist_ok=True)

    warn_missing_policy_config(policy_pool_root, layout, adaptive_policy_config)
    warn_missing_policy_config(policy_pool_root, layout, population_policy_config)

    yml_path = source_dir / f"train-s{population_size}.yml"
    logger.info(f"Writing {yml_path}")
    with yml_path.open("w", encoding="utf-8") as s1_yml:
        s1_yml.write(
            f"""\
{algo}_adaptive:
    policy_config_path: {layout}/policy_config/{adaptive_policy_config}
    featurize_type: ppo
    train: False
"""
        )
        for policy_id in range(1, population_size + 1):
            s1_yml.write(
                f"""\
{algo}{policy_id}:
    policy_config_path: {layout}/policy_config/{population_policy_config}
    featurize_type: ppo
    train: True
"""
            )
    return yml_path


def main() -> None:
    args = parse_args()
    policy_pool_root = resolve_policy_pool_root(args.policy_pool_path)
    layouts = normalize_layout_argument(args.layout)
    logger.info(f"Generate templates for {layouts}")

    for layout in layouts:
        write_population_yaml(
            policy_pool_root=policy_pool_root,
            layout=layout,
            algo=args.algo,
            population_size=args.population_size,
            adaptive_policy_config=args.adaptive_policy_config,
            population_policy_config=args.population_policy_config,
        )


if __name__ == "__main__":
    main()

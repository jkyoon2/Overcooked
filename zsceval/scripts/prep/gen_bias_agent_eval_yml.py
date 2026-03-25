from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from zsceval.utils.hsp_pipeline import normalize_layout_argument, resolve_policy_pool_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bias-agent eval yaml templates.")
    parser.add_argument("layout", type=str, help="Layout name or 'all'.")
    parser.add_argument("--policy_pool_path", type=str, default=None)
    parser.add_argument("--policy_config_name", type=str, default="rnn_policy_config.pkl")
    return parser.parse_args()


def write_eval_template(layout: str, policy_pool_root: Path, policy_config_name: str) -> None:
    num_agents = 3 if layout == "academy_3_vs_1_with_keeper" else 2
    yml_path = policy_pool_root / layout / "hsp" / "s1" / "eval_template.yml"
    yml_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for agent_id in range(num_agents):
        lines.extend(
            [
                f"agent{agent_id}:",
                f"    policy_config_path: {layout}/policy_config/{policy_config_name}",
                "    featurize_type: ppo",
                "    train: False",
                "    model_path:",
                f"        actor: {layout}/hsp/s1/pop/agent{agent_id}_actor.pt",
            ]
        )

    yml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {yml_path}")


def main() -> None:
    args = parse_args()
    policy_pool_root = resolve_policy_pool_root(args.policy_pool_path)
    for layout in normalize_layout_argument(args.layout):
        write_eval_template(layout, policy_pool_root, args.policy_config_name)


if __name__ == "__main__":
    main()

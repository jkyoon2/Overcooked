from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


DEFAULT_LAYOUTS = (
    "2_forced_hard",
    "2_forced_hard_4",
    "2_incentivized_hard",
    "2_incentivized_hard_4",
)


@dataclass(frozen=True)
class ModelSpec:
    layout: str
    policy_type: str
    algorithm: str
    experiment: str
    seed: int
    checkpoint: int
    actor_path: Path

    @property
    def output_stem(self) -> str:
        return f"{self.layout}_seed{self.seed}_{self.checkpoint}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover available shared rMAPPO checkpoints for selected Overcooked "
            "layouts and export reward-suffixed realtime trajectory JSON files."
        )
    )
    parser.add_argument("--layouts", nargs="+", default=list(DEFAULT_LAYOUTS))
    parser.add_argument("--results_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--inventory_path", type=Path, default=None)
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max_ticks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--dry_run", action="store_true", help="Only write the inventory TSV.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if a reward-suffixed JSON already exists for the same seed/checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    results_root = (args.results_root or repo_root / "results").resolve()
    output_root = (args.output_root or repo_root / "json_trajectory").resolve()
    inventory_path = (
        args.inventory_path or output_root / "available_models_2_hard_layouts.tsv"
    ).resolve()
    report_path = (
        args.report_path or output_root / "generation_report_2_hard_layouts.tsv"
    ).resolve()

    specs = discover_specs(results_root=results_root, layouts=args.layouts)
    if args.limit is not None:
        specs = specs[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    write_inventory(inventory_path, specs)
    print(f"MODEL_COUNT={len(specs)}")
    print(f"INVENTORY={inventory_path}")

    if args.dry_run:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "layout",
                "policy_type",
                "algorithm",
                "experiment",
                "seed",
                "checkpoint",
                "status",
                "reward",
                "output_path",
                "error",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        if args.workers <= 1:
            for index, spec in enumerate(specs, start=1):
                row = generate_one(
                    spec=spec,
                    total=len(specs),
                    index=index,
                    output_root=output_root,
                    results_root=results_root,
                    device=args.device,
                    max_ticks=args.max_ticks,
                    overwrite=args.overwrite,
                )
                writer.writerow(row)
                report_file.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        generate_one,
                        spec=spec,
                        total=len(specs),
                        index=index,
                        output_root=output_root,
                        results_root=results_root,
                        device=args.device,
                        max_ticks=args.max_ticks,
                        overwrite=args.overwrite,
                    )
                    for index, spec in enumerate(specs, start=1)
                ]
                for future in as_completed(futures):
                    writer.writerow(future.result())
                    report_file.flush()

    print(f"REPORT={report_path}")


def discover_specs(results_root: Path, layouts: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    overcooked_root = results_root / "Overcooked"
    for layout in layouts:
        layout_root = overcooked_root / layout / "shared" / "rmappo"
        if not layout_root.exists():
            continue

        for experiment_dir in sorted(path for path in layout_root.iterdir() if path.is_dir()):
            for seed_dir in sorted(experiment_dir.glob("seed*"), key=seed_sort_key):
                models_dir = seed_dir / "models"
                if not models_dir.exists():
                    continue

                seed = parse_seed(seed_dir)
                for actor_path in sorted(
                    models_dir.glob("actor_periodic_*.pt"),
                    key=lambda path: parse_checkpoint(path),
                ):
                    checkpoint = parse_checkpoint(actor_path)
                    specs.append(
                        ModelSpec(
                            layout=layout,
                            policy_type="shared",
                            algorithm="rmappo",
                            experiment=experiment_dir.name,
                            seed=seed,
                            checkpoint=checkpoint,
                            actor_path=actor_path.resolve(),
                        )
                    )
    return specs


def write_inventory(path: Path, specs: Sequence[ModelSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layout",
                "policy_type",
                "algorithm",
                "experiment",
                "seed",
                "checkpoint",
                "actor_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "layout": spec.layout,
                    "policy_type": spec.policy_type,
                    "algorithm": spec.algorithm,
                    "experiment": spec.experiment,
                    "seed": spec.seed,
                    "checkpoint": spec.checkpoint,
                    "actor_path": spec.actor_path,
                }
            )


def generate_one(
    spec: ModelSpec,
    total: int,
    index: int,
    output_root: Path,
    results_root: Path,
    device: str,
    max_ticks: Optional[int],
    overwrite: bool,
) -> dict:
    output_dir = output_root / spec.layout
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{spec.output_stem}_*.json"))
    if existing and not overwrite:
        output_path = existing[0]
        reward = reward_from_json(output_path)
        print(f"[{index}/{total}] skip existing {output_path}")
        return report_row(spec, "existing", reward, output_path, "")

    tmp_path = output_dir / f".{spec.output_stem}.tmp.json"
    final_prefix = output_dir / spec.output_stem
    cmd = [
        sys.executable,
        "-m",
        "zsceval.analysis.json.generate_realtime_state_json",
        "--layout_name",
        spec.layout,
        "--algorithm_name",
        spec.algorithm,
        "--experiment_name",
        spec.experiment,
        "--policy_seed",
        str(spec.seed),
        "--checkpoint",
        str(spec.checkpoint),
        "--policy_type",
        spec.policy_type,
        "--results_root",
        str(results_root),
        "--output_path",
        str(tmp_path),
        "--device",
        device,
    ]
    if spec.layout.endswith("_4"):
        cmd.extend(["--num_agents", "4"])
    if max_ticks is not None:
        cmd.extend(["--max_ticks", str(max_ticks)])

    print(
        f"[{index}/{total}] generate layout={spec.layout} exp={spec.experiment} "
        f"seed={spec.seed} checkpoint={spec.checkpoint}"
    )
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        reward = reward_from_json(tmp_path)
        output_path = final_prefix.with_name(f"{final_prefix.name}_{reward}.json")
        if output_path.exists():
            output_path.unlink()
        tmp_path.replace(output_path)
        print(f"[{index}/{total}] saved {output_path}")
        return report_row(spec, "generated", reward, output_path, "")
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        output = ""
        if isinstance(exc, subprocess.CalledProcessError):
            output = (exc.stdout or "").strip().splitlines()[-20:]
            output = " | ".join(output)
        error = f"{type(exc).__name__}: {exc}"
        if output:
            error = f"{error} | {output}"
        print(f"[{index}/{total}] failed {spec.output_stem}: {error}")
        return report_row(spec, "failed", None, None, error)


def report_row(
    spec: ModelSpec,
    status: str,
    reward: Optional[int],
    output_path: Optional[Path],
    error: str,
) -> dict:
    return {
        "layout": spec.layout,
        "policy_type": spec.policy_type,
        "algorithm": spec.algorithm,
        "experiment": spec.experiment,
        "seed": spec.seed,
        "checkpoint": spec.checkpoint,
        "status": status,
        "reward": "" if reward is None else reward,
        "output_path": "" if output_path is None else output_path,
        "error": error,
    }


def reward_from_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        trajectory = json.load(f)
    dynamic_states = trajectory.get("dynamicState", [])
    if not dynamic_states:
        return 0
    return int(dynamic_states[-1].get("score", 0))


def parse_seed(path: Path) -> int:
    return int(path.name.replace("seed", "", 1))


def seed_sort_key(path: Path) -> int:
    try:
        return parse_seed(path)
    except ValueError:
        return 10**9


def parse_checkpoint(path: Path) -> int:
    return int(path.stem.replace("actor_periodic_", "", 1))


if __name__ == "__main__":
    main()

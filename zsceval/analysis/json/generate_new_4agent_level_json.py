from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LAYOUT_LEVELS = {
    "2_forced_hard_4": (40, 80, 120, 160),
    "2_incentivized_hard_4": (60, 120, 160, 240),
}


@dataclass(frozen=True)
class Spec:
    layout: str
    seed: int
    checkpoint: int
    actor_path: Path
    critic_path: Path

    @property
    def output_stem(self) -> str:
        return f"{self.layout}_seed{self.seed}_{self.checkpoint}"

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.layout, self.seed, self.checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate seed>=6 4-agent hard-layout JSON files, discard score 0, and classify into L0-L3."
    )
    parser.add_argument("--layouts", nargs="+", default=sorted(LAYOUT_LEVELS))
    parser.add_argument("--min_seed", type=int, default=6)
    parser.add_argument("--results_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min_age_seconds", type=float, default=60.0)
    parser.add_argument(
        "--skip_report_paths",
        nargs="*",
        type=Path,
        default=None,
        help="Reports whose discarded_zero rows should be reused to avoid re-evaluating deterministic zero-score checkpoints.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    results_root = (args.results_root or repo_root / "results").resolve()
    output_root = (args.output_root or repo_root / "json_trajectory").resolve()
    report_path = (
        args.report_path
        or output_root / "generation_report_new_4agent_level_json.tsv"
    ).resolve()

    skip_report_paths = args.skip_report_paths
    if skip_report_paths is None:
        skip_report_paths = [report_path] if report_path.exists() else []
    skipped_zero_keys = load_discarded_zero_keys(skip_report_paths)

    discovered_specs = discover_specs(
        results_root, args.layouts, args.min_seed, args.min_age_seconds
    )
    specs, skipped_existing, skipped_zero = pending_specs(
        discovered_specs, output_root, skipped_zero_keys
    )
    print(f"DISCOVERED_MODEL_COUNT={len(discovered_specs)}")
    print(f"SKIPPED_EXISTING={skipped_existing}")
    print(f"SKIPPED_REPORT_ZERO={skipped_zero}")
    print(f"MODEL_COUNT={len(specs)}")
    for layout in args.layouts:
        print(f"{layout}={sum(1 for spec in specs if spec.layout == layout)}")

    if args.dry_run:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layout",
                "seed",
                "checkpoint",
                "status",
                "reward",
                "level",
                "output_path",
                "error",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        if args.workers <= 1:
            for index, spec in enumerate(specs, start=1):
                row = generate_one(index, len(specs), spec, results_root, output_root, args.device)
                writer.writerow(row)
                f.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        generate_one,
                        index,
                        len(specs),
                        spec,
                        results_root,
                        output_root,
                        args.device,
                    )
                    for index, spec in enumerate(specs, start=1)
                ]
                for future in as_completed(futures):
                    writer.writerow(future.result())
                    f.flush()
    print(f"REPORT={report_path}")


def discover_specs(
    results_root: Path,
    layouts: list[str],
    min_seed: int,
    min_age_seconds: float,
) -> list[Spec]:
    specs: list[Spec] = []
    now = time.time()

    for layout in layouts:
        if layout not in LAYOUT_LEVELS:
            raise ValueError(f"Unknown level criteria for layout={layout}")
        exp_dir = results_root / "Overcooked" / layout / "shared" / "rmappo" / "sp"
        if not exp_dir.exists():
            continue
        for seed_dir in sorted(exp_dir.glob("seed*"), key=seed_sort_key):
            seed = parse_seed(seed_dir)
            if seed < min_seed:
                continue
            models_dir = seed_dir / "models"
            if not models_dir.exists():
                continue
            for actor_path in sorted(models_dir.glob("actor_periodic_*.pt"), key=parse_checkpoint):
                checkpoint = parse_checkpoint(actor_path)
                critic_path = models_dir / f"critic_periodic_{checkpoint}.pt"
                if not critic_path.exists():
                    continue
                if min(now - actor_path.stat().st_mtime, now - critic_path.stat().st_mtime) < min_age_seconds:
                    continue
                specs.append(
                    Spec(
                        layout=layout,
                        seed=seed,
                        checkpoint=checkpoint,
                        actor_path=actor_path.resolve(),
                        critic_path=critic_path.resolve(),
                    )
                )
    return specs


def pending_specs(
    specs: list[Spec],
    output_root: Path,
    skipped_zero_keys: set[tuple[str, int, int]],
) -> tuple[list[Spec], int, int]:
    pending: list[Spec] = []
    skipped_existing = 0
    skipped_zero = 0
    for spec in specs:
        layout_dir = output_root / spec.layout
        if existing_output(layout_dir, spec.output_stem) is not None:
            skipped_existing += 1
            continue
        if spec.key in skipped_zero_keys:
            skipped_zero += 1
            continue
        pending.append(spec)
    return pending, skipped_existing, skipped_zero


def load_discarded_zero_keys(report_paths: list[Path]) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for report_path in report_paths:
        if not report_path.exists():
            continue
        with report_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("status") != "discarded_zero":
                    continue
                try:
                    keys.add(
                        (
                            row["layout"],
                            int(row["seed"]),
                            int(row["checkpoint"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    return keys


def generate_one(
    index: int,
    total: int,
    spec: Spec,
    results_root: Path,
    output_root: Path,
    device: str,
) -> dict:
    layout_dir = output_root / spec.layout
    for level in ("L0", "L1", "L2", "L3"):
        (layout_dir / level).mkdir(parents=True, exist_ok=True)

    existing = existing_output(layout_dir, spec.output_stem)
    if existing is not None:
        reward = reward_from_json(existing)
        level = existing.parent.name
        print(f"[{index}/{total}] existing {existing}")
        return report_row(spec, "existing", reward, level, existing, "")

    tmp_path = layout_dir / f".{spec.output_stem}.tmp.json"
    cmd = [
        sys.executable,
        "-m",
        "zsceval.analysis.json.generate_realtime_state_json",
        "--layout_name",
        spec.layout,
        "--algorithm_name",
        "rmappo",
        "--experiment_name",
        "sp",
        "--policy_seed",
        str(spec.seed),
        "--checkpoint",
        str(spec.checkpoint),
        "--policy_type",
        "shared",
        "--results_root",
        str(results_root),
        "--output_path",
        str(tmp_path),
        "--device",
        device,
        "--num_agents",
        "4",
    ]

    print(f"[{index}/{total}] generate {spec.layout} seed={spec.seed} checkpoint={spec.checkpoint}")
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        reward = reward_from_json(tmp_path)
        if reward == 0:
            tmp_path.unlink()
            print(f"[{index}/{total}] discard score=0 {spec.output_stem}")
            return report_row(spec, "discarded_zero", reward, "", None, "")
        level = classify(spec.layout, reward)
        output_path = layout_dir / level / f"{spec.output_stem}_{reward}.json"
        if output_path.exists():
            output_path.unlink()
        tmp_path.replace(output_path)
        print(f"[{index}/{total}] saved {output_path}")
        return report_row(spec, "generated", reward, level, output_path, "")
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        output = ""
        if isinstance(exc, subprocess.CalledProcessError):
            output = " | ".join((exc.stdout or "").strip().splitlines()[-20:])
        error = f"{type(exc).__name__}: {exc}"
        if output:
            error = f"{error} | {output}"
        print(f"[{index}/{total}] failed {spec.output_stem}: {error}")
        return report_row(spec, "failed", None, "", None, error)


def existing_output(layout_dir: Path, output_stem: str) -> Optional[Path]:
    for level in ("L0", "L1", "L2", "L3"):
        existing = sorted((layout_dir / level).glob(f"{output_stem}_*.json"))
        if existing:
            return existing[0]
    return None


def classify(layout: str, reward: int) -> str:
    q1, q2, q3, _q4 = LAYOUT_LEVELS[layout]
    if reward <= q1:
        return "L0"
    if reward <= q2:
        return "L1"
    if reward <= q3:
        return "L2"
    return "L3"


def reward_from_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dynamic_states = data.get("dynamicState", [])
    if not dynamic_states:
        return 0
    return int(dynamic_states[-1].get("score", 0))


def report_row(
    spec: Spec,
    status: str,
    reward: Optional[int],
    level: str,
    output_path: Optional[Path],
    error: str,
) -> dict:
    return {
        "layout": spec.layout,
        "seed": spec.seed,
        "checkpoint": spec.checkpoint,
        "status": status,
        "reward": "" if reward is None else reward,
        "level": level,
        "output_path": "" if output_path is None else output_path,
        "error": error,
    }


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

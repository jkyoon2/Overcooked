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


@dataclass(frozen=True)
class LayoutConfig:
    q1: int
    q2: int
    q3: int
    q4: int
    seeds: range
    num_agents: int


LAYOUT_CONFIGS = {
    "2_forced_hard": LayoutConfig(40, 80, 120, 160, range(1, 19), 2),
    "2_incentivized_hard": LayoutConfig(45, 90, 135, 180, range(1, 11), 2),
    "2_forced_hard_4": LayoutConfig(45, 90, 135, 180, range(6, 18), 4),
    "2_incentivized_hard_4": LayoutConfig(60, 120, 180, 240, range(6, 11), 4),
}

LEVELS = ("L0", "L1", "L2", "L3")


@dataclass(frozen=True)
class Spec:
    layout: str
    seed: int
    checkpoint: int
    actor_path: Path
    critic_path: Path

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.layout, self.seed, self.checkpoint)

    @property
    def output_stem(self) -> str:
        return f"{self.layout}_seed{self.seed}_{self.checkpoint}"


@dataclass(frozen=True)
class ExistingFile:
    path: Path
    layout: str
    seed: int
    checkpoint: int

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.layout, self.seed, self.checkpoint)

    @property
    def output_stem(self) -> str:
        return f"{self.layout}_seed{self.seed}_{self.checkpoint}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate shared rMAPPO SP realtime JSON trajectories for requested hard "
            "layouts, then classify positive final scores into L0-L3 folders."
        )
    )
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUT_CONFIGS))
    parser.add_argument("--results_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--inventory_path", type=Path, default=None)
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min_age_seconds", type=float, default=60.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--keep_out_of_scope",
        action="store_true",
        help="Do not delete existing JSON files whose seed is outside the requested range.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate target seed/checkpoint JSONs even when a positive in-range JSON already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    results_root = (args.results_root or repo_root / "results").resolve()
    output_root = (args.output_root or repo_root / "json_trajectory").resolve()
    inventory_path = (
        args.inventory_path
        or output_root / "available_models_requested_level_trajectories.tsv"
    ).resolve()
    report_path = (
        args.report_path
        or output_root / "generation_report_requested_level_trajectories.tsv"
    ).resolve()

    layouts = validate_layouts(args.layouts)
    specs = discover_specs(results_root, layouts, args.min_age_seconds)
    output_root.mkdir(parents=True, exist_ok=True)
    write_inventory(inventory_path, specs)

    existing_files = discover_existing_files(output_root, layouts)
    planned_existing = summarize_existing(existing_files, keep_out_of_scope=args.keep_out_of_scope)
    pending_specs = specs
    if not args.overwrite:
        existing_keys = {
            existing.key
            for existing in existing_files
            if existing.layout in layouts
            and existing.seed in LAYOUT_CONFIGS[existing.layout].seeds
        }
        pending_specs = [spec for spec in specs if spec.key not in existing_keys]

    print(f"DISCOVERED_MODEL_COUNT={len(specs)}")
    for layout in layouts:
        print(f"DISCOVERED {layout} {sum(1 for spec in specs if spec.layout == layout)}")
    print(f"EXISTING_JSON_COUNT={len(existing_files)}")
    print(f"EXISTING_TARGET_KEYS={planned_existing['target_keys']}")
    print(f"EXISTING_OUT_OF_SCOPE={planned_existing['out_of_scope']}")
    print(f"PENDING_MODEL_COUNT={len(pending_specs)}")
    for layout in layouts:
        print(f"PENDING {layout} {sum(1 for spec in pending_specs if spec.layout == layout)}")
    print(f"INVENTORY={inventory_path}")

    if args.dry_run:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
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

        existing_by_key = normalize_existing_files(
            output_root=output_root,
            layouts=layouts,
            keep_out_of_scope=args.keep_out_of_scope,
            writer=writer,
            report_file=report_file,
        )

        if args.overwrite:
            specs_to_generate = specs
        else:
            specs_to_generate = [spec for spec in specs if spec.key not in existing_by_key]

        if args.workers <= 1:
            for index, spec in enumerate(specs_to_generate, start=1):
                row = generate_one(
                    index=index,
                    total=len(specs_to_generate),
                    spec=spec,
                    results_root=results_root,
                    output_root=output_root,
                    device=args.device,
                )
                writer.writerow(row)
                report_file.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        generate_one,
                        index,
                        len(specs_to_generate),
                        spec,
                        results_root,
                        output_root,
                        args.device,
                    )
                    for index, spec in enumerate(specs_to_generate, start=1)
                ]
                for future in as_completed(futures):
                    writer.writerow(future.result())
                    report_file.flush()

    print(f"REPORT={report_path}")


def validate_layouts(layouts: list[str]) -> list[str]:
    unknown = sorted(set(layouts) - set(LAYOUT_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown layouts: {unknown}")
    return layouts


def discover_specs(results_root: Path, layouts: list[str], min_age_seconds: float) -> list[Spec]:
    now = time.time()
    specs: list[Spec] = []
    for layout in layouts:
        config = LAYOUT_CONFIGS[layout]
        exp_dir = results_root / "Overcooked" / layout / "shared" / "rmappo" / "sp"
        if not exp_dir.exists():
            continue
        for seed in config.seeds:
            models_dir = exp_dir / f"seed{seed}" / "models"
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


def write_inventory(path: Path, specs: list[Spec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layout", "seed", "checkpoint", "actor_path", "critic_path"],
            delimiter="\t",
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "layout": spec.layout,
                    "seed": spec.seed,
                    "checkpoint": spec.checkpoint,
                    "actor_path": spec.actor_path,
                    "critic_path": spec.critic_path,
                }
            )


def discover_existing_files(output_root: Path, layouts: list[str]) -> list[ExistingFile]:
    existing: list[ExistingFile] = []
    for layout in layouts:
        layout_dir = output_root / layout
        if not layout_dir.exists():
            continue
        candidates = list(layout_dir.glob("*.json"))
        for level in LEVELS:
            candidates.extend((layout_dir / level).glob("*.json"))
        for path in sorted(candidates):
            parsed = parse_existing_file(path, layouts)
            if parsed is not None:
                existing.append(parsed)
    return existing


def summarize_existing(existing_files: list[ExistingFile], keep_out_of_scope: bool) -> dict[str, int]:
    target_keys = set()
    out_of_scope = 0
    for existing in existing_files:
        if existing.seed in LAYOUT_CONFIGS[existing.layout].seeds:
            target_keys.add(existing.key)
        elif not keep_out_of_scope:
            out_of_scope += 1
    return {"target_keys": len(target_keys), "out_of_scope": out_of_scope}


def normalize_existing_files(
    output_root: Path,
    layouts: list[str],
    keep_out_of_scope: bool,
    writer: csv.DictWriter,
    report_file,
) -> dict[tuple[str, int, int], Path]:
    existing_by_key: dict[tuple[str, int, int], Path] = {}
    existing_files = discover_existing_files(output_root, layouts)
    for existing in existing_files:
        config = LAYOUT_CONFIGS[existing.layout]
        if existing.seed not in config.seeds:
            if keep_out_of_scope:
                continue
            existing.path.unlink()
            row = report_row(
                existing.layout,
                existing.seed,
                existing.checkpoint,
                "pruned_out_of_scope",
                None,
                "",
                None,
                "",
            )
            writer.writerow(row)
            report_file.flush()
            continue

        try:
            reward = reward_from_json(existing.path)
        except Exception as exc:
            row = report_row(
                existing.layout,
                existing.seed,
                existing.checkpoint,
                "failed_existing_read",
                None,
                "",
                existing.path,
                f"{type(exc).__name__}: {exc}",
            )
            writer.writerow(row)
            report_file.flush()
            continue

        level = classify(existing.layout, reward)
        if level is None:
            existing.path.unlink()
            status = "discarded_existing_nonpositive" if reward <= 0 else "discarded_existing_out_of_range"
            row = report_row(
                existing.layout,
                existing.seed,
                existing.checkpoint,
                status,
                reward,
                "",
                None,
                "",
            )
            writer.writerow(row)
            report_file.flush()
            continue

        target_dir = output_root / existing.layout / level
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{existing.output_stem}_{reward}.json"
        kept_path = existing_by_key.get(existing.key)
        if kept_path is not None and kept_path != existing.path:
            existing.path.unlink()
            row = report_row(
                existing.layout,
                existing.seed,
                existing.checkpoint,
                "removed_duplicate_existing",
                reward,
                level,
                kept_path,
                "",
            )
            writer.writerow(row)
            report_file.flush()
            continue

        if existing.path != target_path:
            if target_path.exists():
                existing.path.unlink()
                kept_path = target_path
                status = "removed_duplicate_existing"
            else:
                existing.path.replace(target_path)
                kept_path = target_path
                status = "reclassified_existing"
        else:
            kept_path = existing.path
            status = "existing"

        existing_by_key[existing.key] = kept_path
        row = report_row(
            existing.layout,
            existing.seed,
            existing.checkpoint,
            status,
            reward,
            level,
            kept_path,
            "",
        )
        writer.writerow(row)
        report_file.flush()
    return existing_by_key


def generate_one(
    index: int,
    total: int,
    spec: Spec,
    results_root: Path,
    output_root: Path,
    device: str,
) -> dict:
    config = LAYOUT_CONFIGS[spec.layout]
    layout_dir = output_root / spec.layout
    layout_dir.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        (layout_dir / level).mkdir(parents=True, exist_ok=True)

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
        str(config.num_agents),
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
        level = classify(spec.layout, reward)
        if level is None:
            tmp_path.unlink()
            status = "discarded_nonpositive" if reward <= 0 else "discarded_out_of_range"
            print(f"[{index}/{total}] discard score={reward} {spec.output_stem}")
            return report_row(spec.layout, spec.seed, spec.checkpoint, status, reward, "", None, "")

        output_path = layout_dir / level / f"{spec.output_stem}_{reward}.json"
        if output_path.exists():
            output_path.unlink()
        tmp_path.replace(output_path)
        print(f"[{index}/{total}] saved {output_path}")
        return report_row(
            spec.layout,
            spec.seed,
            spec.checkpoint,
            "generated",
            reward,
            level,
            output_path,
            "",
        )
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
        return report_row(spec.layout, spec.seed, spec.checkpoint, "failed", None, "", None, error)


def classify(layout: str, reward: int) -> Optional[str]:
    config = LAYOUT_CONFIGS[layout]
    if reward <= 0:
        return None
    if reward <= config.q1:
        return "L0"
    if reward <= config.q2:
        return "L1"
    if reward <= config.q3:
        return "L2"
    if reward <= config.q4:
        return "L3"
    return None


def reward_from_json(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    dynamic_states = data.get("dynamicState", [])
    if not dynamic_states:
        return 0
    return int(dynamic_states[-1].get("score", 0))


def parse_existing_file(path: Path, layouts: list[str]) -> Optional[ExistingFile]:
    if path.name.startswith(".") or not path.name.endswith(".json"):
        return None
    for layout in sorted(layouts, key=len, reverse=True):
        prefix = f"{layout}_seed"
        if not path.name.startswith(prefix):
            continue
        stem = path.name[:-5]
        rest = stem[len(prefix) :]
        parts = rest.split("_")
        if len(parts) < 3:
            return None
        try:
            seed = int(parts[0])
            checkpoint = int(parts[1])
        except ValueError:
            return None
        return ExistingFile(path=path, layout=layout, seed=seed, checkpoint=checkpoint)
    return None


def report_row(
    layout: str,
    seed: int,
    checkpoint: int,
    status: str,
    reward: Optional[int],
    level: str,
    output_path: Optional[Path],
    error: str,
) -> dict:
    return {
        "layout": layout,
        "seed": seed,
        "checkpoint": checkpoint,
        "status": status,
        "reward": "" if reward is None else reward,
        "level": level,
        "output_path": "" if output_path is None else output_path,
        "error": error,
    }


def parse_checkpoint(path: Path) -> int:
    return int(path.stem.replace("actor_periodic_", "", 1))


if __name__ == "__main__":
    main()

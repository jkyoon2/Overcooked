from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


LAYOUT_NUM_AGENTS = {
    "2_forced_hard": 2,
    "2_forced_hard_4": 4,
    "2_incentivized_hard": 2,
    "2_incentivized_hard_4": 4,
}
SCORE_BUCKETS = (20, 40, 60, 80, 100)
MAX_TICKS = 200


@dataclass(frozen=True)
class Spec:
    layout: str
    seed: int
    checkpoint: int
    actor_path: Path
    critic_path: Path

    @property
    def key(self) -> tuple[str, int, int]:
        return self.layout, self.seed, self.checkpoint

    @property
    def stem(self) -> str:
        return f"{self.layout}_seed{self.seed}_{self.checkpoint}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic shared/rMAPPO/SP trajectories with old pot "
            "dynamics, capped at 200 action ticks, and group exact final scores "
            "under json_trajectory_200/<layout>/{20,40,60,80,100}/."
        )
    )
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUT_NUM_AGENTS))
    parser.add_argument("--results_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min_age_seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    results_root = (args.results_root or repo_root / "results").resolve()
    output_root = (args.output_root or repo_root / "json_trajectory_200").resolve()
    layouts = validate_layouts(args.layouts)
    specs = discover_specs(results_root, layouts, args.min_age_seconds)

    output_root.mkdir(parents=True, exist_ok=True)
    for layout in layouts:
        for score in SCORE_BUCKETS:
            (output_root / layout / str(score)).mkdir(parents=True, exist_ok=True)

    inventory_path = output_root / "available_models_200.tsv"
    report_path = output_root / "generation_report_200.tsv"
    summary_path = output_root / "verification_summary_200.json"
    write_inventory(inventory_path, specs)

    print(f"MODE=shared/rmappo/sp old_dynamics=true max_ticks={MAX_TICKS}", flush=True)
    print(f"DISCOVERED_MODEL_COUNT={len(specs)}", flush=True)
    for layout in layouts:
        layout_specs = [spec for spec in specs if spec.layout == layout]
        seed_count = len({spec.seed for spec in layout_specs})
        print(
            f"DISCOVERED {layout} models={len(layout_specs)} seeds={seed_count}",
            flush=True,
        )
    print(f"OUTPUT_ROOT={output_root}", flush=True)
    if args.dry_run:
        return

    remove_stale_outputs(output_root, specs, overwrite=args.overwrite)
    tmp_dir = output_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, spec in enumerate(specs, start=1):
            rows.append(
                generate_one(
                    index,
                    len(specs),
                    spec,
                    repo_root,
                    results_root,
                    output_root,
                    tmp_dir,
                    args.device,
                    args.overwrite,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    generate_one,
                    index,
                    len(specs),
                    spec,
                    repo_root,
                    results_root,
                    output_root,
                    tmp_dir,
                    args.device,
                    args.overwrite,
                )
                for index, spec in enumerate(specs, start=1)
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda row: (row["layout"], int(row["seed"]), int(row["checkpoint"])))
    write_report(report_path, rows)
    summary = verify_results(output_root, specs, rows)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for tmp_path in tmp_dir.glob("*.json"):
        tmp_path.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"REPORT={report_path}", flush=True)
    print(f"VERIFICATION_SUMMARY={summary_path}", flush=True)
    print(f"VERIFICATION_PASS={summary['verification_pass']}", flush=True)
    if not summary["verification_pass"]:
        raise RuntimeError(
            "Generation completed but verification failed; inspect verification_summary_200.json"
        )


def validate_layouts(layouts: list[str]) -> list[str]:
    unknown = sorted(set(layouts) - set(LAYOUT_NUM_AGENTS))
    if unknown:
        raise ValueError(f"Unknown layouts: {unknown}")
    if len(set(layouts)) != len(layouts):
        raise ValueError("Duplicate layout arguments are not allowed.")
    return layouts


def discover_specs(
    results_root: Path,
    layouts: list[str],
    min_age_seconds: float,
) -> list[Spec]:
    now = time.time()
    specs: list[Spec] = []
    for layout in layouts:
        experiment_dir = (
            results_root / "Overcooked" / layout / "shared" / "rmappo" / "sp"
        )
        if not experiment_dir.exists():
            continue
        seed_dirs = sorted(experiment_dir.glob("seed*"), key=parse_seed_dir)
        for seed_dir in seed_dirs:
            seed = parse_seed_dir(seed_dir)
            models_dir = seed_dir / "models"
            if not models_dir.is_dir():
                continue
            actor_paths = sorted(
                models_dir.glob("actor_periodic_*.pt"),
                key=parse_checkpoint,
            )
            for actor_path in actor_paths:
                checkpoint = parse_checkpoint(actor_path)
                critic_path = models_dir / f"critic_periodic_{checkpoint}.pt"
                if not critic_path.exists():
                    continue
                checkpoint_age = min(
                    now - actor_path.stat().st_mtime,
                    now - critic_path.stat().st_mtime,
                )
                if checkpoint_age < min_age_seconds:
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


def parse_seed_dir(path: Path) -> int:
    return int(path.name.removeprefix("seed"))


def parse_checkpoint(path: Path) -> int:
    return int(path.stem.removeprefix("actor_periodic_"))


def write_inventory(path: Path, specs: list[Spec]) -> None:
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


def remove_stale_outputs(
    output_root: Path,
    specs: list[Spec],
    overwrite: bool,
) -> None:
    if not overwrite:
        return
    target_stems = {spec.stem for spec in specs}
    for layout in LAYOUT_NUM_AGENTS:
        for score in SCORE_BUCKETS:
            for path in (output_root / layout / str(score)).glob("*.json"):
                file_stem = path.stem
                if any(
                    file_stem.startswith(f"{target_stem}_")
                    for target_stem in target_stems
                ):
                    path.unlink()


def generate_one(
    index: int,
    total: int,
    spec: Spec,
    repo_root: Path,
    results_root: Path,
    output_root: Path,
    tmp_dir: Path,
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    existing = find_existing(output_root, spec)
    if existing is not None and not overwrite:
        try:
            with existing.open("r", encoding="utf-8") as f:
                data = json.load(f)
            metrics = inspect_trajectory(data)
            reward = final_score(data)
            return report_row(spec, "existing", reward, existing, metrics, "")
        except Exception as exc:
            return report_row(
                spec,
                "failed_existing_read",
                None,
                None,
                empty_metrics(),
                f"{type(exc).__name__}: {exc}",
            )

    tmp_path = tmp_dir / f".{spec.stem}.tmp.json"
    command = [
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
        str(LAYOUT_NUM_AGENTS[spec.layout]),
        "--old_dynamics",
        "--max_ticks",
        str(MAX_TICKS),
    ]
    print(
        f"[{index}/{total}] {spec.layout} seed={spec.seed} checkpoint={spec.checkpoint}",
        flush=True,
    )
    try:
        process_env = os.environ.copy()
        process_env["OMP_NUM_THREADS"] = "1"
        process_env["MKL_NUM_THREADS"] = "1"
        process_env["OPENBLAS_NUM_THREADS"] = "1"
        process_env["NUMEXPR_NUM_THREADS"] = "1"
        subprocess.run(
            command,
            cwd=repo_root,
            env=process_env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with tmp_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        reward = final_score(data)
        metrics = inspect_trajectory(data)

        if reward not in SCORE_BUCKETS:
            tmp_path.unlink()
            return report_row(
                spec,
                "discarded_score_not_requested",
                reward,
                None,
                metrics,
                "",
            )

        output_path = (
            output_root
            / spec.layout
            / str(reward)
            / f"{spec.stem}_{reward}.json"
        )
        if output_path.exists():
            output_path.unlink()
        tmp_path.replace(output_path)
        return report_row(spec, "generated", reward, output_path, metrics, "")
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        captured_output = ""
        if isinstance(exc, subprocess.CalledProcessError):
            captured_output = " | ".join(
                (exc.stdout or "").strip().splitlines()[-20:]
            )
        error = f"{type(exc).__name__}: {exc}"
        if captured_output:
            error = f"{error} | {captured_output}"
        print(f"[{index}/{total}] FAILED {spec.stem}: {error}", flush=True)
        return report_row(
            spec,
            "failed",
            None,
            None,
            empty_metrics(),
            error,
        )


def find_existing(output_root: Path, spec: Spec) -> Optional[Path]:
    matches: list[Path] = []
    for score in SCORE_BUCKETS:
        matches.extend(
            (output_root / spec.layout / str(score)).glob(f"{spec.stem}_*.json")
        )
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate existing outputs for {spec.key}: {matches}")
    return matches[0] if matches else None


def final_score(data: dict[str, Any]) -> int:
    states = data.get("dynamicState", [])
    return int(states[-1].get("score", 0)) if states else 0


def inspect_trajectory(data: dict[str, Any]) -> dict[str, int]:
    states = data.get("dynamicState", [])
    if not states:
        raise ValueError("dynamicState is empty")
    timesteps = [int(state["timestep"]) for state in states]
    if timesteps != list(range(len(states))):
        raise ValueError("Timesteps are not consecutive from zero")

    metrics = empty_metrics()
    metrics["last_timestep"] = timesteps[-1]
    metrics["dynamic_state_count"] = len(states)
    for previous, current in zip(states, states[1:]):
        previous_soups = soups_by_position(previous)
        current_soups = soups_by_position(current)
        for position, soup in current_soups.items():
            ingredients = soup.get("ingredients", [])
            if len(ingredients) != 3 or bool(soup.get("isReady", False)):
                continue
            tick = int(soup.get("cookingTick", 0))
            previous_soup = previous_soups.get(position)
            previous_ingredients = (
                [] if previous_soup is None else previous_soup.get("ingredients", [])
            )
            previous_tick = (
                0 if previous_soup is None else int(previous_soup.get("cookingTick", 0))
            )

            if len(previous_ingredients) < 3:
                metrics["third_ingredient_additions"] += 1
                if tick == 1:
                    metrics["automatic_cook_starts_at_tick_1"] += 1
                else:
                    metrics["auto_start_violations"] += 1
            if tick == 0:
                metrics["full_pot_idle_frames"] += 1
            if (
                len(previous_ingredients) == 3
                and 0 < previous_tick < int(soup.get("cookTime", 20))
            ):
                metrics["cooking_tick_increment_steps"] += 1
                if tick != previous_tick + 1:
                    metrics["cooking_tick_increment_violations"] += 1
    return metrics


def empty_metrics() -> dict[str, int]:
    return {
        "last_timestep": -1,
        "dynamic_state_count": 0,
        "third_ingredient_additions": 0,
        "automatic_cook_starts_at_tick_1": 0,
        "auto_start_violations": 0,
        "full_pot_idle_frames": 0,
        "cooking_tick_increment_steps": 0,
        "cooking_tick_increment_violations": 0,
    }


def soups_by_position(state: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    soups: dict[tuple[int, int], dict[str, Any]] = {}
    for obj in state.get("objects", []):
        if obj.get("name") != "soup":
            continue
        position = obj.get("position", {})
        soups[(int(position["x"]), int(position["y"]))] = obj
    return soups


def report_row(
    spec: Spec,
    status: str,
    reward: Optional[int],
    output_path: Optional[Path],
    metrics: dict[str, int],
    error: str,
) -> dict[str, Any]:
    return {
        "layout": spec.layout,
        "seed": spec.seed,
        "checkpoint": spec.checkpoint,
        "status": status,
        "reward": "" if reward is None else reward,
        "output_path": "" if output_path is None else str(output_path),
        **metrics,
        "error": error,
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layout",
        "seed",
        "checkpoint",
        "status",
        "reward",
        "output_path",
        *empty_metrics().keys(),
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def verify_results(
    output_root: Path,
    specs: list[Spec],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_keys = {spec.key for spec in specs}
    report_keys = {
        (str(row["layout"]), int(row["seed"]), int(row["checkpoint"]))
        for row in rows
    }
    if expected_keys != report_keys:
        errors.append("Inventory and report model-key sets differ.")
    if len(report_keys) != len(rows):
        errors.append("Report contains duplicate model keys.")

    generated_paths = {
        Path(str(row["output_path"])).resolve()
        for row in rows
        if row["status"] in {"generated", "existing"}
    }
    actual_paths = {
        path.resolve()
        for layout in LAYOUT_NUM_AGENTS
        for score in SCORE_BUCKETS
        for path in (output_root / layout / str(score)).glob("*.json")
    }
    if generated_paths != actual_paths:
        errors.append(
            f"Report/file mismatch: reported={len(generated_paths)} actual={len(actual_paths)}"
        )

    score_counts: Counter[int] = Counter()
    layout_score_counts: dict[str, Counter[int]] = defaultdict(Counter)
    parse_errors = 0
    metadata_errors = 0
    timestep_errors = 0
    for path in sorted(actual_paths):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            states = data["dynamicState"]
            reward = int(states[-1]["score"])
            directory_score = int(path.parent.name)
            directory_layout = path.parent.parent.name
            layout = str(data["staticInfo"]["layoutName"])
            if reward != directory_score or reward not in SCORE_BUCKETS:
                metadata_errors += 1
            if layout not in LAYOUT_NUM_AGENTS or layout != directory_layout:
                metadata_errors += 1
            if (
                len(states) != MAX_TICKS + 1
                or int(states[0]["timestep"]) != 0
                or int(states[-1]["timestep"]) != MAX_TICKS
            ):
                timestep_errors += 1
            score_counts[reward] += 1
            layout_score_counts[layout][reward] += 1
        except Exception:
            parse_errors += 1

    failed = sum(1 for row in rows if str(row["status"]).startswith("failed"))
    transition_metrics = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "third_ingredient_additions",
            "automatic_cook_starts_at_tick_1",
            "auto_start_violations",
            "full_pot_idle_frames",
            "cooking_tick_increment_steps",
            "cooking_tick_increment_violations",
        )
    }
    report_timestep_errors = sum(
        1
        for row in rows
        if not str(row["status"]).startswith("failed")
        and (
            int(row["last_timestep"]) != MAX_TICKS
            or int(row["dynamic_state_count"]) != MAX_TICKS + 1
        )
    )

    if failed:
        errors.append(f"{failed} generation/report rows failed.")
    if parse_errors:
        errors.append(f"{parse_errors} saved JSON files could not be parsed.")
    if metadata_errors:
        errors.append(f"{metadata_errors} saved JSON score/layout checks failed.")
    if timestep_errors or report_timestep_errors:
        errors.append(
            "200-timestep validation failed: "
            f"saved={timestep_errors} report={report_timestep_errors}"
        )
    if transition_metrics["third_ingredient_additions"] == 0:
        errors.append("No third-ingredient transition was observed.")
    if (
        transition_metrics["third_ingredient_additions"]
        != transition_metrics["automatic_cook_starts_at_tick_1"]
    ):
        errors.append("Not every third ingredient automatically started cooking at tick 1.")
    if transition_metrics["auto_start_violations"]:
        errors.append("Automatic cooking start violations were observed.")
    if transition_metrics["full_pot_idle_frames"]:
        errors.append("Full, uncooked pot frames were observed.")
    if transition_metrics["cooking_tick_increment_violations"]:
        errors.append("Cooking timer increment violations were observed.")

    status_counts = Counter(str(row["status"]) for row in rows)
    seed_counts = Counter((spec.layout, spec.seed) for spec in specs)
    return {
        "verification_pass": not errors,
        "generation_mode": {
            "checkpoint_family": "shared/rmappo/sp",
            "transition_dynamics": "old",
            "automatic_cook_on_third_ingredient": True,
            "max_action_ticks": MAX_TICKS,
            "dynamic_states_per_complete_rollout": MAX_TICKS + 1,
            "directory_structure": "<layout>/<score>",
            "layouts": list(LAYOUT_NUM_AGENTS),
            "score_directories_per_layout": [
                str(score) for score in SCORE_BUCKETS
            ],
            "output_root": str(output_root),
        },
        "coverage": {
            "inventory_rows": len(specs),
            "report_rows": len(rows),
            "unique_model_keys": len(report_keys),
            "layout_seed_counts": {
                layout: len({spec.seed for spec in specs if spec.layout == layout})
                for layout in LAYOUT_NUM_AGENTS
            },
            "rows_per_seed_values": sorted(set(seed_counts.values())),
            "failed": failed,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "saved_files": {
            "count": len(actual_paths),
            "parse_errors": parse_errors,
            "metadata_errors": metadata_errors,
            "timestep_errors": timestep_errors,
            "report_timestep_errors": report_timestep_errors,
            "score_counts": {
                str(score): score_counts[score] for score in SCORE_BUCKETS
            },
            "layout_score_counts": {
                layout: {
                    str(score): layout_score_counts[layout][score]
                    for score in SCORE_BUCKETS
                }
                for layout in LAYOUT_NUM_AGENTS
            },
        },
        "old_dynamics_validation_all_rollouts": transition_metrics,
        "errors": errors,
    }


if __name__ == "__main__":
    main()

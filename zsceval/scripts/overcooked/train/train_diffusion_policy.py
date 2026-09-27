from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zsceval.algorithms.diffusion_policy import (  # noqa: E402
    ID_TO_RECIPE,
    NULL_GOAL_ID,
    RECIPE_TO_ID,
    DDPMScheduler,
    DiffusionOvercookedPolicy,
    OvercookedTrajectoryDataset,
    discover_trajectory_files,
    split_train_val_files,
    split_train_val_test_files,
)
from zsceval.algorithms.diffusion_policy.trainer import (  # noqa: E402
    eval_step,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Overcooked conditional diffusion policy")
    parser.add_argument(
        "--config",
        type=str,
        default="zsceval/scripts/overcooked/config/diffusion_policy.yaml",
    )
    parser.add_argument("--data-root", type=str, default="trajectories")
    parser.add_argument("--layout", type=str, required=True)
    parser.add_argument("--agent-id", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-obs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.n_obs is not None:
        config["data"]["n_obs"] = args.n_obs
    return config


def choose_device(requested: str | None, smoke_test: bool = False) -> torch.device:
    if smoke_test:
        return torch.device("cpu")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets(
    config: dict[str, Any],
    data_root: str | Path,
    layout: str,
    agent_id: int,
    seed: int,
) -> tuple[
    OvercookedTrajectoryDataset,
    OvercookedTrajectoryDataset | None,
    OvercookedTrajectoryDataset | None,
    list,
    list,
    list,
]:
    data_cfg = config["data"]
    files = discover_trajectory_files(data_root, layout=layout, recipes=list(RECIPE_TO_ID))
    if not files:
        raise FileNotFoundError(f"No trajectory files found for layout={layout!r} under {data_root}")
    found_recipes = {item.recipe for item in files}
    missing_recipes = set(RECIPE_TO_ID) - found_recipes
    if missing_recipes:
        raise ValueError(
            f"Layout {layout!r} is missing recipe trajectories: {sorted(missing_recipes)}"
        )

    test_files_per_recipe = int(data_cfg.get("test_files_per_recipe", 0))
    if test_files_per_recipe > 0:
        train_files, val_files, test_files = split_train_val_test_files(
            files,
            train_files_per_recipe=data_cfg.get("train_files_per_recipe"),
            val_files_per_recipe=int(data_cfg.get("val_files_per_recipe", 1)),
            test_files_per_recipe=test_files_per_recipe,
            seed=seed,
        )
    else:
        train_files, val_files = split_train_val_files(
            files,
            val_files_per_recipe=int(data_cfg.get("val_files_per_recipe", 1)),
            seed=seed,
        )
        test_files = []
    train_dataset = OvercookedTrajectoryDataset(
        train_files,
        agent_id=agent_id,
        n_obs=int(data_cfg["n_obs"]),
        horizon=int(data_cfg["horizon"]),
        action_dim=int(data_cfg["action_dim"]),
        label_smoothing=float(data_cfg.get("label_smoothing", 0.0)),
    )
    val_dataset = None
    if val_files:
        val_dataset = OvercookedTrajectoryDataset(
            val_files,
            agent_id=agent_id,
            n_obs=int(data_cfg["n_obs"]),
            horizon=int(data_cfg["horizon"]),
            action_dim=int(data_cfg["action_dim"]),
            label_smoothing=float(data_cfg.get("label_smoothing", 0.0)),
        )
    test_dataset = None
    if test_files:
        test_dataset = OvercookedTrajectoryDataset(
            test_files,
            agent_id=agent_id,
            n_obs=int(data_cfg["n_obs"]),
            horizon=int(data_cfg["horizon"]),
            action_dim=int(data_cfg["action_dim"]),
            label_smoothing=float(data_cfg.get("label_smoothing", 0.0)),
        )
    return train_dataset, val_dataset, test_dataset, train_files, val_files, test_files


def build_model_and_scheduler(
    config: dict[str, Any],
    obs_shape: tuple[int, int, int],
) -> tuple[DiffusionOvercookedPolicy, DDPMScheduler]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]
    model = DiffusionOvercookedPolicy(
        obs_shape=obs_shape,
        n_obs=int(data_cfg["n_obs"]),
        horizon=int(data_cfg["horizon"]),
        n_action_steps=int(data_cfg["n_action_steps"]),
        action_dim=int(data_cfg["action_dim"]),
        obs_emb_dim=int(model_cfg["obs_emb_dim"]),
        goal_emb_dim=int(model_cfg["goal_emb_dim"]),
        down_dims=tuple(int(v) for v in model_cfg["down_dims"]),
        num_goals=5,
        num_diffusion_steps=int(diffusion_cfg["num_diffusion_steps"]),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=int(diffusion_cfg["num_diffusion_steps"]),
        beta_start=float(diffusion_cfg.get("beta_start", 1e-4)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
    )
    model.scheduler = scheduler
    return model, scheduler


def checkpoint_path(config: dict[str, Any], args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root or config["paths"].get("output_root", "results"))
    experiment = args.experiment_name or config["paths"].get("experiment_name", "diffusion_policy")
    return (
        output_root
        / "Overcooked"
        / args.layout
        / "diffusion_policy"
        / experiment
        / f"agent{args.agent_id}"
        / f"seed{args.seed}"
        / "policy.pt"
    )


def best_checkpoint_path(config: dict[str, Any], args: argparse.Namespace) -> Path:
    return checkpoint_path(config, args).with_name("best_policy.pt")


def make_metadata(
    args: argparse.Namespace,
    train_files: list,
    val_files: list,
    test_files: list,
    obs_shape: tuple[int, int, int],
) -> dict:
    return {
        "layout": args.layout,
        "agent_id": args.agent_id,
        "seed": args.seed,
        "obs_shape": obs_shape,
        "recipe_to_id": RECIPE_TO_ID,
        "id_to_recipe": ID_TO_RECIPE,
        "null_goal_id": NULL_GOAL_ID,
        "train_files": [str(item.path) for item in train_files],
        "val_files": [str(item.path) for item in val_files],
        "test_files": [str(item.path) for item in test_files],
    }


def init_wandb(
    config: dict[str, Any],
    args: argparse.Namespace,
    metadata: dict[str, Any],
    ckpt_path: Path,
):
    logging_cfg = config.get("logging", {})
    use_wandb = bool(logging_cfg.get("use_wandb", False)) or args.use_wandb
    if not use_wandb:
        return None

    import wandb

    project = args.wandb_project or logging_cfg.get("project", "Overcooked-Diffusion")
    entity = args.wandb_entity or logging_cfg.get("entity")
    name = (
        args.wandb_name
        or logging_cfg.get("name")
        or f"{args.layout}_agent{args.agent_id}_seed{args.seed}"
    )
    tags = args.wandb_tags if args.wandb_tags is not None else logging_cfg.get("tags", [])
    run_config = {
        "layout": args.layout,
        "agent_id": args.agent_id,
        "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "data": config.get("data", {}),
        "model": config.get("model", {}),
        "diffusion": config.get("diffusion", {}),
        "training": config.get("training", {}),
        "num_train_files": len(metadata["train_files"]),
        "num_val_files": len(metadata["val_files"]),
        "num_test_files": len(metadata["test_files"]),
    }
    init_kwargs = {
        "project": project,
        "name": name,
        "tags": tags,
        "config": run_config,
        "dir": str(ckpt_path.parent),
    }
    if entity:
        init_kwargs["entity"] = entity
    return wandb.init(**init_kwargs)


def run_smoke_test(config: dict[str, Any], args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device, smoke_test=True)
    train_dataset, _, _, train_files, val_files, test_files = build_datasets(
        config,
        args.data_root,
        args.layout,
        args.agent_id,
        args.seed,
    )
    batch_size = min(2, len(train_dataset))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model, scheduler = build_model_and_scheduler(config, train_dataset.obs_shape)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    obs_hist, z_goal, x0_expert = [item.to(device) for item in batch]
    timesteps = scheduler.sample_timesteps(x0_expert.shape[0], device)
    noise = torch.randn_like(x0_expert)
    noised = scheduler.add_noise(x0_expert, noise, timesteps)
    pred = model(noised, timesteps, obs_hist, z_goal)
    expected_shape = (
        batch_size,
        int(config["data"]["horizon"]),
        int(config["data"]["action_dim"]),
    )
    if tuple(pred.shape) != expected_shape:
        raise AssertionError(f"Forward shape mismatch: got {tuple(pred.shape)}, expected {expected_shape}")

    metrics = train_step(
        model,
        batch,
        optimizer,
        scheduler,
        device,
        p_uncond=float(config["training"].get("p_uncond", 0.1)),
        grad_clip_norm=float(config["training"].get("grad_clip_norm", 1.0)),
    )
    actions = model.sample_action_chunk(
        obs_hist.cpu(),
        z_goal.cpu(),
        scheduler=scheduler,
        num_inference_steps=int(config["diffusion"].get("smoke_inference_steps", 4)),
        scheduler_method=str(config["diffusion"].get("scheduler_method", "ddim")),
        eta=float(config["diffusion"].get("eta", 0.0)),
    )
    expected_actions_shape = (batch_size, int(config["data"]["n_action_steps"]))
    if tuple(actions.shape) != expected_actions_shape:
        raise AssertionError(
            f"Action chunk shape mismatch: got {tuple(actions.shape)}, expected {expected_actions_shape}"
        )
    forced_action = int(config["data"]["action_dim"]) - 1
    available_actions = torch.zeros(batch_size, int(config["data"]["action_dim"]), dtype=torch.bool)
    available_actions[:, forced_action] = True
    masked_actions = model.sample_action_chunk(
        obs_hist.cpu(),
        z_goal.cpu(),
        available_actions=available_actions,
        scheduler=scheduler,
        num_inference_steps=int(config["diffusion"].get("smoke_inference_steps", 4)),
        scheduler_method=str(config["diffusion"].get("scheduler_method", "ddim")),
        eta=float(config["diffusion"].get("eta", 0.0)),
    )
    if not torch.all(masked_actions == forced_action):
        raise AssertionError("available_actions mask did not force the only valid action")

    recipes = sorted({item.recipe for item in train_files + val_files + test_files})
    print(
        json.dumps(
            {
                "status": "ok",
                "forward_shape": list(pred.shape),
                "action_shape": list(actions.shape),
                "masked_action_shape": list(masked_actions.shape),
                "mask_forced_action": forced_action,
                "one_step_loss": metrics["loss"],
                "one_step_action_acc": metrics["action_acc"],
                "one_step_exec_action_acc": metrics["exec_action_acc"],
                "num_train_windows": len(train_dataset),
                "num_train_files": len(train_files),
                "num_val_files": len(val_files),
                "num_test_files": len(test_files),
                "n_obs": int(config["data"]["n_obs"]),
                "recipes": recipes,
                "device": str(device),
            },
            indent=2,
        )
    )


@torch.no_grad()
def evaluate_loader(
    model: DiffusionOvercookedPolicy,
    loader: DataLoader,
    scheduler: DDPMScheduler,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    collected: dict[str, list[float]] = {}
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        metrics = eval_step(model, batch, scheduler, device)
        for key, value in metrics.items():
            collected.setdefault(key, []).append(value)
    return {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in collected.items()
    }


def run_training(config: dict[str, Any], args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device, smoke_test=False)
    train_dataset, val_dataset, test_dataset, train_files, val_files, test_files = build_datasets(
        config,
        args.data_root,
        args.layout,
        args.agent_id,
        args.seed,
    )

    train_cfg = config["training"]
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    epochs = args.epochs or int(train_cfg["epochs"])
    max_train_steps = args.max_train_steps
    if max_train_steps is None:
        max_train_steps = train_cfg.get("max_train_steps")
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    model, scheduler = build_model_and_scheduler(config, train_dataset.obs_shape)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    global_step = 0
    start_epoch = 1
    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, map_location=device, model=model, optimizer=optimizer)
        global_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    ckpt_path = checkpoint_path(config, args)
    best_ckpt_path = best_checkpoint_path(config, args)
    metadata = make_metadata(args, train_files, val_files, test_files, train_dataset.obs_shape)
    wandb_run = init_wandb(config, args, metadata, ckpt_path)
    print(
        json.dumps(
            {
                "layout": args.layout,
                "agent_id": args.agent_id,
                "device": str(device),
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset) if val_dataset else 0,
                "test_windows": len(test_dataset) if test_dataset else 0,
                "n_obs": int(config["data"]["n_obs"]),
                "checkpoint": str(ckpt_path),
                "best_checkpoint": str(best_ckpt_path),
                "resume_from": args.resume_from,
                "start_epoch": start_epoch,
                "start_step": global_step,
            },
            indent=2,
        )
    )

    best_val_exec_acc = float("-inf")
    best_epoch = 0
    for epoch in range(start_epoch, epochs + 1):
        epoch_train_metrics: dict[str, list[float]] = {}
        last_metrics = None
        for batch_idx, batch in enumerate(train_loader, start=1):
            metrics = train_step(
                model,
                batch,
                optimizer,
                scheduler,
                device,
                p_uncond=float(train_cfg.get("p_uncond", 0.1)),
                grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            )
            global_step += 1
            last_metrics = metrics
            for key, value in metrics.items():
                epoch_train_metrics.setdefault(key, []).append(value)
            if global_step % int(train_cfg.get("log_interval", 10)) == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={metrics['loss']:.6f} "
                    f"train_acc={metrics['action_acc']:.4f} "
                    f"train_exec_acc={metrics['exec_action_acc']:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/action_acc": metrics["action_acc"],
                            "train/exec_action_acc": metrics["exec_action_acc"],
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
            if max_train_steps is not None and global_step >= int(max_train_steps):
                break

        if last_metrics is None:
            raise RuntimeError("No training steps were run")
        train_epoch_metrics = {
            key: float(np.mean(values))
            for key, values in epoch_train_metrics.items()
        }

        val_metrics = None
        if val_loader is not None:
            eval_max_batches = train_cfg.get("eval_max_batches")
            if eval_max_batches is not None:
                eval_max_batches = int(eval_max_batches)
            val_metrics = evaluate_loader(model, val_loader, scheduler, device, max_batches=eval_max_batches)
            if val_metrics["exec_action_acc"] > best_val_exec_acc:
                best_val_exec_acc = val_metrics["exec_action_acc"]
                best_epoch = epoch
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    metadata,
                    global_step,
                    epoch,
                )
            print(
                f"epoch={epoch} step={global_step} "
                f"train_loss={train_epoch_metrics['loss']:.6f} "
                f"train_acc={train_epoch_metrics['action_acc']:.4f} "
                f"train_exec_acc={train_epoch_metrics['exec_action_acc']:.4f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['action_acc']:.4f} "
                f"val_exec_acc={val_metrics['exec_action_acc']:.4f}"
            )
        else:
            print(
                f"epoch={epoch} step={global_step} "
                f"train_loss={train_epoch_metrics['loss']:.6f} "
                f"train_acc={train_epoch_metrics['action_acc']:.4f} "
                f"train_exec_acc={train_epoch_metrics['exec_action_acc']:.4f}"
            )

        if wandb_run is not None:
            payload = {
                "epoch": epoch,
                "train/epoch_loss": train_epoch_metrics["loss"],
                "train/epoch_action_acc": train_epoch_metrics["action_acc"],
                "train/epoch_exec_action_acc": train_epoch_metrics["exec_action_acc"],
                "best/val_exec_action_acc": best_val_exec_acc if best_epoch else float("nan"),
                "best/epoch": best_epoch,
            }
            if val_metrics is not None:
                payload.update(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/action_acc": val_metrics["action_acc"],
                        "val/exec_action_acc": val_metrics["exec_action_acc"],
                    }
                )
            wandb_run.log(payload, step=global_step)

        save_interval = int(train_cfg.get("save_interval_epochs", 10))
        if save_interval > 0 and epoch % save_interval == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, config, metadata, global_step, epoch)

        if max_train_steps is not None and global_step >= int(max_train_steps):
            break

    save_checkpoint(ckpt_path, model, optimizer, scheduler, config, metadata, global_step, epoch)
    print(f"Saved checkpoint: {ckpt_path}")
    if test_loader is not None:
        test_metrics = evaluate_loader(model, test_loader, scheduler, device)
        print(
            f"final_test step={global_step} "
            f"test_loss={test_metrics['loss']:.6f} "
            f"test_acc={test_metrics['action_acc']:.4f} "
            f"test_exec_acc={test_metrics['exec_action_acc']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "test_final/loss": test_metrics["loss"],
                    "test_final/action_acc": test_metrics["action_acc"],
                    "test_final/exec_action_acc": test_metrics["exec_action_acc"],
                },
                step=global_step,
            )
        if best_ckpt_path.exists():
            load_checkpoint(best_ckpt_path, map_location=device, model=model)
            best_test_metrics = evaluate_loader(model, test_loader, scheduler, device)
            print(
                f"best_test step={global_step} "
                f"best_epoch={best_epoch} "
                f"test_loss={best_test_metrics['loss']:.6f} "
                f"test_acc={best_test_metrics['action_acc']:.4f} "
                f"test_exec_acc={best_test_metrics['exec_action_acc']:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "test_best/loss": best_test_metrics["loss"],
                        "test_best/action_acc": best_test_metrics["action_acc"],
                        "test_best/exec_action_acc": best_test_metrics["exec_action_acc"],
                    },
                    step=global_step,
                )
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    if args.smoke_test:
        run_smoke_test(config, args)
    else:
        run_training(config, args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset


RECIPE_TO_ID = {"TTT": 0, "TTO": 1, "TOO": 2, "OOO": 3}
ID_TO_RECIPE = {v: k for k, v in RECIPE_TO_ID.items()}
NULL_GOAL_ID = 4


@dataclass(frozen=True)
class TrajectoryFile:
    path: Path
    meta_path: Path
    layout: str
    recipe: str
    recipe_id: int
    participant_id: str
    total_steps: int


class TrajectoryRecord(TypedDict):
    obs: np.ndarray
    actions: np.ndarray
    recipe_id: int
    recipe: str
    layout: str
    path: str



def _as_set(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return set(values)


def discover_trajectory_files(
    data_root: str | Path,
    layout: str | Sequence[str] | None = None,
    recipes: Sequence[str] | None = None,
) -> list[TrajectoryFile]:
    data_root = Path(data_root)
    layout_filter = _as_set(layout)
    recipe_filter = _as_set(recipes)
    files: list[TrajectoryFile] = []

    for npz_path in sorted(data_root.glob("*/*.npz")):
        meta_path = npz_path.with_name(f"{npz_path.stem}_meta.json")
        if not meta_path.exists():
            continue
        with meta_path.open("r") as f:
            meta = json.load(f)

        recipe = meta["recipe"]
        if recipe not in RECIPE_TO_ID:
            continue
        file_layout = meta["layout"]
        if layout_filter is not None and file_layout not in layout_filter:
            continue
        if recipe_filter is not None and recipe not in recipe_filter:
            continue

        files.append(
            TrajectoryFile(
                path=npz_path,
                meta_path=meta_path,
                layout=file_layout,
                recipe=recipe,
                recipe_id=RECIPE_TO_ID[recipe],
                participant_id=meta.get("participant_id", ""),
                total_steps=int(meta.get("total_steps", meta.get("horizon", 0))),
            )
        )

    return files


def split_train_val_files(
    files: Sequence[TrajectoryFile],
    val_files_per_recipe: int = 1,
    seed: int = 1,
) -> tuple[list[TrajectoryFile], list[TrajectoryFile]]:
    if val_files_per_recipe <= 0:
        return list(files), []

    rng = random.Random(seed)
    groups: dict[int, list[TrajectoryFile]] = defaultdict(list)
    for item in files:
        groups[item.recipe_id].append(item)

    train_files: list[TrajectoryFile] = []
    val_files: list[TrajectoryFile] = []
    for recipe_id in sorted(groups):
        group = sorted(groups[recipe_id], key=lambda item: str(item.path))
        rng.shuffle(group)
        n_val = min(val_files_per_recipe, max(0, len(group) - 1))
        val_files.extend(group[:n_val])
        train_files.extend(group[n_val:])

    return sorted(train_files, key=lambda item: str(item.path)), sorted(val_files, key=lambda item: str(item.path))


def split_train_val_test_files(
    files: Sequence[TrajectoryFile],
    train_files_per_recipe: int | None = None,
    val_files_per_recipe: int = 1,
    test_files_per_recipe: int = 1,
    seed: int = 1,
) -> tuple[list[TrajectoryFile], list[TrajectoryFile], list[TrajectoryFile]]:
    rng = random.Random(seed)
    groups: dict[int, list[TrajectoryFile]] = defaultdict(list)
    for item in files:
        groups[item.recipe_id].append(item)

    train_files: list[TrajectoryFile] = []
    val_files: list[TrajectoryFile] = []
    test_files: list[TrajectoryFile] = []
    for recipe_id in sorted(groups):
        group = sorted(groups[recipe_id], key=lambda item: str(item.path))
        rng.shuffle(group)
        requested_train = train_files_per_recipe
        requested_total = val_files_per_recipe + test_files_per_recipe
        if requested_train is not None:
            requested_total += requested_train
        if requested_total > len(group):
            raise ValueError(
                f"Recipe {ID_TO_RECIPE[recipe_id]} has {len(group)} files, "
                f"but split requested {requested_total}"
            )

        n_test = test_files_per_recipe
        n_val = val_files_per_recipe
        n_train = len(group) - n_test - n_val
        if requested_train is not None:
            n_train = requested_train

        test_files.extend(group[:n_test])
        val_files.extend(group[n_test : n_test + n_val])
        train_files.extend(group[n_test + n_val : n_test + n_val + n_train])

    return (
        sorted(train_files, key=lambda item: str(item.path)),
        sorted(val_files, key=lambda item: str(item.path)),
        sorted(test_files, key=lambda item: str(item.path)),
    )


def to_rmappo_cnn_obs(obs: np.ndarray) -> np.ndarray:
    """Return the lossless features that RMAPPo's CNN effectively receives after PPO scaling cancels."""
    obs = obs.astype(np.float32, copy=True)
    if obs.shape[-1] != 25:
        raise ValueError(f"Expected 25 observation channels, got shape {obs.shape}")
    return obs


class OvercookedTrajectoryDataset(Dataset):
    """Windowed diffusion-policy dataset from recorded Overcooked trajectories."""

    def __init__(
        self,
        files: Sequence[TrajectoryFile | str | Path],
        agent_id: int,
        n_obs: int = 2,
        horizon: int = 16,
        action_dim: int = 6,
        label_smoothing: float = 0.0,
    ) -> None:
        if agent_id not in (0, 1):
            raise ValueError(f"agent_id must be 0 or 1, got {agent_id}")
        if n_obs < 1:
            raise ValueError(f"n_obs must be >= 1, got {n_obs}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")

        self.files = [self._coerce_file(item) for item in files]
        self.agent_id = agent_id
        self.n_obs = n_obs
        self.horizon = horizon
        self.action_dim = action_dim
        self.label_smoothing = label_smoothing

        self.trajectories: list[TrajectoryRecord] = []
        self.index: list[tuple[int, int]] = []
        self._obs_shape: tuple[int, int, int] | None = None
        self._build_index()

    @staticmethod
    def _coerce_file(item: TrajectoryFile | str | Path) -> TrajectoryFile:
        if isinstance(item, TrajectoryFile):
            return item
        path = Path(item)
        meta_path = path.with_name(f"{path.stem}_meta.json")
        with meta_path.open("r") as f:
            meta = json.load(f)
        recipe = meta["recipe"]
        return TrajectoryFile(
            path=path,
            meta_path=meta_path,
            layout=meta["layout"],
            recipe=recipe,
            recipe_id=RECIPE_TO_ID[recipe],
            participant_id=meta.get("participant_id", ""),
            total_steps=int(meta.get("total_steps", meta.get("horizon", 0))),
        )

    def _build_index(self) -> None:
        for file_info in self.files:
            obs_key = f"obs_p{self.agent_id}"
            with np.load(file_info.path) as loaded:
                if obs_key not in loaded or "actions" not in loaded:
                    raise KeyError(f"{file_info.path} must contain {obs_key} and actions")

                obs = loaded[obs_key]
                actions = loaded["actions"][:, self.agent_id].astype(np.int64)
            obs = to_rmappo_cnn_obs(obs)

            raw_obs_shape = tuple(int(dim) for dim in obs.shape[1:])
            if len(raw_obs_shape) != 3:
                raise ValueError(
                    f"Expected observations with 3 feature dimensions, got shape {obs.shape}"
                )
            obs_shape = (raw_obs_shape[0], raw_obs_shape[1], raw_obs_shape[2])
            if self._obs_shape is None:
                self._obs_shape = obs_shape
            elif self._obs_shape != obs_shape:
                raise ValueError(
                    f"Inconsistent obs shape in {file_info.path}: "
                    f"{obs.shape[1:]} vs {self._obs_shape}"
                )

            traj_idx = len(self.trajectories)
            usable_steps = min(obs.shape[0], actions.shape[0])
            last_start = usable_steps - self.horizon
            for start_t in range(self.n_obs - 1, last_start + 1):
                self.index.append((traj_idx, start_t))

            self.trajectories.append(
                {
                    "obs": obs,
                    "actions": actions,
                    "recipe_id": file_info.recipe_id,
                    "recipe": file_info.recipe,
                    "layout": file_info.layout,
                    "path": str(file_info.path),
                }
            )

        if not self.index:
            raise ValueError("No valid training windows were found")

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        if self._obs_shape is None:
            raise RuntimeError("Dataset has not been indexed")
        return self._obs_shape

    def recipe_counts(self) -> Counter:
        counts = Counter()
        for traj_idx, _ in self.index:
            counts[int(self.trajectories[traj_idx]["recipe_id"])] += 1
        return counts

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_idx, start_t = self.index[idx]
        traj = self.trajectories[traj_idx]
        obs = traj["obs"]
        actions = traj["actions"]
        recipe_id = int(traj["recipe_id"])

        obs_start = start_t - self.n_obs + 1
        obs_hist = obs[obs_start : start_t + 1]
        action_chunk = actions[start_t : start_t + self.horizon]
        x0 = self._one_hot(action_chunk)

        return (
            torch.from_numpy(obs_hist).float(),
            torch.tensor(recipe_id, dtype=torch.long),
            torch.from_numpy(x0).float(),
        )

    def _one_hot(self, action_chunk: np.ndarray) -> np.ndarray:
        if self.label_smoothing > 0.0:
            off_value = self.label_smoothing / max(1, self.action_dim - 1)
            x0 = np.full((self.horizon, self.action_dim), off_value, dtype=np.float32)
            on_value = 1.0 - self.label_smoothing
        else:
            x0 = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
            on_value = 1.0

        rows = np.arange(self.horizon)
        valid = (action_chunk >= 0) & (action_chunk < self.action_dim)
        x0[rows[valid], action_chunk[valid]] = on_value
        return x0


def recipe_id_set(files: Iterable[TrajectoryFile]) -> set[int]:
    return {item.recipe_id for item in files}

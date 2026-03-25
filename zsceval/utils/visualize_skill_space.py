#!/usr/bin/env python
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import seaborn as sns  # noqa: F401
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.utils.bad_skill_generator import BadSkillGenerator
from zsceval.utils.train_util import get_base_run_dir


class SkillSpaceVisualizer(BadSkillGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.latent_records = {"safe": [], "unsafe": [], "policy": []}
        self.sample_counts = defaultdict(int)
        self.max_samples = {"safe": None, "unsafe": None, "policy": None}

    def _set_limits(self, max_safe=None, max_unsafe=None, max_policy=None):
        self.max_samples = {"safe": max_safe, "unsafe": max_unsafe, "policy": max_policy}

    def _can_collect(self, key):
        limit = self.max_samples.get(key)
        if limit is None:
            return True
        return self.sample_counts[key] < limit

    def _append_latent(self, key, z):
        if z is None or not self._can_collect(key):
            return
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        z = np.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        if z.shape[-1] != self.skill_dim:
            z = z.reshape(-1, self.skill_dim)
        limit = self.max_samples.get(key)
        if limit is not None:
            remaining = max(0, limit - self.sample_counts[key])
            if remaining <= 0:
                return
            if z.shape[0] > remaining:
                z = z[:remaining]
        self.latent_records[key].append(z)
        self.sample_counts[key] += z.shape[0]

    def _record_policy_skills(self, skills):
        if skills is None:
            return
        if torch.is_tensor(skills):
            skills_np = skills.detach().cpu().numpy()
        else:
            skills_np = np.asarray(skills)
        if skills_np.size == 0:
            return
        self._append_latent("policy", skills_np.reshape(-1, self.skill_dim))

    def _has_error_in_window(self, error_buffer, start_idx, end_idx, agent_id):
        for idx in range(start_idx, end_idx + 1):
            if error_buffer[idx][agent_id]:
                return True
        return False

    def _classify_and_collect(self, buffer, error_buffer, center_step, safe_sample_interval=1):
        last_idx = len(buffer) - 1
        if center_step - self.segment_pre < 0 or center_step + self.segment_post > last_idx:
            return
        start_idx = center_step - self.segment_pre
        end_idx = center_step + self.segment_post
        if end_idx >= len(error_buffer):
            return
        for agent_id in range(self.num_agents):
            is_unsafe = self._has_error_in_window(error_buffer, start_idx, end_idx, agent_id)
            if is_unsafe:
                if not self._can_collect("unsafe"):
                    continue
                segment = self._extract_segment(buffer, center_step, agent_id)
                if segment is None:
                    continue
                z_bad = self._encode_segment(*segment)
                self._append_latent("unsafe", z_bad)
            else:
                if safe_sample_interval > 1 and center_step % safe_sample_interval != 0:
                    continue
                if not self._can_collect("safe"):
                    continue
                segment = self._extract_segment(buffer, center_step, agent_id)
                if segment is None:
                    continue
                z_safe = self._encode_segment(*segment)
                self._append_latent("safe", z_safe)

    def _reached_all_limits(self):
        has_limit = False
        for key, limit in self.max_samples.items():
            if limit is None:
                continue
            has_limit = True
            if self.sample_counts[key] < limit:
                return False
        return has_limit

    def collect_all_types(
        self,
        num_episodes=20,
        policy_record_every_step=False,
        safe_sample_interval=1,
        max_safe=None,
        max_unsafe=None,
        max_policy=None,
        include_policy=False,
    ):
        include_policy = bool(include_policy)
        self.latent_records = {"safe": [], "unsafe": [], "policy": []}
        self.sample_counts = defaultdict(int)
        if not include_policy:
            max_policy = None
        self._set_limits(max_safe=max_safe, max_unsafe=max_unsafe, max_policy=max_policy)

        obs, share_obs, available_actions = self._reset_envs()
        self._init_rnn_states()
        self.current_skills = np.zeros((self.n_rollout_threads, self.num_agents, self.skill_dim), dtype=np.float32)
        self.high_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.low_masks = np.ones((self.num_agents, self.n_rollout_threads, 1), dtype=np.float32)

        traj_buffers = [[] for _ in range(self.n_rollout_threads)]
        error_buffers = [[] for _ in range(self.n_rollout_threads)]
        total_events_detected = 0
        event_type_counts = defaultdict(int)

        logger.info(
            "Collecting skill latents... target errors: {}",
            self.target_error_types if self.target_error_types else "ALL",
        )

        for _episode in range(num_episodes):
            for step in range(self.episode_length):
                self.current_skills = self._sample_skills(share_obs, obs, step, self.current_skills)
                if include_policy and (
                    policy_record_every_step or step % self.t_seg == 0 or np.any(self.high_masks == 0)
                ):
                    self._record_policy_skills(self.current_skills)

                actions = self._collect_actions(obs, share_obs, available_actions)
                prev_obs = obs
                next_obs, next_share_obs, _rewards, dones, infos, next_available_actions = self.envs.step(actions)

                obs = self._extract_obs_from_infos(next_obs, infos)
                share_obs = self._extract_share_obs_from_infos(next_share_obs, infos)
                available_actions = self._extract_available_actions(next_available_actions, infos)

                dones_arr = np.array(dones)
                if dones_arr.ndim == 1:
                    dones_env = dones_arr
                else:
                    dones_env = np.all(dones_arr, axis=1)

                for env_idx in range(self.n_rollout_threads):
                    traj_buffers[env_idx].append(
                        {
                            "obs": prev_obs[env_idx].copy(),
                            "actions": actions[env_idx, :, 0].copy(),
                        }
                    )

                    detect_error = infos[env_idx].get("detect_error", {}) or {}
                    keys_to_check = (
                        self.target_error_types if self.target_error_types else detect_error.keys()
                    )
                    error_mask = np.zeros(self.num_agents, dtype=bool)
                    for err_type in keys_to_check:
                        counts = detect_error.get(err_type)
                        if counts is None:
                            continue
                        if sum(counts) > 0:
                            event_type_counts[err_type] += 1
                            total_events_detected += 1
                        for agent_id, count in enumerate(counts):
                            if count > 0:
                                error_mask[agent_id] = True

                    error_buffers[env_idx].append(error_mask)

                    # Classify the window whose post-context is now available.
                    current_center_step = len(traj_buffers[env_idx]) - 1 - self.segment_post
                    if current_center_step >= 0:
                        self._classify_and_collect(
                            traj_buffers[env_idx],
                            error_buffers[env_idx],
                            current_center_step,
                            safe_sample_interval=safe_sample_interval,
                        )

                if np.any(dones_env):
                    for env_idx, done_flag in enumerate(dones_env):
                        if not done_flag:
                            continue
                        traj_buffers[env_idx] = []
                        error_buffers[env_idx] = []
                        self._reset_rnn_states(env_idx)

                self._update_masks(dones_arr)

                if self._reached_all_limits():
                    break
            if self._reached_all_limits():
                break

        self.envs.close()
        logger.success("Collection done.")
        logger.success(f"Safe samples   : {self.sample_counts['safe']}")
        logger.success(f"Unsafe samples : {self.sample_counts['unsafe']}")
        if include_policy:
            logger.success(f"Policy samples : {self.sample_counts['policy']}")
        if event_type_counts:
            for k, v in event_type_counts.items():
                logger.info(f"Detected {k}: {v} times")
        else:
            logger.info("No error events detected.")

    def _concat_records(self, key):
        if not self.latent_records[key]:
            return np.empty((0, self.skill_dim), dtype=np.float32)
        return np.concatenate(self.latent_records[key], axis=0)

    def _downsample(self, arr, max_points, rng):
        if max_points is None or arr.shape[0] <= max_points:
            return arr
        idx = rng.choice(arr.shape[0], max_points, replace=False)
        return arr[idx]

    def project_and_plot(
        self,
        methods=("pca", "tsne"),
        tsne_perplexity=30,
        max_points_per_type=None,
        random_state=0,
        include_policy=False,
    ):
        if isinstance(methods, str):
            methods = [methods]
        include_policy = bool(include_policy)
        safe = self._concat_records("safe")
        unsafe = self._concat_records("unsafe")
        if include_policy:
            policy = self._concat_records("policy")
        else:
            policy = np.empty((0, self.skill_dim), dtype=np.float32)

        rng = np.random.default_rng(random_state)
        safe = self._downsample(safe, max_points_per_type, rng)
        unsafe = self._downsample(unsafe, max_points_per_type, rng)
        if include_policy:
            policy = self._downsample(policy, max_points_per_type, rng)

        labels = ["safe"] * safe.shape[0] + ["unsafe"] * unsafe.shape[0]
        all_z_parts = [safe, unsafe]
        if include_policy:
            labels += ["policy"] * policy.shape[0]
            all_z_parts.append(policy)
        if not labels:
            logger.warning("No samples collected for visualization.")
            return []

        all_z = np.concatenate(all_z_parts, axis=0)
        if all_z.shape[0] < 2:
            logger.warning("Not enough samples to project (need at least 2).")
            return []

        outputs = []
        for method in methods:
            method_lower = method.lower()
            if method_lower == "pca":
                reducer = PCA(n_components=2, random_state=random_state)
                z_2d = reducer.fit_transform(all_z)
                out_path = self.run_dir / "skill_space_pca.png"
            elif method_lower == "tsne":
                if all_z.shape[0] < 3:
                    logger.warning("Skipping t-SNE: not enough samples.")
                    continue
                perplexity = min(tsne_perplexity, max(2, all_z.shape[0] - 1))
                if perplexity >= all_z.shape[0]:
                    perplexity = max(2, all_z.shape[0] // 3)
                reducer = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=random_state,
                    init="pca",
                )
                z_2d = reducer.fit_transform(all_z)
                out_path = self.run_dir / "skill_space_tsne.png"
            else:
                logger.warning(f"Unknown projection method: {method}")
                continue

            self._draw_scatter(z_2d, labels, out_path, title=method_upper(method_lower))
            outputs.append(out_path)

        return outputs

    def _draw_scatter(self, data_2d, labels, out_path, title=None):
        if _HAS_SEABORN:
            import seaborn as sns
            sns.set_style("whitegrid")

        palette = {
            "safe": ("#9aa0a6", 0.35),
            "unsafe": ("#d62728", 0.7),
            "policy": ("#1f77b4", 0.7),
        }
        labels_arr = np.asarray(labels)
        plt.figure(figsize=(10, 8))
        for key in ("safe", "unsafe", "policy"):
            mask = labels_arr == key
            if not np.any(mask):
                continue
            color, alpha = palette[key]
            plt.scatter(
                data_2d[mask, 0],
                data_2d[mask, 1],
                s=18,
                c=color,
                alpha=alpha,
                label=key.capitalize(),
                edgecolors="none",
            )
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        logger.info(f"Saved plot to {out_path}")


def method_upper(method):
    return method.upper() if method else ""


def load_yaml_config(yaml_path: Path):
    import yaml
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = get_config()
    parser = get_overcooked_args(parser)
    default_config_path = (
        Path(__file__).resolve().parents[2]
        / "zsceval"
        / "scripts"
        / "overcooked"
        / "config"
        / "gen_bad_skills.yaml"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=str(default_config_path),
        help="Path to YAML config (can reuse gen_bad_skills.yaml).",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--plot_methods", nargs="+", default=None, help="pca tsne")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--max_points_per_type", type=int, default=None)
    parser.add_argument("--policy_record_every_step", action="store_true", default=None)
    parser.add_argument("--safe_sample_interval", type=int, default=None)
    parser.add_argument("--max_safe", type=int, default=None)
    parser.add_argument("--max_unsafe", type=int, default=None)
    parser.add_argument("--max_policy", type=int, default=None)
    parser.add_argument(
        "--include_policy",
        "--policy",
        action="store_true",
        default=None,
        help="Include policy skills in projections/plots.",
    )
    parser.add_argument(
        "--target_error_types",
        nargs="+",
        type=str,
        default=None,
        help="List of error types to visualize as unsafe.",
    )

    args = parser.parse_args()

    cli_overrides = {
        "num_episodes": args.num_episodes,
        "output_dir": args.output_dir,
        "plot_methods": args.plot_methods,
        "policy_record_every_step": args.policy_record_every_step,
        "safe_sample_interval": args.safe_sample_interval,
        "max_safe": args.max_safe,
        "max_unsafe": args.max_unsafe,
        "max_policy": args.max_policy,
        "include_policy": args.include_policy,
        "target_error_types": args.target_error_types,
    }

    if args.config_path:
        yaml_config = load_yaml_config(Path(args.config_path))
        for key, value in yaml_config.items():
            setattr(args, key, value)
            logger.info(f"[Config] {key}: {value}")

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(args, key, value)

    if isinstance(getattr(args, "target_error_types", None), str):
        args.target_error_types = [args.target_error_types]

    args.use_wandb = False

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    num_episodes = getattr(args, "num_episodes", None) or 10
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        run_dir = Path(output_dir)
    else:
        run_dir = (
            Path(get_base_run_dir())
            / args.env_name
            / args.layout_name
            / args.algorithm_name
            / args.experiment_name
            / "skill_space"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "all_args": args,
        "device": device,
        "run_dir": run_dir,
        "num_agents": args.num_agents,
    }

    visualizer = SkillSpaceVisualizer(config)
    include_policy = bool(getattr(args, "include_policy", False))
    visualizer.collect_all_types(
        num_episodes=num_episodes,
        policy_record_every_step=getattr(args, "policy_record_every_step", False),
        safe_sample_interval=getattr(args, "safe_sample_interval", 1) or 1,
        max_safe=getattr(args, "max_safe", None),
        max_unsafe=getattr(args, "max_unsafe", None),
        max_policy=getattr(args, "max_policy", None),
        include_policy=include_policy,
    )

    methods = getattr(args, "plot_methods", None) or ["pca", "tsne"]
    visualizer.project_and_plot(
        methods=methods,
        tsne_perplexity=getattr(args, "tsne_perplexity", 30),
        max_points_per_type=getattr(args, "max_points_per_type", None),
        random_state=getattr(args, "seed", 0),
        include_policy=include_policy,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    main()

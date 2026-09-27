# NeurIPS 재현성 체크리스트 — Overcooked self-play 훈련

이 문서는 다음 네 self-play 설정의 **실제 보존된 W&B 설정·runtime·체크포인트**와 저장소 코드를 대조해 작성했다.

- `2_forced_hard`
- `2_forced_hard_4`
- `2_incentivized_hard`
- `2_incentivized_hard_4`

핵심 결론은 다음과 같다. 모든 run은 shared recurrent MAPPO(rMAPPO)로 고정된 10,000,000 joint environment steps를 학습했다. 학습 종료를 위한 reward threshold나 best-validation checkpoint 선택은 없었다. 대신 모든 주기적 actor/critic checkpoint를 보존했고, 각 seed의 final checkpoint는 10,000,000 step snapshot이다.

## 체크리스트용 요약

### Training method

We trained shared-policy recurrent MAPPO (rMAPPO) agents in the customized `overcooked_new` environment. A shared actor was used by all agents, while a separate centralized critic was trained under CTDE. Each run used 50 rollout environments, an episode horizon of 400, 15 PPO epochs, one minibatch, and a fixed budget of 10 million joint environment steps. We used 20 seeds for `2_forced_hard` and `2_forced_hard_4`, and 10 seeds for `2_incentivized_hard` and `2_incentivized_hard_4`.

### Model selection and reward threshold

There was no reward-threshold-based early stopping or best-validation-checkpoint selection. Checkpoints were saved periodically independent of evaluation reward; the endpoint checkpoint for every run is `actor_periodic_10000000.pt` with its paired critic. Sparse self-play reward was logged during deterministic evaluation, but it was not used to decide whether to stop or retain a run.

### Environment and license

We used a locally vendored and customized multi-recipe fork of Overcooked-AI at `zsceval/envs/overcooked_new/`, invoked with `--overcooked_version new`. Its package metadata identifies upstream `HumanCompatibleAI/overcooked_ai` and declares version `1.1.0`; the vendored environment is MIT licensed (Copyright © 2019 Center for Human-Compatible AI). The surrounding ZSC-Eval framework is also MIT licensed. Because this is a customized in-tree vendor copy, it should not be described as bit-identical to the official upstream `v1.1.0` release.

### Compute

The 60 retained runs comprise 600 million joint environment steps and 265.90 allocated single-GPU wall/GPU-hours, summed from each run's W&B `_runtime`. Jobs used one RTX 3090 per run; the host had two RTX 3090 GPUs and 67.33 GB (62.71 GiB) RAM. FLOPs, GPU utilization, peak VRAM, and peak RAM were not profiled and are therefore reported as unavailable.

## 1. Experiment inventory

| Setting | Agents | Seeds retained | Runs | Training budget/run | Total joint env steps | Final checkpoint |
|---|---:|---|---:|---:|---:|---|
| `2_forced_hard` | 2 | 1–20 | 20 | 10,000,000 | 200M | `actor_periodic_10000000.pt` + paired critic |
| `2_forced_hard_4` | 4 | 1–20 | 20 | 10,000,000 | 200M | `actor_periodic_10000000.pt` + paired critic |
| `2_incentivized_hard` | 2 | 1–10 | 10 | 10,000,000 | 100M | `actor_periodic_10000000.pt` + paired critic |
| `2_incentivized_hard_4` | 4 | 1–10 | 10 | 10,000,000 | 100M | `actor_periodic_10000000.pt` + paired critic |
| **Total** | — | — | **60** | — | **600M** | — |

The four custom layouts use a 13 × 7 grid and a three-onion order. Their definitions are in `zsceval/envs/overcooked_new/src/overcooked_ai_py/data/layouts/`.

## 2. RL architecture

| Component | Specification |
|---|---|
| Algorithm | `rmappo`, experiment name `sp`; recurrent MAPPO with centralized training and decentralized execution (CTDE) |
| Policy sharing | One actor and one critic are shared by all 2 or 4 players in a run |
| Actor input | Agent-local spatial observation: 13 × 7 × 25 |
| Critic input | Centralized spatial observation: 13 × 7 × 20 |
| CNN backbone | Conv2d `32, 3×3, stride 1, padding 1` → ReLU → Conv2d `64, 3×3, 1, 1` → ReLU → Conv2d `32, 3×3, 1, 1` → ReLU |
| Post-CNN trunk | Flatten → FC(64) → ReLU → LayerNorm → FC(64) → ReLU → LayerNorm → FC(64) → ReLU → LayerNorm |
| Recurrent module | One-layer GRU, hidden size 64, followed by LayerNorm; recurrent training chunk length 10 |
| Heads | Actor: 6-action categorical head; critic: scalar value head |
| Initialization / normalisation | Orthogonal initialization, action-head gain 0.01, ValueNorm enabled |
| Disabled components | Attention, influence module, PopArt, policy value head, max pooling, data parallelism |
| Parameter count | Actor: 264,806; critic: 263,041; total: 527,847 trainable parameters (same for 2- and 4-agent runs because of policy sharing) |

The actor and critic are distinct networks. The actor receives local observations; the critic receives the centralized observation. Invalid actions are masked before sampling.

## 3. Optimizer and training hyperparameters

| Category | Value |
|---|---|
| Optimizer | Separate Adam optimizers for actor and critic |
| Actor / critic learning rate | `5e-4` / `5e-4` |
| Adam epsilon / weight decay | `1e-5` / `0` |
| Adam β values | Not explicitly overridden: PyTorch defaults `(0.9, 0.999)`; `amsgrad=False` |
| LR schedule | No linear LR decay; constant base LR |
| PPO epochs / minibatches | 15 / 1 |
| PPO clip parameter | `0.2` |
| Value loss | Clipped value loss, Huber loss (`δ=10`), coefficient `1` |
| Advantage estimation | GAE enabled, `γ=0.99`, `λ=0.95`; normalized advantages |
| Gradient clipping | Enabled, max norm `10` |
| Other loss settings | Policy/value active masks enabled; proper time limits enabled; partial-episode bootstrapping disabled |
| Entropy coefficient | Linear piecewise schedule: `0.20 → 0.05 → 0.01` at `0`, `5M`, and `10M` environment steps |
| Rollout | 50 parallel environments; horizon 400; dummy batch size 2; PyTorch training threads 1 |
| Per-run optimization scale | `400 × 50 = 20,000` joint env steps per rollout update; 500 PPO rollout updates/run; 7,500 actor and 7,500 critic Adam steps/run |
| Evaluation | Enabled every 20 rollout updates; 10 parallel evaluation environments; deterministic policy evaluation |
| Determinism | CUDA requested, deterministic cuDNN requested; launcher limits `OMP`, `MKL`, `OPENBLAS`, and `NUMEXPR` threads to 1 |

All settings above are identical between the four layouts except for `num_agents` and the corresponding per-agent list `agent_policy_names` / `reward_shaping_roles`.

## 4. Reward, threshold, and checkpoint procedure

### Training reward

For non-HSP self-play, the per-agent training reward is:

`r_i = sparse_reward + α(t) × dense_reward_i`.

Every agent uses the `individual` dense-reward preset:

| Dense-reward event | Value |
|---|---:|
| Placement in pot | 3 |
| Dish pickup | 3 |
| Soup pickup | 5 |
| Dish-dispenser, pot, and soup distance shaping | 0 |

`α(0)=1.0`, and it is linearly annealed toward zero over a `100M`-step shaping horizon. Because each run ends at 10M steps, `α(10M)=0.9`; dense shaping is therefore still active at the end of training. Random starts, HSP rewards, and phi rewards are disabled; `old_dynamics=False`.

### Reward threshold

- **No reward threshold was used to terminate training or select a checkpoint.** All runs train to the fixed 10M-step budget.
- `eval_ep_sparse_r` is logged during deterministic self-play evaluation, but it is not used as an early-stopping or selection criterion.
- The persisted config contains `replay_return_threshold=0.75`, but the `overcooked_new` self-play path does not consume it. It must not be reported as a model-selection threshold.

### Checkpoint selection

- There is **no best-validation checkpoint**. Every periodic checkpoint is saved independently of reward.
- Each seed has 52 paired actor/critic snapshots. The final run endpoint is the paired `*_periodic_10000000.pt` checkpoint.
- Snapshot schedule (where one update is 20,000 joint environment steps): updates 0–49 every 2 updates; updates 50–99 every 5 updates; thereafter every 25 updates; the final update is always saved.
- The downstream trajectory inventory contains all valid paired periodic checkpoints (3,120 checkpoint pairs = 60 seeds × 52), not a reward-ranked subset.

There is a distinct **post-training trajectory filter** in `generate_score_trajectories_200.py`: each periodic checkpoint is rolled out deterministically with old pot dynamics for at most 200 action ticks, and only trajectories with exact final scores in `{20, 40, 60, 80, 100}` are retained. This filter is for trajectory export only; it is not a training reward threshold or a checkpoint-selection rule.

If a paper reports one final SP policy per seed, cite the 10M-step endpoint. If it reports score-conditioned trajectory data, cite the all-checkpoint deterministic score-bucket procedure separately.

## 5. Overcooked code, version, and license

| Item | What to report |
|---|---|
| Environment used | `zsceval/envs/overcooked_new/`, selected by `--overcooked_version new` |
| Local environment type | In-tree, customized, multi-recipe vendor fork; not a pip-only dependency or Git submodule |
| Declared upstream | `https://github.com/HumanCompatibleAI/overcooked_ai` |
| Declared package version | `overcooked_ai` `1.1.0` (from `zsceval/envs/overcooked_new/setup.py`) |
| Environment license | MIT License, Copyright © 2019 Center for Human-Compatible AI |
| Surrounding framework | ZSC-Eval; official upstream `https://github.com/sjtu-marl/ZSC-Eval` |
| Framework license | MIT License, Copyright © 2023 xihuai18 |
| Experiment repository recorded by W&B | `https://github.com/jkyoon2/Overcooked.git` |

The vendored environment includes local wrapper, MDP/planner, and custom-layout changes. The package metadata's `1.1.0` declaration identifies its upstream lineage, but an exact upstream source commit cannot be recovered and the local vendor tree differs from the official upstream tag. The accurate wording is therefore “customized vendored fork with package metadata declaring Overcooked-AI v1.1.0,” not “the unmodified official v1.1.0 release.”

### Recorded experiment revisions

W&B metadata records more than one local Git revision across the campaign; do not state a single revision for all 60 runs.

| Setting | Seed range | W&B-recorded commit |
|---|---|---|
| `2_forced_hard` | 1–15 | `bd2a27bf9f7018b87d29865a6be9abd97bdce003` |
|  | 16–20 | `aebcf2a9f986007b52cb0c65f7117f2e31a99701` |
| `2_forced_hard_4` | 1–5 | `556259fc7ce556ffb30aec999189308ba5e8519d` |
|  | 6–15 | `bd2a27bf9f7018b87d29865a6be9abd97bdce003` |
|  | 16–19 | `aebcf2a9f986007b52cb0c65f7117f2e31a99701` |
|  | 20 | `bea6cc3e213004d3782a657c4370fccd2544f8a4` |
| `2_incentivized_hard` | 1–10 | `bd2a27bf9f7018b87d29865a6be9abd97bdce003` |
| `2_incentivized_hard_4` | 1–5 | `556259fc7ce556ffb30aec999189308ba5e8519d` |
|  | 6–10 | `bd2a27bf9f7018b87d29865a6be9abd97bdce003` |

Important audit caveat: the tracked 4-agent layout files were added after the earliest seed 1–5 runs began, so those early runs are documented by W&B configuration and artifacts but are not fully frozen by a clean tracked source tree. In addition, the current working tree has an uncommitted modification to `2_forced_hard.layout`. For archival release, preserve the actual layout files and their SHA-256 hashes alongside each checkpoint set.

## 6. Hardware, runtime, and total computation

### Hardware and software

| Resource | Recorded value |
|---|---|
| CPU | 6 physical cores / 12 logical threads; CPU model not recorded |
| GPU host inventory | 2 × NVIDIA GeForce RTX 3090, 24 GiB VRAM each |
| GPU per job | 1 RTX 3090: the trainer uses `cuda:0`, data parallelism is disabled, and launchers set `CUDA_VISIBLE_DEVICES` to one GPU ID |
| Host RAM | 67,332,755,456 bytes = 67.33 GB = 62.71 GiB |
| CUDA | 13.0 |
| Python | CPython 3.9.23 (`zsceval` conda environment in W&B metadata) |
| OS | Linux 6.14.0-36-generic, x86_64, glibc 2.39 |

The “2 GPUs” field is the host inventory, not a two-GPU training configuration. Each training process used a single visible GPU.

### Actual retained runtime

Runtime is the sum of the `_runtime` field from all preserved per-seed `wandb-summary.json` files. It includes the run's training/evaluation/checkpoint overhead. It is reported both as aggregate allocated single-GPU wall time and GPU-hour accounting; it is **not** a measure of GPU utilization or floating-point operations.

| Setting | Runs | Runtime sum | Mean runtime/run | Training env steps |
|---|---:|---:|---:|---:|
| `2_forced_hard` | 20 | 65.5047 h (65:30:17) | 3.2752 h | 200M |
| `2_forced_hard_4` | 20 | 114.5286 h (114:31:43) | 5.7264 h | 200M |
| `2_incentivized_hard` | 10 | 30.4655 h (30:27:56) | 3.0465 h | 100M |
| `2_incentivized_hard_4` | 10 | 55.3964 h (55:23:47) | 5.5396 h | 100M |
| **Total** | **60** | **265.8952 h (265:53:43)** | — | **600M** |

Derived training-scale quantities:

- 30,000 PPO rollout updates total (60 runs × 500 updates).
- 450,000 actor and 450,000 critic Adam steps total (30,000 × 15 PPO epochs × 1 minibatch).
- 600M is a **joint environment-step** count. Counting a decision for every player separately gives 1.8B agent-action decisions: 300M two-agent joint steps × 2 plus 300M four-agent joint steps × 4.
- Sum-of-runtimes is not the campaign's calendar elapsed time because jobs may have overlapped.

### Quantities not recorded

| Requested quantity | Status |
|---|---|
| CPU model / CPU utilization / CPU core-hours | Not recorded; do not infer from thread limits |
| Peak host RAM | Not recorded |
| Peak GPU memory / GPU utilization | Not recorded |
| FLOPs or energy use | Not profiled; unavailable |

## 7. Primary evidence in this repository

- Actual per-run arguments and hardware: `results/Overcooked/<layout>/shared/rmappo/sp/seed*/wandb/run-*/files/config.yaml` and `wandb-metadata.json`.
- Actual runtime: matching `wandb-summary.json` files (`_runtime` field), with 20/20/10/10 files across the four settings.
- Checkpoint inventory and seed counts: `json_trajectory_200/available_models_200.tsv` and `json_trajectory_200/verification_summary_200.json`.
- Launch settings: `zsceval/scripts/overcooked/shell/train_sp_2agent_seed*.sh` and `train_sp_4agent_seed*.sh`.
- Training entrypoint and environment switch: `zsceval/scripts/overcooked/train/train_sp.py`.
- Architecture and optimizer: `zsceval/algorithms/r_mappo/algorithm/r_actor_critic.py`, `rMAPPOPolicy.py`, `zsceval/algorithms/r_mappo/r_mappo.py`, and `zsceval/algorithms/utils/{cnn,rnn,act}.py`.
- Reward implementation: `zsceval/envs/overcooked_new/Overcooked_Env.py` and `src/overcooked_ai_py/mdp/overcooked_mdp.py`.
- Checkpoint save path and downstream score filtering: `zsceval/runner/shared/overcooked_runner.py` and `zsceval/analysis/json/generate_score_trajectories_200.py`.
- Environment provenance/license: `zsceval/envs/overcooked_new/setup.py`, `zsceval/envs/overcooked_new/LICENSE`, root `README.md`, and root `LICENSE`.

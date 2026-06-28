# HSP Pipeline Analysis — `run_full_hsp_pipeline.sh`

> Last updated: 2026-06-10
> Subject: end-to-end HSP-S2 preparation/training for one Overcooked layout (e.g. `bitl`).
> Reference: `zsceval/scripts/overcooked/shell/run_full_hsp_pipeline.sh`.

---

## 0. What the pipeline produces

For each layout, the final artefact is a **ZSC ego policy** (HSP-S2 adaptive actor) saved at:

```
results/Overcooked/<layout>/shared/adaptive/hsp-S2-s<S>/seed<i>/models/hsp_adaptive/actor_periodic_<step>.pt
```

This actor was trained to collaborate with a curated set of diverse partners (`S = hsp_population_size`, default 12). Diversity comes from two sources stitched into one partner pool:

- **`hsp_k` biased agents** (default 6) drawn from Stage 1 with weighted shaped-reward bias.
- **`S − hsp_k` MEP-population checkpoints** (default 6) drawn at multiple training tags (init/mid/final).

The pipeline orchestrates six stages to (a) train and select those partners and (b) train the ego against them.

---

## 1. Stage map

| # | Stage name (CLI) | Module / script | Output | Parallelism |
|---|---|---|---|---|
| 1 | `bias_train` | `train/train_bias_agent.py` via `shell/train_bias_agents.sh` | `results/.../hsp-S1/seed{1..N}/models/actor_periodic_*.pt` | **seed-parallel** |
| 2 | `bias_extract_eval` | `extract_models/extract_bias_agents_models.py` + `shell/eval_bias_agents_events.sh` (→ `eval/eval.py`) | `policy_pool/.../hsp/s1/hsp/hsp*_{init,mid,final}_w0_actor.pt` + event-stat JSONs in `eval/results/<layout>/bias/*.json` | sequential per agent pair |
| 3 | `mep_train` | `train/train_mep.py` via `shell/train_mep_stage_1.sh` | `results/.../mep/mep-S1-s<P>/seed1/models/mep{1..P}/actor_periodic_*.pt` | seed-parallel (but `mep_seed_end=1` by default) |
| 4 | `mep_extract` | `extract_models/extract_pop_S1_models.py` | `policy_pool/.../mep/s1/mep-S1-s<P>/mep{1..P}_{init,mid,final}_actor.pt` | sequential |
| 5 | `hsp_gen` | `prep/gen_hsp_S2_ymls.py` | `policy_pool/.../hsp/s2/train-s<S>-hsp_mep-S1-s<P>-<seed>.yml` per HSP seed | sequential (fast) |
| 6 | `hsp_train` | `train/train_adaptive.py` via `shell/train_hsp_stage_2.sh` | `results/.../adaptive/hsp-S2-s<S>/seed{1..M}/models/hsp_adaptive/actor_periodic_*.pt` | **seed-parallel** |

The orchestrator (`run_full_hsp_pipeline.sh`) is idempotent: it inspects `results/` and `policy_pool/` and skips stages whose outputs are already present (see `bias_seed_has_training_artifacts`, `mep_policy_pool_ready`, `count_usable_bias_agents`).

---

## 2. Stage-by-stage detail

### Stage 1 — `bias_train` (biased rMAPPO partner factory)

**Driver:** `shell/train_bias_agents.sh <layout> <seed_begin> <seed_max>`
**Worker:** `train/train_bias_agent.py` with `--algorithm_name rmappo --experiment_name hsp-S1`.

What it does for each seed:
- Trains a **2-agent shared-policy rMAPPO run** (`--policy_mode shared`, `--agent_policy_names ppo ppo`, `--use_hsp`).
- Both agents share a single PPO actor/critic; the bias enters through **per-role reward shaping**. The script passes `--w0` and `--w1` weight vectors over Overcooked's SHAPED_INFOS feature list:
  - `w0` (length 34 in the "new" Overcooked code path that `bitl` uses): sampled tokens like `[0:5:10:15]` define a categorical bias family (e.g. "value tomato placement positively" vs "neutral" vs "value onion placement positively"). The seed picks one concrete weight vector from the cartesian product.
  - `w1`: a fixed dense-shaping baseline aligned with `train_sp.sh` (the canonical self-play reward).
- Roles are randomised via `--random_index` and `--reward_shaping_role individual --reward_shaping_roles individual,individual`, so the same policy is trained under both bias profiles by index swap each episode.
- Training horizon: **1e7 env steps**, 50 rollout threads, episode length 400, PPO epochs 15, entropy annealed `0.2 → 0.05 → 0.01` over `[0, 5e6, 1e7]` steps.
- Snapshots saved every `save_interval=25` PPO updates → `actor_periodic_<step>.pt` and `actor_w0/actor_w1` finals.

How many seeds: `bias_seed_begin..bias_seed_end` (default 1..6). The orchestrator can auto-extend up to `bias_seed_cap` (default 72 for new-layout, layout-specific for old) if downstream filtering rejects too many bias agents.

Output that matters downstream: per-seed `actor_periodic_*.pt` checkpoints in `results/Overcooked/<layout>/shared/rmappo/hsp-S1/seed<n>/models/`. The presence of these files is how `bias_seed_has_training_artifacts` knows a seed is done.

### Stage 2 — `bias_extract_eval` (collect + evaluate the bias zoo)

**Driver A:** `extract_models/extract_bias_agents_models.py <layout> Overcooked --wandb_name <name>`.
- Walks `results/Overcooked/<layout>/shared/rmappo/hsp-S1/seed*/models/` and copies selected actors (final + init/mid checkpoints) into the policy pool:
  - `policy_pool/<layout>/hsp/s1/hsp/hsp<n>_<tag>_w0_actor.pt` (biased role)
  - `policy_pool/<layout>/hsp/s1/hsp/hsp<n>_<tag>_w1_actor.pt` (canonical role)

**Driver B:** `shell/eval_bias_agents_events.sh <layout>` → `eval/eval.py --algorithm_name population`.
- For every extracted bias agent, plays the `(_w0, _w1)` pair against itself for 80 episodes (80 eval rollout threads), via the population eval template `policy_pool/.../hsp/s1/eval_template.yml` (auto-generated by `prep/gen_bias_agent_eval_yml.py` if missing).
- Writes per-pair event statistics to `eval/results/<layout>/bias/<exp>.json` (rates of `delivery`, `optimal_placement`, `catastrophic_placement`, recipe-specific deliveries, etc.).

These event stats become the input to the bias-agent **selection** logic in Stage 5.

### Stage 3 — `mep_train` (Maximum-Entropy Population)

**Driver:** `shell/train_mep_stage_1.sh <layout> <population_size> <seed_begin> <seed_max>` (defaults: P=5, seeds=1..1).
**Worker:** `train/train_mep.py` with `--algorithm_name mep --experiment_name mep-S1-s<P> --stage 1`.

What it does:
- Trains **a population of P policies jointly** in a shared-network setting, with an MEP diversity bonus (`--mep_entropy_alpha 0.01`) that pushes the P policies apart in action distribution while each maintains task competence.
- A pre-built population yaml at `policy_pool/<layout>/mep/s1/train-s<P>.yml` (auto-generated by `prep/gen_pop_ymls.py mep -s <P>` if missing) wires up the P policy names (`mep1..mepP`) plus a shared `mep_adaptive` reference.
- Training horizon: **1e7 env steps**, `train_env_batch=125` rollout threads per policy, episode length 400, PPO epochs 15.
- Snapshots: `actor_periodic_<step>.pt` per policy id under `models/mep<id>/`.

Output that matters downstream: full final + intermediate snapshots for `mep1..mepP`. `mep_seed_has_training_artifacts` checks all P sub-folders have at least one `actor_periodic_*.pt`.

### Stage 4 — `mep_extract` (lift MEP into the policy pool)

**Driver:** `extract_models/extract_pop_S1_models.py <layout> Overcooked --algo mep --population_size <P> --experiment_name mep-S1-s<P> --seed <mep_extract_seed>`.

What it does:
- For each MEP policy id (`mep1..mepP`) and each tag (`init`, `mid`, `final`) it copies the corresponding `actor_periodic_*.pt` into:
  - `policy_pool/<layout>/mep/s1/mep-S1-s<P>/mep<id>_<tag>_actor.pt`
- These 3×P checkpoints are what the HSP-S2 yaml will sample from to fill the non-bias slots of the partner pool.

`mep_policy_pool_ready` is the readiness check: all 3×P files must exist.

### Stage 5 — `hsp_gen` (pick bias agents + assemble HSP-S2 yaml)

**Driver:** `prep/gen_hsp_S2_ymls.py -l <layout> -k <hsp_k> -s <mep_pop> -S <hsp_pop>`.

What it does (high level):
1. **Selects** `hsp_k` bias agents from the Stage-2 event-stat JSONs using a diversity / coverage objective (this is what `--count-only` exposes: how many bias agents survive the filter). The default selection target is `hsp_k = 6`.
2. **Fills** the remaining `S − hsp_k` slots from the MEP S1 policy pool, spreading the picks across `init / mid / final` tags so the ego policy sees partners at multiple competence levels.
   - Constraint enforced by the orchestrator: `(hsp_population_size − hsp_k) % 3 == 0` so the tag split is balanced.
3. **Emits** one yaml per HSP seed at `policy_pool/<layout>/hsp/s2/train-s<S>-hsp_mep-S1-s<P>-<seed>.yml`. Each yaml lists `S` partner entries with their on-disk actor paths.

The orchestrator also calls this with `--count-only` *before* Stage 5 to decide whether to auto-extend bias training (`ensure_bias_pool_ready_for_hsp`): if filtered bias agents < `hsp_k`, more bias seeds are scheduled.

### Stage 6 — `hsp_train` (the HSP ego)

**Driver:** `shell/train_hsp_stage_2.sh <layout> <hsp_pop> <seed_begin> <seed_max>` (defaults: S=12, seeds=1..5).
**Worker:** `train/train_adaptive.py` with `--algorithm_name adaptive --experiment_name hsp-S2-s<S> --stage 2 --adaptive_agent_name hsp_adaptive --use_agent_policy_id`.

What it does for each HSP seed:
- Loads the HSP-S2 partner yaml for that seed (Stage 5 output).
- Trains a single **adaptive ego policy** that, at each rollout episode, is paired with a partner sampled from the `S`-policy pool. The ego sees a one-hot partner-id signal (`--use_agent_policy_id`) so it can learn to condition on partner identity at inference time.
- Training scale (population_size 12):
  - 5e7 env steps, 100 rollout threads, episode length 400, PPO epochs 15.
  - Entropy schedule `0.2 → 0.05 → 0.01` over `[0, 2.5e7, 5e7]`.
  - Eval rollout threads = `2 × S` (24), eval episodes = 5, every 20 PPO updates.
- Skipping logic: if `models/hsp_adaptive/actor_periodic_<final_step>.pt` already exists (50M / 80M / 100M depending on S), that seed is skipped.

Output: `results/Overcooked/<layout>/shared/adaptive/hsp-S2-s<S>/seed<n>/models/hsp_adaptive/actor_periodic_*.pt`. The `actor_periodic_<final_step>.pt` is the ZSC-Eval HSP ego policy.

---

## 3. How `run_full_hsp_pipeline.sh` ties it together

- **Layout queue.** It walks `--layouts` (default: `ttt,tto,too,ooo,1_incentivized_hard,1_forced_hard,2_incentivized_hard,2_forced_hard,bitl`) and assigns each layout to one GPU slot from `--gpus` (default `auto`, which queries `nvidia-smi` for idle GPUs under `--free-gpu-mem-max-mb`).
- **Per-layout subshell.** For each layout it spawns a subshell that:
  1. Activates the conda env (`zsceval` by default; you'll want `neurocontroller` here — see §5).
  2. Exports `CUDA_VISIBLE_DEVICES=<that one GPU>` and `WANDB_NAME=<wandb_name>`.
  3. Runs stages within `[--from-stage, --to-stage]` in order: `bias_train → bias_extract_eval → hsp_gen guard → mep_train → mep_extract → hsp_gen → hsp_train`.
- **Idempotency.** Each stage is short-circuited by an on-disk check (file existence in `results/` or `policy_pool/`). If the bias pool already has ≥ `hsp_k` agents, base bias training is skipped; same for MEP and HSP.
- **Auto-extension.** When the filtered bias pool count is below `hsp_k`, the orchestrator schedules additional bias seeds in batches (`ensure_bias_pool_ready_for_hsp`) up to `bias_seed_cap`.
- **Locks.** A per-layout lock dir under `log_root/locks/<layout>.lock` prevents two concurrent invocations from training the same layout.
- **Active-layout skip.** Default-on `--skip-active-layouts` uses `ps + rg` to detect already-running training of the same layout and skips it.

---

## 4. Why only one GPU is used for the single-`bitl` case (today)

The orchestrator's parallelism unit is **one layout = one GPU**. For a single layout (e.g. `bitl`) with `--gpus 0,1` it picks the first GPU, pins `CUDA_VISIBLE_DEVICES` to it for the entire subshell, and never touches GPU 1.

Within that subshell, the seed loops in `train_bias_agents.sh` (6 seeds) and `train_hsp_stage_2.sh` (5 seeds) iterate sequentially, so neither stage exploits the second GPU either.

→ See §5 / §6 for the `--shard-within-layout` mode that fixes this.

---

## 5. Run notes for `bitl`

- The repo's documented conda env name is `zsceval`, but the project standard for this repo (per `zsceval/human_exp/CLAUDE.md`) is `neurocontroller`. The training tree imports the same modules from either; pass `--conda-env neurocontroller` if your conda has only `neurocontroller`.
- `bitl.layout` lives under the new Overcooked path (`zsceval/envs/overcooked_new/.../layouts/bitl.layout`), so the scripts auto-resolve `version="new"` (34-element SHAPED_INFOS bias family, 6 default bias seeds).
- Recipe values: tomato=5, tomato·tomato·onion=10, tomato·onion·onion=15, onion=20 (from `bitl.layout`).
- bitl is not in `default_bias_seed_cap_for_layout`'s explicit list → falls through to default `72`.

---

## 6. Verification commands you can use

```bash
# Dry-run shows exactly which sub-commands will fire, on which GPUs:
bash zsceval/scripts/overcooked/shell/run_full_hsp_pipeline.sh \
  --layouts bitl --gpus 0,1 --shard-within-layout --dry-run

# Check produced bias actors after Stage 1:
find results/Overcooked/bitl/shared/rmappo/hsp-S1 -name 'actor_periodic_*.pt' | head

# Check MEP pool extraction:
ls zsceval/scripts/policy_pool/bitl/mep/s1/mep-S1-s5/

# Check the HSP-S2 yamls before training the ego:
ls zsceval/scripts/policy_pool/bitl/hsp/s2/
```

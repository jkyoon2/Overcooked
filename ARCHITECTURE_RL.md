# Architecture Notes

## ZSC-Eval Overcooked SP RMAPPo Training

This document summarizes the current code path for ordinary self-play training with shared-policy recurrent MAPPO in `zsceval`.

Primary entrypoints:

- Shell launcher: `zsceval/scripts/overcooked/shell/train_sp.sh`
- Python entrypoint: `zsceval/scripts/overcooked/train/train_sp.py`
- Shared runner: `zsceval/runner/shared/overcooked_runner.py`
- Policy wrapper: `zsceval/algorithms/r_mappo/algorithm/rMAPPOPolicy.py`
- Actor and critic modules: `zsceval/algorithms/r_mappo/algorithm/r_actor_critic.py`

The standard `train_sp.sh` path uses:

- Environment: `Overcooked`
- Algorithm: `rmappo`
- Experiment name: `sp`
- Agents: 2
- Agent policy names: `ppo ppo`
- Policy mode: shared policy
- Recurrent policy: enabled
- Overcooked version: `new`
- Episode length: 400
- Rollout threads: 50
- PPO epochs: 15
- Mini-batches: 1
- CNN layer parameters: `32,3,1,1 64,3,1,1 32,3,1,1`
- Entropy coefficient schedule: `0.2 -> 0.05 -> 0.01`

In this mode, both players are controlled by the same trainable policy. There is no opponent pool in the ordinary SP path. The shared policy is evaluated for each agent slot, and experience from all agent slots is stored in the shared replay buffer.

## Data Flow

At rollout time, the runner performs the following loop:

1. Read current `obs`, `share_obs`, recurrent states, masks, and available action masks from `SharedReplayBuffer`.
2. Call `R_MAPPOPolicy.get_actions(...)`.
3. The actor receives local observations and outputs sampled actions plus action log probabilities.
4. The critic receives centralized shared observations and outputs value predictions.
5. The vectorized Overcooked environment steps with the sampled joint action.
6. The next observations, rewards, done flags, info dicts, and available action masks are inserted into the replay buffer.
7. After one rollout segment, returns and advantages are computed, then PPO updates actor and critic.

The main runner methods involved are:

- `warmup()`: initializes first observations in the buffer.
- `collect(step)`: runs the policy to sample actions and values.
- `insert(data)`: writes transitions and masks to the buffer.
- `compute()`: computes bootstrap values and returns.
- `train(total_num_steps)`: performs MAPPO updates.

## Inputs

### Actor Input

The actor receives agent-local `obs`.

For `ppo` featurization in the new Overcooked environment, the observation is a spatial tensor:

```text
(H, W, 25)
```

It is produced by `lossless_state_encoding(...)` and then multiplied by 255 in the Overcooked wrapper for `ppo` agents.

The 25 channels are:

- 10 player-perspective channels:
  - `ego_loc`
  - `partner_loc`
  - 4 ego orientation channels
  - 4 partner orientation channels
- 6 static map channels:
  - `pot_loc`
  - `counter_loc`
  - `onion_disp_loc`
  - `tomato_disp_loc`
  - `dish_disp_loc`
  - `serve_loc`
- 8 object and soup-state channels:
  - `onions_in_pot`
  - `tomatoes_in_pot`
  - `onions_in_soup`
  - `tomatoes_in_soup`
  - `soup_cook_time_remaining`
  - `dishes`
  - `onions`
  - `tomatoes`
- 1 urgency channel:
  - `urgency`

The actor also receives:

- `rnn_states_actor`
- `masks`
- `available_actions`

`available_actions` masks invalid moves and invalid interactions before sampling from the categorical distribution.

### Critic Input

The critic receives centralized `share_obs`.

For the new Overcooked environment, the shared observation is:

```text
(H, W, 20)
```

The wrapper stacks an identical copy of this global observation for every agent.

The 20 channels are:

- 5 global player channels:
  - `all_player_loc`
  - 4 global player orientation channels
- 6 static map channels:
  - `pot_loc`
  - `counter_loc`
  - `onion_disp_loc`
  - `tomato_disp_loc`
  - `dish_disp_loc`
  - `serve_loc`
- 8 object and soup-state channels:
  - `onions_in_pot`
  - `tomatoes_in_pot`
  - `onions_in_soup`
  - `tomatoes_in_soup`
  - `soup_cook_time_remaining`
  - `dishes`
  - `onions`
  - `tomatoes`
- 1 urgency channel:
  - `urgency`

The critic also receives:

- `rnn_states_critic`
- `masks`

## Outputs

### Actor Output

The Overcooked action space is discrete with 6 actions:

```text
0: NORTH
1: SOUTH
2: EAST
3: WEST
4: STAY
5: INTERACT
```

The actor outputs:

- `actions`: sampled action index per agent.
- `action_log_probs`: log probability of the sampled action.
- next `rnn_states_actor`.

During deterministic evaluation, the actor uses the categorical mode instead of sampling.

### Critic Output

The critic outputs:

- `values`: scalar value prediction per agent.
- next `rnn_states_critic`.

## Policy Network

`R_MAPPOPolicy` owns two separate neural networks:

- `R_Actor`
- `R_Critic`

Actor and critic have separate Adam optimizers.

### Actor Architecture

For the standard SP Overcooked setup:

```text
local obs (H, W, 25)
-> CNNBase
   -> Conv2d(25, 32, kernel=3, stride=1, padding=1)
   -> ReLU
   -> Conv2d(32, 64, kernel=3, stride=1, padding=1)
   -> ReLU
   -> Conv2d(64, 32, kernel=3, stride=1, padding=1)
   -> ReLU
   -> Flatten
   -> Linear(flattened_dim, 64)
   -> ReLU
   -> LayerNorm(64)
   -> Linear(64, 64)
   -> ReLU
   -> LayerNorm(64)
   -> Linear(64, 64)
   -> ReLU
   -> LayerNorm(64)
-> GRU(64, 64, num_layers=1)
-> LayerNorm(64)
-> Linear(64, 6)
-> Categorical action distribution
```

The action layer applies the available-action mask by setting invalid logits to `-1e10`.

### Critic Architecture

The critic mirrors the actor backbone, but its input is the centralized shared observation:

```text
share_obs (H, W, 20)
-> CNNBase
   -> Conv2d(20, 32, kernel=3, stride=1, padding=1)
   -> ReLU
   -> Conv2d(32, 64, kernel=3, stride=1, padding=1)
   -> ReLU
   -> Conv2d(64, 32, kernel=3, stride=1, padding=1)
   -> ReLU
   -> Flatten
   -> Linear(flattened_dim, 64)
   -> ReLU
   -> LayerNorm(64)
   -> Linear(64, 64)
   -> ReLU
   -> LayerNorm(64)
   -> Linear(64, 64)
   -> ReLU
   -> LayerNorm(64)
-> GRU(64, 64, num_layers=1)
-> LayerNorm(64)
-> Linear(64, 1)
-> value
```

## Training Techniques

The ordinary SP RMAPPo path combines the following techniques.

### Self-Play With Parameter Sharing

Both agents use the same shared policy network. Rollouts from both agent slots update that same policy.

### CTDE

The code uses centralized training with decentralized execution:

- Actor: local agent observation.
- Critic: centralized shared observation.

This is controlled by `use_centralized_V`, which is enabled by default.

### Recurrent MAPPO

The `rmappo` path requires a recurrent policy. In the standard launcher, `--use_recurrent_policy` enables a GRU in both actor and critic. The buffer stores recurrent states for each timestep, rollout thread, and agent.

### PPO Clipped Objective

Actor updates use the standard PPO clipped surrogate:

```text
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - clip_param, 1 + clip_param) * advantage
policy_loss = -min(surr1, surr2)
```

The default `clip_param` is `0.2`.

### Advantage Normalization

Advantages are computed from returns and value predictions, then normalized by mean and standard deviation before PPO updates.

### GAE

Generalized Advantage Estimation is enabled by default:

- `gamma = 0.99`
- `gae_lambda = 0.95`

### Value Normalization

`ValueNorm` is enabled by default. `PopArt` exists as an option, but is disabled by default and is mutually exclusive with `ValueNorm`.

### Value Loss Clipping And Huber Loss

The critic loss supports clipped value loss. Huber loss is enabled by default.

### Entropy Bonus Schedule

The trainer adapts entropy coefficient based on configured step horizons. The standard launcher uses:

```text
entropy_coefs = 0.2, 0.05, 0.01
entropy_coef_horizons = 0, 5e6, 1e7
```

For `small_corridor`, the middle horizon is `8e6`.

### Gradient Clipping

Actor and critic gradients are clipped by max norm. The default max norm is `10.0`.

### Proper Time Limit Handling

The standard launcher passes `--use_proper_time_limits`. The replay buffer uses `bad_masks` to distinguish true termination from time-limit truncation while computing returns.

### Reward Shaping

The ordinary SP launcher does not enable `--use_hsp`; the HSP flag is commented out. Therefore the default reward per agent is:

```text
reward_i = sparse_reward + reward_shaping_factor * dense_reward_i
```

`reward_shaping_factor` is annealed over `reward_shaping_horizon`.

The launcher sets:

```text
reward_shaping_horizon = 1e8
```

Reward shaping roles are configured through:

- `--reward_shaping_role`
- `--reward_shaping_roles`

If no per-agent roles are given, all agents use the same role.

## Notes And Options

- `use_hsp` changes reward computation to hidden reward weights, but this is not part of the ordinary SP launcher path.
- `use_agent_policy_id` can add one policy-ID channel to shared observations.
- `random_index` can randomize perspective agent index at reset, but it is disabled by default in the standard SP launcher.
- If observations are not spatial tensors, the actor and critic fall back to `MLPBase`; standard Overcooked PPO observations use `CNNBase`.
- `use_policy_vhead` can add a value head to the actor, but it is disabled by default.

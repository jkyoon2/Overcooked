# Task 3 Pre-Investigation

Date: 2026-05-31

Task spec location in this checkout: `tasks/task_3_ai_loader.md`.

## PI-1: HSP Loader Entry Point

The rMAPPO policy class is:

```python
zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy.R_MAPPOPolicy
```

Class name is `R_MAPPOPolicy`, not `rMAPPOPolicy`.

Constructor signature:

```python
R_MAPPOPolicy(args, obs_space, share_obs_space, act_space, device=torch.device("cpu"))
```

The policy requires the saved training/runtime `args`, observation space, shared
observation space, action space, and device. It cannot be reconstructed safely
from actor weights alone.

## PI-2: Checkpoint Restore Sequence

Concrete restore examples exist in:

- `zsceval/runner/shared/base_runner.py::restore()`
- `zsceval/runner/shared/overcooked_runner.py::restore()`
- `zsceval/analysis/core/policy_loader.py::_load_weights()`
- `zsceval/human_exp/agent_pool.py::_load_policy_from_config()`

Restore sequence:

```python
actor_state_dict = torch.load(actor_model_path, map_location=device)
policy.actor.load_state_dict(actor_state_dict)

critic_state_dict = torch.load(critic_model_path, map_location=device)
policy.critic.load_state_dict(critic_state_dict)
```

For the experiment loader, direct `R_MAPPOPolicy` import is preferable to
`make_trainer_policy_cls()` because importing `base_runner.py` currently requires
`tensorboardX`, which is not installed in the `neurocontroller` conda env.

## PI-3: Checkpoint Files

The expected `zsceval/human_exp/data/checkpoints/{tomato,onion}/` directories
are still empty except `.gitkeep`, but Julie confirmed the real checkpoints live
under `results/Overcooked/`.

Layout/checkpoint mapping:

| Condition id | Layout | Meaning |
|---|---|---|
| `tomato` | `ttt` | tomato-only / tomato-converged alias |
| `onion` | `ooo` | onion-only / onion-converged alias |
| `tto` | `tto` | tomato2-onion1 |
| `too` | `too` | tomato1-onion2 |

Confirmed config paths:

```text
results/Overcooked/ttt/shared/rmappo/hsp-S1/seed1/policy_config.pkl
results/Overcooked/tto/shared/rmappo/hsp-S1/seed1/policy_config.pkl
results/Overcooked/too/shared/rmappo/hsp-S1/seed1/policy_config.pkl
results/Overcooked/ooo/shared/rmappo/hsp-S1/seed1/policy_config.pkl
```

Each `seed1/models/` directory contains paired periodic actor/critic weights.
The latest paired checkpoint in these runs is:

```text
actor_periodic_10000000.pt
critic_periodic_10000000.pt
```

Multiple seeds also exist (`seed1` through `seed6` for `hsp-S1` in these
layouts). Task 3 uses `hsp-S1/seed1` and selects the latest paired actor/critic
step automatically.

## PI-4: rMAPPO Action Call Contract

Actor-only deterministic rollout uses:

```python
actions, rnn_states_actor = policy.act(
    obs,
    rnn_states_actor,
    masks,
    available_actions=available_actions,
    deterministic=True,
)
```

Single-agent input shapes for these checkpoints:

- `obs`: `(1, 13, 5, 25)`
- `rnn_states_actor`: `(1, recurrent_N, hidden_size)` = `(1, 1, 64)`
- `masks`: `(1, 1)`
- `available_actions`: `(1, 6)`

The returned `actions` tensor contains an action index `0..5`.

Action index mapping from `Action.INDEX_TO_ACTION`:

```text
0 -> (0, -1)
1 -> (0, 1)
2 -> (1, 0)
3 -> (-1, 0)
4 -> (0, 0)
5 -> "interact"
```

`deterministic=True` reaches the categorical distribution mode path, so it is
argmax rather than sampling.

## PI-5: Actual ttt/tto/too/ooo rMAPPO Spaces

Verified from each `policy_config.pkl` and matching layout MDP:

| Layout | MDP shape | obs_space | share_obs_space | action_space |
|---|---:|---:|---:|---:|
| `ttt` | `(13, 5)` | `(13, 5, 25)` | `(13, 5, 20)` | `Discrete(6)` |
| `tto` | `(13, 5)` | `(13, 5, 25)` | `(13, 5, 20)` | `Discrete(6)` |
| `too` | `(13, 5)` | `(13, 5, 25)` | `(13, 5, 20)` | `Discrete(6)` |
| `ooo` | `(13, 5)` | `(13, 5, 25)` | `(13, 5, 20)` | `Discrete(6)` |

Common config fields:

- `algorithm_name`: `rmappo`
- `num_agents`: `2`
- `episode_length`: `400`
- `old_dynamics`: `False`
- `use_available_actions`: `True`
- `random_index`: `True`
- `recurrent_N`: `1`
- `hidden_size`: `64`

PPO observations are produced by:

```python
mdp.lossless_state_encoding(state, horizon=args.episode_length, old_dynamics=False)[agent_idx] * 255
```

The current backend controls the human as agent `0` and AI as agent `1`, so
Task 3 uses agent index `1` for HSP inference.

## PI-6: Config Requirement

`R_MAPPOPolicy` requires the saved config tuple:

```python
(all_args, obs_space, share_obs_space, act_space)
```

This config exists next to the confirmed checkpoints. Loading only
`actor_periodic_*.pt` is insufficient because network construction depends on
`args`, spaces, recurrent settings, and action-head settings.

## Implementation Notes

No stop condition remains after the corrected checkpoint paths. The Task 3
loader should:

- accept `tomato` as an alias for `ttt`;
- accept `onion` as an alias for `ooo`;
- also accept concrete layout ids `ttt`, `tto`, `too`, `ooo` for inspection;
- load `policy_config.pkl` once at initialization;
- load the latest paired periodic actor/critic files once at initialization;
- call `policy.act(..., deterministic=True)` synchronously;
- keep observation conversion inside `HSPPolicy._preprocess_obs(state)`;
- reject state/layout shape mismatches instead of padding or slicing.

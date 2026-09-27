# Conditional Diffusion Policy — Multi-Recipe Overcooked

Implementation spec. Replaces the RMAPPO actor (`ARCHITECTURE_RL.md`) with a
conditional diffusion policy, reusing the diffusion/U-Net machinery in
`ARCHITECTURE_Diffusion.md`. Same observation/action interface as the RL actor;
only the network family changes.

---

## Input / Output (fixed — everything below elaborates this)

**TRAINING** (one gradient step, per policy):
```text
network inputs:
  noised_chunk   (B, horizon, 6)         # expert action chunk (one-hot) + noise
  t              (B,)                     # diffusion timestep
  obs_hist       (B, n_obs, H, W, 25)    # ego-centric observation history
  z_goal         (B,)                     # recipe id in {0:ttt, 1:tto, 2:too, 3:ooo}
target:
  eps            (B, horizon, 6)          # the noise that was added
```

**INFERENCE** (one policy call):
```text
inputs:
  obs_hist       (B, n_obs, H, W, 25)
  z_goal         (B,)
internal:
  eps ~ N(0, I)  (B, horizon, 6)          # starting tensor, denoised into the chunk
output:
  action_chunk   (B, n_action_steps)      # discrete actions in {0..5}; execute, then re-plan
```

Note the direction: **inference is noise -> clean**. `eps` is the *starting*
tensor that gets carved into the action chunk; it is not added after a
trajectory. The "+noise" step happens only during training corruption.

---

## Scope: one policy per (layout, agent)

- Train a **separate policy per layout**. No weight sharing across layouts.
- Within a layout, train a **separate policy per agent slot**. `agent0` and
  `agent1` are different networks. No parameter sharing between them.
- Total networks = `n_layouts * 2`.
- Each policy sees only its own agent's ego-centric observation (`ego_loc` =
  that agent) and predicts only that agent's actions.

---

## Fixed decisions

- **Goal = conditioning input; strategy = noise seed.** `z_goal` selects the
  recipe. Strategy (subtask ordering / role split for a fixed recipe) is captured
  as multimodality in the `(obs, z_goal)`-conditioned distribution and selected
  by `eps`. There is no `z_strat` conditioning token.
- **Imitation, not RL.** Behavior cloning from demonstrations. No critic, no
  value head, no advantage. Data comes from recipe-specialist rollouts (§7).
- **Discrete actions via continuous relaxation.** Diffuse 6-dim relaxed one-hot
  vectors; mask + argmax at execution.

---

## Dimensions (defaults; configurable)

```text
n_obs          = 2       # observation history length
horizon        = 16      # predicted chunk length
n_action_steps = 8       # executed per call (~ one subtask unit), then re-plan
action_dim     = 6       # NORTH SOUTH EAST WEST STAY INTERACT
obs_emb_dim     = 128    # per-frame CNN feature
goal_emb_dim    = 64
```

---

## Data

Per (layout, agent) demonstration set. Each sample:
```text
obs_hist   (n_obs, H, W, 25)   # ego-centric, 25 channels per ARCHITECTURE_RL.md
                               #   (10 player-perspective + 6 static map + 8 object/soup + 1 urgency)
                               #   static map channels encode the layout
x0_expert  (horizon, 6)        # expert action chunk as one-hot (optionally label-smoothed)
z_goal     scalar in {0,1,2,3} # recipe label from the demo source
```
A NULL index (id 4) is reserved on `z_goal` for classifier-free guidance.

---

## 1. Observation Encoder

Reuse the RL `CNNBase` conv stack, no GRU. Flatten time like the Robomimic
encoder:
```text
(B, n_obs, H, W, 25) -> (B*n_obs, 25, H, W)

Conv2d(25,32,3,1,1)->ReLU -> Conv2d(32,64,3,1,1)->ReLU -> Conv2d(64,32,3,1,1)->ReLU
-> Flatten -> Linear(flat, 128) -> ReLU -> LayerNorm(128)

-> reshape -> obs_feat (B, n_obs*128) = (B, 256)
```

---

## 2. Conditioning

```text
z_goal (B,) -> nn.Embedding(5, 64) -> goal_emb (B, 64)     # 5 = 4 recipes + NULL

global_cond = concat[obs_feat(256), goal_emb(64)] = (B, 320)
```

---

## 3. Noise / Diffusion Variable

The diffused sample and the noise share the chunk shape `(B, horizon, 6)`.

```text
TRAINING (forward):   x_t = add_noise(x0_expert, eps, t)      # predict eps from x_t
INFERENCE (reverse):  x_T = eps ~ N(0,I);  x_{t-1} = step(x_t, model(...));  x_0 = chunk
```

---

## 4. Conditional U-Net

`ConditionalUnet1D` (Chi et al. diffusion-policy style; the CTM U-Net from
`ARCHITECTURE_Diffusion.md` with the `stop time s` input removed for the DDPM
base).
```text
input_dim       = 6
global_cond_dim = 320
down_dims       = [256, 512, 1024]

t -> sinusoidal(128) -> MLP
condition = concat[t_emb(128), global_cond(320)] = (B, 448)   # FiLM per residual block

forward(sample, t, global_cond):
  sample      (B, 16, 6)
  t           (B,)
  global_cond (B, 320)
  -> eps_hat  (B, 16, 6)
```

---

## 5. Execution

```text
x_0 (B, 16, 6)                        # denoised logits over 6 actions per step
slice: chunk = x_0[:, 1:1+n_action_steps, :]   # (B, 8, 6)
per step:
  mask invalid actions with available_actions (logits -> -1e10)   # feasibility (hard, legit)
  a_t = argmax(6)
step env for n_action_steps actions, then re-plan
```

---

## 6. Training

```text
for (obs_hist, z_goal, x0_expert) in demos:        # per (layout, agent) dataset
    z = z_goal, or NULL with prob p_uncond (0.1-0.2)   # classifier-free guidance
    cond = concat[CNN(obs_hist), goal_emb(z)]
    t   = uniform(1..T)
    eps = randn_like(x0_expert)
    x_t = add_noise(x0_expert, eps, t)
    loss = MSE(model(x_t, t, cond), eps)           # eps-prediction (or x0-prediction)
```
DDPM ~100 train steps; DDIM for faster inference. Swap to the consistency
scheduler from `ARCHITECTURE_Diffusion.md` only after the DDPM base recovers
multimodal recipe behavior.

---

## 7. Data Pipeline

```text
per (layout, agent):
  1. train single-recipe RL specialists per recipe        # single-mode, converge cleanly
  2. roll out with multiple seeds / reward-shaping roles   # inject strategy diversity
  3. tag rollouts with z_goal
  4. distill all into one conditional diffusion policy pi(chunk | obs, z_goal)
```
The base model is only as multimodal as the data. A single deterministic
specialist gives one strategy per recipe — deliberately vary seeds/roles so
multiple strategies per recipe exist.

---

## 8. Strategy Alignment (inference-time module — build after §1-7)

Not part of the base network. Applied at inference by correcting `eps` toward a
differentiable strategy score `A_strat(tau) >= 0` (Noise-Space CBF, poster
Algorithm 1, with `z(noise) -> eps`, `h -> A_strat`). Goal stays fixed via
`z_goal`, so every correction is a small within-recipe move.
```text
eps = randn(B, 16, 6)
for k in range(K):
    tau = generate(eps, obs, z_goal)
    H   = A_strat(tau)
    if H >= -eps_tol: break
    g   = vjp(generate, eps, grad_tau A_strat(tau))   # J_G^T grad_tau A_strat
    eps = eps + (-gamma*H/||g||^2)*g  +  (-beta * proj_perp(eps, g))   # push + prior pull
tau_aligned = generate(eps, obs, z_goal)
```
`A_strat` must be differentiable w.r.t. the pre-argmax logits (roll a soft ego
position through the chunk; score role occupancy / subtask ordering). Detailed
`A_strat` spec on request.

---

## Implementation Order

1. Single layout, single agent. Build §1-6, DDPM base. Confirm all 4 recipes are
   recovered (per-recipe mode coverage; no collapse).
2. Both agent slots -> two policies per layout.
3. All layouts -> one policy set per layout.
4. Add §8 strategy alignment.
5. Swap DDPM -> consistency distillation for cheap VJPs.

---

## Code References

- Conv stack: `zsceval/algorithms/r_mappo/algorithm/r_actor_critic.py` (`CNNBase`)
- U-Net to adapt: `sonic_impl/consistency-policy/consistency_policy/ctm_unet.py`
  (drop `stop time s` for DDPM base)
- Scheduler: `sonic_impl/consistency-policy/consistency_policy/diffusion.py`
- Dataset/chunking: `sonic_impl/consistency-policy/consistency_policy/dataset.py`

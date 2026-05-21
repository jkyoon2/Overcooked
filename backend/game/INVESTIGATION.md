# Pre-Investigation Report — Task 2 Game Engine Integration

> Status: COMPLETE — all PI-1 through PI-7 answered.
> Delete this file before final Task 2 commit (per spec).

---

## PI-1: ZSC-Eval + overcooked-ai relationship

ZSC-Eval has overcooked-ai **vendored in-repo as two separate copies** — NOT a submodule, NOT a pip dependency:

| Copy | Path | Version | Type |
|---|---|---|---|
| Single-recipe (original Carroll et al.) | `zsceval/envs/overcooked/overcooked_ai_py/` | 0.0.1 | Single delivery_reward, only onion support |
| Multi-recipe extension | `zsceval/envs/overcooked_new/src/overcooked_ai_py/` | 1.1.0 fork | Per-recipe values, both onion + tomato |

**Critical coupling issue:** All files inside `overcooked_new/src/overcooked_ai_py/` use **absolute imports** with the full namespace (e.g. `from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.actions import ...`). This means the `neurocontroller/` repo root must be on the Python path.

**Resolution adopted:** Install `zsceval` as an editable package from the repo root with `--no-deps`:
```bash
uv pip install -e .. --no-deps
```
This wires `neurocontroller/` into the backend venv's path without pulling in PyTorch or other heavy deps. Additional packages needed: `absl-py>=2.4.0`, `gym>=0.26,<1.0`, `gymnasium>=1.3.0`.

**We use `overcooked_new` (multi-recipe version)**, not `overcooked`. Only `overcooked_new` supports per-recipe reward values required by the 2×2 experiment design (tomato=5, onion=20).

---

## PI-2: MDP / OvercookedEnv import path

| Component | Import path |
|---|---|
| Core MDP (use this) | `from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld` |
| State class | `from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState` |
| Recipe config | `from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import Recipe, EVENT_TYPES` |
| ZSC-Eval wrapper (avoid — too many deps) | `from zsceval.envs.overcooked_new.Overcooked_Env import OvercookedEnv` |

**Key methods on `OvercookedGridworld`:**
- `OvercookedGridworld.from_layout_name(layout_name)` — class method, creates from layout file
- `mdp.get_standard_start_state()` → `OvercookedState`
- `mdp.get_state_transition(state, joint_action)` → `(new_state, infos)`
  - `infos["event_infos"]` — dict of EVENT_TYPES → `[bool, bool]` per agent
  - `infos["sparse_reward_by_agent"]` — `[float, float]` delivery reward per agent

Our `OvercookedEngine.step()` calls `mdp.get_state_transition()` directly. We do NOT use the ZSC-Eval `OvercookedEnv` wrapper (it has heavy ZSC-Eval training dependencies and uses a `gym.Env` interface we don't need).

---

## PI-3: Layout names

**Multi-recipe layouts (in `overcooked_new`) with both tomato + onion:**

| Layout | Grid | Recipes | Notes |
|---|---|---|---|
| `corner_onion_tomato` | 7×6 | ooo (20), ttt (20) | Simple; used as default for Task 2 |
| `simple_o_t` | 5×4 | ooo, ttt | Very small; cramped_room shape |
| `marshmallow_ttt` | 13×5 | ooo (20), ttt (5) | Wide; already has onion=20/tomato=5 |
| `distant_tomato` | 5×7 | ooo, ttt | Tomato dispenser far from pots |
| `h_far_tomato` | 7×5 | ooo, ttt | Horizontal, far tomato |
| `diff_orders` | 5×5 | ooo, ttt, oot, too | 4-recipe variant |
| `cramped_room_tomato` | 5×4 | ooo, ttt, to | 3-recipe variant |

**Recommended layout for experiment:** `corner_onion_tomato` — both dispensers present, readable layout, simple enough for debugging. For the actual study, `marshmallow_ttt` may be better (spacious, more natural).

Single-recipe layouts validated by ZSC-Eval: `random0`, `random1`, `random3`, `unident_s`, `small_corridor`, `corridor` (all onion-only).

---

## PI-4: Pygame renderer

ZSC-Eval's own `StateVisualizer` in `overcooked_new`:
```
from zsceval.envs.overcooked_new.src.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
```

Key methods:
- `StateVisualizer(tile_size=75).render_state(state, grid)` → `pygame.Surface`
- `visualizer.display_rendered_state(state, grid=grid, window_display=True)` — renders into a live pygame window with `run_static_resizeable_window`

**Important:** `StateVisualizer` class body loads images at **class definition time** (e.g. `ARROW_IMG = pygame.image.load(...)`). This means importing the class immediately requires a working display or `SDL_VIDEODRIVER` env var. For headless operation, set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.

The `render_state()` method returns a Surface (no window) — this is what we use for the throwaway pygame window: render in a loop, blit to window surface.

---

## PI-5: Recipe score override

**Single-recipe version:** Has only `delivery_reward=20` in `OvercookedGridworld.__init__` — ONE value for all deliveries. **Cannot differentiate tomato vs onion**. NOT usable for the experiment.

**Multi-recipe version (`overcooked_new`):** `Recipe` class has a `configure()` classmethod:
```python
Recipe.configure({
    'onion_value': 20,
    'tomato_value': 5,
    'onion_time': 20,   # cook time in ticks
    'tomato_time': 20,
})
```
This configures a class-level singleton. **Must be called BEFORE `OvercookedGridworld.from_layout_name()`**. The Recipe config is global (class-level state) — if multiple engine instances are created, all share the same Recipe config. This is acceptable for our single-episode design.

Layout files in `overcooked_new` typically embed `recipe_values` / `onion_value` / `tomato_value`. Our engine's `Recipe.configure()` call **overrides** the layout's values, giving us explicit control.

**Experiment values confirmed:** `Recipe.configure({'onion_value': 20, 'tomato_value': 5, 'onion_time': 20, 'tomato_time': 20})`

---

## PI-6: Action space

**6 actions, encoded as tuples + one string (NOT integers):**

| Name | Value | Key binding |
|---|---|---|
| NORTH | `(0, -1)` | W |
| SOUTH | `(0, 1)` | S |
| EAST | `(1, 0)` | D |
| WEST | `(-1, 0)` | A |
| STAY | `(0, 0)` | (default / no key) |
| INTERACT | `"interact"` | Space |

`Action.ALL_ACTIONS = [(0,-1), (0,1), (1,0), (-1,0), (0,0), 'interact']`

Frontend sends action as a string name (e.g. `"NORTH"`, `"INTERACT"`). Backend maps:
```python
ACTION_MAP = {
    "NORTH": Action.NORTH, "SOUTH": Action.SOUTH,
    "EAST": Action.EAST, "WEST": Action.WEST,
    "STAY": Action.STAY, "INTERACT": Action.INTERACT,
}
```

---

## PI-7: HSP checkpoint loader entry point

Loader entry point: `zsceval.analysis.core.policy_loader` — `build_models_dir()`, `load_policy_context()`

Actor weights: `rMAPPOPolicy.load_state_dict({"actor": path_to_actor_pt})` → `torch.load(ckpt_path["actor"], map_location=device)`

**Task 2 uses `RandomPolicy` only.** Task 3 plugs in `rMAPPOPolicy` via `ai_loader.py`.

The checkpoint loader requires PyTorch, which is NOT installed in the backend venv in Task 2. Do not import any of the loader machinery in Task 2.

---

## B-cat-A Gap: Recipe class is global singleton

`Recipe.configure()` sets class-level variables. If the engine is reset with different reward parameters, `Recipe.configure()` must be called again before creating a new `OvercookedGridworld`. For Task 2 this is not an issue (single config, single engine). Flagging for Task 3/4 in case multi-trial reset changes recipe config.

# Architecture

## The one rule

Policy code never knows whether it is driving PyFlex or a physical arm. Everything above
`registration/` is platform-agnostic; everything below it is an adapter.

```
                    EXPERIMENT CONFIG
             conf/sim_exp/…   |   conf/real_world_exp/…
                              │
                              ▼
                  tool/hydra_train.py
                  tool/hydra_eval.py
                  tool/eval_real_world.py
                              │
                              ▼
              ┌───────────────────────────────┐
              │ controllers/                  │
              │  MAGPIE · LaGarNet ·          │
              │  baselines · demonstrators    │
              └───────────────┬───────────────┘
                              │
                    ═══ registration/ ═══
              agent.py · sim_arena.py · real_arena.py · task.py
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌────────────────────┐        ┌────────────────────────┐
    │ env/softgym_garment│        │ real_robot/            │
    │ PyFlex arenas      │        │ UR / xArm Lite 6       │
    │ action primitives  │        │ RealSense + SAM        │
    └────────────────────┘        └────────────────────────┘
```

`registration/*.py` maps a string in a config to a Python class, through `actoris_harena`'s
`register_agent` / `register_arena`. Swapping `arena` from a simulated one to a real one is the
entire sim-to-real change at the software level — the agent and task configs are reused verbatim.

## The interface itself

Everything crossing the boundary is normalised, so it means the same thing on both sides:

- **Observations** — `info['observation']` with `rgb`, `depth`, `mask`, `semkey_norm_pixel`,
  resized to the arena's configured resolution.
- **Actions** — a dict `{primitive_name: params}` in normalised pixel coordinates on `[-1, 1]`:

  | Primitive | Params |
  |---|---|
  | `norm-pixel-pick-and-fling` | 4 |
  | `norm-pixel-dual-pick-and-place` | 8 |
  | `norm-pixel-single-pick-and-place` | 4 |
  | `no-operation` | 0 |

- **Evaluation** — `info['evaluation']`, `info['reward']`, `info['success']`, `info['goals']`,
  produced by the task, not the arena.

Pixel-normalised actions are what make a policy portable: the simulation arena and the real cell
each convert them into their own metric workspace.

## Execution path

`GarmentEnv.step` → `HybridActionPrimitive.step` (executes in PyFlex) → `_process_info`, which
builds observation, evaluation, success, reward, goals and `done`. `arena.evaluate()` delegates to
`task.evaluate(arena)`. Note that `evaluate()` runs **before** `success()`, so active-subgoal logic
must not depend on the latched `has_succeeded` flag — it lags by one step.

## Deliberate deviations from the group software standard

This repository follows the group's robotics-software guidance in substance, not in package layout.

**No ROS 2, no `src/` package tree.** This is a Hydra + PyFlex research stack. The separation the
standard asks for is already present and enforced by `registration/`; renaming directories into
`robot_control`, `robot_perception` and so on would break every import, every Hydra `config_path`
and every job script while changing nothing about the actual coupling.

**No `robot_bringup` launch files.** `tool/hydra_{train,eval,train_and_transfer}.py` plus
`job_scripts/` are the equivalent entry points. The `mode:=sim|real` switch the standard suggests is
expressed here by choosing a `sim_exp` or a `real_world_exp` config.

**No simulation job in CI.** PyFlex needs a GPU and a manually built extension, neither available on
hosted runners. CI covers byte-compilation, unit tests and full config composition instead — which
catches the failure mode that actually recurs here.

**Two experiment records, not the full back-catalogue.** `experiments/EXP001` and `EXP002`
establish the pattern; historical runs live in `conf/sim_exp/` and are reproducible from there.

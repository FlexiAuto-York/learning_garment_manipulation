# Tests

Three levels, split by what they need to run.

| Level | Needs | Runs in CI |
|---|---|---|
| `unit/` | Python only | yes |
| `integration/` | PyFlex + GPU | no |
| `system/` | PyFlex + GPU, full task rollout | no |

## Unit

```bash
python -m pytest tests/unit -q
```

No GPU, no PyFlex, no dataset. `test_config_composition.py` is the one that earns its keep: it
composes every shipped experiment config and checks that `transfer_eval` targets resolve, which is
the failure mode this repository keeps hitting — a config that only breaks once the job is queued.

Run it after any change under `conf/`.

## Integration

Perception → planning and planning → control across the registration boundary. Requires a built
PyFlex and a GPU, so run locally:

```bash
source ./setup.sh
HEADLESS=1 ./scripts/run_sim.sh -m pytest tests/integration -q
```

Currently a placeholder — the existing arena tests in `tests/unit/` (`test_multi_garment_arena.py`,
`test_eval_initials.py`, `test_collect_low_level_data.py`) belong here once they are made
self-contained.

## System

End-to-end: spawn a crumpled garment, run a policy, verify the final state meets the task's success
threshold. The natural implementation is a short rollout of
`sim_exp/magpie/heuristic_centre_sleeve_folding` — the heuristic policy needs no checkpoint and no
dataset, so it can assert on a real outcome. Not yet implemented.

## Why CI stops at unit tests

PyFlex requires a GPU and a manually compiled extension against a specific CUDA version. Hosted
runners have neither. Rather than pretend otherwise, CI covers byte-compilation, unit tests and
full config composition; simulation checks are run locally before release.

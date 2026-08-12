# Configuration

Hydra composes every run from `conf/`. There is no global config file: the experiment config is the
entry point, named with `--config-name`.

## Groups

| Group | Location | Decides |
|---|---|---|
| `agent` | `conf/agent/{magpie,lagarnet}/` | policy class (`name:`) and its hyperparameters |
| `arena` | `conf/arena/{magpie,lagarnet}/` | garments, camera resolution, workspace, episode horizon |
| `task` | `conf/task/{magpie,lagarnet}/` | reward shaping, goal generation, success thresholds |
| `data_augmenter` | `conf/data_augmenter/` | train-time pixel augmentation (optional) |
| `sim_exp` | `conf/sim_exp/{magpie,lagarnet}/` | a simulation experiment: composes the three above |
| `real_world_exp` | `conf/real_world_exp/` | a hardware experiment: same, with a real arena |
| `transfer_eval` | `conf/transfer_eval/{magpie,lagarnet}/` | zero-shot evaluation sweeps |
| `data_collection` | `conf/data_collection/` | demonstration-collection recipes |

## Anatomy of an experiment config

```yaml
# @package _global_
defaults:
  - /agent/magpie@agent: magpie_ctr_align_all_sim_garments_p4_v126_hindsight
  - /arena/magpie@arena: multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
  - /task/magpie@task:  central_alignment

exp_name:  magpie_ctr_align_all_sim_garments_p4_v126_hindsight   # must match the filename
project_name: bimanual_garment_folding                           # W&B project
save_root: /mnt/ssd/garment_folding_data                         # see below
train_and_eval: train_and_evaluate_single                        # actoris_harena driver
```

`# @package _global_` is required — without it the keys nest under `sim_exp` and nothing resolves.

`exp_name` determines the output directory. Keeping it equal to the filename is what makes a result
traceable back to a runnable config.

## Group references must include the method subfolder

`conf/` is namespaced by method. A reference of the form

```yaml
- /task@task: central_alignment          # ✗ ConfigCompositionException
```

does not resolve, because the file is at `conf/task/magpie/central_alignment.yaml`. Write:

```yaml
- /task/magpie@task: central_alignment   # ✓
```

The same applies to `transfer_eval` entries, which name arenas and tasks as plain values:

```yaml
eval_arenas:
  - arena: magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
    task: magpie/central_alignment
```

`tests/unit/test_config_composition.py` composes every shipped experiment config and fails on any
reference that does not resolve. Run it after touching `conf/`.

## Where output goes

`tool/utils.py:resolve_save_root` decides, in order:

1. `$GARMENT_DATA_ROOT`
2. a recognised lab hostname (`pc282`, `thanos`, `viking`, `labruja`)
3. the config's own `save_root`, if its parent directory exists
4. `./results`, with a warning

The shipped configs carry lab paths in `save_root`. Set `GARMENT_DATA_ROOT` rather than editing
hundreds of configs.

Results are written to `<resolved_root>/<project_name>/<exp_name>/`.

## Command-line overrides

Any key can be overridden without editing a file:

```bash
python tool/hydra_train.py --config-name sim_exp/magpie/<exp> \
    agent.total_update_steps=1000 \
    agent.dataset_config.data_path=my_small_dataset

# print the composed config and exit
python tool/hydra_train.py --config-name sim_exp/magpie/<exp> --cfg job
```

`--cfg job` is the quickest way to check that a new config resolves and that the values you expect
actually landed.

## Adding an experiment

1. Copy the closest config in `conf/sim_exp/<method>/` to a new filename.
2. Set `exp_name` to the new filename.
3. Point `agent`, `arena` and `task` at what you want, including the method subfolder.
4. `python tool/hydra_train.py --config-name sim_exp/<method>/<new> --cfg job` to verify.
5. `python -m pytest tests/unit/test_config_composition.py -q`.

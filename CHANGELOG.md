# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-12

First public release, extracted from the internal `bimanual_garment_folding` research repository
and renamed to `learning_garment_manipulation`.

The Weights & Biases project and output subdirectory (`project_name`) still default to
`bimanual_garment_folding`. That is deliberate: it keeps existing runs, checkpoints and W&B history
addressable. Only the repository, the Python distribution and the documented paths were renamed.

### Added
- MAGPIE multi-primitive flow-matching policy (`controllers/magpie/`).
- LaGarNet goal-conditioned latent-dynamics model and MPC controllers
  (`controllers/rl/lagarnet/`).
- SoftGym/PyFlex simulation arenas, tasks and action primitives
  (`env/softgym_garment/`, `env/tasks/`).
- Real-robot adapters for UR and dual xArm Lite 6 cells (`real_robot/`).
- Baselines: ClothFunnels, ClothMate, DreamerV3, UniFolding.
- Reproducible experiment records under `experiments/`.
- `tests/unit/test_config_composition.py`, which composes every shipped experiment
  config without requiring a GPU.
- Containerised environment under `docker/`, and CI in `.github/workflows/ci.yml`.

### Changed
- `resolve_save_root` now honours `$GARMENT_DATA_ROOT` first, then known lab hostnames,
  then the config's own `save_root`, and finally `./results` with a warning. It previously
  matched on hostname alone and sent every unrecognised machine's output to `./tmp`,
  silently ignoring the configured `save_root`.
- `setup.sh` resolves the repository location from its own path instead of hard-coding
  `../bimanual_garment_folding`, so the clone can be named anything. SoftGym can be
  located with `$SOFTGYM_PATH`, and the conda environment with `$MAGPIE_CONDA_ENV`.

### Fixed
- Repaired ~500 stale Hydra references left over from the reorganisation of `conf/` into
  per-method subfolders. Configs referred to `/task@task: central_alignment` when the file
  had moved to `conf/task/magpie/central_alignment.yaml`, so they raised a composition
  error. This affected experiment configs and `transfer_eval` entries alike.
- Removed experiment configs whose agent or arena no longer existed, and scripts importing
  modules deleted long ago (`tool/train_mp_sac.py`, `tool/train_diffusion.py`,
  `tool/hydra_setup.py`, `tool/collect_demo_data.py`, and two others).

### Removed
- Method families outside MAGPIE/LaGarNet: GPT-Fabric, VLM agents, VCD, MP-SAC, and the
  RoboSuite / dm_control / Raven non-garment benchmarks, together with their configs.
- Paper sources, result notebooks and figures; generated whole-directory source dumps.

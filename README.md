# Learning Garment Manipulation — MAGPIE & LaGarNet

Data-driven **dual-arm garment manipulation**: flattening, canonicalisation-alignment and folding
from crumpled states, in simulation (SoftGym/PyFlex) and on real hardware (UR / xArm Lite 6).

| Method | What it is | Golden-path config |
|---|---|---|
| **MAGPIE** | Multi-primitive imitation policy. An MLP classifier picks one of 4 action primitives; a per-primitive Optimal-Transport Conditional Flow-Matching network generates the continuous parameters. | `sim_exp/magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight` |
| **LaGarNet** | Goal-conditioned latent-dynamics model (RSSM) used for planning; flattening via model-predictive control in latent space. | `sim_exp/lagarnet/final_lagarnet_40000_eps` |

**Author:** Halid A. Kadi · **Contributors:** Houdeyfa Ajrou, Lucy Walsh, Ivan Kapelyukh ·
**Supervisors:** Kasim Terzic, John Oyekan
(York · Sheffield · Imperial · St Andrews · Loughborough)

---

## Contents

1. [Repository map](#1-repository-map)
2. [Installation](#2-installation)
3. [Smoke test — check the install](#3-smoke-test--check-the-install)
4. [Getting the training data](#4-getting-the-training-data)
5. [Training a network policy](#5-training-a-network-policy)
6. [Evaluating a trained policy](#6-evaluating-a-trained-policy)
7. [Configuration system](#7-configuration-system)
8. [Real-robot deployment](#8-real-robot-deployment)
9. [Adding a new agent](#9-adding-a-new-agent)
10. [Troubleshooting](#10-troubleshooting)
11. [Citation & licence](#11-citation--licence)

---

## 1. Repository map

The same policy code runs in simulation and on hardware. Only the layer below the registration
boundary changes.

```
             experiment config  (conf/sim_exp/…  |  conf/real_world_exp/…)
                              │
                              ▼
                    tool/hydra_{train,eval}.py          ← entry points
                              │
                              ▼
                    controllers/            ← policies: MAGPIE, LaGarNet, baselines, demonstrators
                              │
                    ═══ registration/ ═══   ← the stable name → class boundary
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     env/softgym_garment/               real_robot/
     (PyFlex simulation)                (UR / xArm Lite 6, RealSense, SAM)
```

| Path | Responsibility |
|---|---|
| `controllers/` | Policies. `magpie/` (flow-matching), `rl/lagarnet/` (RSSM + MPC), `rl/{cloth_funnels,cloth_mate,dreamer_v3,unifolding}` (baselines), `demonstrators/`, `human/`, `random/` |
| `env/softgym_garment/` | Simulation arenas and the action primitives; `env/tasks/` defines rewards, goals and success |
| `real_robot/` | Hardware adapters: arm drivers, primitives, calibration, camera/perception, loggers |
| `registration/` | `agent.py`, `sim_arena.py`, `real_arena.py`, `task.py` — map config names to classes. **This is the sim/real interface**; both worlds register against it |
| `tool/` | Entry points: `hydra_train.py`, `hydra_eval.py`, `hydra_train_and_transfer.py`, `hydra_collect.py`, `eval_real_world.py` |
| `conf/` | Hydra configs, grouped by method (`magpie/`, `lagarnet/`) — see §7 |
| `experiments/` | Versioned experiment records (`EXP001…`) with a one-command `run.sh` |
| `job_scripts/` | Local background runs and SLURM (Viking) submission |
| `data_augmentation/` | Pixel-space augmenters applied during training |
| `tests/` | `unit/` runs without a GPU; `integration/`, `system/` need PyFlex |
| `docs/` | Architecture, configuration and troubleshooting notes |

---

## 2. Installation

### 2.1 Prerequisites

Three repositories must sit **side by side in the same parent directory**:

```
your-workspace/
├── learning_garment_manipulation/   ← this repository
├── softgym/                         ← https://github.com/halid1020/softgym (branch py3.10)  [simulation only]
└── actoris_harena/                  ← https://github.com/halid1020/actoris_harena (branch develop)
```

`actoris_harena` is the agent/arena framework (`build_agent`, `build_arena`, `evaluate`,
`train_and_evaluate_single`). `softgym` provides PyFlex and is only needed for simulation — skip it
if you are going straight to hardware.

> If SoftGym lives elsewhere, set `SOFTGYM_PATH=/path/to/softgym` before sourcing `setup.sh`.

### 2.2 Environment

```bash
conda create -n magpie python=3.10 -y
conda activate magpie

# framework
cd ../actoris_harena && pip install -e ".[torch]"

# this repository
cd ../learning_garment_manipulation && pip install -e .
```

Optional extras, declared in `pyproject.toml`:

```bash
pip install -e ".[advanced]"    # torch >= 2.0 — required for MAGPIE and LaGarNet
pip install -e ".[clothmate]"   # trimesh, OpenEXR — for the ClothMate baseline
pip install -e ".[dev]"         # pytest, ruff
pip install -e ".[all]"         # everything
```

> MAGPIE and LaGarNet are only registered when `torch >= 2.0.0` — see the version check in
> `registration/agent.py`. If `build_agent` reports an unknown agent name, that check is why.

Build SoftGym's PyFlex bindings by following its own README before continuing.

### 2.3 Assets (simulation only)

Garment meshes, cached goal states and semantic-keypoint annotations. Unzip into the repository
root, so that `assets/` sits next to `conf/`:

```bash
pip install gdown
gdown 1Pwqvu1bGxKbL7Qpt_ots3vQJbzCEBsIa
unzip assets.zip
```

### 2.4 Choose where results go

Checkpoints, evaluation output and logs are written to a single root. Set it once:

```bash
export GARMENT_DATA_ROOT=/path/with/plenty/of/space
```

The shipped configs carry lab-specific `save_root:` paths (e.g. `/mnt/ssd/garment_folding_data`).
`tool/utils.py:resolve_save_root` resolves in this order: `$GARMENT_DATA_ROOT` → a known lab
hostname → the config's own `save_root` → `./results` with a warning. Setting the environment
variable is the reliable option.

Results land in `$GARMENT_DATA_ROOT/<project_name>/<exp_name>/`. Note that `project_name` still
defaults to `bimanual_garment_folding`, the repository's former name — it is the Weights & Biases
project and the output subdirectory, deliberately left unchanged so existing runs, checkpoints and
W&B history stay addressable. Change it in a config if you want a different grouping.

### 2.5 Activate

```bash
cd learning_garment_manipulation
source ./setup.sh          # conda env + PyFlex + PYTHONPATH + MP_FOLD_PATH + REAL_ROBOT_PATH
```

Source this in **every** new shell before running anything. It must be `source`d, not executed.

### 2.6 Docker (alternative)

```bash
cd docker && docker compose build && docker compose run --rm robot
```

Needs the NVIDIA container runtime. See `docker/README.md` for X11 and USB passthrough.

---

## 3. Smoke test — check the install

This runs a **heuristic keypoint policy** — no dataset, no checkpoint, no training required. It is
the fastest way to prove that PyFlex, the assets and the config system all work.

```bash
source ./setup.sh
python tool/hydra_eval.py --config-name sim_exp/magpie/heuristic_centre_sleeve_folding
```

Expected: a PyFlex window (or offscreen buffer) opens, a long-sleeve garment is folded over several
episodes, and per-episode metrics are printed. Results land under
`$GARMENT_DATA_ROOT/bimanual_garment_folding/heuristic_centre_sleeve_folding/`.

**On a hybrid-graphics laptop this will crash on the default Mesa driver** — PyFlex uses legacy
geometry shaders. Force the NVIDIA GPU:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_GPU=0 \
  python tool/hydra_eval.py --config-name sim_exp/magpie/heuristic_centre_sleeve_folding
```

`scripts/run_sim.sh` wraps this for you. On a headless machine also export
`QT_QPA_PLATFORM=offscreen` and `SDL_VIDEODRIVER=dummy`.

---

## 4. Getting the training data

Network policies train **offline from demonstrations**. Each agent config names its dataset:

```yaml
dataset_config:
  data_path: "all_garments_multi_primitive_alignment"   # the dataset name
  data_dir:  "./data/datasets"                          # where it is looked up
```

so the trainer expects `./data/datasets/all_garments_multi_primitive_alignment/`.

| Golden path | Dataset (`data_path`) | Contents |
|---|---|---|
| MAGPIE v126 | `all_garments_multi_primitive_alignment` | Human multi-primitive demos across all garment types |
| LaGarNet 40k | `all_garment_mix` | Mixed flattening trajectories |

### Option A — download the released datasets

```bash
mkdir -p data/datasets && cd data/datasets
gdown <DATASET_FILE_ID>          # TODO: link not yet published
unzip <dataset>.zip
```

> **`<DATASET_FILE_ID>` is a placeholder.** These datasets are not yet publicly hosted — ask the
> maintainer for a copy, or use Option B.

### Option B — collect demonstrations yourself

`tool/hydra_collect.py` drives an arena with a demonstrator policy and writes a `TrajectoryDataset`.
The recipes live in `conf/data_collection/`:

```bash
source ./setup.sh
python tool/hydra_collect.py --config-name data_collection/<recipe>
```

Demonstrators available: `human-multi-primitive` (interactive — you click pick/place points on the
rendered image), `centre_sleeve_folding_policy` and the other keypoint heuristics in
`controllers/demonstrators/`, and `random-multi-primitive` for exploration data.

Collecting the full multi-garment human demo set is a multi-hour job. To verify the pipeline first,
collect a small heuristic set and point a config's `data_path` at it.

### Pretrained checkpoints

To evaluate without training, place the checkpoint in the experiment's own log directory
(§6). Released checkpoints: **`<CHECKPOINT_LINK>` — not yet published.**

---

## 5. Training a network policy

### MAGPIE (golden path)

```bash
source ./setup.sh
export GARMENT_DATA_ROOT=/path/with/space

# foreground ('f'), so you see the output
./job_scripts/submit_training_locally.sh magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight f
```

Identical to running the entry point directly — note the script prepends `sim_exp/`:

```bash
python tool/hydra_train.py \
  --config-name sim_exp/magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight
```

Drop the trailing `f` to run in the background; the log path is printed on start.

**What this config trains** (`conf/agent/magpie/…v126_hindsight.yaml`): ResNet-18 + GroupNorm vision
encoder → 512-d embedding; an MLP primitive classifier over K=4 primitives; per-primitive
`ConditionalMLP1D` flow-matching heads `[1024,1024,1024,512]` with 4-step Euler inference;
hindsight goal relabelling; AdamW lr 3e-4, batch 1024, 120k updates, EMA 0.9999.

### LaGarNet

```bash
./job_scripts/submit_training_locally.sh lagarnet/final_lagarnet_40000_eps f
```

Trains the goal-conditioned RSSM offline (300-d deterministic + 60-d stochastic latent) on 40k
flattening episodes; at evaluation time the learned model is used for MPC.

### On SLURM (Viking)

```bash
./job_scripts/generate_and_submit_viking_job.sh magpie/<exp> -c 6 -m 24G -p gpu -t 72:00:00 -a
```

`-a` uses `tool/hydra_train_and_transfer.py`: train, then run zero-shot transfer evaluation across
the arena/task list in the matching `conf/transfer_eval/` config. Headless rendering variables are
set automatically.

### Monitoring

Training logs to Weights & Biases under `project_name` (default `bimanual_garment_folding`).
Run `wandb offline` to disable.

---

## 6. Evaluating a trained policy

```bash
source ./setup.sh
./job_scripts/submit_evaluating_locally.sh magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight f
```

or directly:

```bash
python tool/hydra_eval.py \
  --config-name sim_exp/magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight
```

**The checkpoint must already be in the experiment's log directory** —
`$GARMENT_DATA_ROOT/<project_name>/<exp_name>/`. Training puts it there; if you downloaded a
checkpoint, copy it there yourself. `eval_checkpoint: -1` in the agent config means "latest".

Clear any previous evaluation output from that directory first, or results will be appended to it.

**Metrics** (defined by the task in `env/tasks/`): normalised coverage, max IoU against the goal
state, and success rate under the task's IoU thresholds.

### Transfer evaluation across garments

To evaluate one trained policy on garments and tasks it was not trained on:

```bash
python tool/hydra_transfer_eval.py --config-name transfer_eval/magpie/<eval_config>
```

Each `conf/transfer_eval/` config names a `train_exp_config` and a list of `eval_arenas`
(arena + task pairs) — for example a long-sleeve-trained MAGPIE evaluated on trousers, skirts and
dresses. Results are written under `<save_root>/transfer_eval/<train_config>/`.

---

## 7. Configuration system

Hydra, with `conf/` as the config root. An experiment config composes three groups:

```yaml
# conf/sim_exp/magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight.yaml
# @package _global_
defaults:
  - /agent/magpie@agent: magpie_ctr_align_all_sim_garments_p4_v126_hindsight   # the policy
  - /arena/magpie@arena: multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
  - /task/magpie@task:  central_alignment                                      # reward, goals, success

exp_name:  magpie_ctr_align_all_sim_garments_p4_v126_hindsight
project_name: bimanual_garment_folding
save_root: /mnt/ssd/garment_folding_data     # overridden by $GARMENT_DATA_ROOT
train_and_eval: train_and_evaluate_single    # which actoris_harena driver to use
```

| Group | Location | Chooses |
|---|---|---|
| `agent` | `conf/agent/{magpie,lagarnet}/` | policy class (`name:`) and all its hyperparameters |
| `arena` | `conf/arena/{magpie,lagarnet}/` | garment set, camera resolution, workspace limits, horizon |
| `task` | `conf/task/{magpie,lagarnet}/` | reward shaping, goals, success thresholds |
| `data_augmenter` | `conf/data_augmenter/` | optional train-time pixel augmentation |

**Configs are namespaced by method.** A reference must include the subfolder —
`- /task/magpie@task: central_alignment`, not `- /task@task: central_alignment`.

To add an experiment, copy the closest existing config in `conf/sim_exp/<method>/`, change
`exp_name` to match the new filename, and point the three groups at the configs you want. Override
anything from the command line:

```bash
python tool/hydra_train.py --config-name sim_exp/magpie/<exp> agent.total_update_steps=1000
```

Print the fully composed config without running it:

```bash
python tool/hydra_train.py --config-name sim_exp/magpie/<exp> --cfg job
```

Action primitives are normalised pixel coordinates in `[-1,1]`, passed as `{primitive_name: params}`:
`norm-pixel-pick-and-fling` (4), `norm-pixel-dual-pick-and-place` (8),
`norm-pixel-single-pick-and-place` (4), `no-operation` (0).

---

## 8. Real-robot deployment

SoftGym is not required. `actoris_harena` and the §2.2 packages are.

1. **SAM weights** — download
   [`sam_vit_h_4b8939.pth`](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth)
   into `real_robot/models/`.
2. **Robot setup** — follow [`tutorials/RunRealWorld.md`](tutorials/RunRealWorld.md) for calibration
   and camera alignment. For the dual xArm Lite 6 cell, `source ./setup.sh xarm` also configures the
   wired NIC (the control boxes sit on a private subnet with no DHCP), then:
   ```bash
   python real_robot/test/test_xarm_lite6_bringup.py --info-only --arm both
   ```
3. **Run a policy:**
   ```bash
   python tool/eval_real_world.py --config-name real_world_exp/magpie_ctr_align_longsleeve_p4_v10
   ```
   As in simulation, the checkpoint must already be in that experiment's log directory.

`conf/real_world_exp/` mirrors `conf/sim_exp/` but composes `/real_world_arena@arena` instead of a
simulated one — the same agent and task configs are reused. That substitution is the whole sim-to-real
change.

---

## 9. Adding a new agent

See [`tutorials/CreateNewAgent.md`](tutorials/CreateNewAgent.md). In short: implement the
`actoris_harena` `Agent` / `TrainableAgent` interface under `controllers/`, register the name in
`registration/agent.py`, and add an agent config under `conf/agent/<method>/` whose `name:` matches.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Segfault / GL crash on startup | PyFlex's geometry shaders on Mesa. Run with `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_GPU=0`, or use `scripts/run_sim.sh` |
| `Directory ../softgym does not exist` | SoftGym is not a sibling. Set `SOFTGYM_PATH=/path/to/softgym` |
| `FileNotFoundError` on a garment mesh or goal | `assets.zip` not unzipped into the repo root (§2.3) |
| Results appear in `./results` unexpectedly | `$GARMENT_DATA_ROOT` is unset and the config's `save_root` does not exist — see the warning line the run prints |
| `ConfigCompositionException` on a new config | A group reference is missing its method subfolder (§7). `python -m pytest tests/unit/test_config_composition.py` checks every shipped config |
| Training starts, then crashes reading data | Dataset missing from `./data/datasets/<data_path>` (§4) |
| Evaluation reports zeros | No checkpoint in the experiment log directory (§6) |
| Qt/SDL errors on a cluster | `export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy` |

---

## 11. Citation & licence

Released under the MIT Licence — see [`LICENSE`](LICENSE).

If you use this code, please cite MAGPIE — see [`CITATION.cff`](CITATION.cff).

```bibtex
@article{kadi2026magpie,
  title  = {Multi-Primitive Bimanual Garment Folding from Crumpled States
            with Pixel-based Flow Matching Policies},
  author = {Kadi, Halid Abdulrahim and Ajrou, Houdeyfa and Walsh, Lucy and
            Kapelyukh, Ivan and Terzi{\'c}, Kasim and Oyekan, John},
  year   = {2026}
}
```

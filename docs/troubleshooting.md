# Troubleshooting

## Rendering and GPU

**Segfault, `GL_INVALID_OPERATION`, or a driver crash as soon as an arena opens.**
PyFlex renders with legacy geometry shaders that recent Mesa versions do not handle. On a
hybrid-graphics laptop the AMD/Intel Mesa driver is selected by default. Force the NVIDIA GPU:

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia EGL_GPU=0 python tool/hydra_eval.py ...
```

or use `./scripts/run_sim.sh tool/hydra_eval.py ...`, which sets all three. `EGL_GPU=0` selects the
NVIDIA EGL device and is the part most often missed.

**`qt.qpa.plugin: could not load the Qt platform plugin "xcb"` on a cluster.**
No X server. Export `QT_QPA_PLATFORM=offscreen` and `SDL_VIDEODRIVER=dummy`, or run
`HEADLESS=1 ./scripts/run_sim.sh ...`. The Viking submission script does this for you.

**CUDA out of memory during training.** Reduce the batch size:
`agent.batch_size=512`. The MAGPIE golden path uses 1024 and expects roughly 24 GB.

## Installation

**`ModuleNotFoundError: No module named 'pyflex'`.** SoftGym's bindings were not built, or
`setup.sh` did not find SoftGym. It looks for a sibling directory; override with
`SOFTGYM_PATH=/path/to/softgym`. Build PyFlex following SoftGym's own README.

**`Directory ../softgym does not exist. Skipping.`** Same cause. Harmless if you only need the real
robot.

**`ModuleNotFoundError: No module named 'actoris_harena'`.** Install the framework in editable mode:
`cd ../actoris_harena && pip install -e ".[torch]"`.

**`ImportError` for `controllers.…` or `env.…`.** `setup.sh` was executed instead of sourced, so
`PYTHONPATH` was never exported. Use `source ./setup.sh`.

## Configuration

**`ConfigCompositionException: Could not find 'task/central_alignment'`.**
The reference is missing its method subfolder. Write `- /task/magpie@task: central_alignment`.
See [configuration.md](configuration.md). `python -m pytest tests/unit/test_config_composition.py`
checks every shipped config at once.

**`In 'sim_exp/…': Key 'agent' not in struct`.** The config is missing `# @package _global_` on its
first line.

**Output appears somewhere unexpected.** Read the `[tool.utils, resolve_save_root]` line the run
prints — it states which of the four rules applied. Set `$GARMENT_DATA_ROOT` to be certain.

## Data and checkpoints

**`FileNotFoundError` on a garment mesh, goal or keypoint file.** `assets.zip` was not unzipped into
the repository root. `assets/` must sit next to `conf/`.

**Training crashes immediately after startup with a dataset path error.** The dataset named by
`agent.dataset_config.data_path` is not under `data/dir` (default `./data/datasets`). See README §4.

**Evaluation reports zeros or random behaviour.** No checkpoint was found in the experiment's log
directory, `<save_root>/<project_name>/<exp_name>/`. Training writes it there; a downloaded
checkpoint must be copied there manually. `eval_checkpoint: -1` means "use the latest".

**Evaluation results look like a previous run.** Old evaluation output in that directory is appended
to, not replaced. Clear it first.

## Hardware

**xArm: `connect socket failed`.** The control boxes are on a private wired subnet with no DHCP, so
a DHCP-configured NIC never gets an address. Run `source ./setup.sh xarm`, which assigns a static
address on that subnet and keeps WiFi as the default route. Verify with:

```bash
python real_robot/test/test_xarm_lite6_bringup.py --info-only --arm both
```

**RealSense not detected in Docker.** The container needs `privileged: true` and the
`/dev/bus/usb` mount — both are in `docker/docker-compose.yml`.

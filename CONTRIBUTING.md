# Contributing

## Before you start

```bash
source ./setup.sh
python -m pytest tests/unit -q
```

`tests/unit` needs neither a GPU nor PyFlex, and takes seconds. Run it before and after your
change.

## Ground rules

**Keep machine-specific settings out of algorithms.** Robot IPs, storage roots, device indices and
camera serials belong in configuration or an environment variable. If you find yourself typing an
absolute path into a `.py` file, put it in `conf/` instead.

**Do not make research code aware of sim vs. real.** Policies talk to `actoris_harena` interfaces;
`registration/` maps names to classes. A new capability that only works in simulation belongs in
`env/softgym_garment/`, and its hardware counterpart in `real_robot/` — behind the same interface.

**Configs are namespaced by method.** New configs go under `conf/<group>/<method>/`, and references
must include the subfolder:

```yaml
- /task/magpie@task: central_alignment      # correct
- /task@task: central_alignment             # will not resolve
```

`tests/unit/test_config_composition.py` enforces this over every shipped config.

**Name experiments after their config.** `exp_name` must match the config's filename, otherwise
results are written to a directory that does not correspond to anything you can re-run.

**Improve components in place.** If a controller needs a variant, add a config or a parameter
rather than copying the module to `..._v2.py`.

## Adding an experiment

Add a folder under `experiments/` with a `README.md` (purpose, robot, metric, known limitations),
an `experiment.yaml` recording the config and commit, and a `run.sh`. Copy `EXP001_*` as a
template. Published results should reference a tag and an experiment ID.

## Pull requests

- Branch from `main` as `feature/<short-name>` or `fix/<short-name>`.
- CI must pass: byte-compilation, `tests/unit`, and config composition.
- Note in the description whether the change affects training behaviour, and whether existing
  checkpoints remain loadable.
- If you change method, architecture, hyperparameters or evaluation, update the corresponding
  section of `README.md` and `CHANGELOG.md` in the same PR.

"""Every shipped experiment config must compose.

This is the cheapest test that catches the failure mode this repository actually suffers from:
configs are namespaced by method (``conf/task/magpie/central_alignment.yaml``), so a reference
written as ``- /task@task: central_alignment`` raises a ConfigCompositionException at run time --
after you have queued the job.

Pure Hydra composition: no PyFlex, no GPU, no dataset. Runs in seconds and in CI.
"""
import os

import pytest

hydra = pytest.importorskip("hydra", reason="hydra-core is required")
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONF_DIR = os.path.join(REPO_ROOT, "conf")


def _configs(group):
    """Config names under conf/<group>, relative to conf/ and without the .yaml suffix."""
    root = os.path.join(CONF_DIR, group)
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".yaml"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), CONF_DIR)
            found.append(rel[: -len(".yaml")])
    return sorted(found)


SIM_EXPERIMENTS = _configs("sim_exp")
REAL_EXPERIMENTS = _configs("real_world_exp")


@pytest.fixture(autouse=True)
def _clean_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(name):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name=name)


def test_config_dir_exists():
    assert os.path.isdir(CONF_DIR), f"conf/ not found at {CONF_DIR}"
    assert SIM_EXPERIMENTS, "no simulation experiment configs found"


@pytest.mark.parametrize("name", SIM_EXPERIMENTS)
def test_sim_experiment_composes(name):
    cfg = _compose(name)
    for key in ("agent", "arena", "task"):
        assert key in cfg, f"{name}: missing '{key}' -- is '# @package _global_' on line 1?"
    assert cfg.agent.get("name"), f"{name}: agent has no 'name', so it cannot be registered"


@pytest.mark.parametrize("name", REAL_EXPERIMENTS)
def test_real_world_experiment_composes(name):
    cfg = _compose(name)
    for key in ("agent", "arena"):
        assert key in cfg, f"{name}: missing '{key}'"


@pytest.mark.parametrize("name", SIM_EXPERIMENTS)
def test_exp_name_matches_filename(name):
    """exp_name determines the output directory; if it drifts from the filename, a result can no
    longer be traced back to a runnable config."""
    cfg = _compose(name)
    exp_name = cfg.get("exp_name")
    if exp_name is None:
        pytest.skip(f"{name} declares no exp_name")
    assert str(exp_name).strip() == os.path.basename(name), (
        f"{name}: exp_name is '{exp_name}' but the file is '{os.path.basename(name)}'"
    )


def test_transfer_eval_targets_resolve():
    """transfer_eval configs name arenas and tasks as plain values rather than through
    `defaults:`, so Hydra composition does not check them. Verify the files exist."""
    missing = []
    for name in _configs("transfer_eval"):
        cfg = OmegaConf.load(os.path.join(CONF_DIR, name + ".yaml"))
        train_exp = cfg.get("train_exp_config")
        if train_exp and not os.path.exists(os.path.join(CONF_DIR, f"{train_exp}.yaml")):
            missing.append(f"{name}: train_exp_config -> {train_exp}")
        for entry in cfg.get("eval_arenas") or []:
            for group in ("arena", "task"):
                value = entry.get(group)
                if value and not os.path.exists(os.path.join(CONF_DIR, group, f"{value}.yaml")):
                    missing.append(f"{name}: {group} -> conf/{group}/{value}.yaml")
    assert not missing, "unresolvable transfer_eval references:\n  " + "\n  ".join(missing)

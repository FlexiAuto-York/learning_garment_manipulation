import os
import socket

# Machines in the original lab setup, kept so existing runs land where they always did.
_KNOWN_HOSTS = {
    "pc282": "./results",
    "thanos": "/data/ah390/bimanual_garment_folding",
    "viking": "/mnt/scratch/users/hcv530/garment_folding_data",
    "labruja": "/data/ah390/bimanual_garment_folding",
}


def resolve_save_root(default_root):
    """
    Where checkpoints, evaluation results and logs are written.

    Precedence:
      1. $GARMENT_DATA_ROOT  -- set this if you are not on one of the lab machines
      2. a known hostname    -- see _KNOWN_HOSTS
      3. the `save_root` declared by the experiment config (`default_root`), if its
         parent directory exists
      4. ./results, with a warning -- so a fresh clone still runs

    The configs ship with lab-specific absolute paths such as /mnt/ssd/garment_folding_data,
    which is why step 4 exists: a newcomer who has not set $GARMENT_DATA_ROOT gets a working
    run in ./results rather than a crash.
    """
    env_root = os.environ.get("GARMENT_DATA_ROOT")
    if env_root:
        print(f"[tool.utils, resolve_save_root] Using $GARMENT_DATA_ROOT: {env_root}")
        return env_root

    hostname = socket.gethostname()
    for known, root in _KNOWN_HOSTS.items():
        if known in hostname:
            print(f"[tool.utils, resolve_save_root] Detected host '{hostname}': {root}")
            return root

    parent = os.path.dirname(os.path.abspath(default_root)) if default_root else ""
    if default_root and os.path.isdir(parent):
        print(f"[tool.utils, resolve_save_root] Using the config's save_root: {default_root}")
        return default_root

    print(
        f"[tool.utils, resolve_save_root] WARNING: host '{hostname}' is unknown, "
        f"$GARMENT_DATA_ROOT is unset, and the config's save_root ({default_root}) does not "
        f"exist. Falling back to ./results -- set GARMENT_DATA_ROOT to choose your own location."
    )
    return "./results"

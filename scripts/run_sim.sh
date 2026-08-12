#!/usr/bin/env bash
# Run any simulation entry point with the GPU/GL settings PyFlex needs.
#
#   ./scripts/run_sim.sh tool/hydra_eval.py --config-name sim_exp/magpie/heuristic_centre_sleeve_folding
#   ./scripts/run_sim.sh tool/hydra_train.py --config-name sim_exp/magpie/<exp>
#
# Why this exists: PyFlex renders with legacy geometry shaders, which crash the Mesa driver that
# hybrid-graphics laptops select by default. These variables force the NVIDIA GPU instead.
# Set HEADLESS=1 on a machine with no X server (clusters, CI).
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python-script> [args...]" >&2
    exit 1
fi

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export EGL_GPU="${EGL_GPU:-0}"

if [ "${HEADLESS:-0}" = "1" ]; then
    export QT_QPA_PLATFORM=offscreen
    export SDL_VIDEODRIVER=dummy
fi

if [ -z "${MP_FOLD_PATH:-}" ]; then
    echo "MP_FOLD_PATH is unset - did you 'source ./setup.sh' first?" >&2
    exit 1
fi

exec python -u "$@"

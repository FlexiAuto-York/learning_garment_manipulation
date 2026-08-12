#!/usr/bin/env bash
# EXP002 - LaGarNet flattening. Run after `source ./setup.sh`.
set -euo pipefail

EXP="lagarnet/final_lagarnet_40000_eps"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [ -z "${MP_FOLD_PATH:-}" ]; then
    echo "Run 'source ./setup.sh' first." >&2
    exit 1
fi
if [ -z "${GARMENT_DATA_ROOT:-}" ]; then
    echo "Warning: GARMENT_DATA_ROOT is unset; output location will be chosen by fallback." >&2
fi

echo "=== EXP002: commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="

echo "--- training ---"
./job_scripts/submit_training_locally.sh "$EXP" f

echo "--- evaluating (MPC; this is slow) ---"
./job_scripts/submit_evaluating_locally.sh "$EXP" f

echo "=== done. Results under \${GARMENT_DATA_ROOT}/bimanual_garment_folding/ ==="

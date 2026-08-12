#!/usr/bin/env bash
# EXP001 - MAGPIE canonicalisation-alignment. Run from the repository root after
# `source ./setup.sh`, or from this directory (it cd's for you).
set -euo pipefail

EXP="magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [ -z "${MP_FOLD_PATH:-}" ]; then
    echo "Run 'source ./setup.sh' first." >&2
    exit 1
fi
if [ -z "${GARMENT_DATA_ROOT:-}" ]; then
    echo "Warning: GARMENT_DATA_ROOT is unset; output location will be chosen by fallback." >&2
fi

echo "=== EXP001: commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ==="

echo "--- training ---"
./job_scripts/submit_training_locally.sh "$EXP" f

echo "--- evaluating ---"
./job_scripts/submit_evaluating_locally.sh "$EXP" f

echo "=== done. Results under \${GARMENT_DATA_ROOT}/bimanual_garment_folding/ ==="

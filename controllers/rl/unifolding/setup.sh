# Resolve the repository root from this script's own location, so the checkout can be
# named anything. Must be sourced, not executed.
_UF_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"

conda activate unifold
_UF_SOFTGYM="${SOFTGYM_PATH:-${_UF_REPO_DIR}/../softgym}"
if [ -d "${_UF_SOFTGYM}" ]; then
  cd "${_UF_SOFTGYM}"
  . ./setup.sh
else
  echo "SoftGym not found at ${_UF_SOFTGYM}. Skipping."
fi

cd "${_UF_REPO_DIR}"

export PYTHONPATH=${PWD}:$PYTHONPATH
export MP_FOLD_PATH=${PWD}
export REAL_ROBOT_PATH="${PWD}/real_robot"
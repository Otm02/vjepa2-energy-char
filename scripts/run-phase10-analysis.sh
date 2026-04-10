#!/bin/bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"
DEFAULT_ANALYSIS_ENV="/home/slurm/comp597/conda/envs/comp597"

cd "${REPO_DIR}"

if [[ -n "${COMP597_ANALYSIS_PYTHON:-}" ]]; then
    ANALYSIS_PYTHON="${COMP597_ANALYSIS_PYTHON}"
elif [[ -d "${COMP597_ANALYSIS_ENV_PREFIX:-${DEFAULT_ANALYSIS_ENV}}" ]]; then
    . "${REPO_DIR}/scripts/conda_init.sh"
    conda activate "${COMP597_ANALYSIS_ENV_PREFIX:-${DEFAULT_ANALYSIS_ENV}}"
    ANALYSIS_PYTHON="$(command -v python3)"
else
    ANALYSIS_PYTHON="$(command -v python3)"
fi

if ! "${ANALYSIS_PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib
for module in ("numpy", "pandas", "matplotlib"):
    importlib.import_module(module)
PY
then
    cat <<EOF
Analysis dependencies are not available in:
  ${ANALYSIS_PYTHON}

Either:
  1. Set COMP597_ANALYSIS_ENV_PREFIX to a conda env that has numpy/pandas/matplotlib, or
  2. Set COMP597_ANALYSIS_PYTHON to a Python interpreter that has those packages.
EOF
    exit 1
fi

"${ANALYSIS_PYTHON}" scripts/analysis/phase10_analysis.py "$@"

echo "Generated analysis artifacts under analysis_outputs/vjepa2/"

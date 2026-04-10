#!/bin/bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPTS_DIR}/.." && pwd)"

python3 "${REPO_DIR}/scripts/run_vjepa2_experiments.py" \
    --launcher slurm \
    --modes baseline \
    --batch-sizes 2 \
    --runs 1 \
    "$@"

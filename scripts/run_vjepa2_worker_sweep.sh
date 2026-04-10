#!/usr/bin/env bash
# Run the worker-count sweep: Mode 3 fine-grained, batch size 2, 5 minutes, 3 reps per worker count.
# Each results root ends with _nw<N> so phase11_extended_analysis.py can infer num_workers.
#
# Usage (from repo root, inside an srun GPU session):
#   bash scripts/smoke_vjepa2_workers.sh   # short test first (recommended)
#   bash scripts/run_vjepa2_worker_sweep.sh
# Optional env: DATASET_CSV, RUN_MINUTES, RESULTS_PARENT, LAUNCHER, MODEL_NAME
# Dataset CSV: export COMP597_JOB_STUDENT_STORAGE_DIR before running, or set
# DATASET_CSV to the manifest path (see smoke_vjepa2_workers.sh).
# MODEL_NAME defaults to vit_huge (matches main V-JEPA2 experiments). On ~12 GiB
# teaching GPUs, use MODEL_NAME=vit_large (same default as smoke_vjepa2_workers.sh)
# or you may hit CUDA OOM at batch size 2.
# Pass --dry-run as first arg to only print commands.

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DRY_RUN="${1:-}"
WORKERS=(0 1 2 4)
RUN_MINUTES="${RUN_MINUTES:-5}"
RESULTS_PARENT="${RESULTS_PARENT:-analysis_inputs}"
MODEL_NAME="${MODEL_NAME:-vit_huge}"

# If we are already inside a Slurm allocation, use local launch to avoid
# nested srun which targets a different node and fails.
if [[ -n "${LAUNCHER:-}" ]]; then
  true  # user explicitly chose
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LAUNCHER="local"
  echo "(detected SLURM_JOB_ID=$SLURM_JOB_ID — using --launcher local)"
else
  LAUNCHER="auto"
fi

# Resolve dataset CSV
if [[ -n "${DATASET_CSV:-}" ]]; then
  CSV="$DATASET_CSV"
elif [[ -n "${COMP597_JOB_STUDENT_STORAGE_DIR:-}" ]]; then
  CSV="${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv"
else
  echo "Need FakeVideo CSV. export COMP597_JOB_STUDENT_STORAGE_DIR or DATASET_CSV" >&2
  exit 1
fi
if [[ ! -f "$CSV" ]]; then
  echo "FakeVideo CSV not found: $CSV" >&2
  exit 1
fi

SWEEP_DIR="${RESULTS_PARENT}/vjepa2_workers_bs2_sweep"
mkdir -p "$SWEEP_DIR"

python3 - <<PY
import json, os
rp = "${RESULTS_PARENT}"
roots = [f"{rp}/vjepa2_workers_bs2_nw{w}" for w in (0, 1, 2, 4)]
manifest = {
    "description": "Worker sweep: fine-grained only, bs=2, ${RUN_MINUTES}-minute runs, 3 repetitions per num_workers",
    "modes": ["finegrained"],
    "batch_sizes": [2],
    "runs": [1, 2, 3],
    "run_minutes": float("${RUN_MINUTES}"),
    "worker_counts": [0, 1, 2, 4],
    "results_roots": roots,
    "note": "Each path must end with _nw<N> for analysis scripts.",
}
os.makedirs("${SWEEP_DIR}", exist_ok=True)
with open("${SWEEP_DIR}/experiment_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

for nw in "${WORKERS[@]}"; do
  ROOT="${RESULTS_PARENT}/vjepa2_workers_bs2_nw${nw}"
  CMD=(
    python3 scripts/run_vjepa2_experiments.py
    --launcher "$LAUNCHER"
    --modes finegrained
    --batch-sizes 2
    --model-name "$MODEL_NAME"
    --runs 1 2 3
    --run-minutes "$RUN_MINUTES"
    --results-root "$ROOT"
    --dataset-csv "$CSV"
    --extra-launch-arg=--model_configs.vjepa2.num_workers
    --extra-launch-arg="$nw"
  )
  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    printf '%q ' "${CMD[@]}"
    echo
  else
    "${CMD[@]}"
  fi
done

echo "Manifest: ${SWEEP_DIR}/experiment_manifest.json"

#!/usr/bin/env bash
# Quick validation before the full worker sweep:
#   1) Spawn import test (src.masks + randaugment shims)
#   2) One fine-grained run with num_workers=2 and a short time budget
#
# Usage (repo root, inside an srun session on a GPU node):
#   bash scripts/smoke_vjepa2_workers.sh
#
# Requires a FakeVideo CSV manifest. Set one of:
#   export COMP597_JOB_STUDENT_STORAGE_DIR=/path   # usual on teaching cluster
#   export DATASET_CSV=/absolute/path/to/videodataset.csv
#
# Optional env:
#   LAUNCHER        — default: local if SLURM_JOB_ID is set, else auto
#   SMOKE_MINUTES   — default: 1
#   SMOKE_MODEL_NAME — default: vit_large (teaching GPUs are often ~12 GiB; vit_huge+bs2 OOMs)
#   SMOKE_BATCH_SIZE — default: 2  (if still OOM, try 1 or a smaller model)

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "== Step 1: spawn import test =="
python3 scripts/test_worker_mask_import_spawn.py

SMOKE_MINUTES="${SMOKE_MINUTES:-1}"
SMOKE_MODEL_NAME="${SMOKE_MODEL_NAME:-vit_large}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
OUT="${REPO}/analysis_inputs/_smoke_vjepa2_workers_nw2"

# If we are already inside a Slurm allocation (SLURM_JOB_ID set), use local
# launch to avoid nested srun which targets a different node and fails.
if [[ -n "${LAUNCHER:-}" ]]; then
  true  # user explicitly chose
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  LAUNCHER="local"
  echo "  (detected SLURM_JOB_ID=$SLURM_JOB_ID — using --launcher local)"
else
  LAUNCHER="auto"
fi

# Resolve dataset CSV
if [[ -n "${DATASET_CSV:-}" ]]; then
  CSV="$DATASET_CSV"
elif [[ -n "${COMP597_JOB_STUDENT_STORAGE_DIR:-}" ]]; then
  CSV="${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv"
else
  echo "Smoke step 2 needs a FakeVideo CSV path." >&2
  echo "  export COMP597_JOB_STUDENT_STORAGE_DIR=/home/slurm/comp597/students/<user>" >&2
  echo "  or: export DATASET_CSV=/full/path/to/videodataset.csv" >&2
  exit 1
fi
if [[ ! -f "$CSV" ]]; then
  echo "FakeVideo CSV not found: $CSV" >&2
  exit 1
fi

echo "== Step 2: short training run (fine-grained, num_workers=2, ${SMOKE_MINUTES} min) =="
echo "    launcher   : $LAUNCHER"
echo "    model_name : $SMOKE_MODEL_NAME"
echo "    batch_size : $SMOKE_BATCH_SIZE"
echo "    dataset    : $CSV"
echo "    output     : $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
python3 scripts/run_vjepa2_experiments.py \
  --launcher "$LAUNCHER" \
  --modes finegrained \
  --batch-sizes "$SMOKE_BATCH_SIZE" \
  --model-name "$SMOKE_MODEL_NAME" \
  --runs 1 \
  --run-minutes "$SMOKE_MINUTES" \
  --results-root "$OUT" \
  --dataset-csv "$CSV" \
  --skip-existing \
  --extra-launch-arg=--model_configs.vjepa2.num_workers \
  --extra-launch-arg=2

BS_DIR="bs${SMOKE_BATCH_SIZE}"
RS="$OUT/mode3_finegrained/${BS_DIR}/run1/run_summary_run1.csv"
if [[ ! -f "$RS" ]] || [[ "$(wc -l < "$RS")" -lt 2 ]]; then
  echo "" >&2
  echo "Smoke test FAILED: expected run_summary with data in $RS" >&2
  echo "Check: $OUT/mode3_finegrained/${BS_DIR}/run1/process.log" >&2
  echo "If you see CUDA out of memory, use a smaller SMOKE_MODEL_NAME (default vit_large) or SMOKE_BATCH_SIZE=1." >&2
  exit 1
fi
echo ""
echo "Smoke test OK: see $OUT/mode3_finegrained/${BS_DIR}/run1/"
echo "If this passed, run: bash scripts/run_vjepa2_worker_sweep.sh"

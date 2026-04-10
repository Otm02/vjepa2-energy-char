# V-JEPA2 workload: usage

This repository wires **Meta V-JEPA2**–style video pretraining into the COMP597 training stack: `launch.py`, Slurm helpers, CodeCarbon, and a fine-grained phase profiler. Use this guide to run the experiment matrix, inspect raw outputs, and regenerate analysis figures.

For general course setup (conda, Slurm paths, extending `launch.py`), see [environments.md](environments.md), [slurm.md](slurm.md), and [programming_guide.md](programming_guide.md).

## Prerequisites

1. **Python environment** with the course dependencies (PyTorch, CodeCarbon, project packages). On the teaching cluster, activate the conda env your course uses before running jobs or local commands.
2. **FakeVideo CSV manifest** — a `videodataset.csv` (or equivalent) listing video clips. The experiment runner defaults to:
   - `${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv`  
   Set `COMP597_JOB_STUDENT_STORAGE_DIR` on the cluster, or pass `--dataset-csv /absolute/path/to/videodataset.csv`.
3. **GPU** — V-JEPA2 is heavy; `vit_huge` with batch size 2 is tuned for larger GPUs. For smaller GPUs, use `--model-name vit_large` (or batch size `1`) via `--extra-launch-arg` or by editing a one-off `launch.py` invocation.

## What gets measured (three modes)

| Mode (CLI)   | Output folder        | Trainer stats      | Purpose |
|-------------|----------------------|--------------------|---------|
| `baseline`  | `mode1_baseline/`    | `noop`             | True baseline: minimal overhead; wall time in `run_metadata.json`. |
| `codecarbon`| `mode2_codecarbon/`  | `codecarbon`       | Coarse energy / power traces (treat **GPU** energy as primary on the class cluster). |
| `finegrained` | `mode3_finegrained/` | `vjepa2_phases` | Phase timings + ~500 ms system timeline samples for utilization and breakdown plots. |

The runner lays out trees under `--results-root` (default `analysis_inputs/vjepa2/`). Per-run artifacts are described in [analysis_inputs/vjepa2/README.md](../analysis_inputs/vjepa2/README.md).

## Run the full experiment matrix

From the **repository root**:

```bash
python3 scripts/run_vjepa2_experiments.py --help
```

Typical full matrix (default: all three modes, batch sizes `2` and `1`, runs `1–3`, Slurm):

```bash
python3 scripts/run_vjepa2_experiments.py \
  --launcher slurm \
  --dataset-csv "${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv"
```

**Local** (no `srun.sh`; uses `python3 launch.py` directly):

```bash
python3 scripts/run_vjepa2_experiments.py --launcher local --dataset-csv /path/to/videodataset.csv
```

**Auto** launcher tries Slurm when `module load slurm` makes `srun` available, otherwise local:

```bash
python3 scripts/run_vjepa2_experiments.py --launcher auto --dataset-csv /path/to/videodataset.csv
```

Useful flags:

- `--modes baseline codecarbon finegrained` — subset of modes.
- `--batch-sizes 2 1` — physical batch sizes (default matches the intended V-JEPA2 study).
- `--runs 1 2 3` — repeat indices per (mode, batch).
- `--results-root analysis_inputs/vjepa2` — raw output root (default).
- `--run-minutes 5.0` — training time budget per run (epochs cap is high so time stops the run).
- `--skip-existing` — skip runs that already have `run_metadata.json` with `status: completed`.
- `--dry-run` — write `experiment_manifest.json` and `command.sh` files only.
- `--extra-launch-arg TOKEN` — repeat to pass through extra `launch.py` arguments (e.g. dataloader or mask-related flags).

### Convenience scripts (single mode, quick tests)

All forward extra args to the runner (`"$@"`):

| Script | Effect |
|--------|--------|
| `./scripts/start-vjepa2.sh` | `baseline`, batch `2`, run `1`, Slurm |
| `./scripts/start-vjepa2-codecarbon.sh` | `codecarbon`, batch `2`, run `1`, Slurm |
| `./scripts/start-vjepa2-finegrained.sh` | `finegrained`, batch `2`, run `1`, Slurm |

Example:

```bash
./scripts/start-vjepa2-finegrained.sh --run-minutes 1 --launcher local
```

### Optional: smoke test (workers / imports)

For a short fine-grained run and import checks (see script header for env vars):

```bash
bash scripts/smoke_vjepa2_workers.sh
```

## Aggregate results and figures

After raw runs exist under `analysis_inputs/vjepa2/` (or a custom `--results-root` you pass consistently):

```bash
./scripts/run-phase10-analysis.sh
```

This runs `scripts/analysis/phase10_analysis.py` with defaults:

- Input: `analysis_inputs/vjepa2`
- Output: `analysis_outputs/vjepa2` (figures, CSVs, `analysis_summary.md`)

Override paths:

```bash
./scripts/run-phase10-analysis.sh --input-root analysis_inputs/vjepa2 --output-root analysis_outputs/vjepa2
```

If structured fine-grained runs are missing, phase 10 can fall back to the checked-in legacy sample under `analysis_inputs/finegrained/` (see `--legacy-finegrained-dir` in `phase10_analysis.py --help`).

**Extended plots** (optional; expects additional input roots you provide):

```bash
python3 scripts/analysis/phase11_extended_analysis.py --help
```

## One-off training without the matrix

Any V-JEPA2 run can be driven directly:

```bash
python3 launch.py --model vjepa2 --trainer simple --help
# Slurm:
./scripts/srun.sh --model vjepa2 --trainer simple --help
```

The matrix runner is a thin layer that fills in `run_metadata.json`, `command.sh`, and mode-specific `trainer_stats` paths.

## Troubleshooting

- **OOM** — Lower `--batch-sizes` to `1`, or use `--model-name vit_large`, or reduce frames / resolution via the runner’s `--num-frames`, `--crop-size`, etc.
- **Wrong dataset path** — Use an absolute `--dataset-csv` or export `COMP597_JOB_STUDENT_STORAGE_DIR` before launching.
- **Analysis Python missing packages** — `run-phase10-analysis.sh` needs NumPy, pandas, matplotlib. Set `COMP597_ANALYSIS_PYTHON` or `COMP597_ANALYSIS_ENV_PREFIX` if your default `python3` is minimal (see the script’s message on failure).

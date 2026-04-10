# V-JEPA2 project — how to run everything

This repository is wired for **V-JEPA2** video self-supervised pretraining on the COMP597 training stack: `launch.py`, Slurm wrappers, **FakeVideo** data, optional **CodeCarbon**, and a **fine-grained phase profiler** (`vjepa2_phases`). Everything below is V-JEPA2–specific unless noted.

---

## What each part of the codebase does

| Area                  | Path                                            | Role                                                                         |
| --------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| CLI entry             | `launch.py`                                     | Local training; `--model vjepa2` selects this workload.                      |
| Slurm entry           | `scripts/srun.sh`                               | Submits or runs under Slurm with the same flags as `launch.py`.              |
| Model + trainer       | `src/models/vjepa2/`                            | V-JEPA2 / JEPA integration and `vjepa2_init`.                                |
| Trainer loop          | `src/trainer/vjepa2_trainer.py`                 | V-JEPA2 training step.                                                       |
| Model config          | `src/config/models/vjepa2/vjepa2_config.py`     | All `model_configs.vjepa2.*` CLI flags (see below).                          |
| Fine-grained stats    | `src/trainer/stats/vjepa2_phases.py`            | Phase timings and system timeline sampling.                                  |
| Dataset / video shims | `src/datasets/`                                 | Re-exports JEPA video utilities so imports work from the repo root.          |
| Mask shims            | `src/masks/`                                    | Re-exports JEPA mask collators for dataloader **spawn** workers.             |
| Experiment matrix     | `scripts/run_vjepa2_experiments.py`             | Runs baseline / CodeCarbon / fine-grained modes into a fixed directory tree. |
| Main analysis         | `scripts/analysis/phase10_analysis.py`          | Aggregates runs → figures + CSVs under `analysis_outputs/vjepa2/`.           |
| Extended analysis     | `scripts/analysis/phase11_extended_analysis.py` | Extra plots when you supply validation + worker-count result roots.          |
| Worker sweep          | `scripts/run_vjepa2_worker_sweep.sh`            | Fine-grained runs for `num_workers` ∈ {0,1,2,4}.                             |
| Smoke test            | `scripts/smoke_vjepa2_workers.sh`               | Spawn import check + one short fine-grained run.                             |
| Import test           | `scripts/test_worker_mask_import_spawn.py`      | Verifies mask / randaugment imports in a **spawn** child process.            |

Upstream JEPA library code lives under `src/models/vjepa2/jepa/` (vendored).

---

## Prerequisites

1. **Python** with PyTorch, this repo’s dependencies, and (for Slurm) cluster modules as required by `scripts/srun.sh`.
2. **FakeVideo CSV** — e.g. `videodataset.csv` listing clips. Default path used by the experiment runner:
   - `${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv`  
   Export `COMP597_JOB_STUDENT_STORAGE_DIR` on the cluster, or pass `--dataset-csv /absolute/path/to/videodataset.csv`.
3. **GPU** with enough memory. Defaults target larger GPUs (`vit_huge`, batch size `2`). On ~12 GiB cards use `vit_large` and/or batch size `1` (see smoke script defaults).

---

## Training: `launch.py` and `scripts/srun.sh`

Both accept the same configuration style. Minimal pattern:

```bash
# Local
python3 launch.py --model vjepa2 --trainer simple --help

# Slurm (from repo root)
./scripts/srun.sh --model vjepa2 --trainer simple --help
```

You must set data and model sections, for example:

- `--data fakevideo`
- `--data_configs.fakevideo.csv_path …`
- `--model_configs.vjepa2.*` (see **Model configuration** below)
- `--trainer_stats noop` | `codecarbon` | `vjepa2_phases` plus matching `trainer_stats_configs.*`

The experiment runner generates full commands for you and saves them as `command.sh` under each run directory.

---

## Three measurement modes (experiment matrix)

| Mode         | CLI name      | Output folder        | `--trainer_stats` | Purpose                                                                      |
| ------------ | ------------- | -------------------- | ----------------- | ---------------------------------------------------------------------------- |
| Baseline     | `baseline`    | `mode1_baseline/`    | `noop`            | Minimal overhead; wall time in `run_metadata.json`.                          |
| CodeCarbon   | `codecarbon`  | `mode2_codecarbon/`  | `codecarbon`      | Coarse energy / power; treat **GPU** energy as primary on the class cluster. |
| Fine-grained | `finegrained` | `mode3_finegrained/` | `vjepa2_phases`   | Per-phase timings + timeline (~500 ms sampling by default).                  |

---

## Experiment matrix: `scripts/run_vjepa2_experiments.py`

Runs combinations of modes × batch sizes × run indices under `--results-root` (default `analysis_inputs/vjepa2/`). Writes `experiment_manifest.json`, per-run `run_metadata.json`, `command.sh`, and logs.

```bash
python3 scripts/run_vjepa2_experiments.py --help
```

**Typical full matrix (Slurm):**

```bash
python3 scripts/run_vjepa2_experiments.py \
  --launcher slurm \
  --dataset-csv "${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv"
```

**Local:**

```bash
python3 scripts/run_vjepa2_experiments.py --launcher local --dataset-csv /path/to/videodataset.csv
```

**Auto** (`--launcher auto`): uses Slurm when `module load slurm` makes `srun` available, otherwise local.

Important flags:

- `--modes baseline codecarbon finegrained` — subset of modes.
- `--batch-sizes 2 1` — default physical batch sizes for the V-JEPA2 study.
- `--runs 1 2 3` — repetition indices per (mode, batch).
- `--results-root` — raw output root (default `analysis_inputs/vjepa2`).
- `--run-minutes` — training time budget per run (epoch cap is high so time stops the run).
- `--epochs-upper-bound` — upper bound on epochs (works with time budget).
- `--sample-interval-secs` — CodeCarbon + fine-grained sampling interval (default `0.5`).
- `--slurm-time-limit` — per-job Slurm limit (must exceed `--run-minutes`).
- `--model-name` — passed to `--model_configs.vjepa2.model_name` (default `vit_huge`).
- `--learning-rate`, `--num-frames`, `--crop-size`, `--patch-size`, `--dtype` — forwarded into model config.
- `--skip-existing` — skip runs whose `run_metadata.json` already has `status: completed`.
- `--dry-run` — write manifests and commands only.
- `--extra-launch-arg TOKEN` — repeat to pass any extra `launch.py` token (e.g. `--model_configs.vjepa2.num_workers` and `2`).

Directory layout for each run is documented in **`analysis_inputs/vjepa2/README.md`**.

---

## Convenience scripts (thin wrappers)

Each forwards extra arguments to `run_vjepa2_experiments.py` (`"$@"`).

| Script                                  | Fixed settings                                        |
| --------------------------------------- | ----------------------------------------------------- |
| `./scripts/start-vjepa2.sh`             | `baseline`, batch `2`, run `1`, `--launcher slurm`    |
| `./scripts/start-vjepa2-codecarbon.sh`  | `codecarbon`, batch `2`, run `1`, `--launcher slurm`  |
| `./scripts/start-vjepa2-finegrained.sh` | `finegrained`, batch `2`, run `1`, `--launcher slurm` |

Example:

```bash
./scripts/start-vjepa2-finegrained.sh --run-minutes 1 --launcher local
```

---

## Worker-count study

DataLoader workers use **spawn**; `src/masks` and `src/datasets` shims exist so worker processes can import JEPA code.

1. **Spawn import test** (no GPU training):

   ```bash
   python3 scripts/test_worker_mask_import_spawn.py
   ```

2. **Smoke** (import test + one short fine-grained run, `num_workers=2`):

   ```bash
   export COMP597_JOB_STUDENT_STORAGE_DIR=…   # or DATASET_CSV=/path/to/videodataset.csv
   bash scripts/smoke_vjepa2_workers.sh
   ```

   Optional env: `LAUNCHER`, `SMOKE_MINUTES`, `SMOKE_MODEL_NAME` (default `vit_large`), `SMOKE_BATCH_SIZE`.

3. **Full sweep** (fine-grained, bs=2, 3 runs per worker count, roots `…/vjepa2_workers_bs2_nw0` … `_nw4`):

   ```bash
   bash scripts/run_vjepa2_worker_sweep.sh
   ```

   Optional env: `DATASET_CSV`, `RUN_MINUTES`, `RESULTS_PARENT`, `LAUNCHER`, `MODEL_NAME`.  
   First argument `--dry-run` prints commands only.  
   If `SLURM_JOB_ID` is set, launcher defaults to **local** to avoid nested `srun`.

---

## Analysis: phase 10 (main)

Aggregates structured inputs into **`analysis_outputs/vjepa2/`** (figures, CSVs, `analysis_summary.md`).

```bash
./scripts/run-phase10-analysis.sh
```

This shell script picks a Python that has NumPy, pandas, and matplotlib (optional conda env via `COMP597_ANALYSIS_ENV_PREFIX` or explicit `COMP597_ANALYSIS_PYTHON`). Then it runs:

```bash
python3 scripts/analysis/phase10_analysis.py "$@"
```

Defaults: `--input-root analysis_inputs/vjepa2`, `--output-root analysis_outputs/vjepa2`.  
If structured fine-grained data is missing, phase 10 can use the legacy sample under `analysis_inputs/finegrained/` (`--legacy-finegrained-dir`, `--legacy-batch-size`).

Full options:

```bash
python3 scripts/analysis/phase10_analysis.py --help
```

---

## Analysis: phase 11 (extended)

For validation comparisons and worker-count roots, after you have produced the corresponding trees under `analysis_inputs/`:

```bash
python3 scripts/analysis/phase11_extended_analysis.py --help
```

Defaults point at `analysis_inputs/vjepa2`, optional `vjepa2_validation_extra`, worker roots `vjepa2_workers_bs2_nw*`, and output `analysis_outputs/vjepa2_extended`. Override with `--main-root`, `--validation-root`, `--worker-roots`, `--output-root`. Missing optional directories are skipped where the code allows empty aggregates.

---

## Model configuration (`model_configs.vjepa2`)

Defined in `src/config/models/vjepa2/vjepa2_config.py`. CLI flags use the dotted form `--model_configs.vjepa2.<name>`. Notable fields:

| Flag                                                                                               | Meaning                                                                   |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `model_name`                                                                                       | e.g. `vit_huge`, `vit_large`                                              |
| `crop_size`, `patch_size`, `num_frames`, `tubelet_size`                                            | Video / ViT geometry                                                      |
| `num_clips`, `sampling_rate`                                                                       | Clip sampling                                                             |
| `num_workers`                                                                                      | DataLoader workers (use with `--extra-launch-arg` from the matrix script) |
| `dtype`                                                                                            | `bfloat16`, `float16`, `float32`                                          |
| `epochs`, `warmup`, `lr`, `weight_decay`, `clip_grad`                                              | Training hyperparameters                                                  |
| `max_runtime_minutes`, `max_steps`                                                                 | Stop conditions (`0` disables)                                            |
| `pred_depth`, `pred_embed_dim`, `use_sdpa`, `loss_exp`, `reg_coeff`, `ema_start`, `ema_end`, `ipe` | Model / optimizer details                                                 |

Use `launch.py --help` (with `--model vjepa2`) to see the full generated list.

---

## Slurm and other scripts

- **`scripts/srun.sh`** — primary way to run V-JEPA2 on the teaching cluster; invoked by the experiment runner when `--launcher slurm`.
- **`scripts/sbatch.sh`**, **`scripts/job.sh`**, **`scripts/bash_srun.sh`**, **`scripts/bash_job.sh`** — other submission patterns; use as your site documents.
- **`scripts/conda_init.sh`** — sourced by `run-phase10-analysis.sh` when activating a conda env for analysis.

---

## Troubleshooting

- **CUDA OOM** — Reduce batch size, use `vit_large`, fewer frames, or smaller `crop_size`.
- **Missing CSV** — Set `COMP597_JOB_STUDENT_STORAGE_DIR` or pass `--dataset-csv` with an absolute path.
- **Nested Slurm failure** — Already inside a GPU allocation: use `--launcher local` or rely on smoke/sweep scripts detecting `SLURM_JOB_ID`.
- **Analysis import errors** — Set `COMP597_ANALYSIS_PYTHON` or `COMP597_ANALYSIS_ENV_PREFIX` so `run-phase10-analysis.sh` uses an interpreter with NumPy, pandas, and matplotlib.

---

## Further documentation

- Raw run tree: **`analysis_inputs/vjepa2/README.md`**
- Course-wide docs (environment, Slurm, code structure): **`docs/ToC.md`**

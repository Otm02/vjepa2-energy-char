#!/usr/bin/env python3
"""Run the V-JEPA2 experiment matrix with deterministic output directories."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_DIR / "scripts"

TRAINING_SUMMARY_RE = re.compile(
    r"TRAINING_SUMMARY\s+global_steps=(?P<global_steps>\d+)\s+"
    r"completed_epochs=(?P<completed_epochs>\d+)\s+"
    r"stop_reason=(?P<stop_reason>\S+)\s+"
    r"wall_time_s=(?P<wall_time_s>[0-9.]+)"
)


@dataclass(frozen=True)
class ModeSpec:
    cli_name: str
    directory_name: str
    trainer_stats: str


MODE_SPECS: Dict[str, ModeSpec] = {
    "baseline": ModeSpec("baseline", "mode1_baseline", "noop"),
    "codecarbon": ModeSpec("codecarbon", "mode2_codecarbon", "codecarbon"),
    "finegrained": ModeSpec("finegrained", "mode3_finegrained", "vjepa2_phases"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_SPECS.keys()),
        default=["baseline", "codecarbon", "finegrained"],
        help="Experiment families to run.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[2, 1],
        help="Physical batch sizes to test. Defaults to the defensible V-JEPA2 sizes expected to fit on RTX 5000 Ada.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Run indices to execute for each mode/batch combination.",
    )
    parser.add_argument(
        "--launcher",
        choices=["auto", "local", "slurm"],
        default="slurm",
        help="How to launch each run. 'slurm' uses scripts/srun.sh; 'local' runs python directly; 'auto' prefers slurm if available after 'module load slurm'.",
    )
    parser.add_argument(
        "--dataset-csv",
        default="${COMP597_JOB_STUDENT_STORAGE_DIR}/vjepa_data/videodataset.csv",
        help=(
            "CSV manifest for the FakeVideo dataset. The default expands "
            "${COMP597_JOB_STUDENT_STORAGE_DIR} when that environment variable is set; "
            "otherwise pass an absolute path or export the variable before running."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=str(REPO_DIR / "analysis_inputs" / "vjepa2"),
        help="Root directory for raw run outputs.",
    )
    parser.add_argument(
        "--run-minutes",
        type=float,
        default=5.0,
        help="Per-run wall-clock training budget in minutes.",
    )
    parser.add_argument(
        "--epochs-upper-bound",
        type=int,
        default=100,
        help="Epoch upper bound used together with the time budget so runs do not stop early because of the epoch count.",
    )
    parser.add_argument(
        "--sample-interval-secs",
        type=float,
        default=0.5,
        help="Sampling interval for CodeCarbon and fine-grained timeline collection.",
    )
    parser.add_argument(
        "--slurm-time-limit",
        default="00:08:00",
        help="Wall-clock time requested from Slurm for each run. Should exceed the 5-minute experiment budget.",
    )
    parser.add_argument(
        "--model-name",
        default="vit_huge",
        help="V-JEPA2 encoder variant.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.000625,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames per clip.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Spatial crop size.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already contain a completed metadata file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests and print commands without executing them.",
    )
    parser.add_argument(
        "--extra-launch-arg",
        action="append",
        default=[],
        help="Additional argument token to pass through to launch.py. Repeat this flag for multiple tokens.",
    )
    return parser.parse_args()


def slurm_available() -> bool:
    probe = subprocess.run(
        [
            "bash",
            "-lc",
            "type module >/dev/null 2>&1 && module load slurm >/dev/null 2>&1 && command -v srun >/dev/null 2>&1",
        ],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def resolve_launcher(requested: str) -> str:
    if requested == "auto":
        return "slurm" if slurm_available() else "local"
    return requested


def build_run_dir(results_root: Path, mode: str, batch_size: int, run_num: int) -> Path:
    mode_dir = MODE_SPECS[mode].directory_name
    return results_root / mode_dir / f"bs{batch_size}" / f"run{run_num}"


def build_launch_args(args: argparse.Namespace, mode: str, batch_size: int, run_num: int, run_dir: Path) -> List[str]:
    launch_args = [
        "--logging.level",
        "INFO",
        "--logging.filename",
        str(run_dir / "training.log"),
        "--model",
        "vjepa2",
        "--trainer",
        "simple",
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--data",
        "fakevideo",
        "--data_configs.fakevideo.csv_path",
        args.dataset_csv,
        "--model_configs.vjepa2.model_name",
        args.model_name,
        "--model_configs.vjepa2.epochs",
        str(args.epochs_upper_bound),
        "--model_configs.vjepa2.max_runtime_minutes",
        str(args.run_minutes),
        "--model_configs.vjepa2.num_frames",
        str(args.num_frames),
        "--model_configs.vjepa2.crop_size",
        str(args.crop_size),
        "--model_configs.vjepa2.patch_size",
        str(args.patch_size),
        "--model_configs.vjepa2.dtype",
        args.dtype,
        "--trainer_stats",
        MODE_SPECS[mode].trainer_stats,
    ]

    if mode == "codecarbon":
        launch_args.extend(
            [
                "--trainer_stats_configs.codecarbon.run_num",
                str(run_num),
                "--trainer_stats_configs.codecarbon.project_name",
                "vjepa2",
                "--trainer_stats_configs.codecarbon.output_dir",
                str(run_dir),
                "--trainer_stats_configs.codecarbon.measure_power_secs",
                str(args.sample_interval_secs),
            ]
        )
    elif mode == "finegrained":
        launch_args.extend(
            [
                "--trainer_stats_configs.vjepa2_phases.run_num",
                str(run_num),
                "--trainer_stats_configs.vjepa2_phases.project_name",
                "vjepa2",
                "--trainer_stats_configs.vjepa2_phases.output_dir",
                str(run_dir),
                "--trainer_stats_configs.vjepa2_phases.measure_power_secs",
                str(args.sample_interval_secs),
            ]
        )

    launch_args.extend(args.extra_launch_arg)
    return launch_args


def build_command(args: argparse.Namespace, launcher: str, launch_args: List[str]) -> List[str]:
    if launcher == "slurm":
        return [str(SCRIPTS_DIR / "srun.sh"), *launch_args]
    return [sys.executable, str(REPO_DIR / "launch.py"), *launch_args]


def parse_training_summary(training_log_path: Path) -> Dict[str, object]:
    if not training_log_path.exists():
        return {}
    content = training_log_path.read_text(encoding="utf-8", errors="replace")
    matches = TRAINING_SUMMARY_RE.findall(content)
    if not matches:
        return {}
    global_steps, completed_epochs, stop_reason, wall_time_s = matches[-1]
    return {
        "global_steps": int(global_steps),
        "completed_epochs": int(completed_epochs),
        "stop_reason": stop_reason,
        "wall_time_s_internal": float(wall_time_s),
    }


def write_manifest(args: argparse.Namespace, results_root: Path, launcher: str) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "launcher": launcher,
        "modes": args.modes,
        "batch_sizes": args.batch_sizes,
        "runs": args.runs,
        "run_minutes": args.run_minutes,
        "epochs_upper_bound": args.epochs_upper_bound,
        "sample_interval_secs": args.sample_interval_secs,
        "model_name": args.model_name,
        "dataset_csv": args.dataset_csv,
        "note": (
            "Primary physical batch-size study defaults to [2, 1]. "
            "This intentionally documents the likely V-JEPA2 fit constraint on RTX 5000 Ada "
            "instead of silently inventing a third physical batch size."
        ),
    }
    (results_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def run_single(
    args: argparse.Namespace,
    launcher: str,
    mode: str,
    batch_size: int,
    run_num: int,
    results_root: Path,
) -> Dict[str, object]:
    run_dir = build_run_dir(results_root, mode, batch_size, run_num)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "run_metadata.json"
    if args.skip_existing and metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed":
                print(f"[skip] {run_dir}")
                return existing
        except Exception:
            pass

    launch_args = build_launch_args(args, mode, batch_size, run_num, run_dir)
    command = build_command(args, launcher, launch_args)
    command_text = shlex.join(command)
    (run_dir / "command.sh").write_text(command_text + "\n", encoding="utf-8")

    metadata: Dict[str, object] = {
        "mode": mode,
        "mode_directory": MODE_SPECS[mode].directory_name,
        "batch_size": batch_size,
        "run_num": run_num,
        "launcher": launcher,
        "command": command,
        "status": "dry_run" if args.dry_run else "pending",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] {command_text}")
        return metadata

    process_log_path = run_dir / "process.log"
    env = os.environ.copy()
    if launcher == "slurm":
        env["COMP597_SLURM_TIME_LIMIT"] = args.slurm_time_limit

    start = time.perf_counter()
    return_code = 1
    with process_log_path.open("w", encoding="utf-8") as process_log:
        process_log.write(command_text + "\n\n")
        process_log.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_DIR),
            env=env,
            stdout=process_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return_code = completed.returncode
    wall_clock_s = time.perf_counter() - start

    training_summary = parse_training_summary(run_dir / "training.log")
    metadata.update(training_summary)
    metadata.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "wall_clock_s_external": round(wall_clock_s, 3),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"[{metadata['status']}] mode={mode} batch_size={batch_size} run={run_num} "
        f"wall_clock_s={metadata['wall_clock_s_external']}"
    )
    return metadata


def main() -> int:
    args = parse_args()
    # So "${COMP597_JOB_STUDENT_STORAGE_DIR}/..." works when the var is set in the shell env.
    args.dataset_csv = os.path.expandvars(args.dataset_csv)
    launcher = resolve_launcher(args.launcher)
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args, results_root, launcher)

    failures = 0
    for mode in args.modes:
        for batch_size in args.batch_sizes:
            for run_num in args.runs:
                metadata = run_single(args, launcher, mode, batch_size, run_num, results_root)
                if metadata.get("status") == "failed":
                    failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

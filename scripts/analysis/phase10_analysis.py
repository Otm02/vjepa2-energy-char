#!/usr/bin/env python3
"""Aggregate V-JEPA2 experiment runs and generate report-ready outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE_LABELS = {
    "baseline": "Mode 1 Baseline",
    "codecarbon": "Mode 2 CodeCarbon",
    "finegrained": "Mode 3 Fine-Grained",
}

MODE_COLORS = {
    "baseline": "#355070",
    "codecarbon": "#6d597a",
    "finegrained": "#b56576",
}

FIG_DPI = 220

# Stacked phase plots: consistent order (compute-heavy phases central)
PHASE_STACK_ORDER = [
    "data_transfer",
    "forward",
    "backward",
    "optimizer_step",
    "ema_update",
    "save_checkpoint",
]

PHASE_STACK_COLORS = {
    "data_transfer": "#8d99ae",
    "forward": "#457b9d",
    "backward": "#e63946",
    "optimizer_step": "#f4a261",
    "ema_update": "#2a9d8f",
    "save_checkpoint": "#6d597a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="analysis_inputs/vjepa2",
        help="Structured raw-run directory produced by scripts/run_vjepa2_experiments.py.",
    )
    parser.add_argument(
        "--output-root",
        default="analysis_outputs/vjepa2",
        help="Directory where figures, aggregate CSVs, and summaries will be written.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Warmup steps to exclude from per-phase steady-state summaries.",
    )
    parser.add_argument(
        "--timeline-interval-secs",
        type=float,
        default=0.5,
        help="Expected timeline sampling interval for aggregation bins.",
    )
    parser.add_argument(
        "--legacy-finegrained-dir",
        default="analysis_inputs/finegrained",
        help="Fallback directory for the checked-in legacy fine-grained sample.",
    )
    parser.add_argument(
        "--legacy-batch-size",
        type=int,
        default=2,
        help="Batch size to assign to the legacy fine-grained sample if used.",
    )
    parser.add_argument(
        "--underutilized-gpu-threshold",
        type=float,
        default=95.0,
        help="Mean GPU utilization threshold used to flag underutilization windows.",
    )
    parser.add_argument(
        "--min-underutilization-duration-s",
        type=float,
        default=1.0,
        help="Minimum contiguous underutilization window to keep in the exported summary.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    csv_dir = output_root / "csv"
    fig_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"csv": csv_dir, "figures": fig_dir}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def first_existing(path_patterns: List[Path]) -> Optional[Path]:
    for path in path_patterns:
        if path.exists():
            return path
    return None


def discover_structured_runs(input_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not input_root.exists():
        return records

    for metadata_path in sorted(input_root.rglob("run_metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = {
            "mode": metadata["mode"],
            "mode_label": MODE_LABELS.get(metadata["mode"], metadata["mode"]),
            "batch_size": metadata["batch_size"],
            "run_num": metadata["run_num"],
            "run_dir": run_dir,
            "metadata": metadata,
            "source": "structured",
        }
        records.append(record)

    return records


def discover_legacy_run(legacy_dir: Path, legacy_batch_size: int) -> List[Dict[str, object]]:
    phase_file = legacy_dir / "phase_timing_run1.csv"
    if not phase_file.exists():
        return []
    return [
        {
            "mode": "finegrained",
            "mode_label": MODE_LABELS["finegrained"],
            "batch_size": legacy_batch_size,
            "run_num": 1,
            "run_dir": legacy_dir,
            "metadata": {"status": "legacy_sample"},
            "source": "legacy",
        }
    ]


def phase_timing_path(record: Dict[str, object]) -> Optional[Path]:
    run_dir = Path(record["run_dir"])
    candidate = first_existing([run_dir / f"phase_timing_run{record['run_num']}.csv", run_dir / "phase_timing_run1.csv"])
    return candidate


def step_summary_path(record: Dict[str, object]) -> Optional[Path]:
    run_dir = Path(record["run_dir"])
    return first_existing([run_dir / f"step_summary_run{record['run_num']}.csv"])


def system_timeline_path(record: Dict[str, object]) -> Optional[Path]:
    run_dir = Path(record["run_dir"])
    return first_existing([run_dir / f"system_timeline_run{record['run_num']}.csv"])


def live_power_path(record: Dict[str, object]) -> Optional[Path]:
    run_dir = Path(record["run_dir"])
    return first_existing([run_dir / f"live_power_run{record['run_num']}.csv", run_dir / "live_power_run1.csv"])


def energy_summary_path(record: Dict[str, object]) -> Optional[Path]:
    run_dir = Path(record["run_dir"])
    return first_existing(
        [
            run_dir / f"summary_run{record['run_num']}.csv",
            run_dir / "summary_run1.csv",
            *(run_dir.glob("run_*_cc_full_rank_*.csv")),
        ]
    )


def summarize_single_run(record: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(record["metadata"])
    phase_df = read_csv_if_exists(phase_timing_path(record) or Path("__missing__"))
    step_df = read_csv_if_exists(step_summary_path(record) or Path("__missing__"))
    run_summary_df = read_csv_if_exists(first_existing([Path(record["run_dir"]) / f"run_summary_run{record['run_num']}.csv"]) or Path("__missing__"))
    live_df = read_csv_if_exists(live_power_path(record) or Path("__missing__"))
    energy_df = read_csv_if_exists(energy_summary_path(record) or Path("__missing__"))
    timeline_df = read_csv_if_exists(system_timeline_path(record) or Path("__missing__"))

    if "duration_ms" in phase_df.columns:
        phase_df["duration_ms"] = pd.to_numeric(phase_df["duration_ms"], errors="coerce")
    if "step" in phase_df.columns:
        phase_df["step"] = pd.to_numeric(phase_df["step"], errors="coerce")
    if "timestamp" in live_df.columns:
        live_df["timestamp"] = pd.to_datetime(live_df["timestamp"], errors="coerce")
    if "elapsed_s" in timeline_df.columns:
        timeline_df["elapsed_s"] = pd.to_numeric(timeline_df["elapsed_s"], errors="coerce")

    steps = metadata.get("global_steps")
    if steps is None:
        if not step_df.empty and "step" in step_df.columns:
            steps = int(step_df["step"].nunique())
        elif not phase_df.empty:
            step_counts = phase_df.loc[phase_df["phase"] == "total_step", "step"].dropna().nunique()
            if step_counts:
                steps = int(step_counts)

    wall_clock_s = metadata.get("wall_clock_s_external")
    if wall_clock_s is None:
        if not run_summary_df.empty and "wall_time_s" in run_summary_df.columns:
            wall_clock_s = pd.to_numeric(run_summary_df["wall_time_s"], errors="coerce").dropna()
            wall_clock_s = float(wall_clock_s.iloc[-1]) if not wall_clock_s.empty else None
        elif not energy_df.empty and "duration" in energy_df.columns:
            duration_col = pd.to_numeric(energy_df["duration"], errors="coerce").dropna()
            wall_clock_s = float(duration_col.iloc[-1]) if not duration_col.empty else None
        elif not live_df.empty and "timestamp" in live_df.columns:
            timestamps = live_df["timestamp"].dropna()
            if len(timestamps) >= 2:
                wall_clock_s = float((timestamps.max() - timestamps.min()).total_seconds())

    gpu_energy_kwh = None
    if not run_summary_df.empty and "codecarbon_gpu_energy_kwh" in run_summary_df.columns:
        col = pd.to_numeric(run_summary_df["codecarbon_gpu_energy_kwh"], errors="coerce").dropna()
        gpu_energy_kwh = float(col.iloc[-1]) if not col.empty else None
    if gpu_energy_kwh is None and not energy_df.empty and "gpu_energy" in energy_df.columns:
        col = pd.to_numeric(energy_df["gpu_energy"], errors="coerce").dropna()
        gpu_energy_kwh = float(col.iloc[-1]) if not col.empty else None
    if gpu_energy_kwh is None and not live_df.empty and "gpu_energy" in live_df.columns:
        col = pd.to_numeric(live_df["gpu_energy"], errors="coerce").dropna()
        gpu_energy_kwh = float(col.iloc[-1]) if not col.empty else None

    throughput = None
    sample_throughput = None
    total_samples = None
    if steps is not None and wall_clock_s not in (None, 0):
        throughput = float(steps) / float(wall_clock_s)
        total_samples = int(steps) * int(record["batch_size"])
        sample_throughput = float(total_samples) / float(wall_clock_s)

    return {
        "mode": record["mode"],
        "mode_label": record["mode_label"],
        "batch_size": int(record["batch_size"]),
        "run_num": int(record["run_num"]),
        "source": record["source"],
        "status": metadata.get("status", "unknown"),
        "run_dir": str(record["run_dir"]),
        "steps": steps,
        "total_samples": total_samples,
        "wall_clock_s": wall_clock_s,
        "throughput_steps_per_s": throughput,
        "throughput_samples_per_s": sample_throughput,
        "gpu_energy_kwh": gpu_energy_kwh,
        "has_phase_timing": not phase_df.empty,
        "has_step_summary": not step_df.empty,
        "has_system_timeline": not timeline_df.empty,
        "has_live_power": not live_df.empty,
    }


def collect_phase_rows(records: List[Dict[str, object]], warmup_steps: int) -> pd.DataFrame:
    frames = []
    for record in records:
        phase_path = phase_timing_path(record)
        if phase_path is None:
            continue
        df = pd.read_csv(phase_path)
        if "duration_ms" not in df.columns:
            continue
        df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
        if "step" in df.columns:
            df["step"] = pd.to_numeric(df["step"], errors="coerce")
            df = df[df["step"] > warmup_steps]
        df["mode"] = record["mode"]
        df["mode_label"] = record["mode_label"]
        df["batch_size"] = record["batch_size"]
        df["run_num"] = record["run_num"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_phase_data(phase_rows: pd.DataFrame) -> pd.DataFrame:
    if phase_rows.empty:
        return pd.DataFrame()
    sub_phases = phase_rows[phase_rows["phase"] != "total_step"].copy()
    per_run = (
        sub_phases.groupby(["mode", "mode_label", "batch_size", "run_num", "phase"], as_index=False)["duration_ms"]
        .mean()
        .rename(columns={"duration_ms": "run_mean_duration_ms"})
    )
    summary = (
        per_run.groupby(["mode", "mode_label", "batch_size", "phase"], as_index=False)
        .agg(
            mean_duration_ms=("run_mean_duration_ms", "mean"),
            std_duration_ms=("run_mean_duration_ms", "std"),
            num_runs=("run_mean_duration_ms", "count"),
        )
    )
    total_per_batch = summary.groupby(["mode", "batch_size"])["mean_duration_ms"].transform("sum")
    summary["time_fraction_pct"] = np.where(total_per_batch > 0, summary["mean_duration_ms"] / total_per_batch * 100.0, np.nan)
    return summary


def build_phase_energy_proxy_tables(
    phase_rows: pd.DataFrame, runs_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate total run GPU energy to phases using time-fraction proxy (energy ∝ phase duration)."""
    empty = pd.DataFrame()
    if phase_rows.empty or runs_df.empty:
        return empty, empty

    sub = phase_rows[phase_rows["phase"] != "total_step"].copy()
    if sub.empty:
        return empty, empty

    run_phase_means = (
        sub.groupby(["batch_size", "run_num", "phase"], as_index=False)["duration_ms"].mean()
    )
    totals = run_phase_means.groupby(["batch_size", "run_num"])["duration_ms"].sum().reset_index(name="total_phase_ms")

    energy_runs = runs_df[
        (runs_df["mode"] == "finegrained") & runs_df["gpu_energy_kwh"].notna()
    ][["batch_size", "run_num", "gpu_energy_kwh"]]

    merged = run_phase_means.merge(totals, on=["batch_size", "run_num"]).merge(
        energy_runs, on=["batch_size", "run_num"], how="inner"
    )
    merged["proxy_phase_energy_kwh"] = merged["gpu_energy_kwh"] * (merged["duration_ms"] / merged["total_phase_ms"])

    agg = (
        merged.groupby(["batch_size", "phase"], as_index=False)
        .agg(
            mean_proxy_energy_kwh=("proxy_phase_energy_kwh", "mean"),
            std_proxy_energy_kwh=("proxy_phase_energy_kwh", "std"),
            num_runs=("proxy_phase_energy_kwh", "count"),
        )
    )
    return agg, merged


def collect_step_timing_rows(records: List[Dict[str, object]], max_step: int = 50) -> pd.DataFrame:
    """Per-step total_step_ms for warmup visualization (includes early steps)."""
    frames = []
    for record in records:
        if record["mode"] != "finegrained":
            continue
        path = step_summary_path(record)
        if path is None:
            continue
        df = read_csv_if_exists(path)
        if df.empty or "total_step_ms" not in df.columns or "step" not in df.columns:
            continue
        df["total_step_ms"] = pd.to_numeric(df["total_step_ms"], errors="coerce")
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df = df[df["step"] <= max_step].copy()
        df["batch_size"] = record["batch_size"]
        df["run_num"] = record["run_num"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_timeline_rows(records: List[Dict[str, object]], timeline_interval_secs: float) -> pd.DataFrame:
    frames = []
    for record in records:
        timeline_path = system_timeline_path(record)
        if timeline_path is None:
            continue
        df = pd.read_csv(timeline_path)
        if "elapsed_s" not in df.columns:
            continue
        df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
        df["time_bin_s"] = (df["elapsed_s"] / timeline_interval_secs).round() * timeline_interval_secs
        df["mode"] = record["mode"]
        df["batch_size"] = record["batch_size"]
        df["run_num"] = record["run_num"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_timeline_rows(timeline_rows: pd.DataFrame) -> pd.DataFrame:
    if timeline_rows.empty:
        return pd.DataFrame()

    metrics = [
        "cpu_utilization_pct",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_power_w",
        "torch_gpu_allocated_mb",
    ]
    for metric in metrics:
        if metric in timeline_rows.columns:
            timeline_rows[metric] = pd.to_numeric(timeline_rows[metric], errors="coerce")

    grouped = timeline_rows.groupby(["mode", "batch_size", "time_bin_s"], as_index=False)
    summary = grouped.agg(
        cpu_utilization_pct_mean=("cpu_utilization_pct", "mean"),
        cpu_utilization_pct_std=("cpu_utilization_pct", "std"),
        gpu_utilization_pct_mean=("gpu_utilization_pct", "mean"),
        gpu_utilization_pct_std=("gpu_utilization_pct", "std"),
        gpu_memory_used_mb_mean=("gpu_memory_used_mb", "mean"),
        gpu_memory_used_mb_std=("gpu_memory_used_mb", "std"),
        gpu_power_w_mean=("gpu_power_w", "mean"),
        gpu_power_w_std=("gpu_power_w", "std"),
        torch_gpu_allocated_mb_mean=("torch_gpu_allocated_mb", "mean"),
        torch_gpu_allocated_mb_std=("torch_gpu_allocated_mb", "std"),
    )
    return summary


def summarize_mode_batch(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    numeric_cols = ["wall_clock_s", "throughput_steps_per_s", "throughput_samples_per_s", "gpu_energy_kwh"]
    for column in numeric_cols:
        runs_df[column] = pd.to_numeric(runs_df[column], errors="coerce")
    summary = (
        runs_df.groupby(["mode", "mode_label", "batch_size"], as_index=False)
        .agg(
            num_runs=("run_num", "count"),
            mean_wall_clock_s=("wall_clock_s", "mean"),
            std_wall_clock_s=("wall_clock_s", "std"),
            mean_throughput_steps_per_s=("throughput_steps_per_s", "mean"),
            std_throughput_steps_per_s=("throughput_steps_per_s", "std"),
            mean_throughput_samples_per_s=("throughput_samples_per_s", "mean"),
            std_throughput_samples_per_s=("throughput_samples_per_s", "std"),
            mean_gpu_energy_kwh=("gpu_energy_kwh", "mean"),
            std_gpu_energy_kwh=("gpu_energy_kwh", "std"),
        )
        .sort_values(["batch_size", "mode"], ascending=[False, True])
    )
    return summary


def build_overhead_summary(mode_batch_summary: pd.DataFrame) -> pd.DataFrame:
    if mode_batch_summary.empty:
        return pd.DataFrame()
    baseline = mode_batch_summary[mode_batch_summary["mode"] == "baseline"][
        ["batch_size", "mean_throughput_steps_per_s", "mean_wall_clock_s"]
    ].rename(
        columns={
            "mean_throughput_steps_per_s": "baseline_throughput_steps_per_s",
            "mean_wall_clock_s": "baseline_wall_clock_s",
        }
    )
    merged = mode_batch_summary.merge(baseline, on="batch_size", how="left")
    merged["throughput_ratio_vs_baseline"] = (
        merged["mean_throughput_steps_per_s"] / merged["baseline_throughput_steps_per_s"]
    )
    merged["throughput_ratio_std_vs_baseline"] = (
        merged["std_throughput_steps_per_s"] / merged["baseline_throughput_steps_per_s"]
    )
    merged["wall_clock_delta_s_vs_baseline"] = (
        merged["mean_wall_clock_s"] - merged["baseline_wall_clock_s"]
    )
    return merged


def detect_underutilization_windows(
    timeline_summary: pd.DataFrame,
    threshold: float,
    min_duration_s: float,
) -> pd.DataFrame:
    if timeline_summary.empty:
        return pd.DataFrame()

    rows = []
    finegrained = timeline_summary[timeline_summary["mode"] == "finegrained"].copy()
    for batch_size, batch_df in finegrained.groupby("batch_size"):
        batch_df = batch_df.sort_values("time_bin_s").reset_index(drop=True)
        active = []
        for _, row in batch_df.iterrows():
            gpu_util = row.get("gpu_utilization_pct_mean")
            if pd.notna(gpu_util) and gpu_util < threshold:
                active.append(row)
            elif active:
                window = _collapse_underutilization_window(batch_size, active, threshold)
                if window["duration_s"] >= min_duration_s:
                    rows.append(window)
                active = []
        if active:
            window = _collapse_underutilization_window(batch_size, active, threshold)
            if window["duration_s"] >= min_duration_s:
                rows.append(window)
    return pd.DataFrame(rows)


def _collapse_underutilization_window(batch_size: int, rows: List[pd.Series], threshold: float) -> Dict[str, object]:
    frame = pd.DataFrame(rows)
    return {
        "batch_size": batch_size,
        "start_s": float(frame["time_bin_s"].min()),
        "end_s": float(frame["time_bin_s"].max()),
        "duration_s": float(frame["time_bin_s"].max() - frame["time_bin_s"].min()),
        "mean_gpu_utilization_pct": float(frame["gpu_utilization_pct_mean"].mean()),
        "mean_cpu_utilization_pct": float(frame["cpu_utilization_pct_mean"].mean()),
        "mean_gpu_power_w": float(frame["gpu_power_w_mean"].mean()),
        "threshold_gpu_utilization_pct": threshold,
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, index=False)


def plot_grouped_bar(
    summary: pd.DataFrame,
    value_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    allowed_modes: Optional[List[str]] = None,
) -> None:
    if summary.empty:
        return
    modes = allowed_modes or sorted(summary["mode"].unique())
    batches = sorted(summary["batch_size"].dropna().unique(), reverse=True)
    if not batches:
        return

    x = np.arange(len(batches))
    width = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    plt.rcParams.update({"font.size": 11})

    for idx, mode in enumerate(modes):
        mode_df = summary[summary["mode"] == mode].set_index("batch_size")
        values = [mode_df.at[batch, value_col] if batch in mode_df.index else np.nan for batch in batches]
        errors = [mode_df.at[batch, std_col] if batch in mode_df.index else np.nan for batch in batches]
        offsets = x + (idx - (len(modes) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, "#7a7a7a"),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"bs={batch}" for batch in batches])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Physical batch size")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_phase_bars(phase_summary: pd.DataFrame, output_dir: Path) -> None:
    if phase_summary.empty:
        return
    for batch_size, batch_df in phase_summary[phase_summary["mode"] == "finegrained"].groupby("batch_size"):
        batch_df = batch_df.sort_values("mean_duration_ms", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            batch_df["phase"],
            batch_df["mean_duration_ms"],
            yerr=batch_df["std_duration_ms"].fillna(0),
            capsize=4,
            color="#457b9d",
            alpha=0.9,
        )
        ax.set_ylabel("Mean duration per step (ms)")
        ax.set_xlabel("Phase")
        ax.set_title(f"Fine-grained V-JEPA2 phase timings (bs={batch_size})")
        ax.set_ylim(bottom=0)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / f"phase_time_bar_bs{batch_size}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


def plot_phase_fraction_stacked(phase_summary: pd.DataFrame, output_path: Path) -> None:
    """100% stacked bars comparing phase time shares at each batch size."""
    if phase_summary.empty:
        return
    fg = phase_summary[phase_summary["mode"] == "finegrained"]
    if fg.empty:
        return

    batch_sizes = sorted(fg["batch_size"].dropna().unique())
    phases = [p for p in PHASE_STACK_ORDER if p in set(fg["phase"].unique())]
    extra = [p for p in sorted(fg["phase"].unique()) if p not in phases]
    phases = phases + extra

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(batch_sizes))
    bottom = np.zeros(len(batch_sizes))

    for phase in phases:
        sub = fg[fg["phase"] == phase].set_index("batch_size")
        heights = [float(sub.loc[bs, "time_fraction_pct"]) / 100.0 if bs in sub.index else 0.0 for bs in batch_sizes]
        heights = np.array(heights, dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottom,
            label=phase.replace("_", " "),
            color=PHASE_STACK_COLORS.get(phase, "#7a7a7a"),
            width=0.55,
        )
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"bs={int(bs)}" for bs in batch_sizes])
    ax.set_ylabel("Fraction of per-step phase time")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase time share (fine-grained, excl. total_step)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_phase_energy_proxy_bars(proxy_df: pd.DataFrame, output_path: Path) -> None:
    """Grouped bars: proxy GPU energy (kWh) attributed to each phase."""
    if proxy_df.empty:
        return

    phases = [p for p in PHASE_STACK_ORDER if p in set(proxy_df["phase"].unique())]
    extra = [p for p in sorted(proxy_df["phase"].unique()) if p not in phases]
    phases = phases + extra
    batch_sizes = sorted(proxy_df["batch_size"].dropna().unique())

    x = np.arange(len(phases))
    width = 0.8 / max(len(batch_sizes), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    bar_colors = ["#457b9d", "#b56576", "#2a9d8f", "#e76f51"]
    for idx, bs in enumerate(batch_sizes):
        sub = proxy_df[proxy_df["batch_size"] == bs].set_index("phase")
        means = [float(sub.loc[p, "mean_proxy_energy_kwh"]) if p in sub.index else 0.0 for p in phases]
        stds = [
            float(sub.loc[p, "std_proxy_energy_kwh"]) if p in sub.index and pd.notna(sub.loc[p, "std_proxy_energy_kwh"]) else 0.0
            for p in phases
        ]
        offsets = x + (idx - (len(batch_sizes) - 1) / 2.0) * width
        ax.bar(
            offsets,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=f"bs={int(bs)}",
            color=bar_colors[idx % len(bar_colors)],
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in phases], fontsize=9)
    ax.set_ylabel("Proxy GPU energy (kWh)")
    ax.set_xlabel("Phase")
    ax.set_title("Time-fraction proxy for GPU energy by phase (fine-grained runs)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_step_warmup_curves(step_rows: pd.DataFrame, output_path: Path, max_step: int = 50) -> None:
    if step_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for batch_size, g in step_rows.groupby("batch_size"):
        sub = g[g["step"] <= max_step]
        curve = sub.groupby("step")["total_step_ms"].agg(["mean", "std"]).reset_index()
        ax.plot(curve["step"], curve["mean"], label=f"bs={int(batch_size)}", linewidth=2)
        ax.fill_between(
            curve["step"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            alpha=0.2,
        )

    ax.set_xlabel("Step index (1-based)")
    ax.set_ylabel("Mean total step time (ms)")
    ax.set_title(f"Warmup: total step time over first {max_step} steps (mean ± std across runs)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_timeline_panels(timeline_summary: pd.DataFrame, output_dir: Path) -> None:
    if timeline_summary.empty:
        return
    finegrained = timeline_summary[timeline_summary["mode"] == "finegrained"]
    for batch_size, batch_df in finegrained.groupby("batch_size"):
        batch_df = batch_df.sort_values("time_bin_s")
        fig, axes = plt.subplots(4, 1, figsize=(12, 12.5), sharex=True)

        axes[0].plot(batch_df["time_bin_s"], batch_df["gpu_utilization_pct_mean"], color="#d62828", linewidth=1.5)
        axes[0].fill_between(
            batch_df["time_bin_s"],
            batch_df["gpu_utilization_pct_mean"] - batch_df["gpu_utilization_pct_std"].fillna(0),
            batch_df["gpu_utilization_pct_mean"] + batch_df["gpu_utilization_pct_std"].fillna(0),
            color="#d62828",
            alpha=0.2,
        )
        axes[0].set_ylabel("GPU util (%)")
        axes[0].set_ylim(bottom=0)

        memory_series = batch_df["gpu_memory_used_mb_mean"].fillna(batch_df["torch_gpu_allocated_mb_mean"])
        memory_std = batch_df["gpu_memory_used_mb_std"].fillna(batch_df["torch_gpu_allocated_mb_std"]).fillna(0)
        axes[1].plot(batch_df["time_bin_s"], memory_series, color="#2a9d8f", linewidth=1.5)
        axes[1].fill_between(
            batch_df["time_bin_s"],
            memory_series - memory_std,
            memory_series + memory_std,
            color="#2a9d8f",
            alpha=0.2,
        )
        axes[1].set_ylabel("GPU memory (MB)")
        axes[1].set_ylim(bottom=0)

        if "gpu_power_w_mean" in batch_df.columns:
            axes[2].plot(batch_df["time_bin_s"], batch_df["gpu_power_w_mean"], color="#bc6c25", linewidth=1.5)
            axes[2].fill_between(
                batch_df["time_bin_s"],
                batch_df["gpu_power_w_mean"] - batch_df["gpu_power_w_std"].fillna(0),
                batch_df["gpu_power_w_mean"] + batch_df["gpu_power_w_std"].fillna(0),
                color="#bc6c25",
                alpha=0.2,
            )
            axes[2].set_ylabel("GPU power (W)")
            axes[2].set_ylim(bottom=0)
        else:
            axes[2].set_visible(False)

        axes[3].plot(batch_df["time_bin_s"], batch_df["cpu_utilization_pct_mean"], color="#264653", linewidth=1.5)
        axes[3].fill_between(
            batch_df["time_bin_s"],
            batch_df["cpu_utilization_pct_mean"] - batch_df["cpu_utilization_pct_std"].fillna(0),
            batch_df["cpu_utilization_pct_mean"] + batch_df["cpu_utilization_pct_std"].fillna(0),
            color="#264653",
            alpha=0.2,
        )
        axes[3].set_ylabel("CPU util (%)")
        axes[3].set_xlabel("Elapsed time (s)")
        axes[3].set_ylim(bottom=0)

        fig.suptitle(f"Fine-grained system telemetry (bs={batch_size})")
        fig.tight_layout()
        fig.savefig(output_dir / f"system_utilization_timeline_bs{batch_size}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


def plot_batch_size_trends(mode_batch_summary: pd.DataFrame, output_path: Path) -> None:
    if mode_batch_summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    legend_handles = []
    legend_labels = []
    for mode, mode_df in mode_batch_summary.groupby("mode"):
        mode_df = mode_df.sort_values("batch_size")
        color = MODE_COLORS.get(mode, "#7a7a7a")
        (h0,) = axes[0].plot(
            mode_df["batch_size"],
            mode_df["mean_throughput_samples_per_s"],
            marker="o",
            color=color,
            linewidth=2,
        )
        axes[1].plot(
            mode_df["batch_size"],
            mode_df["mean_gpu_energy_kwh"],
            marker="o",
            color=color,
            linewidth=2,
        )
        legend_handles.append(h0)
        legend_labels.append(MODE_LABELS.get(mode, mode))

    axes[0].set_title("Training throughput vs. batch size")
    axes[0].set_xlabel("Physical batch size")
    axes[0].set_ylabel("Samples / s")
    axes[0].margins(y=0.12)
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    axes[1].set_title("GPU energy vs. batch size")
    axes[1].set_xlabel("Physical batch size")
    axes[1].set_ylabel("GPU energy (kWh)")
    axes[1].margins(y=0.12)
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=10,
        title="Same color in both panels",
        title_fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    output_root: Path,
    runs_df: pd.DataFrame,
    mode_batch_summary: pd.DataFrame,
    underutilization_windows: pd.DataFrame,
    skipped_items: List[str],
) -> None:
    lines = [
        "# V-JEPA2 Analysis Summary",
        "",
        f"- Runs discovered: {len(runs_df)}",
        f"- Mode/batch aggregates: {len(mode_batch_summary)}",
        "",
    ]

    if not mode_batch_summary.empty:
        lines.append("## Aggregated runs")
        lines.append("")
        lines.append("```text")
        lines.append(mode_batch_summary.to_string(index=False))
        lines.append("```")
        lines.append("")

    if not underutilization_windows.empty:
        lines.append("## Underutilization windows")
        lines.append("")
        lines.append("```text")
        lines.append(underutilization_windows.to_string(index=False))
        lines.append("```")
        lines.append("")

    if skipped_items:
        lines.append("## Skipped outputs")
        lines.append("")
        for item in skipped_items:
            lines.append(f"- {item}")
        lines.append("")

    (output_root / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    out_dirs = ensure_dirs(output_root)
    skipped_items: List[str] = []

    records = discover_structured_runs(input_root)
    if not any(record["mode"] == "finegrained" for record in records):
        records.extend(discover_legacy_run(Path(args.legacy_finegrained_dir), args.legacy_batch_size))

    if not records:
        raise FileNotFoundError(
            "No structured V-JEPA2 runs were found and no legacy fine-grained sample was available."
        )

    runs_df = pd.DataFrame([summarize_single_run(record) for record in records])
    phase_rows = collect_phase_rows(records, args.warmup_steps)
    phase_summary = summarize_phase_data(phase_rows)
    phase_energy_agg, phase_energy_by_run = build_phase_energy_proxy_tables(phase_rows, runs_df)
    step_timing_rows = collect_step_timing_rows(records)
    timeline_rows = collect_timeline_rows(records, args.timeline_interval_secs)
    timeline_summary = summarize_timeline_rows(timeline_rows)
    mode_batch_summary = summarize_mode_batch(runs_df)
    overhead_summary = build_overhead_summary(mode_batch_summary)
    underutilization_windows = detect_underutilization_windows(
        timeline_summary, args.underutilized_gpu_threshold, args.min_underutilization_duration_s
    )

    save_csv(runs_df, out_dirs["csv"] / "run_summary.csv")
    save_csv(mode_batch_summary, out_dirs["csv"] / "mode_batch_summary.csv")
    save_csv(overhead_summary, out_dirs["csv"] / "instrumentation_overhead_summary.csv")
    save_csv(phase_rows, out_dirs["csv"] / "phase_rows.csv")
    save_csv(phase_summary, out_dirs["csv"] / "phase_summary.csv")
    save_csv(phase_energy_agg, out_dirs["csv"] / "phase_energy_proxy_kwh.csv")
    save_csv(phase_energy_by_run, out_dirs["csv"] / "phase_energy_proxy_by_run.csv")
    save_csv(step_timing_rows, out_dirs["csv"] / "step_timing_samples.csv")
    save_csv(timeline_rows, out_dirs["csv"] / "timeline_rows.csv")
    save_csv(timeline_summary, out_dirs["csv"] / "timeline_summary.csv")
    save_csv(underutilization_windows, out_dirs["csv"] / "underutilization_windows.csv")

    if not mode_batch_summary.empty:
        plot_grouped_bar(
            mode_batch_summary,
            value_col="mean_wall_clock_s",
            std_col="std_wall_clock_s",
            ylabel="Wall-clock time (s)",
            title="V-JEPA2 end-to-end wall-clock comparison",
            output_path=out_dirs["figures"] / "end_to_end_time_comparison.png",
        )
    else:
        skipped_items.append("end-to-end wall-clock comparison (no mode/batch summaries)")

    energy_summary = mode_batch_summary.dropna(subset=["mean_gpu_energy_kwh"]) if not mode_batch_summary.empty else pd.DataFrame()
    if not energy_summary.empty:
        plot_grouped_bar(
            energy_summary,
            value_col="mean_gpu_energy_kwh",
            std_col="std_gpu_energy_kwh",
            ylabel="GPU energy (kWh)",
            title="V-JEPA2 total GPU energy comparison",
            output_path=out_dirs["figures"] / "total_gpu_energy_comparison.png",
            allowed_modes=["codecarbon", "finegrained"],
        )
    else:
        skipped_items.append("total GPU energy comparison (no GPU energy summaries found)")

    if not phase_summary.empty:
        plot_phase_bars(phase_summary, out_dirs["figures"])
        plot_phase_fraction_stacked(phase_summary, out_dirs["figures"] / "phase_time_fraction_stacked.png")
    else:
        skipped_items.append("per-phase timing plots (no fine-grained phase timing CSVs found)")

    if not phase_energy_agg.empty:
        plot_phase_energy_proxy_bars(phase_energy_agg, out_dirs["figures"] / "phase_energy_proxy_kwh.png")
    else:
        skipped_items.append("phase energy proxy plot (insufficient fine-grained energy data)")

    if not step_timing_rows.empty:
        plot_step_warmup_curves(step_timing_rows, out_dirs["figures"] / "step_time_warmup.png")
    else:
        skipped_items.append("step warmup plot (no step_summary CSVs)")

    if not timeline_summary.empty:
        plot_timeline_panels(timeline_summary, out_dirs["figures"])
    else:
        skipped_items.append("system utilization timelines (no system_timeline_run*.csv files found)")

    if not mode_batch_summary.empty:
        plot_batch_size_trends(mode_batch_summary, out_dirs["figures"] / "batch_size_trends.png")
    else:
        skipped_items.append("batch-size trend plots (no mode/batch summaries)")

    if not overhead_summary.empty and overhead_summary["throughput_ratio_vs_baseline"].notna().any():
        plot_grouped_bar(
            overhead_summary,
            value_col="throughput_ratio_vs_baseline",
            std_col="throughput_ratio_std_vs_baseline",
            ylabel="Throughput ratio vs. baseline",
            title="Instrumentation overhead relative to the true baseline",
            output_path=out_dirs["figures"] / "instrumentation_overhead.png",
        )
    else:
        skipped_items.append("instrumentation overhead plot (baseline throughput unavailable)")

    write_summary_markdown(output_root, runs_df, mode_batch_summary, underutilization_windows, skipped_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

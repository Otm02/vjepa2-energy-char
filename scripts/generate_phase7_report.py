#!/usr/bin/env python3
"""
generate_phase7_report.py
=========================
Generate Phase 7 preliminary results figures and a Markdown report for the
V-JEPA2 COMP 597 workload.

Figure groups
-------------
A. Training metrics  (x-axis = training step)
   loss, loss_zoomed, learning_rate, grad_norm, throughput, step_time

B. Energy metrics (GPU only)
   breakdown_stacked, total_kwh, efficiency,
   live_power, cumulative_energy

C. Emissions metrics
   co2_per_run, emissions_rate, duration

D. Hardware / system stats
   gpu_memory, mean_power, system_info

CLI arguments
-------------
--codecarbon_dir  DIR   Directory with emissions*.csv files (required)
--live_dir        DIR   Directory with live_power_run*.csv (default: codecarbon_dir)
--log_dir         DIR   Directory with training_log.csv / metrics.csv
--out_dir         DIR   Root output directory (default: docs)
--report_name     FILE  Markdown report filename (default: phase7_preliminary_report.md)

Usage example
-------------
python scripts/generate_phase7_report.py \\
    --codecarbon_dir /path/to/codecarbonlogs \\
    --live_dir       /path/to/codecarbonlogs \\
    --log_dir        /path/to/results \\
    --out_dir        docs \\
    --report_name    phase7_preliminary_report.md
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.35,
    "figure.autolayout":  False,
    "font.size":          10,
})

_FIG_W  = 7.5
_FIG_H  = 3.8
_FIG_SZ = (_FIG_W, _FIG_H)   # standard landscape chart
_FIG_SQ = (5.0, 4.6)          # square / pie

# Colour palette
_C = {
    "cpu":     "#4C72B0",
    "gpu":     "#DD8452",
    "ram":     "#55A868",
    "total":   "#64B5CD",
    "co2":     "#C44E52",
    "duration":"#8172B2",
    "loss":    "#DD8452",
    "lr":      "#4C72B0",
    "grad":    "#937860",
    "tput":    "#55A868",
    "mem":     "#8172B2",
    "cpu_pct": "#4C72B0",
    "power":   "#64B5CD",
}

# ===========================================================================
# I/O helpers
# ===========================================================================

def _save(fig: plt.Figure, path: str) -> str:
    """Save *fig* to *path* (creates parent dirs) and close it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [fig]  {path}")
    return path


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.4g}", fontsize: int = 8) -> None:
    """Add value labels above bar patches."""
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.015,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize,
        )


def _run_labels(n: int) -> list:
    return [f"Run {i}" for i in range(1, n + 1)]


# ===========================================================================
# Data loading
# ===========================================================================

def load_codecarbon(csv_dir: str) -> pd.DataFrame:
    """
    Load CodeCarbon run-level summary CSVs from *csv_dir*.

    Matches  run_*_cc_full_rank_*.csv  (written by SimpleFileOutput.out()).
    Falls back to emissions*.csv for compatibility with the default CodeCarbon
    file writer.  Never loads task-level or substep CSVs.
    """
    p = Path(csv_dir)
    files = (sorted(p.glob("run_*_cc_full_rank_*.csv")) or
             sorted(p.glob("emissions*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {csv_dir!r}.\n"
            "Complete Phase 7 first: ./scripts/start-vjepa2-codecarbon.sh"
        )
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_run"] = len(dfs) + 1
            dfs.append(df)
        except Exception as e:
            print(f"  [warn] Could not read {f.name}: {e}")
    if not dfs:
        raise ValueError("All CSVs failed to parse.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  [cc]   {len(combined)} run record(s) from {len(dfs)} file(s)")
    return combined


def load_live_power(live_dir) -> pd.DataFrame:
    """
    Load live per-measurement power rows written by LivePowerOutput.

    Expected file pattern: live_power_run*.csv
    Expected columns: full EmissionsData field set including
    cpu_power, gpu_power, ram_power, and timestamp.
    """
    if not live_dir:
        return pd.DataFrame()
    p = Path(live_dir)
    files = sorted(p.glob("live_power_run*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        t0 = combined["timestamp"].iloc[0]
        combined["elapsed_s"] = (combined["timestamp"] - t0).dt.total_seconds()
    print(f"  [live] {len(combined)} live-power row(s)")
    return combined


def load_training_log(log_dir) -> pd.DataFrame:
    """
    Load a step-level training metrics CSV.

    Searched names: training_log.csv, metrics.csv, train_metrics.csv.
    Expected columns (any subset):
        step, epoch, loss, lr, grad_norm, throughput,
        step_time_ms, gpu_memory_mb, cpu_percent
    """
    if not log_dir:
        return pd.DataFrame()
    p = Path(log_dir)
    for name in ("training_log.csv", "metrics.csv", "train_metrics.csv"):
        f = p / name
        if f.exists():
            try:
                df = pd.read_csv(f)
                print(f"  [log]  {f.name}  ({len(df)} rows)")
                for alias in ("global_step", "iteration", "iter"):
                    if alias in df.columns and "step" not in df.columns:
                        df = df.rename(columns={alias: "step"})
                return df
            except Exception as e:
                print(f"  [warn] {f.name}: {e}")
    # Fallback: load per-step losses written by CodeCarbonStats.log_stats()
    if log_dir:
        losses_dir = p / "losses"
        if losses_dir.exists():
            loss_files = sorted(losses_dir.glob("run_*_cc_loss_rank_*.csv"))
            if loss_files:
                try:
                    dfs = [pd.read_csv(f) for f in loss_files]
                    loss_df = pd.concat(dfs, ignore_index=True)
                    # Support both column-name formats (named or positional 0/1)
                    if 0 in loss_df.columns:
                        loss_df = loss_df.rename(columns={0: "task_name", 1: "loss"})
                    if "task_name" in loss_df.columns and "step" not in loss_df.columns:
                        loss_df["step"] = (loss_df["task_name"]
                                           .str.extract(r"#(\d+)").astype(float))
                    loss_df["loss"] = pd.to_numeric(loss_df["loss"], errors="coerce")
                    loss_df = loss_df.dropna(subset=["loss"])
                    print(f"  [log]  losses CSV  ({len(loss_df)} rows, loss column only)")
                    return loss_df
                except Exception as e:
                    print(f"  [warn] losses CSV: {e}")
    print("  [info] No training log found -- step-based figures will be skipped.")
    return pd.DataFrame()


def load_step_energy(cc_dir: str) -> pd.DataFrame:
    """
    Load per-step CodeCarbon task data.

    File pattern: run_*_cc_step_rank_*-steps.csv  (written by task_out()).
    Adds a 'step' column parsed from the task_name field.
    """
    p = Path(cc_dir)
    files = sorted(p.glob("run_*_cc_step_rank_*-steps.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if "task_name" in combined.columns:
        combined["step"] = (combined["task_name"]
                            .str.extract(r"#(\d+)").astype(float))
    print(f"  [step_cc] {len(combined)} per-step energy record(s)")
    return combined


def load_substep_energy(cc_dir: str) -> pd.DataFrame:
    """
    Load per-substep CodeCarbon task data.

    File pattern: run_*_cc_substep_rank_*-substeps.csv.
    Adds 'step' and 'substep' columns parsed from task_name.
    task_name examples: 'Forward pass #1', 'Backward pass #1', 'Optimisation step #1'
    """
    p = Path(cc_dir)
    files = sorted(p.glob("run_*_cc_substep_rank_*-substeps.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if "task_name" in combined.columns:
        combined["step"] = (combined["task_name"]
                            .str.extract(r"#(\d+)").astype(float))
        combined["substep"] = (combined["task_name"]
                               .str.replace(r"\s*#\d+$", "", regex=True)
                               .str.strip())
    print(f"  [substep_cc] {len(combined)} per-substep energy record(s)")
    return combined


# ===========================================================================
# GROUP A: Training metrics  (x = training step)
# ===========================================================================

def _get_x(log: pd.DataFrame) -> pd.Series:
    """Return the best x-axis series (step > epoch > row index)."""
    for col in ("step", "epoch"):
        if col in log.columns:
            return log[col]
    return pd.Series(range(len(log)), name="iteration")


def _rolling(s: pd.Series, window: int) -> pd.Series:
    w = max(1, min(window, max(len(s) // 5, 1)))
    return s.rolling(w, center=True, min_periods=1).mean()


def _fix_xaxis(ax: plt.Axes, n_ticks: int = 8) -> None:
    """Cap x-axis tick count to prevent label overplotting on dense step axes."""
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=n_ticks, integer=True))


def plot_loss(log: pd.DataFrame, out: str) -> str:
    """Training loss vs. step (raw + smoothed on same axes)."""
    if log.empty:
        return ""
    loss_col = next((c for c in ("loss", "train_loss", "training_loss")
                     if c in log.columns), None)
    if loss_col is None:
        return ""
    x = _get_x(log)
    y = log[loss_col]
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y, color=_C["loss"], alpha=0.25, linewidth=1.0, label="Raw")
    ax.plot(x, _rolling(y, 20), color=_C["loss"], linewidth=2.0, label="Smoothed (w=20)")
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel("L1 Loss")
    ax.set_title("Training Loss vs. Step")
    ax.legend(fontsize=9)
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "training", "loss.png"))


def plot_loss_zoom(log: pd.DataFrame, out: str) -> str:
    """Smoothed loss only, Y-axis zoomed to [p2, p98] to reveal late-training dynamics."""
    if log.empty:
        return ""
    loss_col = next((c for c in ("loss", "train_loss", "training_loss")
                     if c in log.columns), None)
    if loss_col is None:
        return ""
    x = _get_x(log)
    y_smooth = _rolling(log[loss_col], 20)
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y_smooth, color=_C["loss"], linewidth=2.0)
    lo, hi = np.nanpercentile(y_smooth, [2, 98])
    margin = max((hi - lo) * 0.15, 1e-6)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel("L1 Loss (smoothed)")
    ax.set_title("Training Loss - Zoomed View (p2-p98)")
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "training", "loss_zoomed.png"))


def plot_learning_rate(log: pd.DataFrame, out: str) -> str:
    """Learning rate schedule vs. step."""
    if log.empty or "lr" not in log.columns:
        return ""
    x = _get_x(log)
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, log["lr"], color=_C["lr"], linewidth=1.8)
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule vs. Step")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "training", "learning_rate.png"))


def plot_grad_norm(log: pd.DataFrame, out: str) -> str:
    """Gradient norm vs. step (raw + smoothed)."""
    col = next((c for c in ("grad_norm", "gradient_norm", "grad_norm_total")
                if c in log.columns), None)
    if log.empty or col is None:
        return ""
    x = _get_x(log)
    y = log[col]
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y, color=_C["grad"], alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, _rolling(y, 20), color=_C["grad"], linewidth=2.0, label="Smoothed")
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm vs. Step")
    ax.legend(fontsize=9)
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "training", "grad_norm.png"))


def plot_throughput(log: pd.DataFrame, out: str) -> str:
    """Training throughput (clips/s or samples/s) vs. step."""
    col = next((c for c in ("throughput", "clips_per_sec", "samples_per_sec",
                             "it_per_sec", "steps_per_sec") if c in log.columns), None)
    if log.empty or col is None:
        return ""
    x = _get_x(log)
    y = log[col]
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y, color=_C["tput"], alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, _rolling(y, 10), color=_C["tput"], linewidth=2.0, label="Smoothed")
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel(col.replace("_", " ").title())
    ax.set_title("Training Throughput vs. Step")
    ax.legend(fontsize=9)
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "training", "throughput.png"))


def plot_step_time(log: pd.DataFrame, out: str) -> str:
    """Step wall-clock time (ms) vs. step -- reveals data-loading stalls."""
    col = next((c for c in ("step_time_ms", "batch_time_ms", "iter_time_ms",
                             "step_time", "batch_time") if c in log.columns), None)
    if log.empty or col is None:
        return ""
    x = _get_x(log)
    y = log[col]
    unit = "ms" if "ms" in col else "s"
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y, color=_C["total"], alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, _rolling(y, 10), color=_C["total"], linewidth=2.0, label="Smoothed")
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel(f"Step Time ({unit})")
    ax.set_title("Step Wall-Clock Time vs. Step")
    ax.legend(fontsize=9)
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "training", "step_time.png"))


def plot_step_energy(step_df: pd.DataFrame, out: str) -> str:
    """Line -- total energy consumed (kWh) per training step (from per-step task CSV)."""
    if step_df.empty or "step" not in step_df.columns:
        return ""
    ecol = next((c for c in ("energy_consumed", "gpu_energy") if c in step_df.columns), None)
    if ecol is None:
        return ""
    x = step_df["step"]
    y = step_df[ecol]
    ylabel = "Energy per Step (kWh)"
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y, color=_C["total"], alpha=0.3, linewidth=0.8, label="Raw")
    ax.plot(x, _rolling(y, 10), color=_C["total"], linewidth=2.0, label="Smoothed (w=10)")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-Step Energy ({ecol.replace('_', ' ').title()}) vs. Step")
    ax.legend(fontsize=9)
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "step_energy.png"))


def plot_substep_breakdown(substep_df: pd.DataFrame, out: str) -> str:
    """Bar -- total energy (kWh) by substep type: forward / backward / optimizer."""
    if substep_df.empty or "substep" not in substep_df.columns:
        return ""
    ecol = next((c for c in ("energy_consumed", "gpu_energy") if c in substep_df.columns), None)
    if ecol is None:
        return ""
    totals = substep_df.groupby("substep")[ecol].sum().sort_values(ascending=False)
    style_map = {
        "Forward pass":      _C["gpu"],
        "Backward pass":     _C["cpu"],
        "Optimisation step": _C["ram"],
    }
    colors = [style_map.get(k, "grey") for k in totals.index]
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(range(len(totals)), totals.values, color=colors, alpha=0.88, width=0.5)
    _annotate_bars(ax, bars, "{:.2e}")
    ax.set_xticks(range(len(totals)))
    ax.set_xticklabels(totals.index, rotation=10, ha="right")
    ax.set_ylabel(f"{ecol.replace('_', ' ').title()} (kWh)")
    ax.set_title("Total Energy by Training Substep (Forward / Backward / Optimizer)")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "substep_breakdown.png"))


def plot_substep_per_step(substep_df: pd.DataFrame, out: str) -> str:
    """Stacked area -- per-step energy breakdown by substep type vs. step."""
    if substep_df.empty or "substep" not in substep_df.columns or "step" not in substep_df.columns:
        return ""
    ecol = next((c for c in ("energy_consumed", "gpu_energy") if c in substep_df.columns), None)
    if ecol is None:
        return ""
    pivot = (substep_df
             .pivot_table(index="step", columns="substep", values=ecol, aggfunc="sum")
             .fillna(0))
    if pivot.empty:
        return ""
    style_map = {
        "Forward pass":      _C["gpu"],
        "Backward pass":     _C["cpu"],
        "Optimisation step": _C["ram"],
    }
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    x = pivot.index.values
    bottoms = np.zeros(len(x))
    for col_name in pivot.columns:
        vals = pivot[col_name].values
        color = style_map.get(str(col_name), "grey")
        ax.fill_between(x, bottoms, bottoms + vals, color=color, alpha=0.70, label=col_name)
        bottoms += vals
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Per-Step Energy by Substep Type")
    ax.legend(fontsize=9, loc="upper right")
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "substep_per_step.png"))


# ===========================================================================
# GROUP B: Energy metrics
# ===========================================================================

def plot_energy_stacked(cc: pd.DataFrame, out: str) -> str:
    """Bar -- GPU energy (kWh) per run."""
    if "gpu_energy" not in cc.columns:
        return ""
    n = len(cc)
    x = np.arange(n)
    vals = cc["gpu_energy"].fillna(0).values
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, vals, color=_C["gpu"], alpha=0.88, width=0.5)
    _annotate_bars(ax, bars, "{:.5f}")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n), rotation=0 if n <= 6 else 30, ha="right")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("GPU Energy (kWh) per Run")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "breakdown_stacked.png"))


def plot_energy_pie(cc: pd.DataFrame, out: str) -> str:
    """Pie chart removed — only GPU energy is tracked, so a single-slice pie is meaningless."""
    return ""


def plot_energy_per_run(cc: pd.DataFrame, out: str) -> str:
    """Bar -- GPU energy (kWh) per run."""
    if "gpu_energy" not in cc.columns:
        return ""
    n    = len(cc)
    x    = np.arange(n)
    vals = cc["gpu_energy"].fillna(0).values
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, vals, color=_C["gpu"], alpha=0.88, width=0.5)
    _annotate_bars(ax, bars, "{:.5f}")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n))
    ax.set_ylabel("GPU Energy (kWh)")
    ax.set_title("GPU Energy per Run")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "total_kwh.png"))


def plot_energy_efficiency(cc: pd.DataFrame, out: str) -> str:
    """Bar -- GPU energy per sample (kWh/sample) or per second if sample count unavailable."""
    if "gpu_energy" not in cc.columns:
        return ""
    energy = cc["gpu_energy"].fillna(0).values
    sample_col = next((c for c in ("n_samples", "num_samples", "total_samples")
                       if c in cc.columns), None)
    if sample_col:
        ns = cc[sample_col].fillna(0).values
        ylabel = "GPU kWh / sample"
        title  = "GPU Energy Efficiency (kWh per Sample) per Run"
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(ns > 0, energy / ns, 0.0)
    elif "duration" in cc.columns:
        dur = cc["duration"].fillna(0).values
        ylabel = "GPU kWh / s"
        title  = "Mean GPU Power Proxy (kWh/s) per Run"
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(dur > 0, energy / (dur / 3_600.0), 0.0)
    else:
        return ""
    n = len(cc)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, eff, color=_C["gpu"], alpha=0.88, width=0.5)
    _annotate_bars(ax, bars, "{:.3e}")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "efficiency.png"))


def plot_live_power(live: pd.DataFrame, out: str) -> str:
    """Line -- live GPU power (W) vs. elapsed time (s)."""
    if live.empty or "gpu_power" not in live.columns:
        return ""
    x_col = "elapsed_s" if "elapsed_s" in live.columns else None
    if x_col is None:
        live = live.copy()
        live["elapsed_s"] = range(len(live))
        x_col = "elapsed_s"
    x = live[x_col]
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, live["gpu_power"], color=_C["gpu"], linewidth=1.6, label="GPU Power (W)")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Live GPU Power Draw vs. Time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "live_power.png"))


def plot_live_components(live: pd.DataFrame, out: str) -> str:
    """Removed — duplicates live_power now that only GPU is tracked."""
    return ""


def plot_cumulative_energy(live: pd.DataFrame, out: str) -> str:
    """Line -- cumulative GPU kWh vs. elapsed time (trapezoidal integration)."""
    if live.empty or "elapsed_s" not in live.columns or "gpu_power" not in live.columns:
        return ""
    x        = live["elapsed_s"].values
    gpu_w    = live["gpu_power"].fillna(0).values
    dt       = np.diff(x, prepend=x[0])
    cum_kwh  = np.cumsum(gpu_w * dt / 3_600_000.0)
    fig, ax  = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, cum_kwh, color=_C["gpu"], linewidth=2.0)
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Cumulative GPU Energy (kWh)")
    ax.set_title("Cumulative GPU Energy vs. Time")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "energy", "cumulative_energy.png"))


# ===========================================================================
# GROUP C: Emissions
# ===========================================================================

def plot_co2_per_run(cc: pd.DataFrame, out: str) -> str:
    """Bar -- kg CO2-eq per run (GPU energy x 0.027 kg/kWh Quebec grid)."""
    if "gpu_energy" not in cc.columns:
        return ""
    n    = len(cc)
    x    = np.arange(n)
    vals = cc["gpu_energy"].fillna(0).values * 0.027
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, vals, color=_C["co2"], alpha=0.85, width=0.5)
    _annotate_bars(ax, bars, "{:.2e}")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n))
    ax.set_ylabel("CO2-equivalent (kg)")
    ax.set_title("GPU-Only CO2-Equivalent Emissions per Run")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "emissions", "co2_per_run.png"))


def plot_emissions_rate(cc: pd.DataFrame, out: str) -> str:
    """Bar -- kg CO2-eq/s per run (= gpu_energy * 0.027 / duration)."""
    if "gpu_energy" not in cc.columns or "duration" not in cc.columns:
        return ""
    co2 = cc["gpu_energy"].fillna(0).values * 0.027
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(
            cc["duration"].fillna(0).values > 0,
            co2 / cc["duration"].fillna(0).values,
            0.0,
        )
    n = len(cc)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, rate, color=_C["co2"], alpha=0.70, width=0.5)
    _annotate_bars(ax, bars, "{:.2e}")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n))
    ax.set_ylabel("Emissions Rate (kg CO2eq / s)")
    ax.set_title("GPU-Only Emissions Rate per Run")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "emissions", "emissions_rate.png"))


def plot_duration(cc: pd.DataFrame, out: str) -> str:
    """Bar -- wall-clock duration (minutes) per run."""
    if "duration" not in cc.columns:
        return ""
    n    = len(cc)
    x    = np.arange(n)
    vals = cc["duration"].fillna(0).values / 60.0
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, vals, color=_C["duration"], alpha=0.88, width=0.5)
    _annotate_bars(ax, bars, "{:.1f} m")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n))
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Training Wall-Clock Duration per Run")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "emissions", "duration.png"))


# ===========================================================================
# GROUP D: Hardware / system stats
# ===========================================================================

def plot_gpu_memory(log: pd.DataFrame, out: str) -> str:
    """Line -- GPU memory allocated (MiB) vs. step."""
    col = next((c for c in ("gpu_memory_mb", "gpu_mem_mb", "gpu_memory_mib",
                             "gpu_mem_allocated_mb", "gpu_memory") if c in log.columns), None)
    if log.empty or col is None:
        return ""
    x = _get_x(log)
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, log[col], color=_C["mem"], linewidth=1.8)
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel("GPU Memory (MiB)")
    ax.set_title("GPU Memory Allocated vs. Step")
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "hardware", "gpu_memory.png"))


def plot_cpu_utilization(log: pd.DataFrame, out: str) -> str:
    """Line -- CPU utilization (%) vs. step."""
    col = next((c for c in ("cpu_percent", "cpu_util", "cpu_util_pct")
                if c in log.columns), None)
    if log.empty or col is None:
        return ""
    x = _get_x(log)
    y = log[col]
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    ax.plot(x, y, color=_C["cpu_pct"], alpha=0.35, linewidth=0.8, label="Raw")
    ax.plot(x, _rolling(y, 10), color=_C["cpu_pct"], linewidth=2.0, label="Smoothed")
    ax.set_ylim(0, 105)
    ax.set_xlabel(x.name.capitalize())
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_title("CPU Utilization vs. Step")
    ax.legend(fontsize=9)
    _fix_xaxis(ax)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "hardware", "cpu_utilization.png"))


def plot_mean_power(cc: pd.DataFrame, out: str) -> str:
    """Bar -- mean GPU power (W) per run."""
    if "gpu_power" in cc.columns:
        vals = cc["gpu_power"].fillna(0).values
    elif "gpu_energy" in cc.columns and "duration" in cc.columns:
        dur = cc["duration"].fillna(0).values
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.where(dur > 0, cc["gpu_energy"].fillna(0).values * 3_600_000 / dur, 0.0)
    else:
        print("  [warn] No GPU power data available -- skipping mean_power plot.")
        return ""

    n_runs = len(cc)
    x      = np.arange(n_runs)
    fig, ax = plt.subplots(figsize=_FIG_SZ)
    bars = ax.bar(x, vals, color=_C["gpu"], alpha=0.88, width=0.5)
    _annotate_bars(ax, bars, "{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(_run_labels(n_runs))
    ax.set_ylabel("Power (W)")
    ax.set_title("Mean GPU Power per Run")
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "hardware", "mean_power.png"))


def plot_system_info(cc: pd.DataFrame, out: str) -> str:
    """Text-card figure -- hardware / environment metadata from CodeCarbon."""
    row = cc.iloc[0]

    def _get(col: str, default: str = "--") -> str:
        return str(row[col]) if col in cc.columns and pd.notna(row[col]) else default

    lines = [
        ("GPU Model",       _get("gpu_model")),
        ("GPU Count",       _get("gpu_count")),
        ("CPU Model",       _get("cpu_model")),
        ("CPU Count",       _get("cpu_count")),
        ("RAM Total (GB)",  _get("ram_total_size")),
        ("OS",              _get("os")),
        ("Python",          _get("python_version")),
        ("CodeCarbon",      _get("codecarbon_version")),
        ("Region",          _get("region")),
        ("Country",         _get("country_name")),
        ("Cloud Provider",  _get("cloud_provider")),
        ("Tracking Mode",   _get("tracking_mode")),
        ("PUE",             _get("pue")),
    ]
    fig, ax = plt.subplots(figsize=(7.5, len(lines) * 0.42 + 0.6))
    ax.axis("off")
    col_x = [0.02, 0.45]
    for i, (key, val) in enumerate(lines):
        y = 1.0 - (i + 1) / (len(lines) + 1)
        ax.text(col_x[0], y, key + ":", fontsize=9.5, ha="left", va="center",
                fontweight="bold", transform=ax.transAxes)
        ax.text(col_x[1], y, val, fontsize=9.5, ha="left", va="center",
                transform=ax.transAxes, color="#333333")
    ax.set_title("System / Hardware Information (from CodeCarbon)", fontsize=10, pad=8)
    fig.tight_layout()
    return _save(fig, os.path.join(out, "figures", "hardware", "system_info.png"))


# ===========================================================================
# Summary statistics
# ===========================================================================

def compute_summary(cc: pd.DataFrame) -> dict:
    def _tot(col):
        return float(cc[col].fillna(0).sum())  if col in cc.columns else None
    def _avg(col):
        return float(cc[col].fillna(0).mean()) if col in cc.columns else None
    def _str(col):
        return str(cc[col].iloc[0]) if col in cc.columns else "--"
    total_gpu_kwh = _tot("gpu_energy")
    total_emissions = total_gpu_kwh * 0.027 if total_gpu_kwh is not None else None
    return {
        "n_runs":              len(cc),
        "total_energy_kwh":    total_gpu_kwh,
        "total_emissions_kg":  total_emissions,
        "total_duration_s":    _tot("duration"),
        "mean_gpu_energy_kwh": _avg("gpu_energy"),
        "mean_gpu_power_w":    _avg("gpu_power"),
        "gpu_model":           _str("gpu_model"),
        "cpu_model":           _str("cpu_model"),
        "region":              _str("region"),
        "country":             _str("country_name"),
        "pue":                 _str("pue"),
    }


# ===========================================================================
# Markdown report
# ===========================================================================

def _fmt_kwh(v) -> str:
    return f"{v:.6f} kWh" if v is not None else "_n/a_"

def _fmt_kg(v) -> str:
    return f"{v:.4e} kg CO2eq" if v is not None else "_n/a_"

def _fmt_w(v) -> str:
    return f"{v:.2f} W" if v is not None else "_n/a_"

def _fmt_dur(v) -> str:
    if v is None:
        return "_n/a_"
    h, r = divmod(int(v), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

def _img(key: str, paths: dict, report: str) -> str:
    p = paths.get(key, "")
    if not p:
        return f"*`{key}` not generated (data unavailable).*"
    rel = os.path.relpath(p, os.path.dirname(report)).replace("\\", "/")
    return f"![{key}]({rel})"


def _training_sample_table(log: pd.DataFrame) -> str:
    if log.empty:
        return ""
    cols = [c for c in ("epoch", "step", "loss", "lr", "grad_norm",
                         "throughput", "gpu_memory_mb") if c in log.columns]
    if not cols:
        return ""
    sample = log[cols].head(10)
    try:
        return "\n**First 10 rows of training log:**\n\n" + sample.to_markdown(index=False) + "\n"
    except ImportError:
        lines = [" | ".join(str(v) for v in row)
                 for row in [cols] + sample.values.tolist()]
        return "\n**First 10 rows:**\n\n" + "\n".join(lines) + "\n"


def generate_report(summary: dict, paths: dict,
                    log: pd.DataFrame, out_path: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    I   = lambda k: _img(k, paths, out_path)  # noqa: E731

    def _avail(k: str) -> str:
        return "yes" if paths.get(k) else "no data"

    report = f"""\
# V-JEPA2 - Phase 7 Preliminary Results Report

> **Generated:** {now}
> **Project:** COMP 597 - V-JEPA2 (ViT-Huge, 632 M params)
> **Platform:** Mila / mimi cluster
> **Phase covered:** Phase 7 - CodeCarbon Energy Measurements
>
> **Note:** CPU and RAM energy/power readings from CodeCarbon are discarded
> (TDP mode for CPU, fixed heuristic for RAM). All energy and power figures
> below refer exclusively to **GPU** measurements via NVML.

---

## 1. Overview

This report captures **all metrics collected through Phase 7**.
Fine-grained per-phase analysis (Data Load, Forward Target, Backward, ...) begins in Phase 8.

### Metric / figure availability

| Category | Metric | Status |
|---|---|---|
| Training | Loss vs. step | {_avail("loss")} |
| Training | Loss (zoomed) | {_avail("loss_zoomed")} |
| Training | Learning rate schedule | {_avail("learning_rate")} |
| Training | Gradient norm | {_avail("grad_norm")} |
| Training | Throughput (clips/s) | {_avail("throughput")} |
| Training | Step wall-clock time | {_avail("step_time")} |
| Energy | GPU energy per run | {_avail("breakdown_stacked")} |
| Energy | Total GPU kWh per run | {_avail("total_kwh")} |
| Energy | GPU energy efficiency | {_avail("efficiency")} |
| Energy | Live GPU power | {_avail("live_power")} |
| Energy | Cumulative GPU energy | {_avail("cumulative_energy")} |
| Emissions | CO2eq per run (GPU-only) | {_avail("co2_per_run")} |
| Emissions | Emissions rate (GPU-only) | {_avail("emissions_rate")} |
| Emissions | Duration | {_avail("duration")} |
| Hardware | GPU memory vs. step | {_avail("gpu_memory")} |
| Hardware | Mean GPU power | {_avail("mean_power")} |
| Hardware | System info card | {_avail("system_info")} |

---

## 2. Summary Statistics

| Metric | Value |
|---|---|
| Number of runs | {summary["n_runs"]} |
| Total GPU energy | {_fmt_kwh(summary["total_energy_kwh"])} |
| GPU-only CO2-equivalent | {_fmt_kg(summary["total_emissions_kg"])} |
| Total wall-clock time | {_fmt_dur(summary["total_duration_s"])} |
| Mean GPU energy / run | {_fmt_kwh(summary["mean_gpu_energy_kwh"])} |
| Mean GPU power | {_fmt_w(summary["mean_gpu_power_w"])} |
| GPU | {summary["gpu_model"]} |
| CPU | {summary["cpu_model"]} |
| Region | {summary["region"]}, {summary["country"]} |
| PUE | {summary["pue"]} |

---

## 3. Training Stats

### 3.1 Training Loss

{I("loss")}

{I("loss_zoomed")}

{_training_sample_table(log)}

A monotonically falling L1 loss confirms that V-JEPA2 is optimising the masked
video-prediction objective correctly. The zoomed view crops the y-axis to the
p2-p98 range to reveal late-training dynamics without early-epoch spikes.

### 3.2 Learning Rate Schedule

{I("learning_rate")}

V-JEPA2 uses a cosine-decay schedule with a linear warm-up. The LR curve should
rise steeply for the first ~10% of training then smoothly decay to near zero.

### 3.3 Gradient Norm

{I("grad_norm")}

Gradient clipping (`max_grad_norm`) is applied every step. Large early spikes
are expected; a shrinking trend as loss decreases indicates stable optimisation.

### 3.4 Throughput

{I("throughput")}

Clips per second is the key efficiency baseline for Phase 9 comparisons across
different batch sizes and model dtypes.

### 3.5 Step Wall-Clock Time

{I("step_time")}

Outlier steps with anomalously high step times correspond to checkpointing,
first-step JIT compilation, or data-loading stalls - important context for the
per-phase analysis in Phase 8.

---

## 4. Energy Stats

### 4.1 GPU Energy per Run

{I("breakdown_stacked")}

Only GPU energy (measured via NVML) is reported. CodeCarbon's CPU energy (TDP
model) and RAM energy (fixed heuristic) are unreliable on this cluster and have
been discarded.

### 4.2 Total GPU Energy per Run

{I("total_kwh")}

### 4.3 GPU Energy Efficiency

{I("efficiency")}

Energy per sample normalises throughput by power budget and is the most
comparable metric across different hardware configurations.

### 4.4 Live GPU Power Trace

{I("live_power")}

The live power trace reveals sub-run dynamics: peaks during forward/backward
passes, idle periods during data loading, and GPU idle power between epochs.
Requires `LivePowerOutput` to be registered in `VJepa2PhaseStats`.

### 4.5 Cumulative GPU Energy

{I("cumulative_energy")}

The slope of this curve equals instantaneous GPU power. Steeper sections are
GPU-intensive; flat sections correspond to data loading / checkpointing.

---

## 5. Emissions

### 5.1 CO2-Equivalent per Run

{I("co2_per_run")}

CO2 is computed as GPU energy (kWh) x 0.027 kg CO2eq/kWh (Quebec grid).

### 5.2 Emissions Rate

{I("emissions_rate")}

### 5.3 Wall-Clock Duration

{I("duration")}

Quebec's hydroelectric grid has a carbon intensity of ~0.027 kg CO2eq / kWh -
about 37x lower than a coal-heavy grid. Always report absolute CO2 values
alongside the region to enable cross-site comparisons.

---

## 6. Hardware Stats

### 6.1 System Information

{I("system_info")}

### 6.2 Mean GPU Power

{I("mean_power")}

Mean GPU power = GPU energy / duration. Useful for comparing hardware
configurations without needing the full live power trace.

### 6.3 GPU Memory vs. Step

{I("gpu_memory")}

Peak GPU memory constrains the maximum feasible batch size. A rising trend
across steps may indicate a memory leak (e.g., tensors accumulated in a list).

---

## 7. Preliminary Observations

1. **GPU-only energy tracking** - CPU/RAM readings discarded (TDP / heuristic);
   all energy figures reflect NVML GPU measurements only.
2. **Baseline established** - duration and throughput anchor Phase 9 comparisons.
3. **Low carbon impact** - Quebec's grid keeps per-run CO2eq negligible; cumulative
   impact matters for large Phase 9 sweeps.

### Pre-Phase-8 checklist

- [ ] `emissions*.csv` exists with non-zero `gpu_energy`.
- [ ] Training loss is strictly decreasing (not NaN or flat).
- [ ] GPU peak memory < 32 GB (RTX 5000 Ada, 32 GB).
- [ ] Duration > 60 s for a 2-epoch FakeVideo run.
- [ ] No anomalous step-time spikes beyond the first 2-3 JIT-compilation steps.

---

## 8. Output File Inventory

```
codecarbonlogs/
├── emissions*.csv             <- run-level energy data (required)
└── live_power_run*.csv        <- per-measurement live power (optional)

docs/
├── phase7_preliminary_report.md
└── figures/
    ├── training/
    │   ├── loss.png
    │   ├── loss_zoomed.png
    │   ├── learning_rate.png  (if lr in log)
    │   ├── grad_norm.png      (if grad_norm in log)
    │   ├── throughput.png     (if throughput in log)
    │   └── step_time.png      (if step_time_ms in log)
    ├── energy/
    │   ├── breakdown_stacked.png
    │   ├── total_kwh.png
    │   ├── efficiency.png
    │   ├── live_power.png        (if live CSV present)
    │   └── cumulative_energy.png (if live CSV present)
    ├── emissions/
    │   ├── co2_per_run.png
    │   ├── emissions_rate.png
    │   └── duration.png
    └── hardware/
        ├── system_info.png
        ├── mean_power.png
        └── gpu_memory.png        (if gpu_memory_mb in log)
```

---

*Auto-generated by `scripts/generate_phase7_report.py`.*
"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"  [md]   {out_path}")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Generate Phase 7 preliminary results graphs and Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pa.add_argument("--codecarbon_dir", required=True, metavar="DIR",
                    help="Directory containing CodeCarbon emissions*.csv files.")
    pa.add_argument("--live_dir", default=None, metavar="DIR",
                    help="Directory with live_power_run*.csv (defaults to --codecarbon_dir).")
    pa.add_argument("--log_dir", default=None, metavar="DIR",
                    help="Directory with training_log.csv / metrics.csv.")
    pa.add_argument("--out_dir", default="docs", metavar="DIR",
                    help="Root output directory (figures/ placed inside here).")
    pa.add_argument("--report_name", default="phase7_preliminary_report.md",
                    help="Filename of the Markdown report.")
    args = pa.parse_args()

    live_dir = args.live_dir or args.codecarbon_dir

    print("\n" + "=" * 55)
    print("  V-JEPA2  Phase 7  Preliminary Report Generator")
    print("=" * 55)
    print(f"  CodeCarbon dir : {args.codecarbon_dir}")
    print(f"  Live power dir : {live_dir}")
    print(f"  Training log   : {args.log_dir or '(not provided)'}")
    print(f"  Output dir     : {args.out_dir}")
    print()

    # Load
    try:
        cc   = load_codecarbon(args.codecarbon_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    live      = load_live_power(live_dir)
    log       = load_training_log(args.log_dir)
    step_cc   = load_step_energy(args.codecarbon_dir)
    substep_cc = load_substep_energy(args.codecarbon_dir)
    out  = args.out_dir

    # Figures
    print("\n-- Training ----------------------------------------")
    figs: dict = {}
    figs["loss"]           = plot_loss(log, out)
    figs["loss_zoomed"]    = plot_loss_zoom(log, out)
    figs["learning_rate"]  = plot_learning_rate(log, out)
    figs["grad_norm"]      = plot_grad_norm(log, out)
    figs["throughput"]     = plot_throughput(log, out)
    figs["step_time"]      = plot_step_time(log, out)

    print("\n-- Energy ------------------------------------------")
    figs["step_energy"]        = plot_step_energy(step_cc, out)
    figs["substep_breakdown"]  = plot_substep_breakdown(substep_cc, out)
    figs["substep_per_step"]   = plot_substep_per_step(substep_cc, out)
    figs["breakdown_stacked"]  = plot_energy_stacked(cc, out)
    figs["pie"]                = plot_energy_pie(cc, out)
    figs["total_kwh"]          = plot_energy_per_run(cc, out)
    figs["efficiency"]         = plot_energy_efficiency(cc, out)
    figs["live_power"]         = plot_live_power(live, out)
    figs["live_components"]    = plot_live_components(live, out)
    figs["cumulative_energy"]  = plot_cumulative_energy(live, out)

    print("\n-- Emissions ---------------------------------------")
    figs["co2_per_run"]     = plot_co2_per_run(cc, out)
    figs["emissions_rate"]  = plot_emissions_rate(cc, out)
    figs["duration"]        = plot_duration(cc, out)

    print("\n-- Hardware ----------------------------------------")
    figs["gpu_memory"]      = plot_gpu_memory(log, out)
    figs["cpu_utilization"] = plot_cpu_utilization(log, out)
    figs["mean_power"]      = plot_mean_power(cc, out)
    figs["system_info"]     = plot_system_info(cc, out)

    # Report
    print("\n-- Markdown report ---------------------------------")
    summary     = compute_summary(cc)
    report_path = os.path.join(out, args.report_name)
    generate_report(summary, figs, log, report_path)

    n_gen  = sum(1 for v in figs.values() if v)
    n_skip = sum(1 for v in figs.values() if not v)
    print(f"\n-- Done -- {n_gen} figures generated, {n_skip} skipped (no data)")
    print(f"  Figures -> {out}/figures/")
    print(f"  Report  -> {report_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

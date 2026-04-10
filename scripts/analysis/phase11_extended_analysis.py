#!/usr/bin/env python3
"""Generate report-ready plots for the validation sweep and worker study."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_ROOT = REPO_DIR / "analysis_inputs" / "vjepa2"
DEFAULT_VALIDATION_ROOT = REPO_DIR / "analysis_inputs" / "vjepa2_validation_extra"
DEFAULT_WORKER_ROOTS = [
    REPO_DIR / "analysis_inputs" / "vjepa2_workers_bs2_nw0",
    REPO_DIR / "analysis_inputs" / "vjepa2_workers_bs2_nw1",
    REPO_DIR / "analysis_inputs" / "vjepa2_workers_bs2_nw2",
    REPO_DIR / "analysis_inputs" / "vjepa2_workers_bs2_nw4",
]
DEFAULT_OUTPUT_ROOT = REPO_DIR / "analysis_outputs" / "vjepa2_extended"

FIG_DPI = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", default=str(DEFAULT_MAIN_ROOT))
    parser.add_argument("--validation-root", default=str(DEFAULT_VALIDATION_ROOT))
    parser.add_argument(
        "--worker-roots",
        nargs="+",
        default=[str(path) for path in DEFAULT_WORKER_ROOTS],
        help="Roots for per-worker-count feasibility runs.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    csv_dir = output_root / "csv"
    fig_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"csv": csv_dir, "figures": fig_dir}


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_wall_clock_rows(root: Path, dataset_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for meta_path in sorted(root.glob("mode*/bs*/run*/run_metadata.json")):
        meta = read_json(meta_path)
        rows.append(
            {
                "dataset": dataset_label,
                "mode": meta["mode_directory"],
                "batch_size": int(meta["batch_size"]),
                "run_num": int(meta["run_num"]),
                "wall_clock_s": float(meta.get("wall_clock_s_external", 0.0)),
            }
        )
    return rows


def collect_finegrained_rows(root: Path, dataset_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in sorted(root.glob("mode3_finegrained/bs*/run*")):
        batch_size = int(run_dir.parent.name.replace("bs", ""))
        run_num = int(run_dir.name.replace("run", ""))
        run_summary_path = next(run_dir.glob("run_summary_run*.csv"), None)
        if run_summary_path is not None and run_summary_path.exists():
            run_summary = pd.read_csv(run_summary_path)
            if not run_summary.empty:
                row = run_summary.iloc[0]
                rows.append(
                    {
                        "dataset": dataset_label,
                        "batch_size": batch_size,
                        "run_num": run_num,
                        "gpu_energy_kwh": float(row["codecarbon_gpu_energy_kwh"]),
                        "steps_observed": int(row["steps_observed"]),
                    }
                )

        timeline_path = next(run_dir.glob("system_timeline_run*.csv"), None)
        if timeline_path is not None and timeline_path.exists():
            timeline = pd.read_csv(timeline_path)
            if not timeline.empty:
                rows.append(
                    {
                        "dataset": dataset_label,
                        "batch_size": batch_size,
                        "run_num": run_num,
                        "mean_gpu_util_pct": float(timeline["gpu_utilization_pct"].mean()),
                    }
                )
    return rows


def build_validation_summary(main_root: Path, validation_root: Path) -> pd.DataFrame:
    wall_df = pd.DataFrame(
        collect_wall_clock_rows(main_root, "main")
        + collect_wall_clock_rows(validation_root, "validation")
    )
    fine_df = pd.DataFrame(
        collect_finegrained_rows(main_root, "main")
        + collect_finegrained_rows(validation_root, "validation")
    )

    summary_rows: List[Dict[str, object]] = []
    for dataset_label in ["main", "validation"]:
        for batch_size in [1, 2]:
            coarse = wall_df[
                (wall_df["dataset"] == dataset_label)
                & (wall_df["mode"] == "mode2_codecarbon")
                & (wall_df["batch_size"] == batch_size)
            ]
            fine_wall = wall_df[
                (wall_df["dataset"] == dataset_label)
                & (wall_df["mode"] == "mode3_finegrained")
                & (wall_df["batch_size"] == batch_size)
            ]
            fine = fine_df[
                (fine_df["dataset"] == dataset_label)
                & (fine_df["batch_size"] == batch_size)
            ]
            summary_rows.append(
                {
                    "dataset": dataset_label,
                    "batch_size": batch_size,
                    "coarse_wall_clock_mean_s": coarse["wall_clock_s"].mean(),
                    "coarse_wall_clock_std_s": coarse["wall_clock_s"].std(ddof=1),
                    "fine_wall_clock_mean_s": fine_wall["wall_clock_s"].mean(),
                    "fine_wall_clock_std_s": fine_wall["wall_clock_s"].std(ddof=1),
                    "fine_gpu_energy_mean_kwh": fine["gpu_energy_kwh"].dropna().mean(),
                    "fine_gpu_energy_std_kwh": fine["gpu_energy_kwh"].dropna().std(ddof=1),
                    "fine_gpu_util_mean_pct": fine["mean_gpu_util_pct"].dropna().mean(),
                    "fine_gpu_util_std_pct": fine["mean_gpu_util_pct"].dropna().std(ddof=1),
                }
            )
    return pd.DataFrame(summary_rows)


def plot_validation_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    metrics = [
        ("coarse_wall_clock_mean_s", "coarse_wall_clock_std_s", "CodeCarbon Wall-Clock (s)"),
        ("fine_wall_clock_mean_s", "fine_wall_clock_std_s", "Fine-Grained Wall-Clock (s)"),
        ("fine_gpu_energy_mean_kwh", "fine_gpu_energy_std_kwh", "Fine-Grained GPU Energy (kWh)"),
        ("fine_gpu_util_mean_pct", "fine_gpu_util_std_pct", "Mean GPU Utilization (%)"),
    ]
    colors = {"main": "#355070", "validation": "#b56576"}
    batch_labels = ["bs1", "bs2"]
    x = np.arange(len(batch_labels))
    width = 0.34

    for ax, (mean_col, std_col, title) in zip(axes.flatten(), metrics):
        for offset, dataset_label in [(-width / 2, "main"), (width / 2, "validation")]:
            data = summary[summary["dataset"] == dataset_label].sort_values("batch_size")
            ax.bar(
                x + offset,
                data[mean_col],
                width=width,
                yerr=data[std_col].fillna(0.0),
                capsize=4,
                color=colors[dataset_label],
                label=dataset_label.capitalize(),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(batch_labels)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0, 0].legend(frameon=False, ncol=2, loc="upper center")
    fig.suptitle("Validation Sweep Consistency Against the Main Report Matrix", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_worker_count(root: Path) -> int:
    match = re.search(r"nw(\d+)$", root.name)
    if not match:
        raise ValueError(f"Unable to infer worker count from {root}")
    return int(match.group(1))


def _failure_reason_from_log(process_log: str) -> str:
    pickling_match = re.findall(r"PicklingError: (.*)", process_log)
    if pickling_match:
        return str(pickling_match[-1]).strip()
    mod = re.findall(r"ModuleNotFoundError: (.*)", process_log)
    if mod:
        return str(mod[-1]).strip()[:200]
    attr = re.findall(r"AttributeError: (.*)", process_log)
    if attr:
        return str(attr[-1]).strip()[:200]
    if "DataLoader worker" in process_log and "exited unexpectedly" in process_log:
        return "DataLoader worker exited unexpectedly"
    if "QOSMaxSubmitJobPerUserLimit" in process_log:
        return "Slurm submit limit"
    return ""


def build_worker_per_run_rows(worker_roots: List[Path]) -> pd.DataFrame:
    """One row per (num_workers, run) under mode3_finegrained/bs2/run*."""
    rows: List[Dict[str, object]] = []
    batch_size = 2
    for root in worker_roots:
        if not root.exists():
            continue
        worker_count = parse_worker_count(root)
        bs_dir = root / "mode3_finegrained" / "bs2"
        if not bs_dir.exists():
            continue
        for run_dir in sorted(bs_dir.glob("run*")):
            meta_path = run_dir / "run_metadata.json"
            if not meta_path.exists():
                continue
            meta = read_json(meta_path)
            process_log = ""
            pl = run_dir / "process.log"
            if pl.exists():
                process_log = pl.read_text(encoding="utf-8", errors="replace")

            rs_path = next(run_dir.glob("run_summary_run*.csv"), None)
            run_summary = pd.read_csv(rs_path) if rs_path is not None and rs_path.exists() else pd.DataFrame()

            failure_reason = _failure_reason_from_log(process_log)

            if not run_summary.empty:
                r0 = run_summary.iloc[0]
                steps_observed = int(r0["steps_observed"])
                mean_total_step_ms = float(r0["mean_total_step_ms"])
                gpu_energy_kwh = float(r0["codecarbon_gpu_energy_kwh"]) if pd.notna(r0.get("codecarbon_gpu_energy_kwh")) else np.nan
                status = "success" if steps_observed > 0 else "failed"
            else:
                steps_observed = 0
                mean_total_step_ms = np.nan
                gpu_energy_kwh = np.nan
                status = "failed"

            wall_s = float(meta.get("wall_clock_s_external", 0.0) or 0.0)
            throughput = (
                (steps_observed * batch_size) / wall_s if wall_s > 0 and steps_observed > 0 else np.nan
            )

            tl_path = next(run_dir.glob("system_timeline_run*.csv"), None)
            mean_gpu_util = np.nan
            if tl_path is not None and tl_path.exists():
                tl = pd.read_csv(tl_path)
                if not tl.empty and "gpu_utilization_pct" in tl.columns:
                    mean_gpu_util = float(pd.to_numeric(tl["gpu_utilization_pct"], errors="coerce").mean())

            run_num = int(run_dir.name.replace("run", "")) if run_dir.name.startswith("run") else 0

            rows.append(
                {
                    "num_workers": worker_count,
                    "run_num": run_num,
                    "status": status,
                    "wall_clock_s_external": wall_s,
                    "steps_observed": steps_observed,
                    "mean_total_step_ms": mean_total_step_ms,
                    "throughput_samples_per_s": throughput,
                    "gpu_energy_kwh": gpu_energy_kwh,
                    "mean_gpu_util_pct": mean_gpu_util,
                    "failure_reason": failure_reason if status == "failed" else "",
                }
            )
    columns = [
        "num_workers",
        "run_num",
        "status",
        "wall_clock_s_external",
        "steps_observed",
        "mean_total_step_ms",
        "throughput_samples_per_s",
        "gpu_energy_kwh",
        "mean_gpu_util_pct",
        "failure_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["num_workers", "run_num"])


def build_worker_summary(worker_roots: List[Path]) -> pd.DataFrame:
    """One row per num_workers (aggregated over runs) for legacy feasibility-style table."""
    per_run = build_worker_per_run_rows(worker_roots)
    if per_run.empty:
        return pd.DataFrame(
            columns=[
                "num_workers",
                "status",
                "wall_clock_s_external",
                "global_steps",
                "steps_observed",
                "mean_total_step_ms",
                "failure_reason",
            ]
        )

    agg_rows: List[Dict[str, object]] = []
    for nw, g in per_run.groupby("num_workers"):
        succ = g[g["status"] == "success"]
        if not succ.empty:
            status = "success"
            steps_observed = float(succ["steps_observed"].mean())
            mean_total_step_ms = float(succ["mean_total_step_ms"].mean())
            wall_clock_s_external = float(succ["wall_clock_s_external"].mean())
            failure_reason = ""
        else:
            status = "failed"
            steps_observed = 0.0
            mean_total_step_ms = np.nan
            wall_clock_s_external = float(g["wall_clock_s_external"].mean()) if len(g) else 0.0
            fr = g.loc[g["failure_reason"].astype(str).str.len() > 0, "failure_reason"]
            failure_reason = str(fr.iloc[0]) if len(fr) else ""

        agg_rows.append(
            {
                "num_workers": int(nw),
                "status": status,
                "wall_clock_s_external": wall_clock_s_external,
                "global_steps": int(steps_observed) if status == "success" else 0,
                "steps_observed": int(round(steps_observed)) if status == "success" else 0,
                "mean_total_step_ms": mean_total_step_ms,
                "failure_reason": failure_reason,
            }
        )
    return pd.DataFrame(agg_rows).sort_values("num_workers")


def plot_worker_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    summary = summary.reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), constrained_layout=True)
    colors = {"success": "#2a9d8f", "failed": "#d62828"}
    x = np.arange(len(summary))
    bar_colors = [colors[status] for status in summary["status"]]

    wmax = float(summary["wall_clock_s_external"].max() or 1.0)
    smax = float(summary["steps_observed"].max() or 1.0)

    axes[0].bar(x, summary["wall_clock_s_external"], color=bar_colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in summary["num_workers"]])
    axes[0].set_ylabel("External Wall-Clock (s)")
    axes[0].set_title("Worker-count study at bs=2 (fine-grained; means over runs)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylim(0, max(wmax * 1.18, 10))

    axes[1].bar(x, summary["steps_observed"], color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(v) for v in summary["num_workers"]])
    axes[1].set_xlabel("num_workers")
    axes[1].set_ylabel("Observed Training Steps (mean)")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].set_ylim(0, max(smax * 1.18, 10))

    for i, row in summary.iterrows():
        label = row["status"]
        if row["status"] != "success" and row.get("failure_reason"):
            label = "failed"
        axes[0].text(
            x[i],
            row["wall_clock_s_external"] + max(wmax * 0.02, 0.8),
            str(label),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        if row["status"] != "success" and row.get("failure_reason"):
            axes[1].text(
                x[i],
                0.8,
                "see log",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#5c1d1d",
            )
        elif row["status"] == "success":
            axes[1].text(
                x[i],
                row["steps_observed"] + max(smax * 0.02, 2.0),
                f"{int(row['steps_observed'])} steps",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def build_worker_metrics_aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over successful runs per num_workers."""
    if per_run.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for nw in sorted(per_run["num_workers"].unique()):
        g = per_run[per_run["num_workers"] == nw]
        ok = g[g["status"] == "success"]
        if ok.empty:
            rows.append(
                {
                    "num_workers": int(nw),
                    "n_success_runs": 0,
                    "throughput_mean": np.nan,
                    "throughput_std": np.nan,
                    "mean_gpu_util_mean": np.nan,
                    "mean_gpu_util_std": np.nan,
                    "mean_total_step_ms_mean": np.nan,
                    "mean_total_step_ms_std": np.nan,
                    "gpu_energy_kwh_mean": np.nan,
                    "gpu_energy_kwh_std": np.nan,
                }
            )
            continue
        rows.append(
            {
                "num_workers": int(nw),
                "n_success_runs": int(len(ok)),
                "throughput_mean": float(ok["throughput_samples_per_s"].mean()),
                "throughput_std": float(ok["throughput_samples_per_s"].std(ddof=1))
                if len(ok) > 1
                else 0.0,
                "mean_gpu_util_mean": float(ok["mean_gpu_util_pct"].mean()),
                "mean_gpu_util_std": float(ok["mean_gpu_util_pct"].std(ddof=1))
                if len(ok) > 1
                else 0.0,
                "mean_total_step_ms_mean": float(ok["mean_total_step_ms"].mean()),
                "mean_total_step_ms_std": float(ok["mean_total_step_ms"].std(ddof=1))
                if len(ok) > 1
                else 0.0,
                "gpu_energy_kwh_mean": float(ok["gpu_energy_kwh"].mean()),
                "gpu_energy_kwh_std": float(ok["gpu_energy_kwh"].std(ddof=1)) if len(ok) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_worker_performance(per_run: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    """Line/errorbar plots vs num_workers for successful runs; show failures as gaps."""
    if metrics.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    x = metrics["num_workers"].to_numpy()

    def errbar(ax, y, yerr, ylabel: str, title: str) -> None:
        mask = ~np.isnan(y.astype(float))
        ax.errorbar(
            x[mask],
            y[mask],
            yerr=yerr[mask],
            fmt="o-",
            capsize=4,
            color="#264653",
            linewidth=2,
            markersize=8,
        )
        ax.set_xticks(sorted(per_run["num_workers"].unique()))
        ax.set_xlabel("num_workers")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    errbar(
        axes[0],
        metrics["throughput_mean"].to_numpy(),
        metrics["throughput_std"].fillna(0).to_numpy(),
        "Samples / s",
        "Throughput (success runs)",
    )
    errbar(
        axes[1],
        metrics["mean_gpu_util_mean"].to_numpy(),
        metrics["mean_gpu_util_std"].fillna(0).to_numpy(),
        "Mean GPU util (%)",
        "GPU utilization (timeline mean)",
    )
    errbar(
        axes[2],
        metrics["mean_total_step_ms_mean"].to_numpy(),
        metrics["mean_total_step_ms_std"].fillna(0).to_numpy(),
        "Mean total step time (ms)",
        "Per-step duration (run_summary)",
    )

    for ax in axes:
        for i, nw in enumerate(metrics["num_workers"]):
            if metrics.iloc[i]["n_success_runs"] == 0:
                ax.axvline(nw, color="#d62828", linestyle="--", alpha=0.35)

    fig.suptitle("Worker sweep performance (bs=2); red dashed line = no successful run", fontsize=11, y=1.02)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    validation_summary: pd.DataFrame,
    worker_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    def frame_to_pipe_table(frame: pd.DataFrame) -> str:
        headers = [str(col) for col in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in frame.iterrows():
            values = [str(row[col]) for col in frame.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Extended Analysis Summary",
        "",
        "## Validation Sweep",
        "",
        frame_to_pipe_table(validation_summary.round(4)),
        "",
        "## Worker-Count Feasibility",
        "",
        frame_to_pipe_table(worker_summary.round(4)),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dirs = ensure_dirs(output_root)

    validation_summary = build_validation_summary(Path(args.main_root), Path(args.validation_root))
    validation_summary.to_csv(out_dirs["csv"] / "validation_summary.csv", index=False)
    plot_validation_summary(
        validation_summary,
        out_dirs["figures"] / "validation_consistency.png",
    )

    worker_roots = [Path(path) for path in args.worker_roots]
    worker_per_run = build_worker_per_run_rows(worker_roots)
    worker_per_run.to_csv(out_dirs["csv"] / "worker_runs.csv", index=False)

    worker_summary = build_worker_summary(worker_roots)
    worker_summary.to_csv(out_dirs["csv"] / "worker_feasibility_summary.csv", index=False)
    plot_worker_summary(
        worker_summary,
        out_dirs["figures"] / "worker_feasibility.png",
    )

    worker_metrics = build_worker_metrics_aggregate(worker_per_run)
    worker_metrics.to_csv(out_dirs["csv"] / "worker_metrics_by_nw.csv", index=False)
    if not worker_metrics.empty:
        plot_worker_performance(
            worker_per_run,
            worker_metrics,
            out_dirs["figures"] / "worker_performance_metrics.png",
        )

    write_summary_markdown(
        validation_summary,
        worker_summary,
        output_root / "extended_analysis_summary.md",
    )


if __name__ == "__main__":
    main()

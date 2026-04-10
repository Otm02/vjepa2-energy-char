"""
V-JEPA2 fine-grained measurements.

Records CUDA-synchronised phase timings together with 500 ms sampled
CodeCarbon energy data and rubric-required system-utilisation timelines.
"""

import csv
import logging
import os
import shutil
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency on the cluster image
    psutil = None

from codecarbon import OfflineEmissionsTracker
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData

import src.config as config
import src.trainer.stats.base as base

logger = logging.getLogger(__name__)

trainer_stats_name = "vjepa2_phases"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PhaseFileOutput(BaseOutput):
    """Writes run-level and task-level CodeCarbon outputs to deterministic CSVs."""

    def __init__(self, output_dir: str, run_num: int = 1):
        self.output_dir = output_dir
        self.run_num = run_num
        os.makedirs(output_dir, exist_ok=True)
        self.phase_file = os.path.join(output_dir, f"phases_run{run_num}.csv")
        self.summary_file = os.path.join(output_dir, f"summary_run{run_num}.csv")

    def _write_rows(self, path: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def out(self, total: EmissionsData, delta: EmissionsData):
        self._write_rows(self.summary_file, [dict(total.values)])

    def live_out(self, total: EmissionsData, delta: EmissionsData):
        pass

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str):
        rows = [dict(d.values) for d in data]
        self._write_rows(self.phase_file, rows)


class LivePowerOutput(BaseOutput):
    """Streams CodeCarbon live power samples directly to CSV."""

    def __init__(self, output_dir: str, run_num: int = 1):
        self.file_path = os.path.join(output_dir, f"live_power_run{run_num}.csv")
        self.lock = threading.Lock()
        self.fieldnames: Optional[List[str]] = None

    def _append_row(self, row: Dict[str, Any]) -> None:
        with self.lock:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            file_exists = os.path.exists(self.file_path)
            with open(self.file_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    def out(self, total: EmissionsData, delta: EmissionsData):
        pass

    def live_out(self, total: EmissionsData, delta: EmissionsData):
        self._append_row(dict(total.values))

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str):
        pass


class NVMLSystemReader:
    """Best-effort GPU telemetry reader using NVML, with nvidia-smi fallback."""

    def __init__(self, device: Optional[torch.device]):
        self.device_index = 0
        if device is not None and device.index is not None:
            self.device_index = device.index
        self._source = "unavailable"
        self._handle = None
        self._pynvml = None

        try:  # pragma: no cover - depends on cluster image
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._pynvml = pynvml
            self._source = "pynvml"
        except Exception:
            if shutil.which("nvidia-smi") is not None:
                self._source = "nvidia-smi"

    @property
    def source(self) -> str:
        return self._source

    def close(self) -> None:
        if self._pynvml is not None:  # pragma: no cover - depends on cluster image
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def read(self) -> Dict[str, Any]:
        if self._source == "pynvml":  # pragma: no cover - depends on cluster image
            return self._read_pynvml()
        if self._source == "nvidia-smi":
            return self._read_nvidia_smi()
        return {
            "gpu_utilization_pct": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_power_w": None,
            "gpu_temperature_c": None,
            "gpu_metrics_source": self._source,
        }

    def _read_pynvml(self) -> Dict[str, Any]:  # pragma: no cover - depends on cluster image
        util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        power_w = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        temperature = self._pynvml.nvmlDeviceGetTemperature(
            self._handle, self._pynvml.NVML_TEMPERATURE_GPU
        )
        return {
            "gpu_utilization_pct": float(util.gpu),
            "gpu_memory_used_mb": memory.used / (1024.0 ** 2),
            "gpu_memory_total_mb": memory.total / (1024.0 ** 2),
            "gpu_power_w": power_w,
            "gpu_temperature_c": float(temperature),
            "gpu_metrics_source": self._source,
        }

    def _read_nvidia_smi(self) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id={}".format(self.device_index),
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            parts = [part.strip() for part in result.stdout.strip().split(",")]
            return {
                "gpu_utilization_pct": _safe_float(parts[0]) if len(parts) > 0 else None,
                "gpu_memory_used_mb": _safe_float(parts[1]) if len(parts) > 1 else None,
                "gpu_memory_total_mb": _safe_float(parts[2]) if len(parts) > 2 else None,
                "gpu_power_w": _safe_float(parts[3]) if len(parts) > 3 else None,
                "gpu_temperature_c": _safe_float(parts[4]) if len(parts) > 4 else None,
                "gpu_metrics_source": self._source,
            }
        except Exception:
            return {
                "gpu_utilization_pct": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
                "gpu_power_w": None,
                "gpu_temperature_c": None,
                "gpu_metrics_source": "unavailable",
            }


class SystemTimelineSampler:
    """Background sampler for GPU/CPU utilisation timelines."""

    FIELDNAMES = [
        "timestamp_utc",
        "elapsed_s",
        "sample_interval_s",
        "cpu_utilization_pct",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_power_w",
        "gpu_temperature_c",
        "torch_gpu_allocated_mb",
        "torch_gpu_reserved_mb",
        "gpu_metrics_source",
    ]

    def __init__(
        self,
        output_dir: str,
        run_num: int,
        sample_interval_s: float,
        device: Optional[torch.device] = None,
    ) -> None:
        self.file_path = os.path.join(output_dir, f"system_timeline_run{run_num}.csv")
        self.sample_interval_s = sample_interval_s
        self.device = device
        self.reader = NVMLSystemReader(device)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        with open(self.file_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def start(self) -> None:
        self._start_time = time.time()
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_s * 4.0))
        self.reader.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._write_sample()
            if self._stop_event.wait(self.sample_interval_s):
                break
        self._write_sample()

    def _write_sample(self) -> None:
        if self._start_time is None:
            return

        cpu_pct = psutil.cpu_percent(interval=None) if psutil is not None else None
        gpu_metrics = self.reader.read()
        torch_allocated = None
        torch_reserved = None
        if self.device is not None and self.device.type == "cuda":
            try:
                torch_allocated = torch.cuda.memory_allocated(self.device) / (1024.0 ** 2)
                torch_reserved = torch.cuda.memory_reserved(self.device) / (1024.0 ** 2)
            except Exception:
                torch_allocated = None
                torch_reserved = None

        row = {
            "timestamp_utc": _utc_timestamp(),
            "elapsed_s": time.time() - self._start_time,
            "sample_interval_s": self.sample_interval_s,
            "cpu_utilization_pct": cpu_pct,
            "gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
            "gpu_memory_used_mb": gpu_metrics["gpu_memory_used_mb"],
            "gpu_memory_total_mb": gpu_metrics["gpu_memory_total_mb"],
            "gpu_power_w": gpu_metrics["gpu_power_w"],
            "gpu_temperature_c": gpu_metrics["gpu_temperature_c"],
            "torch_gpu_allocated_mb": torch_allocated,
            "torch_gpu_reserved_mb": torch_reserved,
            "gpu_metrics_source": gpu_metrics["gpu_metrics_source"],
        }

        with open(self.file_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writerow(row)


class VJepa2PhaseStats(base.TrainerStats):
    """Fine-grained phase timings plus run-level utilisation and energy outputs."""

    def __init__(
        self,
        output_dir: str,
        run_num: int = 1,
        project_name: str = "vjepa2",
        measure_power_secs: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.output_dir = output_dir
        self.run_num = run_num
        self.project_name = project_name
        self.measure_power_secs = measure_power_secs
        self.device = device
        self.losses: List[float] = []
        self.step_summaries: List[Dict[str, Any]] = []

        self.current_epoch = 0
        self.current_step = 0
        self._phase_starts: Dict[str, float] = {}
        self._current_step_summary: Optional[Dict[str, Any]] = None
        self._train_start_time: Optional[float] = None
        self._train_end_time: Optional[float] = None

        os.makedirs(output_dir, exist_ok=True)
        self.timing_file = os.path.join(output_dir, f"phase_timing_run{run_num}.csv")
        self.step_summary_file = os.path.join(output_dir, f"step_summary_run{run_num}.csv")
        self.run_summary_file = os.path.join(output_dir, f"run_summary_run{run_num}.csv")

        self._init_csv_files()

        self.output_handler = PhaseFileOutput(output_dir, run_num)
        self.tracker = OfflineEmissionsTracker(
            project_name=project_name,
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[self.output_handler],
            api_call_interval=-1,
            measure_power_secs=measure_power_secs,
            log_level="warning",
            allow_multiple_runs=True,
        )

        self.live_output = LivePowerOutput(output_dir, run_num)
        self.live_tracker = OfflineEmissionsTracker(
            project_name=f"{project_name}_live",
            country_iso_code="CAN",
            region="quebec",
            save_to_file=False,
            output_handlers=[self.live_output],
            api_call_interval=-1,
            measure_power_secs=measure_power_secs,
            allow_multiple_runs=True,
            log_level="warning",
        )

        self.timeline_sampler = SystemTimelineSampler(
            output_dir=output_dir,
            run_num=run_num,
            sample_interval_s=measure_power_secs,
            device=device,
        )

    def _init_csv_files(self) -> None:
        with open(self.timing_file, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "step",
                    "phase",
                    "start_time",
                    "end_time",
                    "duration_ms",
                    "gpu_mem_allocated_mb",
                    "gpu_mem_reserved_mb",
                ]
            )

        with open(self.step_summary_file, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "step",
                    "loss",
                    "total_step_ms",
                    "data_transfer_ms",
                    "forward_ms",
                    "backward_ms",
                    "optimizer_step_ms",
                    "ema_update_ms",
                    "step_start_time",
                    "step_end_time",
                ]
            )

        with open(self.run_summary_file, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "project_name",
                    "run_num",
                    "sample_interval_s",
                    "epochs_observed",
                    "steps_observed",
                    "wall_time_s",
                    "mean_total_step_ms",
                    "std_total_step_ms",
                    "mean_forward_ms",
                    "mean_backward_ms",
                    "mean_optimizer_step_ms",
                    "mean_ema_update_ms",
                    "final_loss",
                    "codecarbon_gpu_energy_kwh",
                ]
            )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def _sync_time(self) -> float:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.time()

    def _record_phase(self, phase_name: str, start: float, end: float) -> None:
        duration_ms = (end - start) * 1000.0
        gpu_mem_alloc = None
        gpu_mem_reserved = None
        if self.device is not None and self.device.type == "cuda":
            try:
                gpu_mem_alloc = torch.cuda.memory_allocated(self.device) / (1024.0 ** 2)
                gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / (1024.0 ** 2)
            except Exception:
                gpu_mem_alloc = None
                gpu_mem_reserved = None

        with open(self.timing_file, "a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    self.current_epoch,
                    self.current_step,
                    phase_name,
                    start,
                    end,
                    f"{duration_ms:.3f}",
                    "" if gpu_mem_alloc is None else f"{gpu_mem_alloc:.1f}",
                    "" if gpu_mem_reserved is None else f"{gpu_mem_reserved:.1f}",
                ]
            )

        if self._current_step_summary is not None:
            self._current_step_summary[f"{phase_name}_ms"] = round(duration_ms, 3)

    def _start_named_phase(self, phase_name: str) -> None:
        self._phase_starts[phase_name] = self._sync_time()

    def _stop_named_phase(self, phase_name: str) -> None:
        start = self._phase_starts.pop(phase_name, None)
        if start is None:
            return
        self._record_phase(phase_name, start, self._sync_time())

    def _step_stat(self, key: str) -> List[float]:
        values = []
        for summary in self.step_summaries:
            value = summary.get(key)
            if value is not None:
                values.append(float(value))
        return values

    def _mean(self, values: List[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def _std(self, values: List[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def _read_codecarbon_gpu_energy(self) -> Optional[float]:
        summary_file = os.path.join(self.output_dir, f"summary_run{self.run_num}.csv")
        if not os.path.exists(summary_file):
            return None
        try:
            with open(summary_file, newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                return None
            return _safe_float(rows[-1].get("gpu_energy"))
        except Exception:
            return None

    def _write_run_summary(self) -> None:
        wall_time_s = 0.0
        if self._train_start_time is not None and self._train_end_time is not None:
            wall_time_s = self._train_end_time - self._train_start_time

        row = {
            "project_name": self.project_name,
            "run_num": self.run_num,
            "sample_interval_s": self.measure_power_secs,
            "epochs_observed": self.current_epoch,
            "steps_observed": self.current_step,
            "wall_time_s": round(wall_time_s, 3),
            "mean_total_step_ms": round(self._mean(self._step_stat("total_step_ms")), 3),
            "std_total_step_ms": round(self._std(self._step_stat("total_step_ms")), 3),
            "mean_forward_ms": round(self._mean(self._step_stat("forward_ms")), 3),
            "mean_backward_ms": round(self._mean(self._step_stat("backward_ms")), 3),
            "mean_optimizer_step_ms": round(self._mean(self._step_stat("optimizer_step_ms")), 3),
            "mean_ema_update_ms": round(self._mean(self._step_stat("ema_update_ms")), 3),
            "final_loss": round(self.losses[-1], 6) if self.losses else None,
            "codecarbon_gpu_energy_kwh": self._read_codecarbon_gpu_energy(),
        }

        with open(self.run_summary_file, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)

    def start_train(self):
        logger.info("Starting V-JEPA2 fine-grained tracking")
        self._train_start_time = time.time()
        self.timeline_sampler.start()
        self.tracker.start()
        self.live_tracker.start()

    def stop_train(self):
        logger.info("Stopping V-JEPA2 fine-grained tracking")
        self._train_end_time = time.time()
        self.timeline_sampler.stop()
        self.live_tracker.stop()
        self.tracker.stop()

    def start_step(self):
        self.current_step += 1
        step_start = self._sync_time()
        self._current_step_summary = {
            "epoch": self.current_epoch,
            "step": self.current_step,
            "loss": None,
            "total_step_ms": None,
            "data_transfer_ms": None,
            "forward_ms": None,
            "backward_ms": None,
            "optimizer_step_ms": None,
            "ema_update_ms": None,
            "step_start_time": step_start,
            "step_end_time": None,
        }
        self._phase_starts["total_step"] = step_start

    def stop_step(self):
        self._stop_named_phase("total_step")
        if self._current_step_summary is not None:
            self._current_step_summary["step_end_time"] = self._sync_time()

    def start_forward(self):
        self._start_named_phase("forward")

    def stop_forward(self):
        self._stop_named_phase("forward")

    def start_backward(self):
        self._start_named_phase("backward")

    def stop_backward(self):
        self._stop_named_phase("backward")

    def start_optimizer_step(self):
        self._start_named_phase("optimizer_step")

    def stop_optimizer_step(self):
        self._stop_named_phase("optimizer_step")

    def start_save_checkpoint(self):
        self._start_named_phase("save_checkpoint")

    def stop_save_checkpoint(self):
        self._stop_named_phase("save_checkpoint")

    def log_step(self):
        if self._current_step_summary is None:
            return

        if self.losses:
            self._current_step_summary["loss"] = round(self.losses[-1], 6)

        row = dict(self._current_step_summary)
        self.step_summaries.append(row)
        with open(self.step_summary_file, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)
        self._current_step_summary = None

    def log_stats(self):
        self._write_run_summary()

    def log_loss(self, loss: torch.Tensor):
        if isinstance(loss, torch.Tensor):
            self.losses.append(float(loss.detach().cpu().item()))
        else:
            self.losses.append(float(loss))

    def start_phase(self, name: str):
        self._start_named_phase(name)

    def stop_phase(self, name: str):
        self._stop_named_phase(name)


def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    trainer_conf = getattr(conf.trainer_stats_configs, "vjepa2_phases", None)
    output_dir = kwargs.get("output_dir", "./vjepa2_energy_logs")
    run_num = 1
    project_name = "vjepa2"
    measure_power_secs = 0.5

    if trainer_conf is not None:
        output_dir = getattr(trainer_conf, "output_dir", output_dir)
        run_num = getattr(trainer_conf, "run_num", run_num)
        project_name = getattr(trainer_conf, "project_name", project_name)
        measure_power_secs = getattr(trainer_conf, "measure_power_secs", measure_power_secs)

    return VJepa2PhaseStats(
        output_dir=output_dir,
        run_num=run_num,
        project_name=project_name,
        measure_power_secs=measure_power_secs,
        device=kwargs.get("device"),
    )

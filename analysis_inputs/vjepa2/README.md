# V-JEPA2 Raw Input Layout

The unified runner writes raw CSV inputs to this directory with a stable layout:

```text
analysis_inputs/vjepa2/
├── experiment_manifest.json
├── mode1_baseline/
│   └── bs2/
│       └── run1/
│           ├── command.sh
│           ├── process.log
│           ├── run_metadata.json
│           └── training.log
├── mode2_codecarbon/
│   └── bs2/
│       └── run1/
│           ├── run_1_cc_full_rank_0.csv
│           ├── run_metadata.json
│           └── training.log
└── mode3_finegrained/
    └── bs2/
        └── run1/
            ├── live_power_run1.csv
            ├── phase_timing_run1.csv
            ├── run_metadata.json
            ├── run_summary_run1.csv
            ├── step_summary_run1.csv
            ├── summary_run1.csv
            ├── system_timeline_run1.csv
            └── training.log
```

Notes:

- `mode1_baseline` is the true no-instrumentation baseline. It uses
  `--trainer_stats noop`, and the authoritative end-to-end time comes from the
  external wall-clock recorded in `run_metadata.json`.
- `mode2_codecarbon` is the coarse GPU-energy mode. Use GPU energy as the
  primary energy metric; CPU and RAM energy columns are present but should not
  be used as primary conclusions on the class cluster.
- `mode3_finegrained` contains the rubric-required phase timings and system
  utilization timelines sampled every 500 ms.
- The legacy checked-in sample remains at `analysis_inputs/finegrained/` and is
  automatically used by the final analysis script when no structured
  fine-grained runs are available yet.

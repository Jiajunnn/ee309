# ee309 Memory Device Profiler

This project compares several memory device technologies (SRAM, DRAM, RRAM, Gain Cell, PCM, FeRAM, NAND Flash, STT-MRAM) against multiple workloads. The main script, `calculate_scores.py`, normalizes each device’s metrics, scores them for Mobile/Embedded, AI/HPC, and Extreme Environment workloads, and prints the top candidates for on-chip and off-chip deployments.

## Repository Contents

- `calculate_scores.py` – CLI tool that ranks devices using predefined metrics, placements, and workload weights.
- `profiling.py` – Synthetic workload profiler for latency/memory measurements (not required for ranking but useful for experimentation).
- `*_summary.csv`, `*_lifetimes.csv` – Sample profiling outputs.

## Device Metrics

The rankings derive from the measured characteristics provided in the assignment:

| Metric | Notes |
| --- | --- |
| Average energy (pJ/bit) | lower is better |
| Area (µm²) | die footprint |
| Latency (ns) | lower is better |
| Bandwidth (GB/s) | higher is better |
| Endurance (cycles) | higher is better |
| Soft error rate (FIT/Mb) | lower is better |
| Retention (years) | data retention |
| Temperature stability (°C) | maximum operating temperature |

Each raw value is min-max normalized into a 1–10 score so the weighted sums remain comparable across workloads.

## Usage

```bash
python3 calculate_scores.py --placement on   # on-chip hierarchy
python3 calculate_scores.py --placement off  # off-chip hierarchy
```

For the selected placement, the script prints three sections (Mobile/Embedded, AI/HPC, Extreme Environment), each listing the top three devices plus their per-workload scores.

## Customizing Weights

- Adjust the `WORKLOAD_WEIGHTS` dictionary in `calculate_scores.py` to experiment with different priorities.
- Each placement has its own weight sets, enabling different optimization criteria for on-chip vs. off-chip use cases.


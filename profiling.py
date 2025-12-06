import csv
import time
import tracemalloc
from collections import defaultdict

import numpy as np

# OPTIONAL: If psutil installed, gather CPU/mem stats
try:
    import psutil
    USE_PSUTIL = True
except ImportError:
    USE_PSUTIL = False


SUMMARY_FIELDS = [
    "avg_latency_ms",
    "p99_latency_ms",
    "peak_memory_kb",
    "sensor_reads_bytes",
    "writes_bytes",
    "cpu_usage_pct",
    "sys_memory_pct",
]


class LifetimeTracker:
    """Tracks how long temporary data structures must live in memory."""

    def __init__(self):
        self._active = {}
        self._samples = defaultdict(list)

    def start(self, name):
        self._active[name] = time.perf_counter()

    def end(self, name):
        start = self._active.pop(name, None)
        if start is not None:
            self._samples[name].append(time.perf_counter() - start)

    def stats(self):
        lifetime_stats = {}
        for name, durations in self._samples.items():
            arr = np.array(durations)
            lifetime_stats[name] = {
                "avg_ms": arr.mean() * 1000,
                "p99_ms": np.percentile(arr, 99) * 1000 if len(arr) > 1 else arr[0] * 1000,
                "max_ms": arr.max() * 1000,
            }
        return lifetime_stats


def _profile_loop(
    workload_fn,
    *,
    iterations,
    lifetime_tracker=None,
    finalize_tracker=None,
    **kwargs,
):
    read_bytes = 0
    write_bytes = 0

    tracemalloc.start()

    latencies = []
    tracker = lifetime_tracker or LifetimeTracker()

    for _ in range(iterations):
        t0 = time.perf_counter()
        rb, wb = workload_fn(tracker, **kwargs)
        read_bytes += rb
        write_bytes += wb
        latencies.append(time.perf_counter() - t0)

    if finalize_tracker:
        finalize_tracker(tracker)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cpu_pct = psutil.cpu_percent() if USE_PSUTIL else None
    mem_info = psutil.virtual_memory() if USE_PSUTIL else None

    return {
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        "peak_memory_kb": peak / 1024,
        "sensor_reads_bytes": read_bytes,
        "writes_bytes": write_bytes,
        "cpu_usage_pct": cpu_pct,
        "sys_memory_pct": mem_info.percent if USE_PSUTIL else None,
        "data_lifetimes_ms": tracker.stats(),
    }


def run_embedded_workload(num_iters=2000, sensor_size=256):
    """Embedded-style sensing + filtering + dot-product logging."""

    def workload_step(tracker, sensor_size=None, **_):
        read_bytes = 0
        write_bytes = 0

        sensor_data = np.random.rand(sensor_size)
        tracker.start("sensor_data")
        read_bytes += sensor_data.nbytes

        filtered = np.convolve(sensor_data, np.ones(5) / 5, mode="valid")
        tracker.end("sensor_data")
        tracker.start("filtered")
        read_bytes += filtered.nbytes

        weight = np.random.rand(filtered.shape[0])
        tracker.start("weight")
        read_bytes += weight.nbytes
        output = np.dot(filtered, weight)
        tracker.end("filtered")
        tracker.end("weight")
        tracker.start("output")
        write_bytes += output.item().__sizeof__()

        log_entry = float(output)
        tracker.end("output")
        tracker.start("log_entry")
        write_bytes += 8
        tracker.end("log_entry")

        return read_bytes, write_bytes

    return _profile_loop(workload_step, iterations=num_iters, sensor_size=sensor_size)


def run_ai_inference_workload(num_iters=300, input_size=1024, hidden_size=512, output_size=128):
    """Heavier workload simulating dense AI layers."""

    def workload_step(
        tracker,
        input_size=None,
        hidden_size=None,
        output_size=None,
        **_,
    ):
        read_bytes = 0
        write_bytes = 0

        x = np.random.rand(input_size)
        tracker.start("ai_input")
        read_bytes += x.nbytes

        w1 = np.random.rand(hidden_size, input_size)
        b1 = np.random.rand(hidden_size)
        tracker.start("ai_w1")
        tracker.start("ai_b1")
        read_bytes += w1.nbytes + b1.nbytes

        hidden = w1 @ x + b1
        tracker.end("ai_input")
        tracker.end("ai_w1")
        tracker.end("ai_b1")
        tracker.start("ai_hidden")
        write_bytes += hidden.nbytes

        relu = np.maximum(hidden, 0)
        tracker.end("ai_hidden")
        tracker.start("ai_relu")
        write_bytes += relu.nbytes

        w2 = np.random.rand(output_size, hidden_size)
        b2 = np.random.rand(output_size)
        tracker.start("ai_w2")
        tracker.start("ai_b2")
        read_bytes += w2.nbytes + b2.nbytes

        logits = w2 @ relu + b2
        tracker.end("ai_w2")
        tracker.end("ai_b2")
        tracker.end("ai_relu")
        tracker.start("ai_logits")
        write_bytes += logits.nbytes

        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        tracker.end("ai_logits")
        tracker.start("ai_output")
        write_bytes += probs.nbytes
        tracker.end("ai_output")

        return read_bytes, write_bytes

    return _profile_loop(
        workload_step,
        iterations=num_iters,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )


def run_physics_sim_workload(num_steps=800, num_particles=2048, timestep=1e-3):
    """Physics-inspired workload updating particle positions and velocities."""

    def workload_step(
        tracker,
        positions=None,
        velocities=None,
        mass=None,
        timestep=None,
        **_,
    ):
        read_bytes = 0
        write_bytes = 0

        forces = np.random.randn(*positions.shape)
        tracker.start("forces")
        read_bytes += forces.nbytes

        accelerations = forces / mass
        tracker.end("forces")
        tracker.start("accelerations")
        write_bytes += accelerations.nbytes

        read_bytes += velocities.nbytes + positions.nbytes + mass.nbytes
        new_velocities = velocities + accelerations * timestep

        tracker.start("new_velocities")
        write_bytes += new_velocities.nbytes

        damped_velocities = 0.99 * new_velocities
        tracker.end("new_velocities")
        tracker.start("damped_velocities")
        write_bytes += damped_velocities.nbytes

        new_positions = positions + damped_velocities * timestep
        tracker.end("damped_velocities")
        tracker.start("new_positions")
        write_bytes += new_positions.nbytes

        np.copyto(velocities, damped_velocities)
        np.copyto(positions, new_positions)
        write_bytes += velocities.nbytes + positions.nbytes
        tracker.end("accelerations")
        tracker.end("new_positions")

        return read_bytes, write_bytes

    positions = np.random.rand(num_particles, 3)
    velocities = np.random.randn(num_particles, 3) * 0.1
    mass = np.ones((num_particles, 3))

    tracker = LifetimeTracker()
    tracker.start("positions_state")
    tracker.start("velocities_state")
    tracker.start("mass_state")

    def finalize(trk):
        trk.end("positions_state")
        trk.end("velocities_state")
        trk.end("mass_state")

    return _profile_loop(
        workload_step,
        iterations=num_steps,
        lifetime_tracker=tracker,
        finalize_tracker=finalize,
        positions=positions,
        velocities=velocities,
        mass=mass,
        timestep=timestep,
    )


def write_stats_to_csv(stats, summary_path="profiling_summary.csv", lifetime_path="data_lifetimes.csv"):
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({field: stats.get(field) for field in SUMMARY_FIELDS})

    lifetimes = stats.get("data_lifetimes_ms") or {}
    if lifetimes:
        with open(lifetime_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["buffer", "avg_ms", "p99_ms", "max_ms"])
            for name, metrics in lifetimes.items():
                writer.writerow([name, metrics["avg_ms"], metrics["p99_ms"], metrics["max_ms"]])


def report_workload(name, stats, summary_path, lifetime_path):
    print(f"\n=== {name} Workload Profiling Results ===")
    print(f"Average latency per iteration: {stats['avg_latency_ms']:.4f} ms")
    print(f"99th percentile latency:       {stats['p99_latency_ms']:.4f} ms")
    print(f"Peak memory usage:             {stats['peak_memory_kb']:.2f} KB")
    print(f"Total bytes read:              {stats['sensor_reads_bytes']/1024:.2f} KB")
    print(f"Total bytes written:           {stats['writes_bytes']/1024:.2f} KB")

    if stats["cpu_usage_pct"] is not None:
        print(f"CPU usage:                     {stats['cpu_usage_pct']} %")
        print(f"System memory usage:           {stats['sys_memory_pct']} %")
    else:
        print("(Install psutil for CPU & system memory profiling)")

    lifetimes = stats["data_lifetimes_ms"]
    if lifetimes:
        print("\nData lifetimes (ms):")
        for buffer_name, metrics in lifetimes.items():
            print(
                f"  {buffer_name:<20s} avg: {metrics['avg_ms']:.6f} | "
                f"p99: {metrics['p99_ms']:.6f} | max: {metrics['max_ms']:.6f}"
            )

    write_stats_to_csv(stats, summary_path=summary_path, lifetime_path=lifetime_path)
    print(f"\nSaved metrics to {summary_path} and {lifetime_path}")

def write_combined_summary(rows, path="all_workloads_summary.csv"):
    if not rows:
        return
    fieldnames = ["workload"] + SUMMARY_FIELDS
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for workload_name, stats in rows:
            row = {"workload": workload_name}
            row.update({field: stats.get(field) for field in SUMMARY_FIELDS})
            writer.writerow(row)
    print(f"\nCombined summary saved to {path}")


def write_combined_lifetimes(rows, path="all_workloads_lifetimes.csv"):
    if not rows:
        return
    fieldnames = ["workload", "buffer", "avg_ms", "p99_ms", "max_ms"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for workload_name, buffer_name, metrics in rows:
            row = {
                "workload": workload_name,
                "buffer": buffer_name,
                "avg_ms": metrics["avg_ms"],
                "p99_ms": metrics["p99_ms"],
                "max_ms": metrics["max_ms"],
            }
            writer.writerow(row)
    print(f"Combined lifetimes saved to {path}")


if __name__ == "__main__":
    workloads = [
        ("Embedded", run_embedded_workload, "embedded", {}),
        ("AI Inference", run_ai_inference_workload, "ai_inference", {}),
        ("Physics Simulation", run_physics_sim_workload, "physics_sim", {}),
    ]

    combined_rows = []
    combined_lifetimes = []
    for name, fn, slug, kwargs in workloads:
        stats = fn(**kwargs)
        report_workload(
            name,
            stats,
            summary_path=f"{slug}_summary.csv",
            lifetime_path=f"{slug}_data_lifetimes.csv",
        )
        combined_rows.append((name, stats))
        lifetimes = stats.get("data_lifetimes_ms") or {}
        for buffer_name, metrics in lifetimes.items():
            combined_lifetimes.append((name, buffer_name, metrics))

    write_combined_summary(combined_rows)
    write_combined_lifetimes(combined_lifetimes)

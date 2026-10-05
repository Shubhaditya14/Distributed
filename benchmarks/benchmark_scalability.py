"""
Benchmark 3: Scalability Analysis

Show how the system behaves as worker count grows (1, 2, 4 workers).

Every worker runs the same fixed number of iterations, so adding workers
does not shorten the run. This is a weak-scaling measurement: it reports
how much per-iteration time grows with worker count, and the aggregate
samples/second that results. Two caveats apply to the current worker:
- each iteration includes a fixed 0.5s demo sleep, which hides most of the
  coordination cost
- all workers seed their batch from the iteration number, so they process
  identical data rather than disjoint shards
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    ProcessManager, save_results, calculate_stats, get_system_info, clean_checkpoints
)

BATCH_SIZE = 32  # From worker.py
NO_CHECKPOINT_INTERVAL = 999999


def run_trial(num_workers: int, num_iterations: int) -> Dict:
    """Run a single scalability trial."""
    print(f"\n{'='*50}")
    print(f"Running trial: {num_workers} workers")
    print(f"{'='*50}")

    clean_checkpoints()
    cluster = ProcessManager(num_workers=num_workers, checkpoint_interval=NO_CHECKPOINT_INTERVAL,
                             num_iterations=num_iterations)

    try:
        cluster.start_master()
        cluster.start_workers()

        start_time = cluster.wait_for_log(cluster.worker_logs, "Starting training", timeout=60)
        completed = start_time is not None and cluster.wait_for_completion(timeout=180)
        end_time = time.time()
    finally:
        cluster.stop_all()

    total_time = end_time - start_time if completed else 0.0
    total_samples = BATCH_SIZE * num_workers * num_iterations

    result = {
        "num_workers": num_workers,
        "completed": completed,
        "total_time": total_time,
        "num_iterations": num_iterations,
        "avg_iteration_time": total_time / num_iterations,
        "samples_per_second": total_samples / total_time if total_time > 0 else 0,
    }

    print(f"  Completed: {completed}")
    print(f"  Training time: {total_time:.2f}s")
    print(f"  Samples/sec: {result['samples_per_second']:.2f}")

    return result


def run_benchmark(worker_counts: List[int], num_trials: int, num_iterations: int = 30) -> Dict:
    """Run full scalability benchmark."""
    results = {
        "system_info": get_system_info(),
        "config": {
            "num_iterations": num_iterations,
            "batch_size_per_worker": BATCH_SIZE,
            "worker_counts": worker_counts,
            "num_trials": num_trials
        },
        "trials_by_workers": {},
        "summary": {},
        "scalability_analysis": {}
    }

    for num_workers in worker_counts:
        worker_key = str(num_workers)
        results["trials_by_workers"][worker_key] = []

        for trial_num in range(num_trials):
            trial_result = run_trial(num_workers, num_iterations)
            trial_result["trial_number"] = trial_num + 1
            results["trials_by_workers"][worker_key].append(trial_result)
            time.sleep(2)

    # Calculate summary statistics
    for worker_key, trials in results["trials_by_workers"].items():
        valid_trials = [t for t in trials if t["completed"]]
        if valid_trials:
            results["summary"][worker_key] = {
                "total_time": calculate_stats([t["total_time"] for t in valid_trials]),
                "avg_iteration_time": calculate_stats([t["avg_iteration_time"] for t in valid_trials]),
                "samples_per_second": calculate_stats([t["samples_per_second"] for t in valid_trials]),
            }

    # Weak scaling relative to a single worker
    if "1" in results["summary"]:
        baseline = results["summary"]["1"]
        for worker_key, stats in results["summary"].items():
            num_workers = int(worker_key)
            throughput_scaling = (stats["samples_per_second"]["mean"]
                                  / baseline["samples_per_second"]["mean"])
            results["scalability_analysis"][worker_key] = {
                "num_workers": num_workers,
                "iteration_time_ms": stats["avg_iteration_time"]["mean"] * 1000,
                "iteration_slowdown_percent": (stats["avg_iteration_time"]["mean"]
                                               / baseline["avg_iteration_time"]["mean"] - 1) * 100,
                "throughput_scaling": throughput_scaling,
                "weak_scaling_efficiency_percent": throughput_scaling / num_workers * 100,
            }

    return results


def main():
    print("=" * 60)
    print("Scalability Benchmark")
    print("=" * 60)

    results = run_benchmark(worker_counts=[1, 2, 4], num_trials=2)

    save_results("scalability_metrics.json", results)

    print("\n" + "=" * 60)
    print("Scalability Summary")
    print("=" * 60)

    print(f"\n{'Workers':<10} {'Iter (ms)':<12} {'Throughput':<12} {'Weak-scaling eff.':<18}")
    print("-" * 55)

    for analysis in results["scalability_analysis"].values():
        print(f"{analysis['num_workers']:<10} {analysis['iteration_time_ms']:<12.1f} "
              f"{analysis['throughput_scaling']:<12.2f} {analysis['weak_scaling_efficiency_percent']:<18.1f}%")

    return results


if __name__ == "__main__":
    main()

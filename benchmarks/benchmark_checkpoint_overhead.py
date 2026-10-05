"""
Benchmark 2: Checkpoint Overhead

Quantify performance impact of checkpointing at different frequencies:
- No checkpointing (baseline)
- Every 25 iterations
- Every 10 iterations (current default)
- Every 5 iterations

Measures training-loop wall time (first worker starts training -> all
workers finished) and checkpoint files written. Each iteration includes the
worker's fixed 0.5s demo sleep, so percentages are relative to a
sleep-dominated baseline; the per-checkpoint cost is the more meaningful
number.
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

NO_CHECKPOINT_INTERVAL = 999999


def checkpoint_files() -> List[Path]:
    """List checkpoint files created by all ranks."""
    return list(Path("/tmp/checkpoints").glob("rank_*/*.pt"))


def run_trial(checkpoint_interval: int, num_workers: int, num_iterations: int) -> Dict:
    """Run a single trial with specified checkpoint interval."""
    print(f"\n{'='*50}")
    print(f"Running trial: checkpoint_interval={checkpoint_interval}")
    print(f"{'='*50}")

    clean_checkpoints()
    cluster = ProcessManager(num_workers=num_workers, checkpoint_interval=checkpoint_interval,
                             num_iterations=num_iterations)

    try:
        cluster.start_master()
        cluster.start_workers()

        start_time = cluster.wait_for_log(cluster.worker_logs, "Starting training", timeout=60)
        completed = start_time is not None and cluster.wait_for_completion(timeout=180)
        end_time = time.time()
    finally:
        cluster.stop_all()

    files = checkpoint_files()
    total_time = end_time - start_time if completed else 0.0
    expected = (num_iterations // checkpoint_interval) * num_workers

    result = {
        "checkpoint_interval": checkpoint_interval,
        "completed": completed,
        "total_time": total_time,
        "num_iterations": num_iterations,
        "avg_iteration_time": total_time / num_iterations,
        "num_checkpoints": len(files),
        "expected_checkpoints": expected,
        "avg_checkpoint_size_kb": sum(f.stat().st_size for f in files) / max(len(files), 1) / 1024,
    }

    print(f"  Completed: {completed}")
    print(f"  Training time: {total_time:.2f}s")
    print(f"  Checkpoints created: {len(files)} (expected {expected})")

    return result


def run_benchmark(checkpoint_intervals: List[int], num_trials: int,
                  num_workers: int = 4, num_iterations: int = 50) -> Dict:
    """Run full benchmark across all checkpoint intervals (0 = no checkpointing)."""
    results = {
        "system_info": get_system_info(),
        "config": {
            "num_workers": num_workers,
            "num_iterations": num_iterations,
            "checkpoint_intervals": checkpoint_intervals,
            "num_trials": num_trials
        },
        "trials_by_interval": {},
        "summary": {}
    }

    for interval in checkpoint_intervals:
        interval_key = str(interval) if interval > 0 else "none"
        results["trials_by_interval"][interval_key] = []

        for trial_num in range(num_trials):
            trial_result = run_trial(interval if interval > 0 else NO_CHECKPOINT_INTERVAL,
                                     num_workers, num_iterations)
            trial_result["trial_number"] = trial_num + 1
            results["trials_by_interval"][interval_key].append(trial_result)
            time.sleep(2)

    # Calculate summary statistics
    for interval_key, trials in results["trials_by_interval"].items():
        valid_trials = [t for t in trials if t["completed"]]
        if valid_trials:
            results["summary"][interval_key] = {
                "total_time": calculate_stats([t["total_time"] for t in valid_trials]),
                "num_checkpoints": valid_trials[0]["num_checkpoints"],
                "avg_checkpoint_size_kb": valid_trials[0]["avg_checkpoint_size_kb"],
            }

    # Calculate overhead compared to baseline
    if "none" in results["summary"]:
        baseline_time = results["summary"]["none"]["total_time"]["mean"]
        for interval_key, stats in results["summary"].items():
            if interval_key == "none":
                continue
            overhead_time = stats["total_time"]["mean"] - baseline_time
            checkpoint_rounds = stats["num_checkpoints"] / num_workers
            stats["overhead_seconds"] = overhead_time
            stats["overhead_percent"] = overhead_time / baseline_time * 100
            stats["overhead_ms_per_checkpoint"] = (
                overhead_time / checkpoint_rounds * 1000 if checkpoint_rounds else 0
            )

    return results


def main():
    print("=" * 60)
    print("Checkpoint Overhead Benchmark")
    print("=" * 60)

    results = run_benchmark(checkpoint_intervals=[0, 25, 10, 5], num_trials=2)

    save_results("checkpoint_overhead.json", results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for interval, stats in results["summary"].items():
        print(f"\nCheckpoint interval: {interval}")
        print(f"  Total time: {stats['total_time']['mean']:.2f}s (+/- {stats['total_time']['std']:.2f})")
        if "overhead_percent" in stats:
            print(f"  Overhead: {stats['overhead_percent']:.2f}% "
                  f"({stats['overhead_ms_per_checkpoint']:.1f} ms per checkpoint)")

    return results


if __name__ == "__main__":
    main()

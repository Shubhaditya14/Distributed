"""
Benchmark 1: Recovery Time Analysis

Measures end-to-end recovery latency for a single worker failure:
- Time from worker kill -> master detects failure
- Time from detection -> all workers restarted and re-registered
- Time from re-registration -> checkpoint loaded and training resumed

The master only detects the failure; it does not restart workers. This
harness plays the operator: as soon as the master reports the failure it
kills the surviving workers and starts new ones (what restart_workers.sh
does, without that script's fixed 3s wait or any human reaction time).
"""

import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    ProcessManager, save_results, calculate_stats, get_system_info, clean_checkpoints
)

# From src/master.py failure_detector_thread
HEARTBEAT_TIMEOUT = 15
DETECTOR_CHECK_INTERVAL = 5


def run_trial(kill_after_checkpoint: int, num_workers: int = 4, checkpoint_interval: int = 10) -> Dict:
    """Kill worker 0 shortly after a checkpoint completes and time the recovery."""
    print(f"\n{'='*60}")
    print(f"Running trial: failure after checkpoint {kill_after_checkpoint}")
    print(f"{'='*60}")

    clean_checkpoints()
    cluster = ProcessManager(num_workers=num_workers, checkpoint_interval=checkpoint_interval)

    try:
        cluster.start_master()
        cluster.start_workers()

        marker = f"All workers completed checkpoint for iteration {kill_after_checkpoint}"
        if cluster.wait_for_log([cluster.master_log], marker, timeout=120) is None:
            return {"error": "timeout_waiting_for_checkpoint"}
        time.sleep(1.5)  # fail part-way into the next checkpoint interval

        _, failure_time = cluster.kill_worker(0)

        detection_time = cluster.wait_for_log([cluster.master_log], "Failure Detected", timeout=60)
        if detection_time is None:
            return {"error": "timeout_waiting_for_detection"}
        checkpoint_iteration = cluster.last_complete_checkpoint()

        cluster.kill_all_workers()
        cluster.start_workers()

        registered_time = cluster.wait_for_log(cluster.worker_logs, "Registered as worker", timeout=60)
        resumed_time = cluster.wait_for_log(cluster.worker_logs, "Starting training from iteration", timeout=60)
        if registered_time is None or resumed_time is None:
            return {"error": "timeout_waiting_for_resume"}

        loaded = cluster.loaded_checkpoints()

        return {
            "kill_after_checkpoint": kill_after_checkpoint,
            "resumed_from_iteration": checkpoint_iteration,
            "all_workers_loaded_checkpoint": all(i == checkpoint_iteration for i in loaded),
            "detection_latency": detection_time - failure_time,
            "restart_latency": registered_time - detection_time,
            "resume_latency": resumed_time - registered_time,
            "total_recovery_time": resumed_time - failure_time,
        }
    finally:
        cluster.stop_all()


def main(num_trials: int = 5):
    print("=" * 60)
    print("Recovery Time Benchmark")
    print("=" * 60)

    results = {
        "system_info": get_system_info(),
        "config": {
            "num_workers": 4,
            "checkpoint_interval": 10,
            "heartbeat_timeout": HEARTBEAT_TIMEOUT,
            "detector_check_interval": DETECTOR_CHECK_INTERVAL,
            "num_trials": num_trials,
        },
        # Detector wakes every 5s and flags heartbeats older than 15s, so a
        # failure is noticed 15-20s after the worker's last heartbeat.
        "detection_latency_bounds": {
            "min": HEARTBEAT_TIMEOUT,
            "max": HEARTBEAT_TIMEOUT + DETECTOR_CHECK_INTERVAL,
        },
        "trials": [],
    }

    for trial_num in range(num_trials):
        trial = run_trial(kill_after_checkpoint=10 * (1 + trial_num % 3))
        trial["trial_number"] = trial_num + 1
        results["trials"].append(trial)
        print(f"  {trial}")
        time.sleep(2)

    valid_trials = [t for t in results["trials"] if "error" not in t]
    phases = ["detection_latency", "restart_latency", "resume_latency", "total_recovery_time"]
    results["aggregate_stats"] = {
        phase: calculate_stats([t[phase] for t in valid_trials]) for phase in phases
    } if valid_trials else {}

    save_results("recovery_time_analysis.json", results)

    print("\n" + "=" * 60)
    print("Recovery Time Analysis Results")
    print("=" * 60)
    for phase, stats in results["aggregate_stats"].items():
        print(f"  {phase}: {stats['mean']:.2f}s (min {stats['min']:.2f}, max {stats['max']:.2f})")

    return results


if __name__ == "__main__":
    main()
    print("\nBenchmark complete!")

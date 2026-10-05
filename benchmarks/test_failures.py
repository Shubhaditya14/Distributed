"""
Part 2: Failure Scenario Tests

Test 1: Single Worker Failure
- Kill one worker after the iteration-30 checkpoint
- Verify all workers resume from that checkpoint and training completes

Test 2: Multiple Sequential Failures
- Kill workers after the iteration-20 and iteration-50 checkpoints
- Verify both recoveries and that training completes

Test 3: Checkpoint Corruption Handling
- Corrupt the latest checkpoint, then kill a worker
- Check whether workers fall back to the previous checkpoint

The master only detects failures; these tests restart the workers
themselves once the master reports the failure, like restart_workers.sh.
Each test passes only if the checks in its "checks" dict all hold.
"""

import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (
    ProcessManager, read_log, save_results, get_system_info, clean_checkpoints
)


class FailureTests:
    """Tests for various failure scenarios."""

    def __init__(self, num_workers: int = 4, checkpoint_interval: int = 10):
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval

    def start_cluster(self, num_iterations: int) -> ProcessManager:
        clean_checkpoints()
        cluster = ProcessManager(num_workers=self.num_workers,
                                 checkpoint_interval=self.checkpoint_interval,
                                 num_iterations=num_iterations)
        cluster.start_master()
        cluster.start_workers()
        return cluster

    def fail_and_restart(self, cluster: ProcessManager, after_checkpoint: int, worker_index: int) -> Dict:
        """Kill one worker after a checkpoint, wait for detection, restart all workers."""
        marker = f"All workers completed checkpoint for iteration {after_checkpoint}"
        if cluster.wait_for_log([cluster.master_log], marker, timeout=120) is None:
            raise RuntimeError(f"Checkpoint {after_checkpoint} never completed")
        time.sleep(1.5)

        log_offset = cluster.master_log_size()
        cluster.kill_worker(worker_index)
        detected = cluster.wait_for_log([cluster.master_log], "Failure Detected",
                                        timeout=60, after=log_offset) is not None
        expected_checkpoint = cluster.last_complete_checkpoint()

        cluster.kill_all_workers()
        cluster.start_workers()
        return {"failure_detected": detected, "expected_checkpoint": expected_checkpoint}

    def test_single_worker_failure(self) -> Dict:
        """Test 1: kill worker 0 after checkpoint 30, verify recovery and completion."""
        print("\n" + "=" * 60)
        print("Test 1: Single Worker Failure")
        print("=" * 60)

        result = {"test_name": "single_worker_failure", "checks": {}}
        cluster = None
        try:
            cluster = self.start_cluster(num_iterations=60)
            failure = self.fail_and_restart(cluster, after_checkpoint=30, worker_index=0)
            completed = cluster.wait_for_completion(timeout=120)

            result["resumed_from_iteration"] = cluster.loaded_checkpoints()
            result["checks"] = {
                "failure_detected": failure["failure_detected"],
                "all_workers_resumed_from_checkpoint":
                    all(i == failure["expected_checkpoint"] for i in result["resumed_from_iteration"]),
                "training_completed": completed,
            }
        except Exception as e:
            result["error"] = str(e)
        finally:
            if cluster:
                cluster.stop_all()

        result["success"] = bool(result["checks"]) and all(result["checks"].values())
        return result

    def test_multiple_failures(self) -> Dict:
        """Test 2: two failures in one run, different workers."""
        print("\n" + "=" * 60)
        print("Test 2: Multiple Sequential Failures")
        print("=" * 60)

        result = {"test_name": "multiple_failures", "failures": [], "checks": {}}
        cluster = None
        start_time = time.time()
        try:
            cluster = self.start_cluster(num_iterations=80)

            for failure_num, (after_checkpoint, worker_index) in enumerate([(20, 0), (50, 1)], start=1):
                failure = self.fail_and_restart(cluster, after_checkpoint, worker_index)
                cluster.wait_for_log(cluster.worker_logs, "Starting training from iteration", timeout=60)
                resumed = cluster.loaded_checkpoints()
                result["failures"].append({
                    "failure_num": failure_num,
                    "killed_worker_index": worker_index,
                    "resumed_from_iteration": resumed,
                })
                result["checks"][f"failure_{failure_num}_detected"] = failure["failure_detected"]
                result["checks"][f"failure_{failure_num}_resumed_from_checkpoint"] = (
                    all(i == failure["expected_checkpoint"] for i in resumed)
                )

            result["checks"]["training_completed"] = cluster.wait_for_completion(timeout=120)
            result["total_time"] = time.time() - start_time
        except Exception as e:
            result["error"] = str(e)
        finally:
            if cluster:
                cluster.stop_all()

        result["success"] = bool(result["checks"]) and all(result["checks"].values())
        return result

    def test_checkpoint_corruption(self) -> Dict:
        """Test 3: corrupt checkpoint 30 on every rank, expect fallback to checkpoint 20."""
        print("\n" + "=" * 60)
        print("Test 3: Checkpoint Corruption Handling")
        print("=" * 60)

        result = {"test_name": "checkpoint_corruption", "corrupted_checkpoint": 30,
                  "expected_fallback": 20, "checks": {}}
        cluster = None
        try:
            cluster = self.start_cluster(num_iterations=60)

            marker = "All workers completed checkpoint for iteration 30"
            if cluster.wait_for_log([cluster.master_log], marker, timeout=120) is None:
                raise RuntimeError("Checkpoint 30 never completed")
            for rank in range(self.num_workers):
                Path(f"/tmp/checkpoints/rank_{rank}/checkpoint_iter_30.pt").write_text("corrupted data")

            failure = self.fail_and_restart(cluster, after_checkpoint=30, worker_index=0)
            completed = cluster.wait_for_completion(timeout=120)

            result["resumed_from_iteration"] = cluster.loaded_checkpoints()
            result["worker_errors"] = sorted({
                line for path in cluster.worker_logs
                for line in read_log(path).splitlines() if line.startswith("Error during training")
            })
            result["checks"] = {
                "failure_detected": failure["failure_detected"],
                "fell_back_to_previous_checkpoint":
                    all(i == result["expected_fallback"] for i in result["resumed_from_iteration"]),
                "training_completed": completed,
            }
        except Exception as e:
            result["error"] = str(e)
        finally:
            if cluster:
                cluster.stop_all()

        result["success"] = bool(result["checks"]) and all(result["checks"].values())
        return result


def run_all_tests():
    """Run all failure scenario tests."""
    print("=" * 60)
    print("Failure Scenario Tests")
    print("=" * 60)

    tests = FailureTests(num_workers=4, checkpoint_interval=10)

    all_results = {
        "system_info": get_system_info(),
        "tests": {}
    }

    all_results["tests"]["single_failure"] = tests.test_single_worker_failure()
    time.sleep(3)

    all_results["tests"]["multiple_failures"] = tests.test_multiple_failures()
    time.sleep(3)

    all_results["tests"]["checkpoint_corruption"] = tests.test_checkpoint_corruption()

    save_results("failure_tests.json", all_results)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in all_results["tests"].items():
        status = "PASSED" if result["success"] else "FAILED"
        print(f"  {test_name}: {status}  {result.get('checks', {})} {result.get('error', '')}")

    return all_results


if __name__ == "__main__":
    run_all_tests()

"""
Benchmark 4: Network Overhead

Measure gRPC heartbeat communication overhead for the current design, where
each worker sends one heartbeat per training iteration (the interval is not
separately configurable).

- Heartbeat rate is measured from a real run (heartbeats handled by the master)
- Message size is the serialized protobuf payload; HTTP/2 and gRPC framing
  are not included
- The interval tradeoff table is a model built from those two numbers and
  the master's failure detector constants, not a measurement
"""

import re
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import orchestrator_pb2

from utils import (
    ProcessManager, read_log, save_results, get_system_info, clean_checkpoints
)

# From src/master.py failure_detector_thread
HEARTBEAT_TIMEOUT = 15
DETECTOR_CHECK_INTERVAL = 5
NO_CHECKPOINT_INTERVAL = 999999


def heartbeat_payload_bytes() -> Dict:
    """Serialized size of a representative heartbeat request and response."""
    request = orchestrator_pb2.HeartbeatRequest(
        worker_id="a1b2c3d4",
        timestamp=int(time.time()),
        current_iteration=50,
        current_loss=0.8731,
        is_training=True,
    )
    response = orchestrator_pb2.HeartbeatResponse(acknowledged=True)
    return {
        "request": len(request.SerializeToString()),
        "response": len(response.SerializeToString()),
    }


def measure_heartbeat_rate(num_workers: int, num_iterations: int) -> Dict:
    """Run a cluster and count the heartbeats the master handled."""
    clean_checkpoints()
    cluster = ProcessManager(num_workers=num_workers, checkpoint_interval=NO_CHECKPOINT_INTERVAL,
                             num_iterations=num_iterations)

    try:
        cluster.start_master()
        cluster.start_workers()

        start_time = cluster.wait_for_log(cluster.worker_logs, "Starting training", timeout=60)
        completed = start_time is not None and cluster.wait_for_completion(timeout=180)
        end_time = time.time()
        heartbeats = len(re.findall(r"^Worker \w+ - iteration: ", read_log(cluster.master_log), re.M))
    finally:
        cluster.stop_all()

    total_time = end_time - start_time if completed else 0.0
    return {
        "completed": completed,
        "num_workers": num_workers,
        "num_iterations": num_iterations,
        "total_time": total_time,
        "total_heartbeats": heartbeats,
        "heartbeats_per_worker_per_sec": heartbeats / num_workers / total_time if total_time else 0,
    }


def model_tradeoff(intervals: List[float], num_workers: int, bytes_per_heartbeat: int) -> Dict:
    """Model payload bandwidth and detection latency for other heartbeat intervals."""
    tradeoff = {}
    for interval in intervals:
        tradeoff[str(interval)] = {
            "interval_seconds": interval,
            "payload_bytes_per_sec": num_workers * bytes_per_heartbeat / interval,
            # Last heartbeat can be up to one interval before the failure; the
            # detector then needs the timeout plus up to one check interval.
            "min_detection_latency": HEARTBEAT_TIMEOUT - interval,
            "max_detection_latency": HEARTBEAT_TIMEOUT + DETECTOR_CHECK_INTERVAL,
        }
    return tradeoff


def main():
    print("=" * 60)
    print("Network Overhead Benchmark")
    print("=" * 60)

    num_workers = 4
    payload = heartbeat_payload_bytes()
    bytes_per_heartbeat = payload["request"] + payload["response"]
    measured = measure_heartbeat_rate(num_workers=num_workers, num_iterations=30)

    results = {
        "system_info": get_system_info(),
        "config": {
            "num_workers": num_workers,
            "heartbeat_timeout": HEARTBEAT_TIMEOUT,
            "detector_check_interval": DETECTOR_CHECK_INTERVAL,
        },
        "heartbeat_payload_bytes": payload,
        "measured": {
            **measured,
            "payload_bytes_per_sec": (measured["total_heartbeats"] * bytes_per_heartbeat
                                      / measured["total_time"]) if measured["total_time"] else 0,
        },
        "modelled_tradeoff": model_tradeoff([0.5, 1.0, 2.0, 5.0], num_workers, bytes_per_heartbeat),
    }

    save_results("network_overhead.json", results)

    print(f"\nHeartbeat payload: {payload['request']} B request + {payload['response']} B response")
    print(f"Measured: {measured['heartbeats_per_worker_per_sec']:.2f} heartbeats/s per worker, "
          f"{results['measured']['payload_bytes_per_sec']:.0f} B/s payload across {num_workers} workers")

    print(f"\n{'Interval':<12} {'Payload B/s':<15} {'Detection latency':<20}")
    print("-" * 50)
    for row in results["modelled_tradeoff"].values():
        print(f"{row['interval_seconds']:<12.1f} {row['payload_bytes_per_sec']:<15.0f} "
              f"{row['min_detection_latency']:.1f}-{row['max_detection_latency']:.1f}s")

    return results


if __name__ == "__main__":
    main()

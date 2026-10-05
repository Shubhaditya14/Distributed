# Distributed Training Orchestrator - Benchmark Report

*Generated: 2026-10-05 15:52:53*
*Environment: macOS-26.3-arm64-arm-64bit, Python 3.10.19, 10 CPUs*

---

## Summary

This benchmark suite evaluates the **Distributed Training Orchestrator**, a gRPC
master that coordinates PyTorch DDP worker processes with heartbeat-based failure
detection and checkpoint-based recovery.

All runs use worker processes on a single machine (CPU, Gloo backend) training a
2,817-parameter MLP on synthetic data, with a fixed 0.5s sleep per iteration.
The numbers characterise the coordination layer, not training performance.

| Metric | Value | Notes |
|--------|-------|-------|
| **Recovery Time** | 18.6s mean (18.1-19.1s) | Worker kill to training resumed, 5 trials, scripted restart |
| **Failure Detection** | 15.6s mean (15.2-16.1s) | 15s heartbeat timeout checked every 5s |
| **Checkpoint Overhead** | +0.4% | Default 10-iteration interval vs a 0.5s/iteration baseline; within run-to-run noise |
| **Per-Iteration Slowdown** | +1.7% | 4 workers vs 1 worker, same iterations per worker |
| **Failure Scenario Tests** | 2/3 passed | See below |

---

## Architecture Overview

### Components

1. **Master (Orchestrator)**
   - gRPC server on port 50051
   - Handles worker registration, heartbeats, checkpoint signalling
   - Failure detection via heartbeat timeout
   - On failure: enters recovery mode and resets the worker registry

2. **Workers**
   - Register with master, receive rank assignment
   - PyTorch DDP for gradient synchronization
   - Checkpoint saving on master request
   - Load the last complete checkpoint when started in recovery mode

Restarting workers after a failure is done outside the master
(`scripts/restart_workers.sh`, or the benchmark harness).

### Communication Layers

| Layer | Protocol | Purpose |
|-------|----------|---------|
| Coordination | gRPC | Registration, heartbeats, checkpoint signals |
| Training | PyTorch DDP (gloo) | Gradient AllReduce |

---

## Benchmark 1: Recovery Time Analysis

### Methodology
One worker is killed with SIGKILL shortly after a checkpoint completes. The
harness waits for the master to report the failure, then immediately kills the
remaining workers and starts new ones (the job `scripts/restart_workers.sh`
does by hand, without its fixed 3s wait or any human reaction time).

### Results

| Phase | Mean | Min | Max | Description |
|-------|------|-----|-----|-------------|
| Failure Detection | 15.65s | 15.17s | 16.08s | Worker kill to master reporting the failure |
| Worker Restart | 2.32s | 2.27s | 2.41s | Kill survivors, start new workers, re-register |
| DDP Init + Checkpoint Load | 0.68s | 0.51s | 1.27s | Barrier, process group setup, state restoration |
| **Total** | 18.65s | 18.13s | 19.09s | Worker kill to training resumed |

5 of 5 trials completed; in 5 of them every worker resumed from
the last complete checkpoint.

### Detection Latency Bounds

The detector thread wakes every 5s and flags any worker whose last
heartbeat is older than 15s, so a failure is detected between
**15s and 20s** after the worker's last heartbeat. Detection dominates recovery time.

The harness starts each trial the same way, so the kill lands at a similar point in the
detector's 5s cycle every time; the measured spread is narrower than these bounds.

---

## Benchmark 2: Checkpoint Overhead

### Methodology
Compared training-loop wall time at different checkpoint frequencies against a
run with checkpointing disabled.

### Results

4 workers, 50 iterations, 2 trials per setting.

| Frequency | Checkpoint Files | Total Time | Overhead |
|-----------|------------------|------------|----------|
| None (baseline) | 0 | 26.47s | - |
| Every 25 iterations | 8 | 26.42s | -0.21% |
| Every 10 iterations | 20 | 26.57s | +0.35% |
| Every 5 iterations | 40 | 26.50s | +0.09% |

Each checkpoint file is about 14 KB (model + optimizer state). A checkpoint costs one
`torch.save` plus one extra heartbeat RPC per worker.

### Interpreting the Percentages
The baseline is dominated by the worker's fixed 0.5s sleep per iteration and the model
is tiny, so checkpoint cost is too small to resolve: differences of a fraction of a
percent (including negative ones) are run-to-run noise. These figures say nothing about
checkpoint overhead for a real model.

---

## Benchmark 3: Scalability Analysis

### Methodology
Ran the same number of iterations with 1, 2, and 4 workers and measured
training-loop wall time. Every worker runs the full iteration count, so this is
a weak-scaling measurement: adding workers adds samples per iteration rather
than shortening the run.

### Results

| Workers | Iteration Time | Slowdown vs 1 Worker | Samples/sec | Weak-Scaling Efficiency |
|---------|----------------|----------------------|-------------|-------------------------|
| 1 | 522.9 ms | +0.0% | 61 | 100.0% |
| 2 | 524.2 ms | +0.2% | 122 | 99.8% |
| 4 | 531.9 ms | +1.7% | 241 | 98.3% |

### Caveats
- No wall-clock speedup is measured or claimed: the run takes the same number of iterations at any worker count.
- Each iteration includes a fixed 0.5s sleep, which hides most of the gRPC and AllReduce cost
  and inflates the efficiency figure.
- All workers seed their batch from the iteration number, so they process identical data,
  not disjoint shards. Samples/sec counts those duplicate samples.

---

## Benchmark 4: Network Overhead

### Methodology
Workers send one heartbeat per training iteration; the interval is not separately
configurable. Heartbeat rate was measured from a real run, and message size is the
serialized protobuf payload (HTTP/2 and gRPC framing excluded).

### Results

| Metric | Value |
|--------|-------|
| Heartbeat request payload | 25 bytes |
| Heartbeat response payload | 2 bytes |
| Heartbeat rate | 1.87 per worker per second |
| Payload bandwidth (4 workers) | 201 bytes/s |

### Modelled Heartbeat Interval Tradeoff

Calculated from the payload size and the detector constants, not measured:

| Interval | Payload Bandwidth | Detection Latency After Failure |
|----------|-------------------|---------------------------------|
| 0.5s | 216 bytes/s | 14.5-20.0s |
| 1.0s | 108 bytes/s | 14.0-20.0s |
| 2.0s | 54 bytes/s | 13.0-20.0s |
| 5.0s | 22 bytes/s | 10.0-20.0s |

Heartbeat bandwidth is negligible at any of these intervals; detection latency is set
almost entirely by the 15s timeout and 5s check interval, not by heartbeat frequency.

---

## Failure Scenario Tests

Each test restarts the workers itself once the master reports a failure, and
passes only if every listed check holds.

### Test 1: Single Worker Failure

**Scenario**: Kill worker 0 after the iteration-30 checkpoint; all workers should resume from it and finish training.

- ✅ `failure_detected`
- ✅ `all_workers_resumed_from_checkpoint`
- ✅ `training_completed`

**Result**: PASSED

### Test 2: Multiple Sequential Failures

**Scenario**: Kill worker 0 after checkpoint 20 and worker 1 after checkpoint 50 in the same run; both recoveries should resume from the last complete checkpoint.

- ✅ `failure_1_detected`
- ✅ `failure_1_resumed_from_checkpoint`
- ✅ `failure_2_detected`
- ✅ `failure_2_resumed_from_checkpoint`
- ✅ `training_completed`

**Result**: PASSED

### Test 3: Checkpoint Corruption Handling

**Scenario**: Corrupt checkpoint 30 on every rank, then kill a worker; workers should fall back to checkpoint 20.

- ✅ `failure_detected`
- ❌ `fell_back_to_previous_checkpoint`
- ❌ `training_completed`
- Worker output: `Error during training: pickle data was truncated`

**Result**: FAILED

Corruption fallback is not implemented: `load_checkpoint` has no retry with an earlier checkpoint, so a worker that cannot load its checkpoint exits.

---

## Limitations

1. **Manual restart**: the master detects failures but does not restart workers; an operator or script must.
2. **Detection latency**: 15-20 seconds after the last heartbeat (15s timeout, 5s check interval).
3. **Full restart**: one worker failure requires restarting all workers; iterations after the last complete checkpoint are redone.
4. **No corruption fallback**: an unreadable checkpoint stops the worker instead of falling back.
5. **Single master**: the master is a single point of failure, and its recovery state lives in memory.
6. **Single machine**: workers connect to `localhost`; multi-host operation is untested.
7. **Fixed cluster size**: the worker count is fixed at master startup (default 4).
8. **No security**: insecure gRPC channel, no authentication or encryption.
9. **Synthetic workload**: tiny MLP on random data with a 0.5s sleep per iteration; results do not predict real training performance.

---

## Appendix: Running the Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Run individual benchmarks
cd benchmarks
python benchmark_recovery_time.py
python benchmark_checkpoint_overhead.py
python benchmark_scalability.py
python benchmark_network_overhead.py

# Run failure tests
python test_failures.py

# Generate this report
python generate_report.py
```

Or run everything with `python run_all_benchmarks.py --all`.

---

*This report was generated by `benchmarks/generate_report.py` from `benchmarks/results/*.json`.*

# Distributed Training Orchestrator

A **fault-tolerant training orchestration prototype** built with gRPC and PyTorch DDP (Distributed Data Parallel). A master process coordinates DDP workers with heartbeat-based failure detection, coordinated checkpointing, and checkpoint-based recovery after a worker restart.

Everything runs as processes on a single machine (CPU, Gloo backend) with a small synthetic workload. It demonstrates the coordination protocol; it is not a production system.

---

## Measured Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Recovery Time** | 18.6s mean (18.1-19.1s) | Worker kill to training resumed, 5 trials, scripted restart |
| **Failure Detection** | 15.6s mean, 15-20s by design | 15s heartbeat timeout checked every 5s |
| **Checkpoint Overhead** | Below measurement noise (<0.5%) | 14 KB checkpoints against a 0.5s/iteration synthetic workload |
| **Per-Iteration Slowdown** | +1.7% at 4 workers vs 1 | Weak scaling; no wall-clock speedup is claimed |
| **Failure Scenario Tests** | 2 of 3 pass | Corruption fallback is not implemented |

Measured on one machine (macOS arm64, 10 CPUs, Python 3.10) with 4 worker processes.

Numbers come from `benchmarks/results/*.json`; see [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for methodology and caveats.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Master (port 50051)                              │
│                                                                          │
│   - Worker registry & rank assignment                                   │
│   - Heartbeat-based failure detection (15s timeout)                     │
│   - Checkpoint signalling & completion tracking                         │
│   - Recovery mode (which checkpoint to resume from)                     │
│   - Live training dashboard                                             │
└─────────────────────────────────────────────────────────────────────────┘
          ▲                 ▲                 ▲                 ▲
          │ gRPC            │ gRPC            │ gRPC            │ gRPC
          │ Heartbeats      │ Heartbeats      │ Heartbeats      │ Heartbeats
          ▼                 ▼                 ▼                 ▼
     ┌─────────┐       ┌─────────┐       ┌─────────┐       ┌─────────┐
     │ Worker  │       │ Worker  │       │ Worker  │       │ Worker  │
     │ Rank 0  │       │ Rank 1  │       │ Rank 2  │       │ Rank 3  │
     └─────────┘       └─────────┘       └─────────┘       └─────────┘
          ▲                 ▲                 ▲                 ▲
          └─────────────────┴────────┬────────┴─────────────────┘
                                     │
                         PyTorch DDP (port 29500)
                         Gloo Backend - AllReduce
                         Gradient Synchronization
```

### Communication Layers

| Layer | Protocol | Port | Purpose |
|-------|----------|------|---------|
| **Coordination** | gRPC | 50051 | Registration, heartbeats, checkpoint signals |
| **Training** | PyTorch DDP (Gloo) | 29500 | Gradient AllReduce synchronization |

Both layers use `localhost`; the addresses are hardcoded in `src/worker.py`.

---

## Components

### Master (Orchestrator Server)
**Location:** `src/master.py`

- Manages worker registry and automatic rank assignment (0-3 with the default 4 workers)
- Monitors worker health via heartbeat processing
- Runs failure detector thread (5s check interval, 15s timeout)
- Signals checkpoints and tracks which workers have completed them
- On failure: enters recovery mode and resets the registry so restarted workers can re-register
- Prints a training dashboard with per-worker status on every heartbeat

The master does **not** kill or restart workers. After it reports a failure, run `scripts/restart_workers.sh`.

### Workers
**Location:** `src/worker.py`

- Register with master and receive unique rank
- Block at a barrier until all workers have registered
- Initialize PyTorch DDP process group
- Execute training loop, sending one heartbeat per iteration
- Save a checkpoint when the master's heartbeat response requests one
- Load the last complete checkpoint when started while the master is in recovery mode

### Model
- 3-layer feedforward neural network (10 -> 64 -> 32 -> 1), 2,817 parameters
- Wrapped with `DistributedDataParallel` for gradient sync
- MSE loss with SGD optimizer (lr 0.01)
- Synthetic random data, batch size 32, with a fixed 0.5s sleep per iteration
- All workers seed their batch from the iteration number, so they train on identical data

### Configuration

Set through environment variables:

| Variable | Read by | Default | Purpose |
|----------|---------|---------|---------|
| `EXPECTED_WORKERS` | master | 4 | Workers required before training starts |
| `CHECKPOINT_INTERVAL` | master | 10 | Iterations between checkpoints |
| `NUM_ITERATIONS` | worker | 100 | Training iterations |

The shell scripts always launch 4 workers.

---

## Fault Tolerance

### Failure Detection
- **Mechanism:** Heartbeat timeout
- **Check Interval:** Every 5 seconds
- **Timeout:** 15 seconds (hardcoded in `src/master.py`)
- **Detection Latency:** 15-20 seconds after the worker's last heartbeat

### Checkpoint System
- **Frequency:** `CHECKPOINT_INTERVAL` (default: every 10 iterations)
- **Storage:** `/tmp/checkpoints/rank_{rank}/checkpoint_iter_{iteration}.pt`
- **Contents:** Model state, optimizer state, iteration number
- **Coordination:** Each worker is told to checkpoint on its heartbeat for the checkpoint iteration (DDP keeps workers in lockstep) and acknowledges on its next heartbeat. A checkpoint counts as complete, and becomes the recovery point, only once every worker has acknowledged it.

### Recovery Flow

```
1. Master detects missing heartbeat (15s timeout)
           ↓
2. Master enters recovery mode and clears the worker registry
           ↓
3. Operator runs restart_workers.sh:
   kills remaining workers, starts 4 new ones
           ↓
4. Workers re-register with master
           ↓
5. Workers call GetRecoveryInfo() for the last complete checkpoint
           ↓
6. Workers load that checkpoint and resume training
```

Surviving workers do not exit on their own when a peer dies (they block in AllReduce), so all workers are restarted together.

### Measured Recovery Phases

| Phase | Mean | Range | Description |
|-------|------|-------|-------------|
| Failure Detection | 15.6s | 15.2-16.1s | Worker kill to master reporting the failure |
| Worker Restart | 2.3s | 2.3-2.4s | Kill survivors, start new workers, re-register |
| DDP Init + Checkpoint Load | 0.7s | 0.5-1.3s | Barrier, process group setup, state restoration |
| **Total** | **18.6s** | 18.1-19.1s | Worker kill to training resumed |

Five trials with a scripted restart that reacts the moment the master reports the failure. `restart_workers.sh` adds a fixed 3s wait, and a human adds reaction time. Detection can take up to 20s depending on where the failure falls in the detector's 5s cycle.

---

## Benchmark Results

### Scalability (weak scaling)

Every worker runs the same number of iterations, so adding workers does not shorten the run. The table shows how per-iteration time grows.

| Workers | Iteration Time | Slowdown vs 1 Worker | Samples/sec |
|---------|----------------|----------------------|-------------|
| 1 | 522.9 ms | - | 61 |
| 2 | 524.2 ms | +0.2% | 122 |
| 4 | 531.9 ms | +1.7% | 241 |

About 500 ms of each iteration is the worker's fixed sleep, and all workers process identical batches, so these numbers show only that coordination cost is small next to that sleep.

### Checkpoint Overhead

4 workers, 50 iterations, 2 trials per setting.

| Frequency | Checkpoint Files | Total Time | Overhead |
|-----------|------------------|------------|----------|
| None (baseline) | 0 | 26.47s | - |
| Every 25 iterations | 8 | 26.42s | -0.21% |
| Every 10 iterations | 20 | 26.57s | +0.35% |
| Every 5 iterations | 40 | 26.50s | +0.09% |

The differences are run-to-run noise: a 14 KB checkpoint is too cheap to resolve against a 0.5s/iteration baseline.

### Heartbeat Overhead

| Metric | Value |
|--------|-------|
| Heartbeat payload | 25 bytes request, 2 bytes response (protobuf only, no gRPC framing) |
| Heartbeat rate | 1.87 per worker per second (one per iteration) |
| Payload bandwidth, 4 workers | 201 bytes/s |

### Failure Scenario Tests

| Test | Scenario | Result |
|------|----------|--------|
| Single Worker Failure | Kill worker 0 after checkpoint 30; all workers resume from iteration 30 and finish | PASSED |
| Multiple Sequential Failures | Kill worker 0 after checkpoint 20, worker 1 after checkpoint 50; resume from 20, then 50, and finish | PASSED |
| Checkpoint Corruption | Corrupt checkpoint 30 on all ranks, then kill a worker | FAILED: workers exit with a load error, no fallback to iteration 20 |

---

## Project Structure

```
distributed/
├── protos/
│   └── orchestrator.proto          # gRPC service definitions
├── src/
│   ├── master.py                   # Orchestrator server
│   ├── worker.py                   # DDP training worker
│   ├── orchestrator_pb2.py         # Generated protobuf messages
│   └── orchestrator_pb2_grpc.py    # Generated gRPC stubs
├── scripts/
│   ├── run.sh                      # Launch master + 4 workers
│   ├── kill_worker.sh              # Simulate worker failure
│   ├── restart_workers.sh          # Restart workers after failure
│   └── generate_proto.sh           # Regenerate proto files
├── benchmarks/
│   ├── results/                    # Benchmark output JSON (gitignored)
│   ├── benchmark_*.py              # Benchmark scripts
│   ├── test_failures.py            # Failure scenario tests
│   ├── generate_report.py          # Builds BENCHMARK_REPORT.md from results/
│   ├── run_all_benchmarks.py       # Runs everything
│   └── utils.py                    # Shared process management helpers
├── compare_checkpoints.py          # Demo: diff two checkpoint files (uses dummy data)
├── requirements.txt
├── futurescope.md                  # Roadmap ideas
├── BENCHMARK_REPORT.md             # Detailed benchmark report
└── README.md
```

---

## gRPC Service Definition

| RPC Method | Request | Response | Purpose |
|------------|---------|----------|---------|
| `Register` | worker_id | assigned_rank, world_size, success, message | Worker joins cluster |
| `SendHeartbeat` | worker_id, timestamp, current_iteration, current_loss, is_training, checkpointed_iteration | acknowledged, should_checkpoint | Progress updates, checkpoint signal and acknowledgment |
| `CanStartTraining` | worker_id | ready, world_size | Barrier synchronization (blocks until all workers register) |
| `GetRecoveryInfo` | worker_id | in_recovery_mode, checkpoint_iteration | Recovery state query |

---

## Setup

### 1. Create Environment
The scripts activate a conda environment named `myproject`:
```bash
conda create -n myproject python=3.10
conda activate myproject
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Proto Files (if needed)
```bash
./scripts/generate_proto.sh
```

---

## Usage

### Run Everything (Master + 4 Workers)
```bash
./scripts/run.sh
```
This deletes `/tmp/checkpoints` and starts from scratch.

### Run Manually
```bash
# Terminal 1: Start Master
cd src && python master.py

# Terminals 2-5: Start Workers
cd src && python worker.py
cd src && python worker.py
cd src && python worker.py
cd src && python worker.py
```
Start all workers within about 15 seconds of each other; see Limitations.

### Simulate Failure
```bash
# Kill a worker by index 0-3 (default: 0)
./scripts/kill_worker.sh 1

# Once the master prints "Failure Detected", restart all workers
./scripts/restart_workers.sh
```

### Run Benchmarks
```bash
cd benchmarks
python benchmark_recovery_time.py
python benchmark_checkpoint_overhead.py
python benchmark_scalability.py
python benchmark_network_overhead.py
python test_failures.py
python generate_report.py        # rebuild BENCHMARK_REPORT.md
```
No cluster may be running while benchmarks run; they use the same ports.

---

## Training Flow

1. **Master Startup** - gRPC server on port 50051, spawns failure detector
2. **Worker Registration** - Each worker gets assigned rank (0-3)
3. **Barrier Wait** - Workers block in `CanStartTraining()` until all have registered
4. **DDP Init** - Workers initialize distributed process group
5. **Recovery Check** - Workers query `GetRecoveryInfo()` for resume point
6. **Training Loop**:
   - Generate batch -> Forward -> Loss -> Backward
   - DDP AllReduce synchronizes gradients
   - Send heartbeat with iteration/loss
   - Checkpoint if signaled by master
7. **Live Dashboard** - Master prints worker status on every heartbeat

---

## Limitations & Future Work

### Current Limitations
- **Manual Restart:** The master detects failures but an operator must run `restart_workers.sh`
- **Full Restart:** One failed worker means restarting all workers
- **Detection Latency:** 15-20 second failure detection
- **Re-computation:** Iterations after the last complete checkpoint are redone
- **No Corruption Fallback:** A worker that cannot read its checkpoint exits instead of trying an earlier one
- **False Positives:** Any worker silent for 15s is treated as failed. This includes workers waiting at the barrier (if the others take too long to start) and workers that have finished training, so the master reports a "failure" about 15-20s after a run completes
- **Single Master:** Master is a single point of failure; recovery state is in memory (`/tmp/orchestrator_state.json` is written but never read back)
- **Single Machine:** Workers connect to `localhost`; multi-host operation is untested
- **Fixed Cluster Size:** Worker count is fixed at master startup
- **No Security:** Insecure gRPC channel, no authentication or encryption
- **Synthetic Workload:** Tiny model, random data, no data sharding across workers

### Potential Improvements
- Automatic worker restart by the master
- Faster failure detection via TCP keepalive or gossip protocol
- Async checkpointing overlapped with training
- Checkpoint validation with fallback to earlier checkpoints
- Master redundancy for production use
- Elastic scaling for dynamic worker addition/removal
- TLS/mTLS for secure gRPC communication

See [futurescope.md](futurescope.md) for a longer roadmap.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- gRPC 1.50+
- Unix-like OS (for shell scripts)

---

## License

MIT License

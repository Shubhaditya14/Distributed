"""
Generate benchmark report in Markdown format.

Creates BENCHMARK_REPORT.md from the JSON files in benchmarks/results/.
Every number in the report is read from those files; sections whose
benchmark has not been run say so instead of showing placeholder values.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
REPORT_PATH = PROJECT_ROOT / "BENCHMARK_REPORT.md"

NOT_RUN = "*No results found. Run the benchmark to populate this section.*\n"


def load_results(filename: str) -> Optional[Dict]:
    """Load results from JSON file."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def generate_summary() -> str:
    """Generate summary section from whichever results exist."""
    recovery = load_results("recovery_time_analysis.json")
    checkpoint = load_results("checkpoint_overhead.json")
    scalability = load_results("scalability_metrics.json")
    tests = load_results("failure_tests.json")

    rows = []
    if recovery and recovery.get("aggregate_stats"):
        stats = recovery["aggregate_stats"]
        total, detection = stats["total_recovery_time"], stats["detection_latency"]
        rows.append(f"| **Recovery Time** | {total['mean']:.1f}s mean ({total['min']:.1f}-{total['max']:.1f}s) "
                    f"| Worker kill to training resumed, {total['count']} trials, scripted restart |")
        rows.append(f"| **Failure Detection** | {detection['mean']:.1f}s mean ({detection['min']:.1f}-{detection['max']:.1f}s) "
                    f"| 15s heartbeat timeout checked every 5s |")
    if checkpoint and "10" in checkpoint.get("summary", {}):
        stats = checkpoint["summary"]["10"]
        rows.append(f"| **Checkpoint Overhead** | {stats['overhead_percent']:+.1f}% "
                    f"| Default 10-iteration interval vs a 0.5s/iteration baseline; within run-to-run noise |")
    if scalability and "4" in scalability.get("scalability_analysis", {}):
        analysis = scalability["scalability_analysis"]["4"]
        rows.append(f"| **Per-Iteration Slowdown** | {analysis['iteration_slowdown_percent']:+.1f}% "
                    f"| 4 workers vs 1 worker, same iterations per worker |")
    if tests:
        results = tests["tests"].values()
        passed = sum(1 for t in results if t["success"])
        rows.append(f"| **Failure Scenario Tests** | {passed}/{len(results)} passed | See below |")

    table = "\n".join(rows) if rows else "| *(no results)* | | |"

    return f"""## Summary

This benchmark suite evaluates the **Distributed Training Orchestrator**, a gRPC
master that coordinates PyTorch DDP worker processes with heartbeat-based failure
detection and checkpoint-based recovery.

All runs use worker processes on a single machine (CPU, Gloo backend) training a
2,817-parameter MLP on synthetic data, with a fixed 0.5s sleep per iteration.
The numbers characterise the coordination layer, not training performance.

| Metric | Value | Notes |
|--------|-------|-------|
{table}
"""


def generate_architecture_section() -> str:
    """Generate architecture overview section."""
    return """## Architecture Overview

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
"""


def generate_recovery_section() -> str:
    """Generate recovery time analysis section."""
    data = load_results("recovery_time_analysis.json")

    section = """## Benchmark 1: Recovery Time Analysis

### Methodology
One worker is killed with SIGKILL shortly after a checkpoint completes. The
harness waits for the master to report the failure, then immediately kills the
remaining workers and starts new ones (the job `scripts/restart_workers.sh`
does by hand, without its fixed 3s wait or any human reaction time).

### Results

"""
    if not data or not data.get("aggregate_stats"):
        return section + NOT_RUN

    stats = data["aggregate_stats"]
    descriptions = [
        ("detection_latency", "Failure Detection", "Worker kill to master reporting the failure"),
        ("restart_latency", "Worker Restart", "Kill survivors, start new workers, re-register"),
        ("resume_latency", "DDP Init + Checkpoint Load", "Barrier, process group setup, state restoration"),
        ("total_recovery_time", "**Total**", "Worker kill to training resumed"),
    ]
    section += "| Phase | Mean | Min | Max | Description |\n|-------|------|-----|-----|-------------|\n"
    for key, name, description in descriptions:
        s = stats[key]
        section += f"| {name} | {s['mean']:.2f}s | {s['min']:.2f}s | {s['max']:.2f}s | {description} |\n"

    trials = [t for t in data["trials"] if "error" not in t]
    resumed_ok = sum(1 for t in trials if t["all_workers_loaded_checkpoint"])
    bounds = data["detection_latency_bounds"]
    section += f"""
{len(trials)} of {len(data['trials'])} trials completed; in {resumed_ok} of them every worker resumed from
the last complete checkpoint.

### Detection Latency Bounds

The detector thread wakes every {data['config']['detector_check_interval']}s and flags any worker whose last
heartbeat is older than {data['config']['heartbeat_timeout']}s, so a failure is detected between
**{bounds['min']}s and {bounds['max']}s** after the worker's last heartbeat. Detection dominates recovery time.

The harness starts each trial the same way, so the kill lands at a similar point in the
detector's {data['config']['detector_check_interval']}s cycle every time; the measured spread is narrower than these bounds.
"""
    return section


def generate_checkpoint_section() -> str:
    """Generate checkpoint overhead section."""
    data = load_results("checkpoint_overhead.json")

    section = """## Benchmark 2: Checkpoint Overhead

### Methodology
Compared training-loop wall time at different checkpoint frequencies against a
run with checkpointing disabled.

### Results

"""
    if not data or "none" not in data.get("summary", {}):
        return section + NOT_RUN

    config, summary = data["config"], data["summary"]
    section += (f"{config['num_workers']} workers, {config['num_iterations']} iterations, "
                f"{config['num_trials']} trials per setting.\n\n")
    section += "| Frequency | Checkpoint Files | Total Time | Overhead |\n"
    section += "|-----------|------------------|------------|----------|\n"
    section += f"| None (baseline) | 0 | {summary['none']['total_time']['mean']:.2f}s | - |\n"
    for key, stats in summary.items():
        if key == "none":
            continue
        section += (f"| Every {key} iterations | {stats['num_checkpoints']} | {stats['total_time']['mean']:.2f}s "
                    f"| {stats['overhead_percent']:+.2f}% |\n")

    sizes = [s["avg_checkpoint_size_kb"] for k, s in summary.items() if k != "none"]
    section += f"""
Each checkpoint file is about {max(sizes):.0f} KB (model + optimizer state). A checkpoint costs one
`torch.save` plus one extra heartbeat RPC per worker.

### Interpreting the Percentages
The baseline is dominated by the worker's fixed 0.5s sleep per iteration and the model
is tiny, so checkpoint cost is too small to resolve: differences of a fraction of a
percent (including negative ones) are run-to-run noise. These figures say nothing about
checkpoint overhead for a real model.
"""
    return section


def generate_scalability_section() -> str:
    """Generate scalability analysis section."""
    data = load_results("scalability_metrics.json")

    section = """## Benchmark 3: Scalability Analysis

### Methodology
Ran the same number of iterations with 1, 2, and 4 workers and measured
training-loop wall time. Every worker runs the full iteration count, so this is
a weak-scaling measurement: adding workers adds samples per iteration rather
than shortening the run.

### Results

"""
    if not data or not data.get("scalability_analysis"):
        return section + NOT_RUN

    section += "| Workers | Iteration Time | Slowdown vs 1 Worker | Samples/sec | Weak-Scaling Efficiency |\n"
    section += "|---------|----------------|----------------------|-------------|-------------------------|\n"
    for key, analysis in data["scalability_analysis"].items():
        throughput = data["summary"][key]["samples_per_second"]["mean"]
        section += (f"| {analysis['num_workers']} | {analysis['iteration_time_ms']:.1f} ms "
                    f"| {analysis['iteration_slowdown_percent']:+.1f}% | {throughput:.0f} "
                    f"| {analysis['weak_scaling_efficiency_percent']:.1f}% |\n")

    section += """
### Caveats
- No wall-clock speedup is measured or claimed: the run takes the same number of iterations at any worker count.
- Each iteration includes a fixed 0.5s sleep, which hides most of the gRPC and AllReduce cost
  and inflates the efficiency figure.
- All workers seed their batch from the iteration number, so they process identical data,
  not disjoint shards. Samples/sec counts those duplicate samples.
"""
    return section


def generate_network_section() -> str:
    """Generate network overhead section."""
    data = load_results("network_overhead.json")

    section = """## Benchmark 4: Network Overhead

### Methodology
Workers send one heartbeat per training iteration; the interval is not separately
configurable. Heartbeat rate was measured from a real run, and message size is the
serialized protobuf payload (HTTP/2 and gRPC framing excluded).

### Results

"""
    if not data or "measured" not in data:
        return section + NOT_RUN

    payload, measured = data["heartbeat_payload_bytes"], data["measured"]
    section += f"""| Metric | Value |
|--------|-------|
| Heartbeat request payload | {payload['request']} bytes |
| Heartbeat response payload | {payload['response']} bytes |
| Heartbeat rate | {measured['heartbeats_per_worker_per_sec']:.2f} per worker per second |
| Payload bandwidth ({measured['num_workers']} workers) | {measured['payload_bytes_per_sec']:.0f} bytes/s |

### Modelled Heartbeat Interval Tradeoff

Calculated from the payload size and the detector constants, not measured:

| Interval | Payload Bandwidth | Detection Latency After Failure |
|----------|-------------------|---------------------------------|
"""
    for row in data["modelled_tradeoff"].values():
        section += (f"| {row['interval_seconds']}s | {row['payload_bytes_per_sec']:.0f} bytes/s "
                    f"| {row['min_detection_latency']:.1f}-{row['max_detection_latency']:.1f}s |\n")

    section += """
Heartbeat bandwidth is negligible at any of these intervals; detection latency is set
almost entirely by the 15s timeout and 5s check interval, not by heartbeat frequency.
"""
    return section


def generate_failure_tests_section() -> str:
    """Generate failure scenario tests section."""
    data = load_results("failure_tests.json")

    section = """## Failure Scenario Tests

Each test restarts the workers itself once the master reports a failure, and
passes only if every listed check holds.

"""
    if not data:
        return section + NOT_RUN

    scenarios = {
        "single_failure": ("Test 1: Single Worker Failure",
                           "Kill worker 0 after the iteration-30 checkpoint; all workers should resume "
                           "from it and finish training."),
        "multiple_failures": ("Test 2: Multiple Sequential Failures",
                              "Kill worker 0 after checkpoint 20 and worker 1 after checkpoint 50 in the "
                              "same run; both recoveries should resume from the last complete checkpoint."),
        "checkpoint_corruption": ("Test 3: Checkpoint Corruption Handling",
                                  "Corrupt checkpoint 30 on every rank, then kill a worker; workers should "
                                  "fall back to checkpoint 20."),
    }

    for key, (title, scenario) in scenarios.items():
        result = data["tests"].get(key)
        if result is None:
            continue
        section += f"### {title}\n\n**Scenario**: {scenario}\n\n"
        for check, ok in result.get("checks", {}).items():
            section += f"- {'✅' if ok else '❌'} `{check}`\n"
        if result.get("error"):
            section += f"- ❌ error: {result['error']}\n"
        for error in result.get("worker_errors", []):
            section += f"- Worker output: `{error}`\n"
        section += f"\n**Result**: {'PASSED' if result['success'] else 'FAILED'}\n\n"

    corruption = data["tests"].get("checkpoint_corruption")
    if corruption and not corruption["success"]:
        section += ("Corruption fallback is not implemented: `load_checkpoint` has no retry with an "
                    "earlier checkpoint, so a worker that cannot load its checkpoint exits.\n")
    return section


def generate_conclusions() -> str:
    """Generate conclusions section."""
    return """## Limitations

1. **Manual restart**: the master detects failures but does not restart workers; an operator or script must.
2. **Detection latency**: 15-20 seconds after the last heartbeat (15s timeout, 5s check interval).
3. **Full restart**: one worker failure requires restarting all workers; iterations after the last complete checkpoint are redone.
4. **No corruption fallback**: an unreadable checkpoint stops the worker instead of falling back.
5. **Single master**: the master is a single point of failure, and its recovery state lives in memory.
6. **Single machine**: workers connect to `localhost`; multi-host operation is untested.
7. **Fixed cluster size**: the worker count is fixed at master startup (default 4).
8. **No security**: insecure gRPC channel, no authentication or encryption.
9. **Synthetic workload**: tiny MLP on random data with a 0.5s sleep per iteration; results do not predict real training performance.
"""


def generate_report():
    """Generate the complete benchmark report."""
    print("=" * 60)
    print("Generating Benchmark Report")
    print("=" * 60)

    system_info = (load_results("recovery_time_analysis.json") or {}).get("system_info", {})
    environment = (f"*Environment: {system_info['platform']}, Python {system_info['python_version']}, "
                   f"{system_info['cpu_count']} CPUs*\n" if system_info else "")

    sections = [
        generate_summary(),
        generate_architecture_section(),
        generate_recovery_section(),
        generate_checkpoint_section(),
        generate_scalability_section(),
        generate_network_section(),
        generate_failure_tests_section(),
        generate_conclusions(),
    ]
    body = "\n---\n\n".join(sections)

    report = f"""# Distributed Training Orchestrator - Benchmark Report

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
{environment}
---

{body}
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
"""

    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    print(f"Report saved to: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    generate_report()

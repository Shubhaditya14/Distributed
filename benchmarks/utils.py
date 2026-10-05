"""Utility functions for benchmarking distributed training orchestrator."""

import os
import re
import sys
import json
import time
import signal
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
VIZ_DIR = PROJECT_ROOT / "benchmarks" / "visualizations"
MODELS_DIR = PROJECT_ROOT / "benchmarks" / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ProcessManager:
    """Manages master and worker processes for benchmarking.

    Process output goes to per-process log files (never a pipe: the master
    prints a dashboard on every heartbeat and would block on a full pipe).
    Timings are taken by polling those logs.
    """

    POLL_INTERVAL = 0.02

    def __init__(self, num_workers: int = 4, checkpoint_interval: int = 10,
                 num_iterations: int = 100):
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        self.num_iterations = num_iterations
        self.master_process: Optional[subprocess.Popen] = None
        self.worker_processes: List[subprocess.Popen] = []
        self.log_dir = Path(tempfile.mkdtemp(prefix="orchestrator_bench_"))
        self.master_log = self.log_dir / "master.log"
        self.worker_logs: List[Path] = []
        self.generation = 0
        self._log_files = []

    def _spawn(self, script: str, log_path: Path) -> subprocess.Popen:
        env = os.environ.copy()
        env["EXPECTED_WORKERS"] = str(self.num_workers)
        env["CHECKPOINT_INTERVAL"] = str(self.checkpoint_interval)
        env["NUM_ITERATIONS"] = str(self.num_iterations)
        env["PYTHONUNBUFFERED"] = "1"

        log_file = open(log_path, "w")
        self._log_files.append(log_file)
        return subprocess.Popen(
            [sys.executable, str(SRC_DIR / script)],
            cwd=str(SRC_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    def start_master(self) -> int:
        """Start the master process and return its PID."""
        self.master_process = self._spawn("master.py", self.master_log)
        if self.wait_for_log([self.master_log], "Orchestrator server started", timeout=30) is None:
            raise RuntimeError(f"Master failed to start, see {self.master_log}")
        return self.master_process.pid

    def start_workers(self) -> List[int]:
        """Start a fresh generation of worker processes and return their PIDs."""
        self.generation += 1
        self.worker_processes = []
        self.worker_logs = []

        for i in range(self.num_workers):
            log_path = self.log_dir / f"worker_gen{self.generation}_{i}.log"
            self.worker_logs.append(log_path)
            self.worker_processes.append(self._spawn("worker.py", log_path))
            time.sleep(0.5)

        return [p.pid for p in self.worker_processes]

    def kill_worker(self, index: int = 0) -> Tuple[int, float]:
        """Kill a specific worker and return (PID, timestamp)."""
        if index >= len(self.worker_processes):
            raise ValueError(f"Worker index {index} out of range")

        pid = self.worker_processes[index].pid
        kill_time = time.time()
        os.kill(pid, signal.SIGKILL)
        return pid, kill_time

    def kill_all_workers(self):
        """Kill all workers and wait for them to exit."""
        for proc in self.worker_processes:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        self.worker_processes = []

    def stop_all(self):
        """Stop all processes."""
        self.kill_all_workers()

        if self.master_process:
            try:
                self.master_process.kill()
                self.master_process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        self.master_process = None

        for log_file in self._log_files:
            log_file.close()
        self._log_files = []

    def wait_for_log(self, log_paths: List[Path], pattern: str,
                     timeout: float = 120, after: int = 0) -> Optional[float]:
        """Wait until `pattern` appears in every log; return the timestamp, or None on timeout.

        `after` skips that many leading characters (used for the master log,
        which spans several worker generations).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(pattern in read_log(path)[after:] for path in log_paths):
                return time.time()
            time.sleep(self.POLL_INTERVAL)
        return None

    def wait_for_completion(self, timeout: float = 300) -> bool:
        """Wait for workers to exit; True only if every worker finished training."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(p.poll() is not None for p in self.worker_processes):
                break
            time.sleep(self.POLL_INTERVAL)
        return all("Training completed!" in read_log(path) for path in self.worker_logs)

    def master_log_size(self) -> int:
        return len(read_log(self.master_log))

    def last_complete_checkpoint(self) -> int:
        """Last iteration the master reported as checkpointed by all workers."""
        matches = re.findall(r"All workers completed checkpoint for iteration (\d+)",
                             read_log(self.master_log))
        return int(matches[-1]) if matches else 0

    def loaded_checkpoints(self) -> List[Optional[int]]:
        """Checkpoint iteration each current worker resumed from (None if it did not)."""
        loaded = []
        for path in self.worker_logs:
            match = re.search(r"Loaded checkpoint from iteration (\d+)", read_log(path))
            loaded.append(int(match.group(1)) if match else None)
        return loaded


def read_log(path: Path) -> str:
    """Read a log file, returning '' if it does not exist yet."""
    try:
        return Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return ""


class BenchmarkTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


def save_results(filename: str, data: Dict):
    """Save benchmark results to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")


def load_results(filename: str) -> Dict:
    """Load benchmark results from JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'r') as f:
        return json.load(f)


def get_system_info() -> Dict:
    """Get system information for reproducibility."""
    import platform

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def clean_checkpoints():
    """Clean up checkpoint directories."""
    checkpoint_dir = Path("/tmp/checkpoints")
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)

    state_file = Path("/tmp/orchestrator_state.json")
    if state_file.exists():
        state_file.unlink()


def calculate_stats(values: List[float]) -> Dict:
    """Calculate statistics for a list of values."""
    import statistics
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "count": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "count": len(values)
    }

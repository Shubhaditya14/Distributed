#!/bin/bash

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
PID_FILE="$PROJECT_ROOT/.pids"

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate myproject 2>/dev/null

# Clean up any existing checkpoints and state for fresh start
rm -rf /tmp/checkpoints
rm -f /tmp/orchestrator_state.json

# Change to src directory
cd "$SRC_DIR"

# Start master in background
echo "Starting master..."
python master.py &
MASTER_PID=$!
sleep 3

# Start 4 workers in background
echo "Starting 4 workers..."
WORKER_PIDS=()
for i in {1..4}; do
    python worker.py &
    WORKER_PIDS+=($!)
    sleep 0.5
done

# Save PIDs to file for recovery scripts
echo "MASTER_PID=$MASTER_PID" > "$PID_FILE"
echo "WORKER_PIDS=(${WORKER_PIDS[@]})" >> "$PID_FILE"

echo ""
echo "Master PID: $MASTER_PID"
echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo "PIDs saved to: $PID_FILE"
echo ""
echo "To simulate failure:  ./scripts/kill_worker.sh <worker_index>"
echo "To restart workers:   ./scripts/restart_workers.sh"
echo ""
echo "Press Ctrl+C to stop all processes..."

# Handle Ctrl+C - kill all processes
cleanup() {
    echo ""
    echo "Stopping all processes..."
    kill ${WORKER_PIDS[@]} 2>/dev/null
    kill $MASTER_PID 2>/dev/null
    rm -f "$PID_FILE"
    wait
    echo "Done."
    exit 0
}

trap cleanup SIGINT

# Wait for master process
wait $MASTER_PID

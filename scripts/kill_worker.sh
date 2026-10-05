#!/bin/bash

# Simulate worker failure by killing a specific worker
# Usage: ./kill_worker.sh [worker_index]
#   worker_index: 0-3 (default: 0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/.pids"

if [ ! -f "$PID_FILE" ]; then
    echo "Error: PID file not found. Is the cluster running?"
    echo "Start the cluster first with: ./scripts/run.sh"
    exit 1
fi

# Source the PID file to get WORKER_PIDS array
source "$PID_FILE"

WORKER_INDEX=${1:-0}

if [ "$WORKER_INDEX" -lt 0 ] || [ "$WORKER_INDEX" -ge ${#WORKER_PIDS[@]} ]; then
    echo "Error: Invalid worker index. Valid range: 0-$((${#WORKER_PIDS[@]}-1))"
    exit 1
fi

WORKER_PID=${WORKER_PIDS[$WORKER_INDEX]}

if kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "Killing worker $WORKER_INDEX (PID: $WORKER_PID)..."
    kill -9 "$WORKER_PID"
    echo "Worker killed. Master should detect failure within 15-20 seconds."
    echo ""
    echo "After master detects failure, run: ./scripts/restart_workers.sh"
else
    echo "Worker $WORKER_INDEX (PID: $WORKER_PID) is not running."
fi

#!/bin/bash

# Start master in background
echo "Starting master..."
python3 master.py &
MASTER_PID=$!
sleep 2

# Start 4 workers in background
echo "Starting 4 workers..."
for i in {1..4}; do
    python3 worker.py &
    WORKER_PIDS+=($!)
    sleep 0.5
done

echo ""
echo "Master PID: $MASTER_PID"
echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo ""
echo "Press Ctrl+C to stop all processes..."

# Handle Ctrl+C - kill all processes
cleanup() {
    echo ""
    echo "Stopping all processes..."
    kill ${WORKER_PIDS[@]} 2>/dev/null
    kill $MASTER_PID 2>/dev/null
    wait
    echo "Done."
    exit 0
}

trap cleanup SIGINT

# Wait for master process
wait $MASTER_PID

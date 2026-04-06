#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/runs"
NUM_EPISODES=${1:-1}   # defaults to 1 if no argument given

for i in $(seq 1 $NUM_EPISODES); do
    echo "[Total Episodes] Run $i of $NUM_EPISODES"

    latest_file=$(ls -t "$RUNS_DIR"/*.zip 2>/dev/null | head -n 1 | sed 's/\.zip$//')

    if [ -n "$latest_file" ]; then
        echo "[Total Episodes] Resuming from checkpoint: $latest_file"
    else
        echo "[Total Episodes] No checkpoint found — starting fresh"
    fi

    echo "$latest_file" | python "$SCRIPT_DIR/baseline_fast_v3.py"
done

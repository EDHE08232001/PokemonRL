#!/bin/bash

# Resolve the directory this script lives in (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/runs"

# Find the latest checkpoint (.zip) in the runs directory
latest_file=$(ls -t "$RUNS_DIR"/*.zip 2>/dev/null | head -n 1 | sed 's/\.zip$//')

if [ -n "$latest_file" ]; then
    echo "[This Training Run] Resuming from checkpoint: $latest_file"
else
    echo "[This Training Run] No checkpoint found — starting fresh"
fi

echo "$latest_file" | python "$SCRIPT_DIR/baseline_fast_v3.py"

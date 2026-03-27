#!/bin/bash

# Resolve the directory this script lives in (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/runs"


# latest file with .zip extension
latest_file=$(ls -t "$RUNS_DIR"/*.zip 2>/dev/null | head -n 1 | sed 's/\.zip$//')
echo "starting with checkpoint: $latest_file"
echo "$latest_file" | python "$SCRIPT_DIR/baseline_fast_v3.py"
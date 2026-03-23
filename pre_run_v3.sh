#!/bin/bash
# =============================================================================
# pre_run_v3.sh — Pre-run cleanup for PokemonRL V3 training
#
# Run automatically by train_v3.sh before go_forever.sh starts.
# Prevents disk quota failures (50GB limit) by pruning large output files
# that accumulate across training runs.
#
# What it cleans:
#   - curframe_*.jpeg     : per-step game snapshots (64 envs × every 50 steps)
#   - final_states/*.jpeg : episode-end screenshots older than 2 days
#   - poke_*.zip          : old checkpoints (keeps the 3 most recent only)
#   - *.err / *.out       : old SLURM logs from previous jobs (keeps last 5)
#
# What it NEVER touches:
#   - The 3 most recent checkpoints (needed for go_forever.sh to resume)
#   - TensorBoard event files (needed for analysis)
#   - Recent final_states/ images (last 2 days kept for inspection)
# =============================================================================

set -euo pipefail

RUNS_DIR="$HOME/PokemonRL/src/v3/runs"
CONSOLE_DIR="$HOME/PokemonRL/console_outputs"

echo "============================================="
echo " PokemonRL V3 — Pre-run Cleanup"
echo " $(date)"
echo "============================================="

# ── Helper: human-readable size ──────────────────────────────────────────────
report_size() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        local sz
        sz=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  $label: $sz"
    fi
}

echo ""
echo "[INFO] Disk usage before cleanup:"
report_size "~/PokemonRL total" "$HOME/PokemonRL"
report_size "  runs/"           "$RUNS_DIR"
report_size "  console_outputs" "$CONSOLE_DIR"
df -h "$HOME" | tail -1 | awk '{print "  Home partition: " $3 " used / " $2 " total (" $5 " full)"}'

echo ""

# ── 1. Delete ALL curframe snapshots ─────────────────────────────────────────
# These are written every 50 steps per env = ~64 files regenerated constantly.
# They are purely for interactive inspection and have zero training value.
CURFRAME_COUNT=$(find "$RUNS_DIR" -maxdepth 1 -name "curframe_*.jpeg" 2>/dev/null | wc -l)
if [ "$CURFRAME_COUNT" -gt 0 ]; then
    find "$RUNS_DIR" -maxdepth 1 -name "curframe_*.jpeg" -delete
    echo "[CLEAN] Deleted $CURFRAME_COUNT curframe_*.jpeg snapshots"
else
    echo "[OK   ] No curframe snapshots found"
fi

# ── 2. Delete old final_states screenshots (older than 2 days) ───────────────
# Keep recent ones for inspection, prune the rest.
FS_DIR="$RUNS_DIR/final_states"
if [ -d "$FS_DIR" ]; then
    OLD_FRAMES=$(find "$FS_DIR" -name "*.jpeg" -mtime +2 2>/dev/null | wc -l)
    if [ "$OLD_FRAMES" -gt 0 ]; then
        find "$FS_DIR" -name "*.jpeg" -mtime +2 -delete
        echo "[CLEAN] Deleted $OLD_FRAMES final_states/*.jpeg older than 2 days"
    else
        echo "[OK   ] No old final_states images to prune"
    fi

    # Safety cap: if still more than 200 images, remove the oldest ones
    REMAINING=$(find "$FS_DIR" -name "*.jpeg" 2>/dev/null | wc -l)
    if [ "$REMAINING" -gt 200 ]; then
        EXCESS=$(( REMAINING - 200 ))
        find "$FS_DIR" -name "*.jpeg" -printf '%T+ %p\n' 2>/dev/null \
            | sort | head -n "$EXCESS" | awk '{print $2}' | xargs rm -f
        echo "[CLEAN] Hard-capped final_states/ to 200 images (removed $EXCESS excess)"
    fi
else
    echo "[OK   ] final_states/ directory does not exist yet"
fi

# ── 3. Keep only the 3 most recent checkpoints ───────────────────────────────
# go_forever.sh only needs the most recent one. We keep 3 as a safety buffer
# in case the latest checkpoint is corrupt.
CHECKPOINT_COUNT=$(find "$RUNS_DIR" -maxdepth 1 -name "poke_*.zip" 2>/dev/null | wc -l)
if [ "$CHECKPOINT_COUNT" -gt 3 ]; then
    TO_DELETE=$(( CHECKPOINT_COUNT - 3 ))
    # Sort by modification time (oldest first), delete the excess
    find "$RUNS_DIR" -maxdepth 1 -name "poke_*.zip" -printf '%T+ %p\n' 2>/dev/null \
        | sort | head -n "$TO_DELETE" | awk '{print $2}' | xargs rm -f
    echo "[CLEAN] Pruned $TO_DELETE old checkpoint(s), kept 3 most recent"
else
    echo "[OK   ] Only $CHECKPOINT_COUNT checkpoint(s) present — no pruning needed"
fi

# Print which checkpoints are kept
echo "[INFO] Active checkpoints:"
find "$RUNS_DIR" -maxdepth 1 -name "poke_*.zip" 2>/dev/null \
    | sort -t_ -k2 -n | while read -r f; do
    sz=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "         $sz  $(basename "$f")"
done

# ── 4. Prune old SLURM console logs (keep last 5 .out and 5 .err) ────────────
if [ -d "$CONSOLE_DIR" ]; then
    for EXT in out err; do
        LOG_COUNT=$(find "$CONSOLE_DIR" -maxdepth 1 -name "v3_train-*.${EXT}" 2>/dev/null | wc -l)
        if [ "$LOG_COUNT" -gt 5 ]; then
            TO_DELETE=$(( LOG_COUNT - 5 ))
            find "$CONSOLE_DIR" -maxdepth 1 -name "v3_train-*.${EXT}" -printf '%T+ %p\n' \
                | sort | head -n "$TO_DELETE" | awk '{print $2}' | xargs rm -f
            echo "[CLEAN] Pruned $TO_DELETE old .${EXT} log(s), kept 5 most recent"
        fi
    done
else
    echo "[OK   ] console_outputs/ not found — skipping log pruning"
fi

# ── Final disk report ─────────────────────────────────────────────────────────
echo ""
echo "[INFO] Disk usage after cleanup:"
report_size "~/PokemonRL total" "$HOME/PokemonRL"
report_size "  runs/"           "$RUNS_DIR"
df -h "$HOME" | tail -1 | awk '{print "  Home partition: " $3 " used / " $2 " total (" $5 " full)"}'
echo ""
echo "[DONE] Pre-run cleanup complete — starting training"
echo "============================================="

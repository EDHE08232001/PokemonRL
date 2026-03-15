# PokemonRL V2 (Recommended) — Coordinate-Based Exploration

V2 is the recommended training approach. It replaces V1's frame-based KNN exploration with **coordinate counting** and uses a **structured dict observation** that provides game state directly to the policy.

---

## How It Works

- Uses PPO with `MultiInputPolicy` (CNN + MLP feature extraction)
- Tracks unique `(x, y, map_id)` tiles visited — exploration reward scales with tile count
- Applies a stuck penalty (-0.05) if the agent revisits the same tile 600+ times
- Runs 64 parallel Pokemon Red emulators
- Dict observation: screens, HP, level (Fourier-encoded), badges, events, local map, recent actions

**Initial game state**: `init.state` — starts from the very beginning of the game.

---

## Training

```bash
# Install core dependencies
pip install .

# Start training
python src/v2/baseline_fast_v2.py
```

### Resume from Checkpoint

Pipe the checkpoint path via stdin:

```bash
echo "runs/poke_26214400_steps" | python src/v2/baseline_fast_v2.py
```

### Continuous Training

Use the wrapper script to automatically resume from the latest checkpoint in a loop:

```bash
src/v2/go_forever.sh
```

---

## Outputs & Logs

All outputs are saved to a single directory:

```
src/v2/runs/
├── poke_<steps>_steps.zip                          # Model checkpoints
├── curframe_<id>.jpeg                              # Current frame snapshots
├── PPO_1/events.out.tfevents.*                     # TensorBoard event files
└── final_states/                                   # Screenshots at episode end
    ├── frame_r<reward>_<count>_explore_map.jpeg    # Local exploration map
    ├── frame_r<reward>_<count>_full_explore_map.jpeg  # Full exploration map
    └── frame_r<reward>_<count>_full.jpeg           # Full game screen
```

**TensorBoard logs** are in the same `runs/` directory:

```bash
tensorboard --logdir src/v2/runs/
```

TensorBoard metrics include:
- `env_stats/`: mean values per episode (coords, levels, badges, deaths)
- `env_stats_max/`: max values across parallel envs
- `env_stats_distribs/`: histograms
- `trajectory/explore_map`: aggregated exploration map image
- `trajectory/all_flags`: JSON of all event flags set

---

## Checkpoints

- **Format**: `poke_<total_steps>_steps.zip`
- **Location**: `src/v2/runs/`
- **Frequency**: Every `ep_length // 2` steps (81,920 steps)
- Checkpoints are standard Stable Baselines 3 `.zip` files

---

## Running a Pre-trained Model

1. Place your checkpoint `.zip` file inside the runs directory:
   ```
   src/v2/runs/poke_<STEPS>_steps.zip
   ```

2. Run:
   ```bash
   python src/v2/run_pretrained_interactive.py
   ```

The script **auto-detects the most recent checkpoint** in `src/v2/runs/` — no manual path editing needed.

**Toggle AI control** by creating/editing `src/v2/agent_enabled.txt`:
- Write `yes` to let the AI play
- Write `no` (or delete the file) to play manually

---

## When Training Stops

- The latest checkpoint `.zip` remains in `src/v2/runs/` and can be used to resume
- Episode screenshots are saved to `final_states/`
- Use `go_forever.sh` for unattended long-term training that auto-resumes after interruption

---

## File Structure

```
src/v2/
├── red_gym_env_v2.py               # Gym environment (coordinate exploration)
├── baseline_fast_v2.py             # Training script (64 parallel envs)
├── run_pretrained_interactive.py   # Play with trained model (auto-detects checkpoint)
├── global_map.py                   # Map coordinate conversion
├── tensorboard_callback.py         # TensorBoard logging callback
├── stream_agent_wrapper.py         # WebSocket live map streaming
├── events.json                     # Event flag names
├── map_data.json                   # Map region coordinate data
└── go_forever.sh                   # Continuous training wrapper script
```

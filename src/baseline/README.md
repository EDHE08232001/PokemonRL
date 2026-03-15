# PokemonRL Baseline (V1) — KNN Frame Exploration

The Baseline is the original training approach. It uses **visual novelty** via an approximate nearest-neighbor index (HNSW) over downsampled game frames to drive exploration.

---

## How It Works

- Uses PPO with `CnnPolicy` (convolutional neural network)
- Builds an HNSW index (`hnswlib`) of downsampled RGB frames (36x40)
- A frame is "novel" if its L2 distance to the nearest stored frame exceeds 2,000,000
- Exploration reward scales with the number of novel frames discovered
- Runs 16 parallel Pokemon Red emulators

**Initial game state**: `has_pokedex_nballs.state` — starts with Pokedex and Pokeballs already obtained.

---

## Training

```bash
# Install baseline extras
pip install .[baseline]

# Start training
python src/baseline/run_baseline_parallel_fast.py
```

### Resume from Checkpoint

Pipe the checkpoint path via stdin:

```bash
echo "session_<UUID>/poke_38207488_steps" | python src/baseline/run_baseline_parallel_fast.py
```

---

## Outputs & Logs

When training starts, a session directory is created:

```
src/baseline/session_<UUID>/
├── poke_<steps>_steps.zip       # Model checkpoints (saved every 20,480 steps)
├── curframe_<id>.jpeg           # Current frame snapshot (saved every 50 steps)
├── all_runs_<id>.json           # Reward breakdown per episode
├── agent_stats_<id>.csv.gz      # Detailed per-step agent statistics
└── final_states/                # Screenshots at episode end
    ├── frame_r<reward>_<count>_small.jpeg
    └── frame_r<reward>_<count>_full.jpeg
```

**TensorBoard logs** are saved in the same session directory:

```bash
tensorboard --logdir src/baseline/session_<UUID>/
```

---

## Checkpoints

- **Format**: `poke_<total_steps>_steps.zip`
- **Location**: `src/baseline/session_<UUID>/`
- **Frequency**: Every `ep_length` steps (20,480 steps)
- Checkpoints are standard Stable Baselines 3 `.zip` files containing the full model state

---

## Running a Pre-trained Model

1. Place your checkpoint `.zip` file inside the session directory:
   ```
   src/baseline/session_<SESSION_ID>/poke_<STEPS>_steps.zip
   ```

2. Edit `file_name` in `run_pretrained_interactive.py` to point to your checkpoint path (relative to `src/baseline/`), e.g.:
   ```python
   file_name = "session_4da05e87/poke_439746560_steps"
   ```

3. Run:
   ```bash
   python src/baseline/run_pretrained_interactive.py
   ```

> **Note**: Unlike V2/V3, the baseline interactive script does **not** auto-detect checkpoints. You must manually set the path.

---

## When Training Stops

- The latest checkpoint `.zip` remains in the session directory and can be used to resume
- Episode screenshots and stats are saved to `final_states/` and JSON/CSV files
- Stable Baselines 3 handles graceful shutdown on Ctrl+C

---

## File Structure

```
src/baseline/
├── red_gym_env.py                  # Gym environment (KNN frame exploration)
├── red_gym_env_minimal.py          # Minimal env variant
├── run_baseline_parallel_fast.py   # Training script (16 parallel envs)
├── run_baseline_parallel.py        # Alt training script (44 parallel envs)
├── run_pretrained_interactive.py   # Play with trained model
├── baseline_fast_minimal.py        # Minimal training script
├── run_recorded_actions.py         # Replay recorded actions
├── memory_addresses.py             # Game Boy memory address constants
├── global_map.py                   # Map coordinate conversion
├── tensorboard_callback.py         # TensorBoard logging callback
├── stream_agent_wrapper.py         # WebSocket live map streaming
├── events.json                     # Event flag names
├── map_data.json                   # Map region coordinate data
└── ray_exp/                        # Experimental Ray RLlib training
    ├── red_gym_env_ray.py
    └── train_ray.py
```

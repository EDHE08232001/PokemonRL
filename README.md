# PokemonRL - Train RL Agents to Play Pokemon Red

Reinforcement learning agents that learn to play Pokemon Red using [PyBoy](https://github.com/Baekalfen/PyBoy) emulator and [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) PPO.

Three training approaches are included: a **Baseline** (original, frame-based KNN exploration), **V2** (improved, coordinate-based exploration), and **V3** (experimental — recurrent memory, semantic text rewards, and topological graph navigation). V2 is recommended for stable training; V3 is the active research frontier.

---

## Table of Contents

- [RL Method Overview](#rl-method-overview)
- [Version Descriptions](#version-descriptions)
  - [Baseline (V1)](#baseline-v1---knn-frame-exploration)
  - [V2 (Recommended)](#v2-recommended---coordinate-based-exploration)
  - [V3 (Experimental)](#v3-experimental---recurrent-memory--semantic-text--topological-graph)
- [Comparison Table](#comparison-between-versions)
- [Quick Start](#quick-start)
- [Detailed Training Instructions](#detailed-training-instructions)
- [Running a Trained Model](#running-a-trained-model)
- [Project Structure](#project-structure)
- [ROM Setup](#rom-setup)
- [Monitoring Training](#monitoring-training)
- [Training Broadcast](#training-broadcast)
- [Requirements](#requirements)
- [Related Work](#related-work)

---

## RL Method Overview

Both versions use **Proximal Policy Optimization (PPO)** from Stable Baselines 3. PPO is an **on-policy, actor-critic** algorithm — it is **not** Q-Learning, DQN, or classic REINFORCE.

### Why PPO? (vs. other RL methods)

| Method | Type | Why not used here |
|--------|------|-------------------|
| **REINFORCE** | On-policy, policy gradient | High variance, no value baseline, sample-inefficient |
| **Q-Learning / DQN** | Off-policy, value-based | Struggles with large visual observation spaces; requires replay buffers that grow with KNN frame stores |
| **A2C** | On-policy, actor-critic | Less stable than PPO; no clipping to prevent large policy updates |
| **PPO** (used) | On-policy, actor-critic | Stable training via clipped surrogate objective; works well with both CNN and multi-input policies; efficient with parallel environments |

### How PPO Works in This Project

1. **Multiple parallel environments** run the Pokemon Red emulator simultaneously (16 for Baseline, 64 for V2).
2. Each environment collects a rollout of experience (state, action, reward, next state).
3. PPO computes advantage estimates using Generalized Advantage Estimation (GAE) and updates the policy network by optimizing a **clipped surrogate loss** — this prevents destructively large policy updates.
4. The policy network outputs a probability distribution over the 6–7 Game Boy buttons (D-pad + A/B/Start).
5. The value network estimates expected future reward from the current state.

### Reward Shaping

Both versions use **dense, shaped rewards** rather than sparse game-completion signals. The reward is a composite of:
- **Event flags**: game story progress (talking to NPCs, completing quests)
- **Exploration**: novelty of visited states (frames or coordinates)
- **Badges**: gym badge collection
- **Healing**: recovering HP at Pokemon Centers
- **Opponent levels**: encountering stronger opponents
- **Death penalty**: negative reward for party wipeouts

---

## Version Descriptions

### Baseline (V1) — KNN Frame Exploration

The Baseline approach is the original training method. It uses **visual novelty** to drive exploration.

**Core idea**: Build an approximate nearest-neighbor index (HNSW via `hnswlib`) of downsampled game frames. A new frame is "novel" if its L2 distance to the nearest stored frame exceeds a threshold (`sim_frame_dist = 2,000,000`). The exploration reward scales with the number of novel frames discovered.

**Observation space** (flat `Box`):
- 3 stacked downsampled RGB frames (36×40×3 each)
- 2 visual "memory bars" encoding recent reward deltas and exploration progress
- Total observation shape: `(128, 40, 3)` — treated as an image by `CnnPolicy`

**Policy**: `CnnPolicy` — a convolutional neural network that processes the stacked frames as a single image. Shared CNN feature extractor feeds into separate policy (actor) and value (critic) heads.

**Reward components**:
- `event`: max event flags triggered (×reward_scale)
- `level`: party level sum (scaled down above 22)
- `heal`: cumulative healing amount (×4 per heal fraction)
- `op_lvl`: max opponent level (×0.2)
- `dead`: death count (×-0.1 penalty)
- `badge`: badge count (×5)
- `explore`: KNN index size (pre/post level-threshold weighting)

**PPO hyperparameters**:
- `n_steps`: `ep_length // 8` (2,560 steps per rollout)
- `batch_size`: 128
- `n_epochs`: 3
- `gamma`: 0.998
- `num_envs`: 16

**Initial state**: `has_pokedex_nballs.state` — the game starts with the Pokedex and Pokeballs already obtained, skipping early-game grind.

**Limitations**:
- KNN index consumes significant memory (~20K frames × 4,320 floats each)
- Frame-based novelty can be confused by minor visual changes (text boxes, menu states)
- CnnPolicy must learn all game state from pixels alone

---

### V2 (Recommended) — Coordinate-Based Exploration

V2 replaces frame-based exploration with **coordinate counting** and uses a **structured dict observation** that explicitly provides game state information to the policy.

**Core idea**: Track every unique `(x, y, map_id)` tile the agent visits. Exploration reward scales linearly with the number of unique tiles discovered. A "stuck" penalty is applied if the agent revisits the same tile 600+ times.

**Observation space** (dict — `MultiInputPolicy`):
- `screens`: 3 stacked grayscale frames (72×80×3) — more resolution than Baseline
- `health`: HP fraction [0, 1]
- `level`: Fourier-encoded party level sum (8 sine components)
- `badges`: 8-bit binary vector
- `events`: all event flag bits (~1,720 bits)
- `map`: local crop of the global exploration map (48×48×1), showing nearby visited tiles
- `recent_actions`: last 3 actions taken

**Policy**: `MultiInputPolicy` — processes each observation component with an appropriate sub-network (CNN for screens/map, MLP for scalars/vectors), then combines features.

**Reward components**:
- `event`: max event flags (×4 ×reward_scale)
- `heal`: cumulative healing (×10 ×reward_scale, quadratic per heal)
- `badge`: badge count (×10 ×reward_scale)
- `explore`: unique coordinate count (×0.1 ×explore_weight ×reward_scale)
- `stuck`: -0.05 penalty when current tile visited 600+ times

**PPO hyperparameters**:
- `n_steps`: `ep_length // 64` (2,560 steps per rollout)
- `batch_size`: 512
- `n_epochs`: 1
- `gamma`: 0.997
- `ent_coef`: 0.01 (entropy bonus for exploration)
- `num_envs`: 64

**Initial state**: `init.state` — starts from the very beginning of the game (no items pre-obtained).

**Advantages over Baseline**:
- No KNN index → much lower memory usage
- Dict observation provides structured game state → faster learning
- Coordinate exploration is deterministic and interpretable
- More parallel environments (64 vs 16) → faster wall-clock training
- Stuck penalty prevents the agent from oscillating in one location

---

### V3 (Experimental) — Recurrent Memory, Semantic Text, & Topological Graph

V3 extends V2 with three new systems aimed at pushing the agent beyond pure spatial exploration toward narrative understanding and structural map reasoning. **Full details: [v3/README.md](v3/README.md)**

**Key additions**:
1. **Recurrent Memory (LSTM)**: Replaces PPO with RecurrentPPO (`sb3-contrib`). The `MultiInputLstmPolicy` maintains hidden state across the episode, giving the agent working memory for multi-step tasks (building navigation, dialogue sequences). The `recent_actions` observation is removed since the LSTM handles temporal action history natively.
2. **Semantic Text Rewards**: Hooks into Game Boy WRAM (`0xCF4B`) to read and decode the active text buffer using the Gen 1 character map. New unique dialogue strings grant a one-time +0.5 intrinsic reward, incentivizing NPC interaction and sign reading.
3. **Topological Graph Navigation**: Builds a directed graph (`networkx.DiGraph`) of `(map_id, row, col)` nodes during exploration. Detects "warp edges" (door/teleport transitions between map IDs) and grants a +2.0 bonus for discovering new maps. The shortest-path distance from Pallet Town is fed into the observation as `graph_distance`.

**New dependencies**: `sb3-contrib`, `networkx`

```bash
pip install -r requirements/requirements-v3.txt
cd v3
python baseline_fast_v3.py
```

---

## Comparison Between Versions

| Feature | Baseline (V1) | V2 (Recommended) | V3 (Experimental) |
|---------|---------------|-------------------|--------------------|
| **RL Algorithm** | PPO (CnnPolicy) | PPO (MultiInputPolicy) | RecurrentPPO (MultiInputLstmPolicy) |
| **Exploration method** | KNN over downsampled frames (HNSW, L2 distance) | Coordinate counting (unique tiles visited) | Coordinate counting + topological graph + text |
| **Observation type** | Flat RGB image (stacked frames + memory bars) | Dict: screens, HP, level, badges, events, map, actions | Dict: screens, HP, level, badges, events, map, text_hash, graph_distance |
| **Screen format** | 36×40 RGB, 3 stacked | 72×80 grayscale, 3 stacked | 72×80 grayscale, 3 stacked |
| **Policy architecture** | CNN → shared features → actor/critic | CNN (screens, map) + MLP (scalars) → combined → actor/critic | CNN + MLP → LSTM (128h, 1L) → actor/critic |
| **Parallel envs** | 16 | 64 | 64 |
| **PPO epochs/update** | 3 | 1 | 1 |
| **Batch size** | 128 | 512 | 512 |
| **Discount (gamma)** | 0.998 | 0.997 | 0.997 |
| **Entropy coef** | 0 (default) | 0.01 | 0.01 |
| **Memory usage** | Higher (KNN index ~20K frames) | Lower (coordinate dict only) | Moderate (coords + graph + LSTM states) |
| **Training speed** | Slower | Faster | Moderate (LSTM overhead) |
| **Initial game state** | Has Pokedex + Pokeballs | Start of game | Start of game |
| **Stuck penalty** | No | Yes (-0.05 at 600+ visits) | Yes (-0.05 at 600+ visits) |
| **Level encoding** | Raw level sum in reward | Fourier-encoded in observation | Fourier-encoded in observation |
| **Text understanding** | None | None | Gen 1 WRAM text decoding + reward |
| **Map structure** | None | Flat exploration map | Exploration map + directed graph |
| **PyBoy version** | v1.x (`botsupport_manager`, `get_memory_value`) | v2.x (`screen.ndarray`, `memory[]`) | v2.x (`screen.ndarray`, `memory[]`) |
| **Result** | Reaches ~Cerulean City | Reaches Cerulean City, trains faster | Under development |

---

## Quick Start

```bash
# 1. Place your Pokemon Red ROM in the ROM_INPUT folder
cp /path/to/PokemonRed.gb ROM_INPUT/PokemonRed.gb

# 2. Install dependencies (pick one)
pip install -r requirements/requirements-v2.txt          # V2 (recommended)
pip install -r requirements/requirements-baseline.txt    # Baseline

# 3. Run
python main.py
```

`main.py` presents a menu to train or run either approach:

```
  [1] Train Baseline
  [2] Run Baseline    (play with trained model)
  [3] Train V2
  [4] Run V2          (play with trained model)
```

---

## Detailed Training Instructions

### Prerequisites

1. **Python 3.10+** is required.
2. **ffmpeg** must be available on your PATH (for video recording).
3. A legally obtained **Pokemon Red** Game Boy ROM file (see [ROM Setup](#rom-setup)).
4. **Hardware**: Training benefits significantly from multiple CPU cores. V2 defaults to 64 parallel environments. Reduce `num_cpu` in the training script if you have fewer cores.
5. **GPU**: Optional. PyTorch with CUDA will speed up PPO updates. The emulators run on CPU regardless.

### Option A: Using the Menu (Recommended)

```bash
# Install V2 dependencies
pip install -r requirements/requirements-v2.txt

# Place ROM
cp /path/to/PokemonRed.gb ROM_INPUT/PokemonRed.gb

# Launch menu
python main.py

# Select [3] to train V2, or [1] for Baseline
```

### Option B: Running Training Scripts Directly

#### Train V2

```bash
pip install -r requirements/requirements-v2.txt
cd v2
python baseline_fast_v2.py
```

This will:
1. Spin up 64 parallel Pokemon Red emulators
2. Begin PPO training with `MultiInputPolicy`
3. Save checkpoints to `v2/runs/` (every `ep_length // 2` steps)
4. Log to TensorBoard in `v2/runs/`

To resume from a checkpoint, pipe the checkpoint path via stdin:
```bash
echo "runs/poke_26214400_steps" | python baseline_fast_v2.py
```

#### Train Baseline

```bash
pip install -r requirements/requirements-baseline.txt
cd baseline
python run_baseline_parallel_fast.py
```

This will:
1. Spin up 16 parallel Pokemon Red emulators
2. Begin PPO training with `CnnPolicy`
3. Save checkpoints to `baseline/session_<id>/`
4. Log to TensorBoard in the session directory

To resume from a checkpoint, edit `file_name` in `run_baseline_parallel_fast.py` to point to your checkpoint.

### Training Configuration

Key parameters you may want to adjust (in the training scripts):

| Parameter | Baseline Default | V2 Default | Description |
|-----------|-----------------|------------|-------------|
| `num_cpu` | 16 | 64 | Number of parallel environments (reduce if you have fewer cores) |
| `ep_length` | 20,480 | 163,840 | Steps per episode per environment |
| `batch_size` | 128 | 512 | PPO minibatch size |
| `n_epochs` | 3 | 1 | PPO update epochs per rollout |
| `gamma` | 0.998 | 0.997 | Discount factor |
| `reward_scale` | 4 | 0.5 | Scales all reward components |
| `explore_weight` | 3 | 0.25 | Weight of exploration reward |
| `action_freq` | 24 | 24 | Emulator ticks per agent step |
| `use_wandb_logging` | False | False | Enable Weights & Biases logging |

### Enable Weights & Biases Logging

Set `use_wandb_logging = True` in the training script. You will need a W&B account and `wandb` installed (`pip install wandb`).

---

## Running a Trained Model

### V2 (Recommended)

A pre-trained V2 checkpoint is included in `v2/runs/`.

```bash
cd v2

# Auto-loads the most recent checkpoint from runs/
python run_pretrained_interactive.py
```

### Baseline

```bash
cd baseline

# Edit file_name in run_pretrained_interactive.py to point to your checkpoint
python run_pretrained_interactive.py
```

### Interactive Controls

- **SDL2 window**: The game renders in a visible window. You can use arrow keys and keyboard to play manually.
- **AI toggle**: Edit `agent_enabled.txt` in the working directory:
  - Write `yes` to let the AI play
  - Write `no` (or delete the file) to play manually
- The game runs at 6× emulation speed for comfortable viewing.

---

## Project Structure

```
PokemonRL/
├── main.py                  # Entry point - menu to train or run models
├── ROM_INPUT/               # Place PokemonRed.gb here
├── saves/                   # Game save states for initializing training
│   ├── init.state           # Early game state (used by V2)
│   ├── has_pokedex.state
│   ├── has_pokedex_nballs.state  # Has Pokedex + Pokeballs (used by Baseline)
│   └── fast_text_start.state
│
├── baseline/                # Baseline approach (V1)
│   ├── red_gym_env.py       # Gym environment (KNN frame exploration)
│   ├── run_baseline_parallel_fast.py  # Training script (16 CPUs)
│   ├── run_baseline_parallel.py       # Alternate training script (44 CPUs)
│   ├── run_pretrained_interactive.py  # Play with trained model
│   ├── red_gym_env_minimal.py         # Minimal env variant (experimental)
│   ├── baseline_fast_minimal.py       # Minimal training script
│   ├── memory_addresses.py  # Game Boy memory address constants
│   ├── global_map.py        # Map coordinate conversion
│   ├── tensorboard_callback.py  # TensorBoard logging callback
│   ├── stream_agent_wrapper.py  # WebSocket live map streaming
│   └── ...
│
├── v2/                      # V2 approach (recommended)
│   ├── red_gym_env_v2.py    # Gym environment (coordinate exploration)
│   ├── baseline_fast_v2.py  # Training script (64 CPUs)
│   ├── run_pretrained_interactive.py  # Play with trained model
│   ├── global_map.py        # Map coordinate conversion
│   ├── tensorboard_callback.py  # TensorBoard logging callback
│   ├── stream_agent_wrapper.py  # WebSocket live map streaming
│   ├── events.json          # Event flag names (parsed from pokered)
│   ├── map_data.json        # Map region coordinate data
│   └── runs/                # Pre-trained checkpoint included
│       └── poke_26214400.zip
│
├── v3/                      # V3 approach (experimental)
│   ├── red_gym_env_v3.py    # Gym environment (LSTM + text + graph)
│   ├── baseline_fast_v3.py  # Training script (RecurrentPPO, 64 CPUs)
│   ├── tensorboard_callback_v3.py  # TensorBoard logging (+ V3 metrics)
│   ├── global_map.py        # Map coordinate conversion
│   ├── stream_agent_wrapper.py  # WebSocket live map streaming
│   ├── events.json          # Event flag names
│   ├── map_data.json        # Map region coordinate data
│   └── README.md            # V3 architecture documentation
│
├── requirements/            # All pip dependency files (centralized)
│   ├── requirements-base.txt              # Shared core deps (unpinned)
│   ├── requirements-baseline.txt          # Baseline pinned deps
│   ├── requirements-baseline-unfrozen.txt # Baseline unpinned deps
│   ├── requirements-v2.txt               # V2 pinned deps (Linux/CUDA)
│   ├── requirements-v2-macos.txt         # V2 pinned deps (macOS)
│   ├── requirements-v3.txt              # V3 deps (adds sb3-contrib, networkx)
│   └── requirements-ray.txt             # Ray RLlib experiment deps
│
├── visualization/           # Map and progress visualization scripts
└── assets/                  # Images and media for documentation
```

---

## ROM Setup

You need a legally obtained **Pokemon Red** Game Boy ROM file.

1. Rename it to `PokemonRed.gb`
2. Place it in the `ROM_INPUT/` directory
3. Verify the SHA1 checksum: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`

```bash
shasum ROM_INPUT/PokemonRed.gb
```

---

## Monitoring Training

### TensorBoard

```bash
# For V2
tensorboard --logdir v2/runs/
# Open http://localhost:6006

# For Baseline
tensorboard --logdir baseline/session_<id>/
```

TensorBoard will show:
- **env_stats/**: mean values per episode (coords visited, levels, badges, deaths, etc.)
- **env_stats_max/**: max values across parallel envs
- **env_stats_distribs/**: histograms of stats across envs
- **trajectory/explore_map**: aggregated exploration map image
- **trajectory/all_flags**: JSON of all event flags set
- Standard PPO metrics (loss, entropy, explained variance, etc.)

### Session Directories

Training creates `session_<uuid>/` directories containing:
- `curframe_<id>.jpeg`: periodically saved current game frame
- `final_states/`: screenshots at episode end
- `all_runs_<id>.json`: reward breakdown per episode
- `agent_stats_<id>.csv.gz`: detailed per-step agent statistics

---

## Training Broadcast

Stream training progress to a shared live map using the `StreamWrapper`:

```python
from stream_agent_wrapper import StreamWrapper

env = StreamWrapper(
    env,
    stream_metadata={
        "user": "your-username",
        "env_id": rank,
        "color": "#0033ff",
        "extra": "",
    }
)
```

View the live map: https://pwhiddy.github.io/pokerl-map-viz/

---

## Requirements

- Python 3.10+
- ffmpeg (available on PATH)
- All dependency files are in `requirements/`:
  - `requirements-v2.txt` — V2 pinned (Linux/CUDA, recommended)
  - `requirements-v2-macos.txt` — V2 pinned (macOS, no NVIDIA packages)
  - `requirements-v3.txt` — V3 (adds sb3-contrib, networkx)
  - `requirements-baseline.txt` — Baseline pinned
  - `requirements-baseline-unfrozen.txt` — Baseline unpinned (flexible)
  - `requirements-base.txt` — Shared core deps (unpinned)
  - `requirements-ray.txt` — Ray RLlib experiment

### Key Dependencies

| Package | Baseline | V2 | V3 | Purpose |
|---------|----------|-----|-----|---------|
| `pyboy` | 1.6.9 | 2.4.0 | ≥2.4.0 | Game Boy emulator |
| `stable-baselines3` | 2.0.0 | 2.3.2 | ≥2.3.2 | PPO implementation |
| `sb3-contrib` | — | — | ≥2.3.0 | RecurrentPPO (LSTM policy) |
| `networkx` | — | — | ≥3.0 | Topological graph navigation |
| `torch` | 2.0.1 | 2.5.0 | ≥2.5.0 | Neural network backend |
| `gymnasium` | 0.28.1 | 0.29.1 | ≥0.29.1 | Environment interface |
| `hnswlib` | 0.7.0 | — | — | KNN index (Baseline only) |
| `scikit-image` | 0.21.0 | 0.24.0 | ≥0.24.0 | Frame downsampling |
| `einops` | 0.6.1 | 0.8.0 | ≥0.8.0 | Tensor reshaping |
| `mediapy` | custom fork | 1.2.2 | ≥1.2.2 | Video recording |
| `websockets` | — | 13.1 | ≥13.1 | Live map streaming |

---

## Related Work

- [Pokemon Red via Reinforcement Learning (arXiv)](https://arxiv.org/abs/2502.19920)
- [Pokemon RL Edition](https://drubinstein.github.io/pokerl/)
- [PokeGym](https://github.com/PufferAI/pokegym)
- [Live Map Visualization](https://github.com/pwhiddy/pokerl-map-viz/)

## Built With

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms (PPO)
- [PyTorch](https://pytorch.org/) - Neural network framework
- [Gymnasium](https://gymnasium.farama.org/) - Environment interface standard

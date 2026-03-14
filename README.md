# PokemonRL - Train RL Agents to Play Pokemon Red

Reinforcement learning agents that learn to play Pokemon Red using [PyBoy](https://github.com/Baekalfen/PyBoy) emulator and [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) PPO.

Two training approaches are included: a **Baseline** (original, frame-based KNN exploration) and **V2** (improved, coordinate-based exploration). V2 is recommended.

## Quick Start

```bash
# 1. Place your Pokemon Red ROM in the ROM_INPUT folder
cp /path/to/PokemonRed.gb ROM_INPUT/PokemonRed.gb

# 2. Install dependencies (pick one)
pip install -r v2/requirements.txt          # V2 (recommended)
pip install -r baseline/requirements.txt    # Baseline

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
│   ├── run_pretrained_interactive.py  # Play with trained model
│   ├── memory_addresses.py  # Game Boy memory address constants
│   ├── global_map.py        # Map coordinate conversion
│   ├── tensorboard_callback.py
│   ├── stream_agent_wrapper.py
│   ├── requirements.txt
│   └── ...
│
├── v2/                      # V2 approach (recommended)
│   ├── red_gym_env_v2.py    # Gym environment (coordinate exploration)
│   ├── baseline_fast_v2.py  # Training script (64 CPUs)
│   ├── run_pretrained_interactive.py  # Play with trained model
│   ├── global_map.py        # Map coordinate conversion
│   ├── tensorboard_callback.py
│   ├── stream_agent_wrapper.py
│   ├── requirements.txt
│   └── runs/                # Pre-trained checkpoint included
│       └── poke_26214400.zip
│
├── visualization/           # Map and progress visualization notebooks/scripts
├── experiments/             # CLIP-based location recognition experiments
└── assets/                  # Images and media for documentation
```

## ROM Setup

You need a legally obtained **Pokemon Red** Game Boy ROM file.

1. Rename it to `PokemonRed.gb`
2. Place it in the `ROM_INPUT/` directory
3. Verify the SHA1 checksum: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`

```bash
shasum ROM_INPUT/PokemonRed.gb
```

## Approaches

### Baseline (Original)

- **Exploration**: KNN index over downsampled game frames. Rewards novelty via visual distance.
- **Policy**: `CnnPolicy` (CNN over RGB frame stacks + memory visualization bars)
- **Training**: 16 parallel environments, batch size 128, 3 epochs per update
- **Initial state**: `has_pokedex_nballs.state` (starts with Pokedex and Pokeballs)
- **Result**: Reaches Cerulean City

```bash
cd baseline
python run_baseline_parallel_fast.py
```

### V2 (Recommended)

- **Exploration**: Coordinate-based. Tracks visited (x, y, map) positions on a global map.
- **Policy**: `MultiInputPolicy` (dict observations: screens, health, level, badges, events, local map, recent actions)
- **Training**: 64 parallel environments, batch size 512, 1 epoch per update
- **Initial state**: `init.state` (early game)
- **Result**: Reaches Cerulean City, trains faster with less memory

```bash
cd v2
python baseline_fast_v2.py
```

### Key Differences

| Feature | Baseline | V2 |
|---------|----------|----|
| Exploration | KNN over frames | Coordinate-based |
| Policy | CnnPolicy | MultiInputPolicy |
| Parallel envs | 16 | 64 |
| Epochs/update | 3 | 1 |
| Memory usage | Higher (KNN index) | Lower |
| Training speed | Slower | Faster |

## Playing with a Trained Model

Both approaches include interactive play scripts. A pre-trained V2 checkpoint is included in `v2/runs/`.

```bash
cd v2
python run_pretrained_interactive.py
```

- Arrow keys to move, `a` = A button, `s` = B button
- Toggle AI control by editing `agent_enabled.txt` (`yes` / `no`)

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

## Tracking Progress

Session directories contain rendered frames of the current game state. Use TensorBoard:

```bash
cd <session_directory>
tensorboard --logdir .
# Open localhost:6006
```

Set `use_wandb_logging = True` in training scripts for Weights & Biases integration.

## Requirements

- Python 3.10+
- ffmpeg (available on PATH)
- See `baseline/requirements.txt` or `v2/requirements.txt` for Python packages
- macOS users: use `v2/macos_requirements.txt`

## Related Work

- [Pokemon Red via Reinforcement Learning (arXiv)](https://arxiv.org/abs/2502.19920)
- [Pokemon RL Edition](https://drubinstein.github.io/pokerl/)
- [PokeGym](https://github.com/PufferAI/pokegym)
- [Live Map Visualization](https://github.com/pwhiddy/pokerl-map-viz/)

## Built With

- [PyBoy](https://github.com/Baekalfen/PyBoy) - Game Boy emulator
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms

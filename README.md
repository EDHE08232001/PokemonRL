# CSI 5340 / ELG 5214 — Project Submission Guide

**Project:** Evaluating Symbolic Coordinate Discovery as a Robust Exploration Metric in Noisy Environments  
**Group:** Group 3  
**Members:** Edward He, Haozheng Wang, Sree Sahithya T.R., Xujia Fan  
**Repository:** https://github.com/EDHE08232001/PokemonRL

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
  - [Prerequisites](#prerequisites)
  - [ROM Setup](#rom-setup)
  - [Installing Dependencies](#installing-dependencies)
- [Running the Code](#running-the-code)
  - [Quick Start (Interactive Menu)](#quick-start-interactive-menu)
  - [Train V1 Baseline](#train-v1-baseline)
  - [Train V2 (Recommended)](#train-v2-recommended)
  - [Train V3 (Our Contribution)](#train-v3-our-contribution)
  - [Running on Morningstar HPC (SLURM)](#running-on-morningstar-hpc-slurm)
- [Verifying the Environment](#verifying-the-environment)
- [Viewing Results](#viewing-results)
  - [TensorBoard](#tensorboard)
  - [Exploration Map Screenshots](#exploration-map-screenshots)
  - [Training Logs](#training-logs)
- [Reproducing Key Results](#reproducing-key-results)
  - [V1 Baseline](#v1-baseline)
  - [V2 Symbolic Coordinate Discovery](#v2-symbolic-coordinate-discovery)
  - [V3 Semantic Reasoning](#v3-semantic-reasoning)
- [Running a Pre-Trained Model Interactively](#running-a-pre-trained-model-interactively)
- [Determinism and Seeds](#determinism-and-seeds)

---

## Repository Structure

```
PokemonRL/
├── main.py                          # Entry point — interactive menu
├── check_env.py                     # Environment verification script
├── pyproject.toml                   # Dependency specification (all versions pinned)
├── train_v3.sh                      # SLURM job script for V3 on Morningstar HPC
├── pre_run_v3.sh                    # Pre-run disk cleanup script (HPC only)
├── ROM_INPUT/                       # Place PokemonRed.gb here (user-supplied)
├── saves/
│   ├── init.state                   # Start-of-game save state (used by V2, V3)
│   └── has_pokedex_nballs.state     # Pre-equipped save state (used by V1)
└── src/
    ├── baseline/                    # V1 — Pixel-KNN (CnnPolicy)
    ├── v2/                          # V2 — Coordinate exploration (MultiInputPolicy)
    └── v3/                          # V3 — LSTM + text + graph (RecurrentPPO)
```

---

## Environment Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| ffmpeg | Any recent (must be on `PATH`) |
| Pokemon Red ROM | SHA1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a` |

Verify ffmpeg is available:
```bash
ffmpeg -version
```

### ROM Setup

You must supply a legally obtained Pokemon Red (English) Game Boy ROM file.

```bash
# Place the ROM in the correct directory
cp /path/to/PokemonRed.gb ROM_INPUT/PokemonRed.gb

# Verify the checksum
shasum ROM_INPUT/PokemonRed.gb
# Expected: ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

### Installing Dependencies

The project uses optional dependency groups. Install only what you need:

```bash
# V2 only (recommended for replication)
pip install .

# V1 Baseline only
pip install .[baseline]

# V3 (our contribution) — requires sb3-contrib and networkx
pip install .[v3]

# macOS (Apple Silicon / Intel) — no CUDA packages
pip install .[macos]

# Windows with NVIDIA GPU
pip install .[windows]

# Everything
pip install .[all]
```

> **Note for Morningstar HPC:** See [Running on Morningstar HPC](#running-on-morningstar-hpc-slurm) below and `v3_setup_morning_star_hpc.md` for the offline wheel installation procedure required behind the cluster firewall.

---

## Running the Code

### Quick Start (Interactive Menu)

The simplest way to train or run any version:

```bash
python main.py
```

This presents the following menu:

```
  [1] Train Baseline  (KNN exploration, CnnPolicy)
  [2] Run Baseline    (play with trained Baseline model)

  [3] Train V2        (coordinate exploration, MultiInputPolicy)
  [4] Run V2          (play with trained V2 model)

  [5] Train V3        (LSTM + text + graph, RecurrentPPO)
  [6] Run V3          (play with trained V3 model)

  [c] Check Environment (verify deps, ROM, hardware)
  [q] Quit
```

---

### Train V1 Baseline

```bash
pip install .[baseline]
python src/baseline/run_baseline_parallel_fast.py
```

**What happens:**
- Spawns 16 parallel Pokemon Red emulators
- Trains PPO with `CnnPolicy` and HNSW pixel-KNN exploration
- Saves checkpoints to `src/baseline/session_<UUID>/`
- Logs to TensorBoard in the same session directory

**To resume from a checkpoint**, edit `file_name` at the top of `run_baseline_parallel_fast.py` to point to your checkpoint path.

---

### Train V2 (Recommended)

```bash
pip install .
python src/v2/baseline_fast_v2.py
```

**What happens:**
- Spawns 64 parallel Pokemon Red emulators
- Trains PPO with `MultiInputPolicy` and coordinate-based exploration
- Saves checkpoints to `src/v2/runs/` every `ep_length // 2` steps
- Logs to TensorBoard in `src/v2/runs/`

**To resume from a checkpoint:**
```bash
echo "runs/poke_26214400_steps" | python src/v2/baseline_fast_v2.py
```

**To run continuously across multiple episodes:**
```bash
bash src/v2/go_forever.sh
```

---

### Train V3 (Our Contribution)

```bash
pip install .[v3]
python src/v3/baseline_fast_v3.py
```

**What happens:**
- Spawns 64 parallel Pokemon Red emulators
- Trains `RecurrentPPO` with `MultiInputLstmPolicy`
- Adds LSTM memory, WRAM text decoding, and topological graph navigation on top of V2
- Saves checkpoints to `src/v3/runs/` every `ep_length // 2` steps
- Logs to TensorBoard in `src/v3/runs/`

**To resume from a checkpoint:**
```bash
echo "runs/poke_XXXXXXX_steps" | python src/v3/baseline_fast_v3.py
```

**To run multiple episodes back-to-back (auto-resume):**
```bash
# Run 5 episodes sequentially, auto-resuming from latest checkpoint
bash src/v3/go_v3.sh 5
```

---

### Running on Morningstar HPC (SLURM)

**One-time environment setup** (run from a login node):
```bash
# Follow the complete offline installation guide
cat v3_setup_morning_star_hpc.md
```

**Submitting a training job:**
```bash
# Ensure output directory exists
mkdir -p console_outputs

# Submit V3 training job (24 hours, 1 GPU, 256GB RAM)
sbatch train_v3.sh
```

**Monitoring the job:**
```bash
# Check job status
squeue -u $USER

# Tail the live training output
tail -f console_outputs/v3_train-<JOBID>.out
```

The SLURM script automatically:
1. Runs `pre_run_v3.sh` to free disk space before training
2. Finds the latest checkpoint and resumes, or starts fresh
3. Saves console output to `console_outputs/v3_train-<JOBID>.out`

---

## Verifying the Environment

Before training, run the full environment check:

```bash
# Check all versions
python check_env.py

# Check a specific version
python check_env.py v2
python check_env.py v3
python check_env.py baseline
```

This verifies:
- Python version (≥3.10)
- All required packages and versions
- ROM file presence and accessibility
- Save state files
- GPU/CUDA availability
- V3-specific correctness checks (fix verification)

**Expected output for a correctly configured environment:**
```
[OK  ]  All checks passed — ready to train!
```

Exit code `0` = all checks passed. Exit code `1` = one or more failures.

---

## Viewing Results

### TensorBoard

All training metrics are logged to TensorBoard automatically.

```bash
# V2 training metrics
tensorboard --logdir src/v2/runs/

# V3 training metrics
tensorboard --logdir src/v3/runs/

# V1 Baseline metrics
tensorboard --logdir src/baseline/session_<UUID>/

# Then open in your browser:
# http://localhost:6006
```

**Key metrics to examine:**

| TensorBoard Path | What It Shows |
|-----------------|---------------|
| `rollout/ep_rew_mean` | Mean episode reward over time |
| `train/explained_variance` | Value network quality (target: >0.6) |
| `train/entropy_loss` | Policy exploration diversity |
| `train/clip_fraction` | PPO trust region health (target: <0.1) |
| `train/approx_kl` | KL divergence between policy updates |
| `env_stats/coord_count` | Unique tiles explored |
| `env_stats/badge` | Gym badges earned |
| `env_stats/deaths` | Party blackouts per episode |
| `trajectory/explore_map` | Heatmap image of explored tiles |
| `v3/mean_dialogue_count` | Unique NPC dialogues decoded (V3 only) |
| `v3/mean_graph_nodes` | Topological graph size (V3 only) |
| `v3/mean_maps_discovered` | Unique map regions reached (V3 only) |
| `reward_components/*` | Per-component reward breakdown (V3 only) |

---

### Exploration Map Screenshots

Screenshots are saved automatically during training:

| File | Location | When Saved |
|------|----------|------------|
| `curframe_<id>.jpeg` | `src/v*/runs/` | Every 5,000 steps per environment |
| `frame_r<reward>_<N>_explore_map.jpeg` | `src/v*/runs/final_states/` | End of each episode |
| `frame_r<reward>_<N>_full_explore_map.jpeg` | `src/v*/runs/final_states/` | End of each episode |
| `frame_r<reward>_<N>_full.jpeg` | `src/v*/runs/final_states/` | End of each episode |

The `explore_map` images are aggregated heatmaps showing all tiles visited across all 64 parallel environments. The filename encodes the total reward achieved that episode.

---

### Training Logs

**V3 prints a formatted summary table to stdout at every rollout end**, visible in SLURM `.out` files or your terminal. The table shows per-environment stats including reward, tiles explored, badges, deaths, dialogue count, and graph distance.

**V1 Baseline** additionally saves:
- `all_runs_<id>.json` — episode reward breakdown
- `agent_stats_<id>.csv.gz` — detailed per-step statistics

---

## Reproducing Key Results

### V1 Baseline

The V1 results in our paper were obtained with the following configuration (already set in `run_baseline_parallel_fast.py`):

| Parameter | Value |
|-----------|-------|
| `num_cpu` | 16 |
| `ep_length` | 20,480 |
| `batch_size` | 128 |
| `n_epochs` | 3 |
| `gamma` | 0.998 |
| `reward_scale` | 4 |
| `explore_weight` | 3 |

Expected behaviour: training reward swings between 20–72 due to the Noisy TV problem, with exploration reward collapsing after visual novelty saturates.

---

### V2 Symbolic Coordinate Discovery

The V2 results in our paper (1,040–1,113 FPS, explained variance converging to ~0.9) were obtained with:

| Parameter | Value |
|-----------|-------|
| `num_cpu` | 64 |
| `ep_length` | 163,840 |
| `batch_size` | 512 |
| `n_epochs` | 1 |
| `gamma` | 0.997 |
| `ent_coef` | 0.01 |
| `reward_scale` | 0.5 |
| `explore_weight` | 0.25 |

These are the current defaults in `src/v2/baseline_fast_v2.py`.

---

### V3 Semantic Reasoning

The **post-fix** V3 results reported in Section 5.4 of the paper use the following configuration (current defaults in `src/v3/baseline_fast_v3.py`):

| Parameter | Before Fix | After Fix (Current) |
|-----------|-----------|---------------------|
| `n_epochs` | 3 | **1** |
| `gamma` | 0.995 | **0.997** |
| `ent_coef` | 0.01 | **0.02** |
| `vf_coef` | 0.5 | **0.75** |
| `clip_range` | 0.2 | **0.15** |
| `badge_multiplier` | 10 | **25** |
| `op_lvl reward` | disabled | **enabled** |
| `explore_weight` | 0.25 | **1.0** |

> All fixes are committed to the repository. The pre-fix configuration that produced the regression described in Section 5.2 is documented in the paper but is **not** the current code state.

**To reproduce the 10.5M step results** (Section 4.3):
```bash
pip install .[v3]
python src/v3/baseline_fast_v3.py
# Let it run for approximately 10.5M timesteps (~2-3 SLURM jobs at 24h each)
```

Expected results at 10.5M steps:
- Value loss: ~0.002
- Explained variance: ~0.76
- Throughput: ~264 FPS
- Best environment: ~1,772 unique tiles, ~15 maps, ~19 dialogue strings

---

## Running a Pre-Trained Model Interactively

To watch a trained agent play (or to play manually in the same window):

**V2:**
```bash
# Auto-detects most recent checkpoint in src/v2/runs/
python src/v2/run_pretrained_interactive.py
```

**V3:**
```bash
# Auto-detects most recent checkpoint in src/v3/runs/
python src/v3/run_pretrained_interactive.py
```

The game opens in an SDL2 window at 6× emulation speed.

**Toggling AI control at runtime:**
```bash
# Let the AI play
echo "yes" > src/v3/agent_enabled.txt

# Take manual control
echo "no" > src/v3/agent_enabled.txt
# (or simply delete the file)
```

---

## Determinism and Seeds

- Seeds are set via `set_random_seed(seed + rank)` in each environment closure, where `rank` is the environment index (0–63).
- The base seed defaults to `0` in all training scripts. To change it, modify the `seed` argument in `make_env()`.
- PyBoy loads a fixed binary `.state` file at the start of each episode, ensuring identical initial game state across all runs.
- All parallel environments are managed by `SubprocVecEnv` with deterministic process ordering.

To run with a different seed:
```bash
# Edit the seed variable in the training script, then run:
python src/v3/baseline_fast_v3.py
```

---

*For questions about the codebase, see the per-version READMEs in `src/baseline/`, `src/v2/`, and `src/v3/`.*

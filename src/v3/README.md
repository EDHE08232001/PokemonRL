# PokemonRL V3 — Recurrent Memory, Semantic Text, & Topological Graph

V3 is a research extension of the V2 coordinate-based exploration agent. It adds three new architectural components designed to push the agent beyond spatial novelty toward **narrative understanding** and **structural map reasoning**.

---

## Architectural Differences from V2

| Feature | V2 | V3 |
|---------|-----|-----|
| **RL Algorithm** | PPO (`MultiInputPolicy`) | RecurrentPPO (`MultiInputLstmPolicy`) |
| **Temporal Memory** | Recent actions buffer (3-step) | LSTM (128 hidden, 1 layer) |
| **Text Understanding** | None | RAM text hooking + Gen 1 hex decoding |
| **Map Representation** | Flat coordinate counting | Coordinate counting + directed graph (NetworkX) |
| **Warp Handling** | Implicit (coordinate jumps) | Explicit warp edge detection + discovery reward |
| **Observation Space** | screens, health, level, badges, events, map, recent_actions | screens, health, level, badges, events, map, **text_hash**, **graph_distance** |
| **New Dependencies** | — | `sb3-contrib`, `networkx` |

---

## How Each Component Works

### 1. Recurrent Memory (LSTM)

**Problem**: V2 uses a fixed-length `recent_actions` buffer (3 steps). The agent has no memory of what happened before those 3 steps, making it difficult to navigate multi-room buildings, complete multi-step NPC dialogues, or remember which doors it has already tried.

**Solution**: Replace PPO with **RecurrentPPO** from `sb3-contrib`, which wraps the policy network with an LSTM layer. The LSTM maintains a hidden state across the entire episode, giving the agent an implicit working memory.

- **Policy**: `MultiInputLstmPolicy` — processes dict observations through CNN/MLP feature extractors, then feeds the combined features through an LSTM before the actor/critic heads.
- **Config**: `lstm_hidden_size=128`, `n_lstm_layers=1` — kept lightweight to maintain high throughput with 64 parallel environments.
- **Observation change**: The `recent_actions` key is removed from the observation dict since the LSTM natively captures action history in its hidden state.

### 2. Semantic Exploration via RAM Text Hooking

**Problem**: V2 rewards spatial novelty (new tiles visited) but is completely blind to text content. The agent has no incentive to read signs, talk to NPCs, or progress the story — it only cares about stepping on new coordinates.

**Solution**: Hook into the Game Boy's WRAM to read the active text buffer, decode it using the Gen 1 character map, and reward the agent for encountering new dialogue.

**How it works**:
1. **RAM Hook**: On every step, read 20 bytes starting at WRAM address `0xCF4B` (the active text window buffer).
2. **Gen 1 Decoding**: Translate raw hex values to characters:
   - `0x7F` → Space
   - `0x80–0x99` → A–Z
   - `0x9A–0xB3` → a–z
   - `0xF6–0xFF` → 0–9
   - `0x50` → String terminator
3. **Novelty Check**: If the decoded string is >3 characters and hasn't been seen before, add it to `seen_dialogue` and grant a **+0.5 intrinsic reward**.
4. **Observation**: The current text string is hashed (MD5) into an 8-byte `uint8` vector (`text_hash`) and included in the observation dict.

### 3. Topological Graph Navigation

**Problem**: V2's exploration map is a flat 2D grid. When the agent enters a door (warp zone), its coordinates teleport to a completely different location. The agent has no structural understanding of how maps connect.

**Solution**: Build a **directed graph** (NetworkX `DiGraph`) where nodes are `(map_id, row, col)` tuples and edges represent observed movement transitions.

**How it works**:
1. **Node Tracking**: Every step, compute `current_node = (map_id, y, x)`. If it differs from `previous_node`, add a directed edge.
2. **Warp Detection**: If the map ID changed between consecutive nodes, the edge is flagged as a "warp edge". Discovering a new map ID via a warp grants a **+2.0 intrinsic reward**.
3. **Graph Distance**: Compute shortest path length from the root node (starting position in Pallet Town) to the current node. This integer distance is clamped to [0, 255] and fed into the observation as `graph_distance`.
4. **Purpose**: The graph distance gives the agent a sense of topological progression — "how far am I from home through the map structure?" — independent of pixel coordinates.

---

## Running V3

All scripts use **absolute path resolution** — they work correctly from any working directory.

### Install Dependencies

```bash
pip install .[v3]
```

### Place ROM

```bash
cp /path/to/PokemonRed.gb ROM_INPUT/PokemonRed.gb
```

### Train

```bash
python src/v3/baseline_fast_v3.py
```

This will:
1. Spin up 64 parallel Pokemon Red emulators
2. Begin RecurrentPPO training with `MultiInputLstmPolicy`
3. Save checkpoints to `src/v3/runs/` (every `ep_length // 2` steps)
4. Log to TensorBoard in `src/v3/runs/`

### Resume from Checkpoint

```bash
echo "runs/poke_XXXXXXX_steps" | python src/v3/baseline_fast_v3.py
```

### Run a Trained Model Interactively

```bash
python src/v3/run_pretrained_interactive.py
```

This will:
1. Auto-detect the most recent `.zip` checkpoint from `src/v3/runs/`
2. Load the `RecurrentPPO` model with LSTM state tracking
3. Open an SDL2 window at 6× emulation speed

**Toggle AI control** by creating/editing `src/v3/agent_enabled.txt`:
- Write `yes` on the first line to let the AI play
- Write `no` (or delete the file) to play manually via the SDL2 window

The interactive script properly maintains LSTM hidden states across steps, passing `state` and `episode_start` to `model.predict()` for correct recurrent inference.

### Continuous Training

```bash
src/v3/go_forever.sh
```

This loops indefinitely, finding the latest checkpoint and resuming training.

### Monitor Training

```bash
tensorboard --logdir src/v3/runs/
# Open http://localhost:6006
```

V3 adds these TensorBoard metrics under `v3/`:
- `v3/mean_dialogue_count` / `v3/max_dialogue_count` — unique dialogue strings discovered
- `v3/mean_graph_nodes` / `v3/max_graph_nodes` — topological graph size
- `v3/mean_maps_discovered` / `v3/max_maps_discovered` — unique map IDs reached via warps

---

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sb3-contrib` | ≥2.3.0 | RecurrentPPO (LSTM policy for Stable Baselines 3) |
| `networkx` | ≥3.0 | Directed graph for topological map navigation |

All other dependencies are shared with V2 (see `pyproject.toml` at the project root).

---

## File Structure

```
src/v3/
├── red_gym_env_v3.py              # Gym environment (LSTM obs, text hooks, graph nav)
├── baseline_fast_v3.py            # Training script (RecurrentPPO, 64 parallel envs)
├── run_pretrained_interactive.py  # Interactive play with trained model (LSTM state tracking)
├── tensorboard_callback_v3.py     # TensorBoard logging (+ V3 metrics)
├── global_map.py                  # Map coordinate conversion (shared with V2)
├── stream_agent_wrapper.py        # WebSocket live map streaming (shared with V2)
├── events.json                    # Event flag names
├── map_data.json                  # Map region coordinate data
├── go_forever.sh                  # Continuous training wrapper
└── README.md                      # This file
```

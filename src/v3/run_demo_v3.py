"""
V3 Demo Runner — records a full-color side-by-side video:

  LEFT  │  Game screen (RGB, 3× upscale → 480 × 432)
  RIGHT │  Full-world exploration heatmap (inferno colormap, same height)

Stats (step count, tiles visited, maps discovered, events, deaths) are
overlaid in the top-left corner of the game panel every frame.

Usage
-----
    # Auto-detect latest checkpoint in src/v3/runs/
    python src/v3/run_demo_v3.py

    # Explicit checkpoint
    python src/v3/run_demo_v3.py --checkpoint path/to/ckpt.zip

    # Longer demo (default = 2^16 ≈ 65 k steps)
    python src/v3/run_demo_v3.py --steps 131072 --speed 2

Output
------
    demo_v3.mp4   (project root, or --output /path/to/file.mp4)
"""

import sys
import os
import argparse
import glob
import logging
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import mediapy as media

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))   # expose src/
sys.path.insert(0, str(_SCRIPT_DIR))          # expose src/v3/

logging.getLogger("pyboy").setLevel(logging.CRITICAL + 1)

from v3.red_gym_env_v3 import RedGymEnv           # noqa: E402
from sb3_contrib import RecurrentPPO              # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────
GAME_H   = 432    # 144 × 3  (native GB resolution upscaled 3×)
GAME_W   = 480    # 160 × 3
MAP_W    = GAME_H  # map panel — square, same height
VIDEO_FPS   = 20  # frames per second in the output video
RECORD_EVERY = 4  # capture one video frame every N agent steps


# ── helpers ──────────────────────────────────────────────────────────────────

def _inferno(arr_f32: np.ndarray) -> np.ndarray:
    """
    Fast inferno colormap without importing matplotlib at runtime.
    arr_f32: float32 in [0, 1]  →  uint8 RGB (H, W, 3)
    """
    from matplotlib.cm import inferno
    return (inferno(arr_f32)[:, :, :3] * 255).astype(np.uint8)


def colorize_explore_map(explore_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize the full global exploration map (484 × 476 uint8) to (target_h, target_w)
    and apply inferno colormap so visited tiles glow orange/yellow on a dark background.
    """
    pil = Image.fromarray(explore_map, mode="L").resize(
        (target_w, target_h), Image.NEAREST
    )
    normed = np.array(pil).astype(np.float32) / 255.0
    return _inferno(normed)


def make_composite_frame(
    game_rgb: np.ndarray,     # (144, 160, 3) from pyboy.screen.ndarray[:,:,:3]
    explore_map: np.ndarray,  # (484, 476) global map
    step: int,
    stats: dict,
) -> np.ndarray:
    """
    Build one (GAME_H × (GAME_W + MAP_W), 3) composite frame.
    """
    # ── left panel: upscale game ─────────────────────────────────────────────
    game_pil = Image.fromarray(game_rgb, mode="RGB").resize(
        (GAME_W, GAME_H), Image.NEAREST
    )

    # ── right panel: colorized exploration heatmap ───────────────────────────
    map_rgb = colorize_explore_map(explore_map, GAME_H, MAP_W)
    map_pil = Image.fromarray(map_rgb, mode="RGB")

    # ── stitch side-by-side ──────────────────────────────────────────────────
    canvas = Image.new("RGB", (GAME_W + MAP_W, GAME_H), (10, 10, 10))
    canvas.paste(game_pil, (0, 0))
    canvas.paste(map_pil,  (GAME_W, 0))

    draw = ImageDraw.Draw(canvas)

    # ── stats overlay (top-left of game panel) ───────────────────────────────
    lines = [
        f"Step     {step:>8,}",
        f"Tiles    {stats.get('coord_count', 0):>8,}",
        f"Maps     {stats.get('maps_discovered', 0):>8}",
        f"Nodes    {stats.get('graph_nodes', 0):>8,}",
        f"Events   {stats.get('event', 0):>8.2f}",
        f"Deaths   {stats.get('deaths', 0):>8}",
    ]
    for i, line in enumerate(lines):
        y = 8 + i * 20
        draw.text((9, y + 1), line, fill=(0, 0, 0))      # shadow
        draw.text((8, y),     line, fill=(180, 255, 100)) # text

    # ── panel labels (bottom centre) ─────────────────────────────────────────
    draw.text((GAME_W // 2 - 40,  GAME_H - 22), "Agent View",     fill=(190, 190, 190))
    draw.text((GAME_W + MAP_W // 2 - 60, GAME_H - 22), "Exploration Map", fill=(190, 190, 190))

    return np.array(canvas)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Record a V3 demo video.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .zip checkpoint (auto-detects latest in src/v3/runs/ if omitted).",
    )
    parser.add_argument(
        "--steps", type=int, default=2 ** 16,
        help="Max agent steps to record (default 65536, ~3–5 min at 2× speed).",
    )
    parser.add_argument(
        "--speed", type=int, default=2,
        help="PyBoy emulation speed multiplier (default 2).",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(_PROJECT_ROOT / "demo_v3.mp4"),
        help="Output .mp4 path (default: <project_root>/demo_v3.mp4).",
    )
    args = parser.parse_args()

    # ── locate checkpoint ─────────────────────────────────────────────────────
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        zips = glob.glob(str(_SCRIPT_DIR / "runs" / "*.zip"))
        if not zips:
            print("ERROR: No checkpoint found in src/v3/runs/.  Use --checkpoint.")
            sys.exit(1)
        checkpoint = max(zips, key=os.path.getmtime)
    print(f"[demo] Checkpoint : {checkpoint}")
    print(f"[demo] Max steps  : {args.steps:,}")
    print(f"[demo] Speed      : {args.speed}×")
    print(f"[demo] Output     : {args.output}")

    # ── environment ───────────────────────────────────────────────────────────
    sess_path = _SCRIPT_DIR / f"demo_session_{str(uuid.uuid4())[:8]}"
    env_config = {
        "headless":        False,
        "save_final_state": False,
        "early_stop":      False,
        "action_freq":     24,
        "init_state":      str(_PROJECT_ROOT / "saves" / "init.state"),
        "max_steps":       args.steps,
        "sound":           False,
        "print_rewards":   False,
        "save_video":      False,   # we record ourselves (color)
        "fast_video":      False,
        "session_path":    sess_path,
        "gb_path":         str(_PROJECT_ROOT / "ROM_INPUT" / "PokemonRed.gb"),
        "debug":           False,
        "reward_scale":    0.5,
        "explore_weight":  1.0,
    }

    set_random_seed(42)
    env = RedGymEnv(env_config)
    env.pyboy.set_emulation_speed(args.speed)

    # ── load model ────────────────────────────────────────────────────────────
    model = RecurrentPPO.load(
        checkpoint, env=env,
        custom_objects={"lr_schedule": 0, "clip_range": 0},
    )
    print("[demo] Model loaded — starting recording loop ...")

    obs, _info = env.reset()
    lstm_states   = None
    episode_start = True
    frames        = []

    for step in range(args.steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states,
            episode_start=episode_start, deterministic=False,
        )
        episode_start = False
        obs, _reward, terminated, truncated, _info = env.step(action)
        env.render()  # keep SDL2 window live

        # ── capture composite frame ──────────────────────────────────────────
        if step % RECORD_EVERY == 0:
            game_rgb     = env.pyboy.screen.ndarray[:, :, :3].copy()  # RGB color
            explore_gray = env.explore_map.copy()
            stats        = env.agent_stats[-1] if env.agent_stats else {}
            frames.append(make_composite_frame(game_rgb, explore_gray, step, stats))

        if step % 2000 == 0 and step > 0:
            print(f"[demo]   step {step:,}/{args.steps:,}  — {len(frames)} frames")

        if terminated or truncated:
            print(f"[demo] Episode ended at step {step:,}.")
            break

    env.close()

    # ── write video ───────────────────────────────────────────────────────────
    if not frames:
        print("ERROR: No frames captured.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[demo] Writing {len(frames)} frames at {VIDEO_FPS} fps → {out_path}")

    frame_shape = (GAME_H, GAME_W + MAP_W)   # (height, width) in pixels
    with media.VideoWriter(str(out_path), frame_shape, fps=VIDEO_FPS) as writer:
        for f in frames:
            writer.add_image(f)

    print(f"[demo] Done!  →  {out_path}")


if __name__ == "__main__":
    main()

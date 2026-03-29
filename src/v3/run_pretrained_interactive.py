"""
Interactive play with a trained V3 RecurrentPPO model.

Loads the most recent checkpoint from runs/ and runs the agent visibly.
Toggle AI control by writing 'yes'/'no' to agent_enabled.txt.
When AI is disabled, the emulator just ticks (manual play via SDL2 window).
"""

import sys
import os
import logging
from os.path import exists
from pathlib import Path
import uuid
import time
import glob

# Silence PyBoy's sound module — it emits CRITICAL "Buffer overrun!" every tick,
# which inflates SLURM .out files to 100+ GB over a long training run.
logging.getLogger("pyboy").setLevel(logging.CRITICAL + 1)

# Resolve absolute paths relative to this script's location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Ensure the script's directory and src/ are on sys.path for local imports
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from v3.red_gym_env_v3 import RedGymEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import set_random_seed


def make_env(rank, env_conf, seed=0):
    """Create a closure that initializes a RedGymEnv."""
    def _init():
        env = RedGymEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init


def get_most_recent_zip_with_age(folder_path):
    """Find the most recent .zip checkpoint and return (path, age_in_hours)."""
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))

    if not zip_files:
        return None, None

    most_recent_zip = max(zip_files, key=os.path.getmtime)

    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600

    return most_recent_zip, age_in_hours


if __name__ == '__main__':

    sess_path = _SCRIPT_DIR / f'session_{str(uuid.uuid4())[:8]}'
    ep_length = 2**23  # very long episode for interactive play

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': str(_PROJECT_ROOT / 'saves' / 'init.state'), 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': str(_PROJECT_ROOT / 'ROM_INPUT' / 'PokemonRed.gb'), 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25
    }

    # Single environment for interactive mode
    num_cpu = 1
    env = make_env(0, env_config)()

    # Auto-detect most recent checkpoint
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age(str(_SCRIPT_DIR / "runs"))
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"using checkpoint: {file_name}, which is {time_since:.2f} hours old")
    else:
        print("ERROR: No checkpoint found in runs/ directory")
        sys.exit(1)

    print('\nloading checkpoint')
    model = RecurrentPPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    obs, info = env.reset()
    # RecurrentPPO requires LSTM states to be tracked across steps
    lstm_states = None
    episode_start = True

    while True:
        # Toggle AI control at runtime via agent_enabled.txt
        try:
            with open(_SCRIPT_DIR / "agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=False)
            episode_start = False
            obs, rewards, terminated, truncated, info = env.step(action)
        else:
            # Manual mode: just tick emulator (player uses SDL2 window controls)
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1
        env.render()
        if truncated:
            break
    env.close()

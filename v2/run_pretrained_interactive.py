"""
Interactive play with a trained V2 PPO model.

Loads the most recent checkpoint from runs/ and runs the agent visibly.
Toggle AI control by writing 'yes'/'no' to agent_enabled.txt.
When AI is disabled, the emulator just ticks (manual play via SDL2 window).
"""

import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank, env_conf, seed=0):
    """Create a closure that initializes a RedGymEnv for SubprocVecEnv."""
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

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23  # very long episode for interactive play

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../saves/init.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../ROM_INPUT/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
    }

    # Single environment for interactive mode
    num_cpu = 1
    env = make_env(0, env_config)()

    # Auto-detect most recent checkpoint
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age("runs")
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"using checkpoint: {file_name}, which is {time_since} hours old")

    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    obs, info = env.reset()
    while True:
        # Toggle AI control at runtime via agent_enabled.txt
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
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

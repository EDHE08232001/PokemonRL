"""
Interactive play with a trained Baseline PPO model.

Loads a checkpoint and runs the agent in a visible window.
Toggle AI control by writing 'yes'/'no' to agent_enabled.txt.
"""

from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
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


if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23  # very long episode for interactive play

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../../saves/has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../../ROM_INPUT/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
    }

    # Single environment (no parallelism) for interactive mode
    num_cpu = 1
    env = make_env(0, env_config)()

    file_name = 'session_4da05e87_main_good/poke_439746560_steps'

    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    obs, info = env.reset()
    while True:
        action = 7  # default: pass (no-op)
        # Read agent_enabled.txt to toggle AI control at runtime
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
    env.close()

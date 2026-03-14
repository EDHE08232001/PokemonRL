"""
Baseline training script (fast variant).

Trains PPO with CnnPolicy on 16 parallel Pokemon Red environments.
Uses KNN frame-based exploration. Supports checkpoint resumption and W&B logging.
"""

from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


def make_env(rank, env_conf, seed=0):
    """Create a closure that initializes a RedGymEnv for SubprocVecEnv."""
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    use_wandb_logging = False
    ep_length = 2048 * 10  # steps per episode per environment
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../saves/has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../ROM_INPUT/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
        'explore_weight': 3
    }

    print(env_config)

    num_cpu = 16  # parallel environments
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                             name_prefix='poke')

    callbacks = [checkpoint_callback, TensorboardCallback()]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # Resume from checkpoint if available
    file_name = 'session_e41c9eff/poke_38207488_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # PPO with CnnPolicy: 3 epochs per update, batch size 128
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log=sess_path)

    # Train for ~5000 episodes
    model.learn(total_timesteps=(ep_length) * num_cpu * 5000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()

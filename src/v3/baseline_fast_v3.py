"""
V3 training script.

Trains RecurrentPPO (LSTM) with MultiInputLstmPolicy on parallel Pokemon Red
environments. Adds temporal memory, semantic text rewards, and topological
graph navigation on top of V2's coordinate-based exploration.

Key changes from V2:
- RecurrentPPO from sb3-contrib instead of PPO from stable-baselines3
- MultiInputLstmPolicy instead of MultiInputPolicy
- LSTM config: 128 hidden units, 1 layer (lightweight for high env throughput)
"""

import sys
from os.path import exists
from pathlib import Path

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
from v3.stream_agent_wrapper import StreamWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from v3.tensorboard_callback_v3 import TensorboardCallback


def make_env(rank, env_conf, seed=0):
    """Create a closure that initializes a RedGymEnv wrapped with StreamWrapper."""
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf),
            stream_metadata={
                "user": "v3-default",
                "env_id": rank,
                "color": "#994477",
                "extra": "",
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    use_wandb_logging = False
    ep_length = 2048 * 80  # steps per episode per environment
    sess_id = "runs"
    sess_path = _SCRIPT_DIR / sess_id

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': str(_PROJECT_ROOT / 'saves' / 'init.state'), 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': str(_PROJECT_ROOT / 'ROM_INPUT' / 'PokemonRed.gb'), 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25
    }

    print(env_config)

    num_cpu = 64  # parallel environments
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length // 2, save_path=sess_path,
                                             name_prefix="poke")

    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v3-a",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # Read checkpoint path from stdin (for pipeline use), or start fresh
    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()
        # Resolve relative checkpoint paths against the script directory
        if file_name and not Path(file_name).is_absolute():
            file_name = str(_SCRIPT_DIR / file_name)

    train_steps_batch = ep_length // 64  # n_steps per PPO update

    # LSTM policy kwargs — lightweight for high parallel throughput
    policy_kwargs = dict(
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = RecurrentPPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        # RecurrentPPO with MultiInputLstmPolicy: 1 epoch per update, batch size 512
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=512,
            n_epochs=3,
            gamma=0.995,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=sess_path,
        )

    print(model.policy)

    model.learn(total_timesteps=(ep_length) * num_cpu * 10000, callback=CallbackList(callbacks), tb_log_name="poke_rppo")

    if use_wandb_logging:
        run.finish()

import gym
import stable_baselines3
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from stable_baselines3.common.env_util import make_vec_env
import env
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='source', type=str, help='Training environment')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()

def main():

    source_env = make_vec_env('CustomHopper-source-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    train_env = source_env

    # PPO model
    model = PPO('MlpPolicy', train_env, learning_rate=0.00025, verbose=1, device=args.device)
    
    # Train
    model.learn(total_timesteps=3000000)
    
    # Save the model
    model.save('model_5')


if __name__ == '__main__':
    main()

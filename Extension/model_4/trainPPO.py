"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import stable_baselines3
import argparse
from stable_baselines3 import PPO
from env.custom_hopper import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='source', type=str, help='Training environment')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--n-episodes', default=1000000, type=int, help='Number of training episodes') # Re change numepisodes in 100000
    parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()

def main():
    source_env = make_vec_env('CustomHopper-source-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    target_env = make_vec_env('CustomHopper-target-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    if args.train == 'source':
        train_env = source_env
    else:
        train_env = target_env

    model = PPO('MlpPolicy', env=train_env, verbose=1, device='cpu')
    model.learn(total_timesteps=3000000)
    model.save('Model_4')


if __name__ == '__main__':
    main()
"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym
import os
import env
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Model_4', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')
    parser.add_argument('--test', default='source', type=str, help='Testing environment')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()
def main():
    if args.test == 'source':
        test_env = gym.make('CustomHopper-source-v0')
    else:
        test_env = gym.make('CustomHopper-target-v0')
    model = PPO.load(args.model)

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = test_env.reset()
        
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = test_env.step(action)
            if args.render:
                test_env.render()
                
            test_reward += reward
        print(f"Episode: {episode} | Return: {test_reward}")
	

if __name__ == '__main__':
	main()    
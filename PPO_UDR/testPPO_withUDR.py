"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym
import os
import env
from env.custom_hopper import *
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='modelPPO_UDR', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()

def main():
    test_env = gym.make('CustomHopper-target-v0')
    model = PPO.load(args.model)
    returns = np.zeros(args.episodes)
    # Test loop
    for episode in range(args.episodes):
        done = False
        test_return = 0
        state = test_env.reset()
        
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = test_env.step(action)
            if args.render:
                test_env.render()
                
            test_return += reward
        returns[episode] = test_return
        print(f"Episode: {episode} | Return: {test_return}")
    mean_reward = np.mean(returns)
    print(f"Mean return on test episodes is:{mean_reward}")

if __name__ == '__main__':
	main()    
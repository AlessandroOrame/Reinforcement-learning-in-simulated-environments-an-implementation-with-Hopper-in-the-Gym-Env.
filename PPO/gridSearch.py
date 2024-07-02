import itertools
import time
import gym
import stable_baselines3
import argparse
from stable_baselines3 import PPO
from env.custom_hopper import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os
import torch.nn as nn 

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
    target_env = make_vec_env('CustomHopper-target-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    if args.train == 'source':
        train_env = source_env
        eval_env = gym.make('CustomHopper-source-v0')
    else:
        train_env = target_env
        eval_env = gym.make('CustomHopper-target-v0')
   
   # Parameters
    param_grid = {
        'learning_rate': [0.025, 0.0025, 0.00025],
        'gamma': [0.90, 0.95, 0.99]
    }

    # Grid of parameters
    param_combinations = list(itertools.product(*(param_grid[key] for key in param_grid)))

    best_mean_reward = -np.inf
    best_params = None
    start = time.time()

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Testing parameters: {param_dict}")
        
        model = PPO(
            'MlpPolicy',
            env=train_env,
            learning_rate=param_dict['learning_rate'],
            gamma = param_dict['gamma'],
            verbose=0,
            device=args.device
        )

        model.learn(total_timesteps=1000000)
        end= time.time()
        print(f"Time:{end-start}")

        mean_reward = evaluate_model(model, eval_env)
        print(f"Mean reward: {mean_reward}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_params = param_dict
            best_model = model

    print(f"Best parameters: {best_params}")
    print(f"Best mean reward: {best_mean_reward}")


# Function for validation 
def evaluate_model(model, env, n_episodes=100):
    all_rewards = []
    for _ in range(n_episodes):
        done = False
        state = env.reset()
        episode_rewards = 0
        while not done:
            action, _ = model.predict(state, deterministic=True) # applied in a deterministic way
            state, reward, done, _ = env.step(action)
            episode_rewards += reward
        all_rewards.append(episode_rewards)
    return np.mean(all_rewards)

if __name__ == '__main__':
    main() 
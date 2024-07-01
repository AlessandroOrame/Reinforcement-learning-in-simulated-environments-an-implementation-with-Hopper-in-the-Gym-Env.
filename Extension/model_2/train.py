import os
import env
import gym
import argparse
import stable_baselines3
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='source', type=str, help='Training environment')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()

def make_env(env_id):
    def _init():
        return gym.make(env_id)
    return _init

def main():
    if args.train == 'source':
        env_id = 'CustomHopper-source-v0'
    else:
        env_id = 'CustomHopper-target-v0'

    envs = [make_env(env_id) for _ in range(N_ENVS)]
    train_env = DummyVecEnv(envs)

    # PPO model
    model = PPO('MlpPolicy', train_env, learning_rate=0.00025, verbose=1, device=args.device)
    
    # Train
    model.learn(total_timesteps=3000000)
    
    # Save the model
    model.save('model_2')


if __name__ == '__main__':
    main()

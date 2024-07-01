"""Test an RL agent on the OpenAI Gym Hopper environment"""
import os
import env
import gym
import torch
import argparse
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model_3', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')
    parser.add_argument('--test', default='source', type=str, help='Testing environment')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()
def main():

    test_env = gym.make('CustomHopper-source-v0')
    
    model = PPO.load(args.model) 

    #initialize values for output statistics
    metric_obstacle_1 = 0
    metric_obstacle_2 = 0
    metric_obstacle_3 = 0

    obstacles_positions = []
    for i in range (3):
        obstacles_positions.append(test_env.env.sim.data.get_body_xpos(f'obstacle{i+1}'))

    mean_return = 0

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
    
        #print(f"Episode: {episode} | Return: {test_reward}")
        mean_return += test_reward

        # index of Hopper foot
        foot_id = test_env.model.body_name2id("foot")
        # foot position
        foot_position = test_env.sim.data.body_xpos[foot_id]
        # x-coordinate of foot 
        foot_x_position = foot_position[0]

        # compute statistics for obstalce jumping
        if foot_x_position > test_env.env.sim.data.get_body_xpos('obstacle3')[0]+0.15:
            metric_obstacle_1 += 1
            metric_obstacle_2 += 1
            metric_obstacle_3 += 1
        elif foot_x_position > test_env.env.sim.data.get_body_xpos('obstacle2')[0]+0.15:
            metric_obstacle_1 += 1
            metric_obstacle_2 += 1
        elif foot_x_position > test_env.env.sim.data.get_body_xpos('obstacle1')[0]+0.15:
            metric_obstacle_1 += 1

    print(f'Average Return:{mean_return/args.episodes}')        
    print(f'percentage of success obstacle1:{metric_obstacle_1/(args.episodes)*100}%') 
    print(f'percentage of success obstacle2:{metric_obstacle_2/(args.episodes)*100}%')
    print(f'percentage of success obstacle3:{metric_obstacle_3/(args.episodes)*100}%')       

if __name__ == '__main__':
	main()    

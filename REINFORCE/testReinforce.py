"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agentReinforce import Agent, Policy
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='BestParamsModel_REINFORCE.mdl', type=str, help='Model path') 
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():
	
	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)  # Upload the policy of the trained model 
	agent = Agent(policy, device=args.device)

	rewards = np.ones(args.episodes)

	# Test loop
	for episode in range(args.episodes):
		done = False
		test_return = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True) # Evaluation=True to apply policy in a deterministic way

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_return += reward

		rewards[episode] = test_return

		print(f"Episode: {episode} | Return: {test_return}")
	
	mean_return_test = np.mean(rewards)
	print(f"Mean return on {args.episodes} test episodes is:{mean_return_test}")
	

if __name__ == '__main__':
	main()
"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agentActorCritic import Agent, ActorPolicy, CriticPolicy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actormodel', default='models/BestActorParamsModel_100mila.mdl', type=str, help='Actor path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')
    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	'''print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())'''
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	APolicy = ActorPolicy(observation_space_dim, action_space_dim)
	CPolicy = CriticPolicy(observation_space_dim, action_space_dim)

	APolicy.load_state_dict(torch.load(args.actormodel), strict=True)
	CPolicy.load_state_dict(torch.load('models/BestCriticParamsModel_100mila.mdl'), strict=True)

	agent = Agent(APolicy, CPolicy, device=args.device)

	mean_return = 0
	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")
		mean_return += test_reward
	
	print(f"Average Return: {mean_return/args.episodes}")

if __name__ == '__main__':
	main()
"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import torch
import gym
from env.custom_hopper import *
from agentActorCritic import Agent, ActorPolicy, CriticPolicy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--actormodel', default='ActorModel.mdl', type=str, help='Actor path')
	parser.add_argument('--criticmodel', default='CriticModel.mdl', type=str, help='Critic path')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
	parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')
	
	return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	#initialize policy
	APolicy = ActorPolicy(observation_space_dim, action_space_dim)
	CPolicy = CriticPolicy(observation_space_dim, action_space_dim)

	#load the train actor and critic models
	APolicy.load_state_dict(torch.load(args.actormodel), strict=True)
	CPolicy.load_state_dict(torch.load(args.criticmodel), strict=True)

	#initialize the agent with the trained policy
	agent = Agent(APolicy, CPolicy, device=args.device)

	mean_return = 0
	for episode in range(args.episodes):
		done = False
		test_return = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_return += reward

		print(f"Episode: {episode} | Return: {test_return}")
		mean_return += test_return
	
	print(f"Average Return: {mean_return/args.episodes}")

if __name__ == '__main__':
	main()

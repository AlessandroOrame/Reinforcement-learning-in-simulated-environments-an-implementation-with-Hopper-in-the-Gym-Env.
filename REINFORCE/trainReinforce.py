"""Train an RL agent on the OpenAI Gym Hopper environment using REINFORCE algorithms
"""

import os
import csv
import gym
import torch
import time
import argparse
import numpy as np
from env.custom_hopper import *
from agentReinforce import Agent, Policy


def parse_args():
	parser = argparse.ArgumentParser() 
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--n-episodes', default=50000, type=int, help='Number of training episodes')
	parser.add_argument('--val-episodes', default=10, type=int, help='Number of validation episodes')
	parser.add_argument('--val-every', default=1000, type=int, help='Validate model every <> episodes')
	return parser.parse_args()

args = parse_args()

# Directory for validation logs
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
Reinforce_log_file_path = os.path.join(log_dir, 'Reinforce_policy_convergence.csv')


def main():

	env = gym.make('CustomHopper-source-v0')
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

	# Initialize log files
	with open(Reinforce_log_file_path, mode='w', newline='') as policy_convergence_log_file:
		policy_convergence_log_writer = csv.writer(policy_convergence_log_file)
		policy_convergence_log_writer.writerow(['Episode', 'Return'])

	# Initialize values to find the best model  
	best_reward = 0
	best_params = None 
	best_iter = 0
	start_time = time.time()
    
	# Training loop
	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over
			action, action_probabilities = agent.get_action(state)
			previous_state = state
			state, reward, done, info = env.step(action.detach().cpu().numpy())
			agent.store_outcome(previous_state, state, action_probabilities, reward, done)
			train_reward += reward

		agent.update_policy(baseline = 0) # Change input to change baseline
		agent.restart()

		# Validation for logs
		if (episode+1)%args.val_every == 0:
			val_rewards = np.zeros(args.val_episodes)
			# Validation loop
			for val_episode in range(args.val_episodes):
				val_done = False
				val_reward = 0
				state = env.reset()
			
				while not val_done:
					action, _ = agent.get_action(state, evaluation=True)
					state, reward, val_done, info = env.step(action.detach().cpu().numpy())
					val_reward += reward

				val_rewards[val_episode] = val_reward

			# Write the mean validation return in the log file
			mean_return = val_rewards.mean()
			with open(Reinforce_log_file_path, mode='a', newline='') as policy_convergence_log_file:
				policy_convergence_log_writer = csv.writer(policy_convergence_log_file)
				policy_convergence_log_writer.writerow([episode, mean_return])

			# Update the best model if necessary
			if mean_return > best_reward:
				best_reward = mean_return
				best_params = agent.policy.state_dict()
				best_iter = episode 

	end_time = time.time()
	total_time = end_time - start_time
	print(f'Computational time for training and validation: {total_time: .3f} seconds') 
	
	# Save the last weights values
	last_params = agent.policy.state_dict()
	torch.save(last_params, 'LastParamsModel_REINFORCE.mdl')

	# Save the best performing weights values
	torch.save(best_params, "BestParamsModel_REINFORCE.mdl")
	print(f"best training iteration: {best_iter}, with reward: {best_reward}")

if __name__ == '__main__':
	main()
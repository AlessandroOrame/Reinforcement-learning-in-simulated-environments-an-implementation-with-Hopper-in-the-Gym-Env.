import os
import csv
import time
import gym
import torch
import argparse
import numpy as np
from env.custom_hopper import *
import matplotlib.pyplot as plt
from agentActorCritic import Agent, ActorPolicy, CriticPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=50000, type=int, help='Number of training episodes')
    parser.add_argument('--validate-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
AC_policy_convergence_log_file_path = os.path.join(log_dir, 'ActorCritic.csv')


def main():

    env = gym.make('CustomHopper-source-v0')

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    A_policy = ActorPolicy(observation_space_dim, action_space_dim)
    C_policy = CriticPolicy(observation_space_dim, action_space_dim)
    agent = Agent(A_policy, C_policy, device=args.device)

    # Initialize log files
    with open(AC_policy_convergence_log_file_path, mode='w', newline='') as AC_policy_convergence_log_file:
        AC_policy_convergence_log_writer = csv.writer(AC_policy_convergence_log_file)
        AC_policy_convergence_log_writer.writerow(['Episode', 'Return'])

    best_reward = -np.inf
    best_actor_params = None
    best_critic_params = None
    best_trainEpisode = 0
    start_time= time.time()

    # Interleave data collection to policy updates
    # Training loop
    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()
        current_step = 0
        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            prev_statevalue = agent.get_statevalue(previous_state)
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            current_step += 1
            curr_statevalue = agent.get_statevalue(state)
            agent.store_outcome(previous_state, state, action_probabilities, reward, done, prev_statevalue, curr_statevalue)

            if (current_step % 50 == 0) or (done):
                agent.update_policy()
                agent.restart()

            train_reward += reward

        if (episode+1)%args.validate_every == 0:
            
            val_rewards = np.zeros(10)

            # Validation loop
            for val_episode in range(10):
                val_done = False
                test_reward = 0
                state = env.reset()

                while not val_done:
                    action, _ = agent.get_action(state, evaluation=True)
                    state, reward, val_done, info = env.step(action.detach().cpu().numpy())
                    test_reward += reward

                val_rewards[val_episode] = test_reward

            mean_reward = val_rewards.mean()

            with open(AC_policy_convergence_log_file_path, mode='a', newline='') as policy_convergence_log_file:
                policy_convergence_log_writer = csv.writer(policy_convergence_log_file)
                policy_convergence_log_writer.writerow([episode, mean_reward])
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_actor_params = agent.actor_policy.state_dict()
                best_critic_params = agent.critic_policy.state_dict()
                best_trainEpisode = episode
            
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Computational time for training: {total_time: .3f} seconds')

    # Save the last weights values
    last_actor_params = agent.actor_policy.state_dict()
    last_critic_params = agent.critic_policy.state_dict()
    torch.save(last_actor_params, 'LastActorModel.mdl')
    torch.save(last_critic_params, 'LastCriticModel.mdl')

    torch.save(best_actor_params, "BestActorParamsModel.mdl")
    torch.save(best_critic_params, "BestCriticParamsModel.mdl")
    print(f"Best training iteration: {best_trainEpisode}, with reward: {best_reward}")
    

if __name__ == '__main__':
    main()

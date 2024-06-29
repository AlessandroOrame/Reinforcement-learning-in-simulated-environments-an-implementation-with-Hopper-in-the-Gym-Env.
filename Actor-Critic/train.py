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
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--validate-every', default=10000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
AC_policy_convergence_log_file_path = os.path.join(log_dir, 'ActorCritic_10mila.csv')


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

    # TASK 2 and 3: interleave data collection to policy updates

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
    torch.save(last_actor_params, 'LastActorModel_10mila.mdl')
    torch.save(last_critic_params, 'LastCriticModel_10mila.mdl')

    torch.save(best_actor_params, "BestActorParamsModel_10mila.mdl")
    torch.save(best_critic_params, "BestCriticParamsModel_10mila.mdl")
    print(f"Best training iteration: {best_trainEpisode}, with reward: {best_reward}")
    
    #plot_policy_convergence(AC_policy_convergence_log_file_path)
    


def plot_policy_convergence(AC_policy_convergence_log_file_path):
    episodes = []
    returns = []

    # Read the CSV file
    with open(AC_policy_convergence_log_file_path, mode='r') as AC_policy_convergence_log_file:
        AC_policy_convergence_log_reader = csv.reader(AC_policy_convergence_log_file)
        next(AC_policy_convergence_log_reader)  # Skip the header
        for row in AC_policy_convergence_log_reader:
            episodes.append(int(row[0]))
            returns.append(float(row[1]))

    # Convert lists to numpy arrays for convenience
    episodes = np.array(episodes)
    returns = np.array(returns)

    # Calculate a running average for smoother visualization
    window_size = 100
    running_avg = np.convolve(returns, np.ones(window_size) / window_size, mode='valid')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, returns, label='Episode Return', color='blue', linewidth=0.85)
    plt.plot(episodes[window_size-1:], running_avg, color='red', label='Moving Average', linewidth=2)

    # Adding titles and labels
    plt.title('Actor Critic Policy Convergence', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Return', fontsize=14)

    # Adding a legend
    plt.legend(loc='upper left', fontsize=12)

    # Adding a grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Enhancing the overall look
    plt.style.use('seaborn-darkgrid')

    plt.show()


if __name__ == '__main__':
    main()

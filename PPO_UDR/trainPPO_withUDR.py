import gym
import stable_baselines3
import argparse
from stable_baselines3 import PPO
from env.custom_hopper import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, CallbackList
import os
import csv
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--n-episodes', default=10750, type=int, help='Number of training episodes')
    parser.add_argument('--val-episodes', default=10, type=int, help='Number of validation episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()
N_ENVS = os.cpu_count()
ENV_EPS = int(np.ceil(args.n_episodes/N_ENVS))

# Callback useful to evaluate training convergence
class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=args.val_episodes, log_path=None, verbose=1):
        super(CustomEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_count = 0
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Return"])

    def _on_step(self) -> bool:
        # Check if an episode is done
        if any(self.locals.get("dones")):
            self.episode_count += 1
            self._evaluate()
        return True

    def _evaluate(self):
        episode_returns = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_return = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_return += reward
            episode_returns.append(episode_return)

        mean_return = np.mean(episode_returns)

        if self.log_path is not None:
            with open(self.log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.episode_count, mean_return])

def main():
    train_env = make_vec_env('CustomHopper-source-random-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    target_eval_env = gym.make('CustomHopper-target-v0')
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_path_target = os.path.join(log_dir, 'PPO_TrainOnSourceWithUDR_ValOnTarget.csv')
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=ENV_EPS, verbose=1) # Callback for stopping criteria
    target_eval_callback = CustomEvalCallback(eval_env=target_eval_env, n_eval_episodes=args.eval_episodes, 
                                              log_path= log_path_target)  # Validation during training
    callback_list = [stop_callback, target_eval_callback]
    callback = CallbackList(callback_list)
    model = PPO('MlpPolicy', env=train_env, learning_rate=0.00025, gamma=0.99, verbose=1, device=args.device)
    start = time.time()
    model.learn(total_timesteps=1e9, callback=callback)
    end = time.time()
    model.save('ModelPPO_UDR')
    total_time = end - start
    print(f"Computational Time: {total_time}")

if __name__ == '__main__':
    main()

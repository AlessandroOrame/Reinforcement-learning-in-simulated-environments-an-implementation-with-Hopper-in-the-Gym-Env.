import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class ActorPolicy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean_actor = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean_actor, sigma)

        return normal_dist
    
class CriticPolicy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_mean = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):

        """
            Critic
        """
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_mean(x_critic)
        
        return state_value


class Agent(object):
    def __init__(self, actor_policy, critic_policy, device='cpu'):
        self.train_device = device
        self.actor_policy = actor_policy.to(self.train_device)
        self.critic_policy = critic_policy.to(self.train_device)
        self.actor_optimizer = torch.optim.Adam(actor_policy.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(critic_policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.prev_states = []
        self.states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.V = []
        self.Vprev = []



    def update_policy(self):
        steps = len(self.action_log_probs)
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        V_t = torch.stack(self.V, dim=0).to(self.train_device).squeeze(-1)
        V_prev = torch.stack(self.Vprev, dim=0).to(self.train_device).squeeze(-1)

        actor_loss = 0
        critic_loss = 0
        for i in range(steps):
            adv_term = rewards[i] + (1-done[i])*self.gamma * V_t[i] - V_prev[i]
            if steps > 1: 
                actor_loss += -action_log_probs[i] * adv_term.detach()
            else:
                actor_loss += -action_log_probs * adv_term.detach()
            critic_loss += F.mse_loss((rewards[i] + (1-done[i]) * self.gamma * V_t[i]).detach(), V_prev[i]) 
        
        actor_loss = actor_loss/steps
        critic_loss = critic_loss/steps
        
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph = True)
        self.critic_optimizer.step()


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.actor_policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()                
                
            return action, action_log_prob
    
    
    def get_statevalue(self, state):
        x = torch.from_numpy(state).float().to(self.train_device)
        val = self.critic_policy(x)
        return val


    def store_outcome(self, prev_state, state, action_log_prob, reward, done, prev_statevalue, curr_statevalue):
        self.prev_states.append(torch.from_numpy(prev_state).float())
        self.states.append(torch.from_numpy(state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        self.Vprev.append(prev_statevalue)
        self.V.append(curr_statevalue)
        

    def restart(self):
        self.action_log_probs = []
        self.rewards = []
        self.prev_states = []
        self.states = []
        self.done = []
        self.V = []
        self.Vprev = []
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Sequence, Tuple, Dict, Callable, List
from functools import partial
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a)) # TODO: Bug 2 is here.


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class QtOptAgent:
    def __init__(self,
                optimizer_cfg: Dict,
                state_dim: int,
                action_dim: int,
                action_lb: Sequence[float],
                action_ub: Sequence[float],
                critic_path,
                device='cuda'
                ):
        
        self.actor = Actor(state_dim, action_dim, action_ub[0]).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.load_critic_checkpoint(critic_path)

        self.optimizer_cfg = optimizer_cfg
        self.action_lb, self.action_ub = (torch.tensor(action_lb, device=device, dtype=torch.float32),
                                          torch.tensor(action_ub, device=device, dtype=torch.float32))
        
        self.action_dim = action_dim
        self.population_size = optimizer_cfg['population_size']
        self.elite_num = int(self.population_size * optimizer_cfg['elite_ratio'])
        self.num_iterations = optimizer_cfg['num_iterations']
        self.return_mean_elite = optimizer_cfg['return_mean_elite']
        self.alpha = optimizer_cfg['alpha']

        self.device = device

    def load_critic_checkpoint(self, path):
        # load pre-train critic network from path
        policy_dicts = torch.load(path)
        
        self.critic.load_state_dict(policy_dicts['critic'])
        self.actor.load_state_dict(policy_dicts['actor'])

    def act(self, obs: torch.Tensor):
        mu, dispersion = self._init_population()

        best_solution = torch.empty_like(mu)
        best_value = -np.inf

        population = torch.zeros((self.population_size, self.action_dim)).to(device=self.device)
        
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)

            with torch.no_grad():
                q1, q2 = self.critic(obs.repeat((self.population_size, 1)), population)
                values = ((q1 + q2) / 2.).squeeze(dim=-1)
            best_values, elite_idx = values.topk(self.elite_num, dim=0)

            elite = population[elite_idx]

            mu, dispersion = self._update_population_params(elite, mu, dispersion)

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()

        return mu if self.return_mean_elite else best_solution

    def _init_population(self):
        mean = torch.zeros((self.action_dim,)).to(self.device)
        dispersion = torch.ones_like(mean).to(self.device)
        return mean, dispersion

    def _sample_population(self, mu, dispersion, population):

        pop  = mu + dispersion * torch.randn_like(population)
        # TODO: do we need to change this? Double check it
        pop = torch.where(pop > self.action_lb, pop, self.action_lb)
        population = torch.where(pop < self.action_ub, pop, self.action_ub)

        return population

    def _update_population_params(self, elite, mu, dispersion):
        new_mu = torch.mean(elite, dim=0)
        new_dispersion = torch.std(elite, dim=0)

        mu = self.alpha * mu + (1 - self.alpha) * new_mu
        dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
        return mu, dispersion

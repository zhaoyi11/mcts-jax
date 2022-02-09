import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(np.prod(state_shape), 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)
        

    def forward(self, state):
        state_flatten = torch.flatten(state, start_dim=1)
        a = F.elu(self.l1(state_flatten))
        a = F.elu(self.l2(a))
        return self.l3(a)


class Critic(nn.Module):
    def __init__(self, state_shape):
        super(Critic, self).__init__()
        state_dim = np.prod(state_shape)
        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        state_flatten = torch.flatten(state, start_dim=1)
        q = F.relu(self.l1(state_flatten))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

class PolicyValue(nn.Module):
    def __init__(self, state_shape: tuple, num_actions: int):
        super(PolicyValue, self).__init__()

        state_dim = np.prod(state_shape)

        self.l1 = nn.Linear(state_dim, 50)
        self.l2 = nn.Linear(50, 50)
        self.policy = nn.Linear(50, num_actions)
        self.value = nn.Linear(50, 1)

    def forward(self, state):
        state_flatten = torch.flatten(state, start_dim=1)

        x = F.elu(self.l1(state_flatten))
        x = F.elu(self.l2(x))

        logits = self.policy(x)
        value = self.value(x)

        return logits, value
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy

from typing import Sequence, Tuple, Dict, Callable, List
from functools import partial
import copy
from .search import mcts, puct, visit_count_policy
from .networks import Actor, Critic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AZAgent:
    def __init__(self, 
                state_shape: tuple,
                num_actions: int,
                model,
                lr: float,
                discount: float,
                num_simulations: int,
                ):

        self.actor = Actor(state_shape, num_actions).to(device)
        self.critic = Critic(state_shape).to(device)
        self.model = model
        self.optimizer = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()),
                            lr=lr, weight_decay=1e-4)

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.discount = discount
        self.num_actions = num_actions
        self.num_simulations = num_simulations

    def _eval_fn(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(np.expand_dims(observation, axis=0)).to(device)
            logit = self.actor(observation)
            value = self.critic(observation)
            
        logits = logit.cpu().numpy().squeeze(axis=0)
        value = value.item()
        probs = scipy.special.softmax(logits)
        return probs, value

    def act(self, observation):
        """ Compute the agent's policy via MCTS. """
        if self.model.needs_reset:
            self.model.reset()
        
        # compute a fresh MCTS plan.
        root = mcts(
            observation,
            model = self.model,
            search_policy = puct,
            evaluation = self._eval_fn,
            num_simulations = self.num_simulations,
            num_actions = self.num_actions,
            discount = self.discount
        )

        probs = visit_count_policy(root)

        action = np.random.choice(self.num_actions, p=probs)

        return action, probs.astype(np.float32)

    def update(self, data): 
        """ Do a gradient update step on the loss. """     

        logits = self.actor(data.state) # logtis, without passing softmax, shape: [B, A] 
        value = self.critic(data.state) # [B, 1]

        with torch.no_grad():
            next_value = self.critic(data.next_state)
            target_value = data.reward + self.discount * data.not_done * next_value
        
        value_loss = torch.square(value - target_value).mean(-1) # shpae: [B,]

        # policy loss distills MCTS policy into the polic network
        policy_loss = self.criterion(logits, data.extra['pi']) 
        loss = (value_loss + policy_loss).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

        

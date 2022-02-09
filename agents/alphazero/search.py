""" Reference: https://github.com/deepmind/acme/blob/master/acme/agents/tf/mcts/search.py"""

import dataclasses
from typing import Callable, Dict
import numpy as np
import gym

from wrappers import DMC2GYMWrapper, SimulatorWrapper
from utils import types

@dataclasses.dataclass
class Node:
    """ A MCTS Node."""
    reward: float = 0
    visit_count: int = 0
    terminal: bool = False
    prior: float = 1.
    total_value: float = 0.
    children: Dict[types.Action, 'Node'] = dataclasses.field(default_factory=dict)

    def expand(self, prior: np.ndarray):
        """ Expands this node, adding child nodes. """
        assert prior.ndim == 1  # prior should be a flat vector 
        for a, p in enumerate(prior):
            self.children[a] = Node(prior=p)

    @property
    def value(self) -> types.Value: # Q(s,a)
        """ Returns the value from this node. """
        if self.visit_count:
            return self.total_value / self.visit_count
        return 0.

    @property
    def children_visits(self) -> np.ndarray:
        """Return array of visit counts of visited children."""
        return np.array([c.visit_count for c in self.children.values()])

    @property
    def children_values(self) -> np.ndarray:
        """Return array of values of visited children."""
        return np.array([c.value for c in self.children.values()])


def mcts(
    observation: types.Observation,
    model: gym.Env,
    search_policy: Callable[[Node], types.Action],
    evaluation: types.EvaluationFn,
    num_simulations: int,
    num_actions: int,
    discount: float = 1.,
    dirichlet_alpha: float = 1.,
    exploration_fraction: float = 0.,
  ) -> Node:
    """ Does MCTS. """
    
    # Evaluate the prior policy for this state.
    prior, value = evaluation(observation)
    assert prior.shape == (num_actions,)

    # Add exploration noise to the prior
    noise = np.random.dirichlet(alpha=[dirichlet_alpha] * num_actions)
    prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    # Create a fresh tree search
    root = Node()
    root.expand(prior)

    model.save_checkpoint()

    for _ in range(num_simulations):
        # start a new simulation from the top.
        trajectory = [root]
        node = root

        obs = None

        # Generate a trajectory
        while node.children:
            # Select an action according to the search policy
            action = search_policy(node)

            # Point the node at the corresponding child
            node = node.children[action]

            # Step the simulation and add this timestep to the node.
            obs, reward, done, _ = model.step(action)
            node.reward = reward or 0.
            node.terminal = done
            trajectory.append(node)

        if obs is None:
            raise ValueError('Generated an empty rollout; this should not happen.')

        if node.terminal:
            value = 0.
        else:
            # Bootstrap from this node with value function
            prior, value = evaluation(obs)

            # Expand this node for the next time
            node.expand(prior)

        # Load the saved model state.
        model.load_checkpoint()

        # Mote Carlo back-up with bootstrap from value function.
        ret = value
        while trajectory:
            node = trajectory.pop()

            # Accumulate the discounted return
            ret *= discount
            ret += node.reward

            # Update the node.
            node.total_value += ret
            node.visit_count += 1
    
    return root


def bfs(node: Node) -> types.Action:
    """Breadth-first search policy."""
    visit_counts = np.array([c.visit_count for c in node.children.values()])
    return argmax(-visit_counts)


def puct(node: Node, ucb_scaling: float = 1.) -> types.Action:
    """PUCT search policy, i.e. UCT with 'prior' policy."""
    # Action values Q(s,a).
    value_scores = np.array([child.value for child in node.children.values()])
    check_numerics(value_scores)

    # Policy prior P(s,a).
    priors = np.array([child.prior for child in node.children.values()])
    check_numerics(priors)

    # Visit ratios.
    visit_ratios = np.array([
        np.sqrt(node.visit_count) / (child.visit_count + 1)
        for child in node.children.values()
    ])
    check_numerics(visit_ratios)

    # Combine.
    puct_scores = value_scores + ucb_scaling * priors * visit_ratios
    return argmax(puct_scores)


def visit_count_policy(root: Node, temperature: float = 1.) -> types.Probs:
    """Probability weighted by visit^{1/temp} of children nodes."""
    visits = root.children_visits

    if np.sum(visits) == 0:  # uniform policy for zero visits
        visits += 1
    rescaled_visits = visits**(1 / temperature)
    probs = rescaled_visits / np.sum(rescaled_visits)
    check_numerics(probs)

    return probs


def argmax(values: np.ndarray) -> types.Action:
    """Argmax with random tie-breaking."""
    check_numerics(values)
    max_value = np.max(values)
    return np.int32(np.random.choice(np.flatnonzero(values == max_value)))


def check_numerics(values: np.ndarray):
    """Raises a ValueError if any of the inputs are NaN or Inf."""
    if not np.isfinite(values).all():
        raise ValueError('check_numerics failed. Inputs: {}. '.format(values))
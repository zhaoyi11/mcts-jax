"""Reference: https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/planning/trajectory_opt.py"""
from typing import Sequence, Tuple, Dict, Callable, List
from functools import partial
import copy

import ray
import numpy as np

@ray.remote
def rollout(model, checkpoint, action_sequence):
    model.load_checkpoint(checkpoint)

    terminated = False
    total_reward = 0

    for action in action_sequence:
        obs, reward, done, _ = model.step(action)
        reward = 0 if terminated else reward

        terminated |= bool(done)

        # TODO: fix this, potintial bug
        if done:
            model.reset()

        total_reward += reward    
    
    return total_reward

class CEMOptimizer:
    def __init__(self, 
                optimizer_cfg: Dict,
                lower_bound, upper_bound,    
                ):
        self.num_iterations = optimizer_cfg.get("num_iterations")
        self.elite_ratio = optimizer_cfg.get("elite_ratio")
        self.population_size = optimizer_cfg.get("population_size")
        self.elite_num = int(self.population_size * self.elite_ratio)
        self.alpha = optimizer_cfg.get("alpha")
        self.return_mean_elite = optimizer_cfg.get("return_mean_elite")
        self.clipped_normal = optimizer_cfg.get("clipped_normal")
        self.lower_bound, self.upper_bound = lower_bound, upper_bound

    def _init_population(self, prior_solution):
        # initialize the mean and dispersion of population
        mean = copy.copy(prior_solution)

        if self.clipped_normal:
            dispersion = np.ones_like(mean)
        else: # truncated normal
            dispersion = ((self.upper_bound - self.lower_bound) ** 2) / 16
        return mean, dispersion

    def _sample_population(self, mean, dispersion, population): # TODO: remove needs of population
        # fills population with random samples
        # for truncated normal, dispersion should be the variance
        # for clipped normal, dispersion should be the standard deviation

        if self.clipped_normal:
            pop = mean + dispersion * np.random.randn(*population.shape)
            population = np.clip(pop, a_min=self.lower_bound, a_max=self.upper_bound)
        else:
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean

            mv = np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2))
            constrained_var = np.minimum(mv, dispersion)

            pop = utils.truncated_normal_(population)
            population = mean + np.square(constrained_var) * pop
        return population

    def _update_population_params(self, elite, mu, dispersion):
        # update mu and dispersion according to elites
        new_mu = np.mean(elite, axis=0)
        if self.clipped_normal:
            new_dispersion = np.std(elite, axis=0)
        else:
            new_dispersion = np.var(elite, axis=0)
        
        mu = self.alpha * mu + (1 - self.alpha) * new_mu
        dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion

        return mu, dispersion

    def optimize(self, objective_fn, prior_solution):
        mu, dispersion = self._init_population(prior_solution)
        best_solution = copy.copy(mu)
        best_value = -np.inf
        population = np.zeros((self.population_size,) + prior_solution.shape) # init population (sample action sequences)
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)

            values = objective_fn(action_sequences=population) # evaluate action sequences
            values = np.nan_to_num(values, -1e-10)  # filter out NaN values # shape: [population_size, 1]

            elite_idx = np.argpartition(values, -self.elite_num, axis=None)[-self.elite_num :]
            
            best_values = values[elite_idx]
            elite = population[elite_idx]

            mu, dispersion = self._update_population_params(elite, mu, dispersion)
            
            # get the best value of populations
            best_idx = np.argpartition(best_values, -1, axis=None)[-1]
            if best_values[best_idx] > best_value:
                best_value = best_values[best_idx]
                best_solution = copy.copy(elite[best_idx])

        return mu if self.return_mean_elite else best_solution


class TrajectoryOptimizer:
    def __init__(self,
                optimizer_cfg: Dict,
                action_lb: np.ndarray,
                action_ub: np.ndarray,
                planning_horizon: int,
                replan_freq: int = 1,
                keep_last_solution: bool = True):
        """Class for using generic optimizers on trajectory optimization problems.

        - This is a convenience class that sets up optimization problem for trajectories, given only
        action bounds and the length of the horizon. 
        - Using this class, the concern of handling appropriate tensor shapes for the optimization 
        problem is hidden from the users, which only need to provide a function that is capable of
        evaluating trajectories of actions. 
        - It also takes care of shifting previous solution for the next optimization call, if the user desires.

        The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
        represent planning horizon and action dimension, respectively. The initial solution for the
        optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

        Args:
            optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
            action_lb (np.ndarray): the lower bound for actions.
            action_ub (np.ndarray): the upper bound for actions.
            planning_horizon (int): the length of the trajectories that will be optimized.
            replan_freq (int): the frequency of re-planning. This is used for shifting the previous
            solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
            keep_last_solution (bool): if ``True``, the last solution found by a call to
                :meth:`optimize` is kept as the initial solution for the next step. This solution is
                shifted ``replan_freq`` time steps, and the new entries are filled using the initial
                solution. Defaults to ``True``.
        """
        self.optimizer = CEMOptimizer(optimizer_cfg,
                                    lower_bound=np.tile(action_lb, (planning_horizon, 1)), # TODO: to list?
                                    upper_bound=np.tile(action_ub, (planning_horizon, 1)),)

        self.initial_solution = (action_lb + action_ub) / 2.
        self.initial_solution = np.tile(self.initial_solution, (planning_horizon, 1)) # shape: Plan_horizon, act_dim

        self.previous_solution = copy.copy(self.initial_solution)

        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.planning_horizon = planning_horizon

    def optimize(self, trajectory_eval_fn) -> np.ndarray:
        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            prior_solution=self.previous_solution,
        )

        # if need to reuse previous planned results
        if self.keep_last_solution:
            self.previous_solution = np.roll(best_solution, -self.replan_freq, axis=0)
            self.previous_solution[-self.replan_freq:] = self.initial_solution[0]
        
        return best_solution 

    def reset(self,):
        self.previous_solution = copy.copy(self.initial_solution)


class CEMAgent:
    def __init__(self, 
                optimizer_cfg: Dict, 
                action_lb: Sequence[float],
                action_ub: Sequence[float],
                planning_horizon: int = 1,
                replan_freq: int = 1,
                verbose: bool = False,
                keep_last_solution: bool = True):
        """Agent that performs trajectory optimization on a given objective function for each action.
        This class uses an internal :class:`TrajectoryOptimizer` object to generate
        sequence of actions, given a user-defined trajectory optimization function.
        Args:
            optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
                the trajectory optimizer.
            action_lb (sequence of floats): the lower bound of the action space.
            action_ub (sequence of floats): the upper bound of the action space.
            planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
            replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
                generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
                Defaults to 1.
            verbose (bool): if ``True``, prints the planning time on the console.
            keep_last_solution (bool): if ``True``, the last solution found by a call to
                :meth:`optimize` is kept as the initial solution for the next step. This solution is
                shifted ``replan_freq`` time steps, and the new entries are filled using the initial
                solution. Defaults to ``True``.
        Note:
            After constructing an agent of this type, the user must call
            :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
            be automatically instantiated with Hydra (which in turn makes it easy to replace this
            agent with an agent of another type via config-only changes).
        """   
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb), np.array(action_ub), 
            planning_horizon,
            replan_freq,
            keep_last_solution
        )
        self.optimizer_cfg = optimizer_cfg
        self.action_lb, self.action_ub = np.array(action_lb), np.array(action_ub)
        self.planning_horizon, self.replan_freq = planning_horizon, replan_freq

        self.actions_to_use: List[np.ndarray] = []

        self.verbose = verbose
        self.keep_last_solution = keep_last_solution

    def reset(self, planning_horizon: int = None):
        if planning_horizon: # need to set new planning_horizon
            self.optimizer = TrajectoryOptimizer(
                self.optimizer_cfg,
                self.action_lb, self.action_ub,
                planning_horizon,
                replan_freq = self.replan_freq,
            )
        else:
            self.optimizer.reset()

    def act(self, model, checkpoint)->np.ndarray:
        """Issues an action given an observation.
        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.
        Args:
            obs (np.ndarray): the observation for which the action is needed.
            optimizer_callback (callable, optional): a callback function
                to pass to the optimizer.
        Returns:
            (np.ndarray): the action.
        """
        plan_time = 0.0
        
        if not self.actions_to_use: # replan is necessary
            
            plan = self.optimizer.optimize(partial(self.trajectory_eval_fn, checkpoint=checkpoint, model=model))

            # record the first self.replan_freq
            self.actions_to_use.extend([a for a in plan[:self.replan_freq]])

        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def trajectory_eval_fn(self, model, checkpoint, action_sequences):
        """Evaluates a batch of action sequences on the model.
        Current version doesn't support run with multiple particals since we use deterministic 
        simulator here. When using a learned dynamics, we could follow https://github1s.com/facebookresearch/mbrl-lib/blob/main/mbrl/models/model_env.py

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        assert len(action_sequences.shape) == 3
        population_size, planning_horizon, action_dim = action_sequences.shape

        total_rewards = [rollout.remote(model, checkpoint, action_sequences[i])
                         for i in range(population_size)]
        total_rewards = np.array(ray.get(total_rewards))

        # reset model's state
        model.load_checkpoint(checkpoint)
        return total_rewards.reshape((population_size, 1))

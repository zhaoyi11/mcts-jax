import copy
from typing import Union
import dataclasses
import numpy as np

from gym import core
from .base_wrapper import EnvWrapper

@dataclasses.dataclass
class Checkpoint:
    """Holds the checkpoint state for the environment simulator."""
    needs_reset: bool
    ep_len: int
    ep_rew: int
    env_state: np.ndarray


class SimulatorWrapper(EnvWrapper):
    """A simulator model, which wraps a copy of the true environment.
    
    Assumptions:
    - The environment (including RNG) is fully copyable via `deepcopy`.
    - Environment dynamics (modulo episode resets) are deterministic.
    """
    _checkpoint: Checkpoint
    # _env: core.Env

    def __init__(self, env: core.Env):
        super().__init__(env)
        assert isinstance(env, core.Env)
        self._needs_reset = True

        self._ep_len = 0
        self._ep_rew = 0
        self.save_checkpoint()

    def save_checkpoint(self):
        self._checkpoint = Checkpoint(
            needs_reset=self._needs_reset,
            ep_len=self._ep_len, # important, this will determine the termination.
            ep_rew=self._ep_rew,
            env_state = self._env.physics.get_state(),
        )

        return self._checkpoint

    def load_checkpoint(self, checkpoint:Checkpoint=None):
        if checkpoint is not None:
            self._env.physics.set_state(checkpoint.env_state)
            # self._env._step_count=checkpoint.ep_len # not really useful, after setting, the env can't teriminate automatically after max_steps
            self._ep_len = checkpoint.ep_len

            self._ep_rew = checkpoint.ep_rew
            self._needs_reset = checkpoint.needs_reset

        else: # load checkpoint from internal checkpoint
            self._env.physics.set_state(self._checkpoint.env_state)
            # self._env._step_count=self._checkpoint.ep_len
            self._ep_len = self._checkpoint.ep_len

            self._ep_rew = self._checkpoint.ep_rew
            self._needs_reset = self._checkpoint.needs_reset 


    def step(self, action):
        if self._needs_reset:
            raise ValueError("This model needs to be explicitly reset.")
        obs, r, d, info = self._env.step(action) # won't reset after > max_steps (1000)
        
        self._needs_reset = d
        self._ep_rew += r
        self._ep_len += 1 # TODO: only plus 1, should we change to + skip_steps?
    
        return obs, r, d, info
    

    def reset(self, ):
        self._needs_reset = False
        return self._env.reset()


    @property
    def needs_reset(self,):
        return self._needs_reset
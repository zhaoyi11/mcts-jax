from gym import core
from .base_wrapper import EnvWrapper

class DiscretizeWrapper(EnvWrapper):
    """ Discretize the action space following """
#%%
import gym
import dm_control
from dm_control import suite
import time

from mcgs.wrappers.dmc2gym import DMC2GYMWrapper
# %%
env = DMCWrapper('walker', 'walk', {'random': 1,}, visualize_reward=True)
env.reset()

#%%
start_time = time.time()

for i in range(20):
    env.reset()
    for j in range(1000):
        action = env.action_space.sample()
        obs, r, d, _ = env.step(action)

duration = time.time() - start_time
print(duration/20)

#%%
env.physics.get_state()
# %%
# test dm_control
import numpy as np

env = suite.load('walker', 'walk')
start_time = time.time()

for i in range(20):
    env.reset()
    for i in range(1000):
        action = np.float32(np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, 
                                        size=env.action_spec().shape))
        timestep = env.step(action)

duration = time.time() - start_time
print(duration/20)
# %%
# test dm2gym
import dm2gym
import copy
env = gym.make('dm2gym:WalkerWalk-v0')
copy.deepcopy(env)

# %%
start_time = time.time()

for i in range(20):
    env.reset()
    for j in range(1000):
        action = env.action_space.sample()
        obs, r, d, _ = env.step(action)

duration = time.time() - start_time
# %%
print(duration/20)
# %%
# %%
env = suite.load('walker', 'walk', {'random': 1})

gym_env = DMC2GYMWrapper(env, frame_skip=2)

# gym_env._step_count = 100

#%%
import simulator
model = simulator.SimulatorWrapper(gym_env)

model.reset()
initial_state = model.save_checkpoint()

for i in range(10):
    model.step(model.action_space.sample())

intermediate_state = model.save_checkpoint()

#%%
for i in range(10):
    model.step(model.action_space.sample())

print(intermediate_state, initial_state)
print(f'after addition 10 steps, {model._env._step_count}, {model._env.physics.get_state()}')

#%%
model.load_checkpoint()
print(f'Implicitly load the intermediate state, {model._env._step_count}, {model._env.physics.get_state()}')

#%%
model.load_checkpoint(initial_state)
print(f'Load the initial state, {model._env._step_count}, {model._env.physics.get_state()}')

#%%
for i in range(10):
    model.step(model.action_space.sample())
print(model._env._step_count, model._env.physics.get_state())
#%%
import copy
copy.deepcopy(gym_env)
#%%
import simulator
model = simulator.Simulator(gym_env)

# %%
gym_env.physics.get_state()
# %%
gym_env.action_space
# %%
gym_env.observation_space
# %%
action = gym_env.action_space.sample()
# %%
gym_env.step(action)
# %%
gym_env._env.__dict__.keys()
# %%
gym_env._env._step_count = 999
# %%
gym_env._env._step_count
# %%
from gym import core
isinstance(gym_env, DMC2GYMWrapper)
# %%
import dm_env
isinstance(env, dm_env.Environment)
# %%
env._step_count
# %%
env.physics
# %%
gym_env.physics
# %%
timestep = gym_env.step(gym_env.action_space.sample())
# %%
timestep
# %%

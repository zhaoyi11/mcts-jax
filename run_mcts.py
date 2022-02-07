# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example running MCTS on BSuite in a single process."""

from typing import Tuple

from absl import app
from absl import flags

import bsuite
import dm_env
import gym
import utils
from agents import AZAgent
from wrappers import DMC2GYMWrapper, SimulatorWrapper, BS2GYMWrapper


# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', './bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')

FLAGS = flags.FLAGS


def make_env_and_model(
    bsuite_id: str,):
    """Create environment and corresponding model (learned or simulator)."""
    env = bsuite.load_from_id(bsuite_id)
    env = BS2GYMWrapper(env)
    model = SimulatorWrapper(env)  # pytype: disable=attribute-error

    return env, model


def main(_):
    # Create an environment and environment model.
    env, model = make_env_and_model(
        bsuite_id=FLAGS.bsuite_id,
    )

    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    buffer = utils.ReplayBuffer(state_shape, action_dim, 10000)

    agent = AZAgent(
        state_shape = state_shape,
        num_actions = action_dim,
        model = model,
        lr = 1e-3,
        discount = 0.99,
        num_simulations = 50,
    )

    
    for i in range(env.bsuite_num_episodes):
        ep_reward = 0
        obs, done = env.reset(), False
        while not done:
            action, probs = agent.act(obs)
            next_obs, rew, done, _ = env.step(action)

            ep_reward += rew
            buffer.add(obs, action, next_obs, rew, done, {'pi': probs})

            upate_info = agent.update(buffer.sample(batch_size=16))
            
            obs = next_obs

        print(f'Episode: {i}, Episode reward: {ep_reward}')

    
if __name__ == '__main__':
  app.run(main)
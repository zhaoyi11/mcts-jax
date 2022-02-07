import os
os.environ["MUJOCO_GL"] = "egl"
import time
from pathlib import Path

import ray
import utils
from dm_control import suite
import torch

from wrappers import DMC2GYMWrapper
from agents import QtOptAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_env(env_name, seed=0, num_repeats=1):
    domain_name, task_name = env_name.split("-")
    env = suite.load(domain_name, task_name, {"random": seed})
    env = DMC2GYMWrapper(env, frame_skip = num_repeats)

    return env


if __name__ == "__main__":
    seed = int(time.time())

    env = make_env("walker-walk", num_repeats=2, seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    optimizer_cfg = {
        'num_iterations': 2,
        'elite_ratio': 0.1,
        'population_size': 100,
        'alpha': 0.1,
        'return_mean_elite': False,
    }

    agent = QtOptAgent(optimizer_cfg,
                    action_lb = min_action,
                    action_ub = max_action,
                    state_dim = state_dim,
                    action_dim = action_dim,
                    critic_path="./walker-walk_td3.pth",
                    )
    
    video_recorder = utils.VideoRecorder(Path.cwd(), 
                                    camera_id = 0,
                                    use_wandb=False)

    obs = env.reset()

    video_recorder.init(env, enabled=True)

    total_rewards, done = 0, False
    while not done:
        action = agent.act(torch.FloatTensor(obs).to(device))
        # with torch.no_grad():
        #     action = agent.actor(torch.FloatTensor(obs).to(device))
        obs, reward, done, _ = env.step(action.cpu().numpy())
        video_recorder.record(env)

        total_rewards += reward
    
        print(f'cumulative_reward: {total_rewards}, perstep_reward: {reward}')
    
    video_recorder.save(f'walker_walk_qtopt_{seed}.mp4')
    print(f'total_rewards: {total_rewards}')

import os
os.environ["MUJOCO_GL"] = "egl"
import time
from pathlib import Path

import ray
import utils
from dm_control import suite

from wrappers import DMC2GYMWrapper, SimulatorWrapper
from agents import CEMAgent

def make_simulator(env_name, seed=0, num_repeats=1):
    domain_name, task_name = env_name.split("-")
    env = suite.load(domain_name, task_name, {"random": seed})
    env = DMC2GYMWrapper(env, frame_skip = num_repeats)
    model = SimulatorWrapper(env)

    return model


if __name__ == "__main__":
    seed = int(time.time())

    model = make_simulator("walker-walk", num_repeats=2, seed=seed)

    ray.init()
    
    state_dim = model.observation_space.shape[0]
    action_dim = model.action_space.shape[0]
    max_action = model.action_space.high
    min_action = model.action_space.low

    optimizer_cfg = {
        'num_iterations': 5,
        'elite_ratio': 0.1,
        'population_size': 300,
        'alpha': 0.1,
        'return_mean_elite': True,
        'clipped_normal': True, # if False, uses truncked normal
    }

    agent = CEMAgent(optimizer_cfg,
                    action_lb = min_action,
                    action_ub = max_action,
                    planning_horizon = 12,
                    replan_freq = 1,
                    verbose = False, 
                    keep_last_solution = True,
                    )
    agent.optimizer.reset()
    
    video_recorder = utils.VideoRecorder(Path.cwd(), 
                                    camera_id = 0,
                                    use_wandb=False)

    model.reset()
    checkpoint = model.save_checkpoint()

    video_recorder.init(model, enabled=True)

    total_rewards, done = 0, False
    for i in range(200):
        action = agent.act(model, checkpoint)
        obs, reward, done, _ = model.step(action)
        checkpoint = model.save_checkpoint()
        video_recorder.record(model)

        total_rewards += reward
    
        print(f' i: {i}, cumulative_reward: {total_rewards}, perstep_reward: {reward}')
    
    video_recorder.save(f'walker_walk_cem_{seed}.mp4')
    print(f'total_rewards: {total_rewards}')

    ray.shutdown()


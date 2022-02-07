import argparse
import tqdm
import copy
import time
import random

import pybullet_envs
from dm_control import suite
from wrappers import DMC2GYMWrapper
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########### Helper Functions ##########
def eval_policy(policy, eval_env, eval_episodes=10):
    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = np.array(state).reshape(1, -1)
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    return {'returns': avg_reward}


def fill_initial_buffer(env, replay_buffer, n_random_timesteps):
    # prefill initial exploration data
    state, done = env.reset(), False
    episode_timesteps = 0
    for _ in range (n_random_timesteps):
        episode_timesteps += 1
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        done_bool = float(1 - info['discount']) 
        replay_buffer.add(state, action, next_state, reward, done_bool)
        
        state = next_state

        if done:
            state, done = env.reset(), False
            episode_timesteps = 0
            
    return replay_buffer


########## Define Agent ##########
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a)) # TODO: Bug 2 is here.


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


########## TD3 ##########
class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = batch

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q, _ = self.critic(state, pi)
            
            actor_loss = -Q.mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"critic_loss": critic_loss.item(),
                "critic": current_Q1.mean().item()}

    def save(self, filename):
        torch.save({'critic': self.critic.state_dict(),
                    'actor': self.actor.state_dict(),}, filename + "_td3.pth")

    def load(self, filename):
        policy_dicts = torch.load(filename + "_td3.pth")
        
        self.critic.load_state_dict(policy_dicts['critic'])
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(policy_dicts['actor'])
        self.target_actor = copy.deepcopy(self.actor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', default='TD3')
    parser.add_argument('--env', default='walker-walk') 
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--max_timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--n_random_timesteps', type=int, default=10_000,
                        help='num of inital random data to pre-fill the replay buffer')
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    # Algorithm specific arguments
    parser.add_argument("--expl_noise", default=0.1, type=float,
                        help="Std of Gaussian exploration noise")
    parser.add_argument("--discount", default=0.99, type=float,
                        help="Discount factor.")
    parser.add_argument("--tau", default=0.005, type=float,
                        help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float,
                        help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, type=float,
                        help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, 
                        help="Frequency of delayed policy updates") 
    # options
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of the experiment')  
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    if args.seed == 0:
        args.seed = int(time.time())

    experiment_name = f"{args.env}_{args.algo_name}_{args.seed}_{int(time.time())}"
    
    wandb.init(project="rl_project", config=vars(args), name=experiment_name)

    # Init env and seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    domain_name, task_name = args.env.split("-")
    env = suite.load(domain_name, task_name, {"random": args.seed})
    env = DMC2GYMWrapper(env)

    eval_env = suite.load(domain_name, task_name, {"random": args.seed+100})
    eval_env = DMC2GYMWrapper(eval_env)
    

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0] 

    max_action = float(env.action_space.high[0])
    

    replay_buffer = ReplayBuffer(state_shape, action_dim)
    # prefill random initialization data
    replay_buffer = fill_initial_buffer(env, replay_buffer, args.n_random_timesteps) # TODO: remove this latter
    

    # init td3
    td3_kwargs = {
        "state_dim": state_shape[0],
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq, 
    }
    td3 = TD3(**td3_kwargs)


    state, done = env.reset(), False
    episode_timesteps = 0
    for t in tqdm.tqdm(range(args.max_timesteps)):
        episode_timesteps += 1

        action = (
                td3.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
		).clip(-max_action, max_action)
        
        next_state, reward, done, info = env.step(action)
        
        done_float = float(1 - info['discount']) 

        replay_buffer.add(state, action, next_state, reward, done_float)
        
        state = next_state

        if done:
            state, done = env.reset(), False
            episode_timesteps = 0

        # update policy per data point
        policy_update_info = td3.train(replay_buffer.sample(args.batch_size))
        wandb.log({"train/": policy_update_info})

        # Evaluate episode
        if t % args.eval_freq == 0:
            eval_info = eval_policy(td3, eval_env)
            eval_info.update({'timesteps': t}) # record timesteps
            print(f"Time steps: {t}, Eval_info: {eval_info}")
            wandb.log({"eval/": eval_info}) 

    if args.save_model:
        td3.save(f"./{experiment_name}")

    env.close()
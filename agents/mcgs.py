import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.stats import multivariate_normal
import os
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from itertools import cycle
import sys
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import time


N_LAYERS = 15
N_LAYERS_2D_NAV_ENV = 5
ROLLOUT_LENGTH_SECONDS = 1.5
MAX_N_EXPS = 500
MIN_N_DATA_PER_Q_NODE = max(5, int(0.03 * MAX_N_EXPS))
PW_C = 0.5
PW_ALPHA = 0.475


def plot_pw():
	x = np.arange(0, MAX_N_EXPS)
	y = PW_C * np.power(x, PW_ALPHA)
	rc('figure', figsize=(20, 10))
	fig, ax = plt.subplots(1, 1)
	ax.set_xlim([0, MAX_N_EXPS])
	ax.set_ylim([0, MAX_N_EXPS / 8])
	ax.set_xticks(np.arange(11) * MAX_N_EXPS // 10)
	ax.set_yticks(np.arange(0, MAX_N_EXPS / 8, 2))
	ax.plot(x, y)
	plt.grid()
	plt.show()


# plot_pw()


class MCGS:
	def __init__(self, env, alg):
		global N_LAYERS
		if "2d-navigation" in env.env_name:
			N_LAYERS = N_LAYERS_2D_NAV_ENV

		self.env = env
		self.alg = alg

		self.action_space = env.action_space
		self.observation_space = env.observation_space

		self.rollout_length = int(ROLLOUT_LENGTH_SECONDS / env.dt)
		assert self.rollout_length > N_LAYERS

		self.layers = None

		state_dim = env.observation_space.low.shape[0]
		self.exp_owner = -np.ones((N_LAYERS, MAX_N_EXPS), dtype=np.int)
		self.states = np.zeros((N_LAYERS, MAX_N_EXPS, state_dim))
		self.states2 = np.zeros((N_LAYERS, MAX_N_EXPS, state_dim))
		self.actions = np.zeros((N_LAYERS, MAX_N_EXPS, env.action_space.low.shape[0]))
		self.rewards = np.zeros((N_LAYERS, MAX_N_EXPS))
		self.n_exps = np.zeros(N_LAYERS, dtype=np.int)

		self.best_action = None
		self.best_rew = None

		self.postpone_clustering = np.zeros(N_LAYERS)

		self.time_budget_total = 0
		self.time_budget_env_step = 0
		self.time_budget_clustering = 0
		self.time_action_bandit = 0
		self.time_q_bandit = 0

	def search(self, simulation_budget):
		time_total_start = time.perf_counter()
		env = self.env
		master_state = env.get_state()

		self.layers = []
		for l in range(N_LAYERS):
			self.layers.append([Q_Node(action_space=env.action_space, state_dim=env.observation_space.low.shape[0])])
		self.exp_owner *= 0
		self.exp_owner -= 1
		self.states *= 0
		self.states2 *= 0
		self.actions *= 0
		self.rewards *= 0
		self.n_exps *= 0

		self.best_action = np.copy(0.5 * (env.action_space.low + env.action_space.high))
		self.best_rew = -np.inf

		self.postpone_clustering *= 0

		timesteps = 0
		while timesteps < simulation_budget:
			env.set_state(master_state)
			# Graph policy
			exps, done, ep_ret, traj_len = self.graph_policy(env)
			if not done:
				# Default policy
				ep_ret, traj_len = self.default_policy(env, ep_ret, traj_len)
			ep_ret += env.critic() * np.power(env.gamma, traj_len)
			# Backpropagation
			self.backpropagate(exps, ep_ret)
			timesteps += traj_len

		action = self.best_action
		env.set_state(master_state)
		self.time_budget_total += time.perf_counter() - time_total_start
		return action

	def graph_policy(self, env):
		q = 0
		exps = []
		s1 = env.obs
		sum_r = 0
		traj_len = 0
		done = False
		for l in range(N_LAYERS):
			cur_q = self.layers[l][q]
			time_action_bandit_start = time.perf_counter()
			a = cur_q.action_bandit()
			self.time_action_bandit += time.perf_counter() - time_action_bandit_start
			time_step_start = time.perf_counter()
			s2, r, done, info = env.step(a, render=False)
			self.time_budget_env_step += time.perf_counter() - time_step_start
			exps.append({'q': q, 'exp': [s1, a, s2]})
			sum_r += r * np.power(env.gamma, l)
			traj_len += 1
			if done or l + 1 == N_LAYERS:
				break
			time_q_bandit_start = time.perf_counter()
			q = self.q_bandit(l + 1, s2)
			self.time_q_bandit += time.perf_counter() - time_q_bandit_start
			s1 = s2
		return exps, done, sum_r, traj_len

	def q_bandit(self, l, s):
		n_q = len(self.layers[l])
		if n_q == 1:
			return 0
		max_score = -np.inf
		max_score_i = -1
		# n_parent = min(self.n_exps[l - 1], MAX_N_EXPS)
		for q in range(n_q):
			pdf = self.layers[l][q].pdf.pdf(s)
			pdf = np.mean(pdf)
			# c = 0.25
			# n_child = np.sum(self.exp_owner[l, :] == q)
			# score = pdf + c * np.sqrt(n_parent / n_child)
			score = pdf
			score += np.random.random_sample() * 0.01  # Break ties randomly
			if score > max_score:
				max_score = score
				max_score_i = q
		return max_score_i

	def default_policy(self, env, ep_ret, traj_len):
		assert traj_len == N_LAYERS
		for t in range(self.rollout_length - traj_len):
			time_step_start = time.perf_counter()
			ob, r, done, info = env.step(env.action_space.sample(), render=False)
			self.time_budget_env_step += time.perf_counter() - time_step_start
			ep_ret += r * np.power(env.gamma, N_LAYERS + t)
			traj_len += 1
			if done:
				break
		return ep_ret, traj_len

	@ignore_warnings(category=ConvergenceWarning)
	def backpropagate(self, exps, rew):
		assert 0 < len(exps) <= N_LAYERS

		# Update the best found action if needed
		if rew > self.best_rew:
			self.best_rew = rew
			self.best_action[:] = exps[0]['exp'][1][:]

		for l in range(len(exps)):
			q = exps[l]['q']
			exp = exps[l]['exp']

			idx = self.n_exps[l] % MAX_N_EXPS
			self.exp_owner[l, idx] = q
			self.states[l, idx, :] = exp[0][:]
			self.actions[l, idx, :] = exp[1][:]
			self.states2[l, idx, :] = exp[2][:]
			self.rewards[l, idx] = rew
			self.n_exps[l] += 1

			time_cluster_start = time.perf_counter()
			if self.postpone_clustering[l] > 0:
				self.postpone_clustering[l] -= 1
			desired_n_clusters = self.n_exps[l] // MIN_N_DATA_PER_Q_NODE  # int(PW_C * np.power(self.n_exps[l], PW_ALPHA))
			clustered = True
			if l > 0 and self.n_exps[l] > (desired_n_clusters + 1) * MIN_N_DATA_PER_Q_NODE \
					and len(self.layers[l]) < desired_n_clusters and self.postpone_clustering[l] == 0:
				# Try to make a new cluster
				cur_layer_data = self.states[l, :self.n_exps[l], :]
				clustering_alg = self.alg[5:]
				if clustering_alg == "kmeans":
					clusters_idx = KMeans(n_clusters=desired_n_clusters, random_state=0).fit(cur_layer_data).labels_
				elif clustering_alg == "agglomerative":
					clusters_idx = AgglomerativeClustering().fit(cur_layer_data).labels_
				else:
					assert clustering_alg == "gmm"
					clusters_idx = GaussianMixture(n_components=desired_n_clusters, covariance_type="full", random_state=0).fit(cur_layer_data).predict(cur_layer_data)
				for c in range(desired_n_clusters):
					if np.sum(clusters_idx == c) < MIN_N_DATA_PER_Q_NODE:
						clustered = False
						break
				if clustered:
					while len(self.layers[l]) < desired_n_clusters:
						self.layers[l].append(Q_Node(action_space=self.action_space, state_dim=self.observation_space.low.shape[0]))
					self.exp_owner[l, :self.n_exps[l]] = clusters_idx
					for c in range(desired_n_clusters):
						indices = self.exp_owner[l, :] == c
						self.layers[l][c].update(self.states[l, indices, :], self.actions[l, indices, :], self.rewards[l, indices])
				else:
					# Postpone the clustering for better performance
					self.postpone_clustering[l] = MIN_N_DATA_PER_Q_NODE
			else:
				clustered = False
			if not clustered:
				# No new clustering needed. Updated the Q node regularly.
				indices = self.exp_owner[l, :] == q
				if np.sum(indices) >= MIN_N_DATA_PER_Q_NODE:
					self.layers[l][q].update(self.states[l, indices, :], self.actions[l, indices, :], self.rewards[l, indices])
			self.time_budget_clustering += time.perf_counter() - time_cluster_start

	def plot(self, frame_counter):
		fig, ax = self.env.env.plot()

		cycol = cycle('bgrcmk')
		n_steps = self.env.env._max_episode_steps

		# Now we can plot the search graph
		for l in range(N_LAYERS):
			offset = frame_counter + l
			if offset >= n_steps - 1:
				break
			n_q = len(self.layers[l])
			ax.annotate('No. Qs: %d' % n_q, (offset - 0.2, 5.15), color='k', fontsize=14)
			for q in range(n_q):
				q_node = self.layers[l][q]
				c = next(cycol)
				# State distribution
				idx = self.exp_owner[l, :] == q
				n_exps = np.sum(idx)
				N = 25
				state_mean = 3 + 2 * q_node.state_mean[1]
				state_std = 2 * q_node.state_std[1]
				X = np.linspace(state_mean - state_std, state_mean + state_std, N)
				P = norm(state_mean, state_std).pdf(X)
				P -= np.min(P)
				if np.max(P) > 0.35:
					P = P / np.max(P) * 0.35
				ax.plot(offset - P, X, color=c)
				ax.annotate('n=%d' % n_exps, (offset - np.max(P) - 0.2, state_mean + 0.03), color=c, fontsize=10)
				if n_exps > 0:
					rew_mean = np.mean(self.rewards[l, idx])
					rew_std = np.std(self.rewards[l, idx])
					ax.annotate('r=%.1f+/-%.1f' % (rew_mean, rew_std), (offset - np.max(P) - 0.2, state_mean - 0.03), color=c, fontsize=10)
					ax.plot(offset * np.ones([n_exps]), 3 + 2 * self.states[l, idx, 1], 'x', color=c)

				# Action distribution
				xs, ys = [offset], [state_mean]
				for mult in [-1, 0, 1]:
					v = np.array([1, q_node.action_mean[0] + mult * q_node.action_sd[0]])
					v = v / np.linalg.norm(v) * 0.25
					xs.append(offset + v[0])
					ys.append(state_mean + v[1])
					if mult == 0:
						ax.add_patch(plt.arrow(offset, state_mean, v[0], v[1], width=0.01, color=c, zorder=3))
				ax.fill(xs, ys, color=c, alpha=0.25, zorder=3)
		plt.suptitle('Frame %d' % frame_counter, fontsize=50)
		fig.tight_layout()
		time_tag = datetime.now().strftime("%Y-%m-%d---%H-%M-%S-%f")[:-3]
		# fig.savefig(os.path.join('images', '%s-%d.png' % (time_tag, frame_counter)), bbox_inches='tight', pad_inches=0.1, dpi=200)
		fig.savefig(os.path.join('images', 'frame-%d.png' % frame_counter), bbox_inches='tight', pad_inches=0.1, dpi=200)
		# plt.show()
		plt.close(fig)

	def report_time_budget(self):
		print('Time budget report:')
		print('\tTotal search time: %.2f' % self.time_budget_total)
		print('\t\tEnv step ratio: %d%%' % (100 * self.time_budget_env_step / self.time_budget_total))
		print('\t\tClustering ratio: %d%%' % (100 * self.time_budget_clustering / self.time_budget_total))
		print('\t\tAction bandit ratio: %d%%' % (100 * self.time_action_bandit / self.time_budget_total))
		print('\t\tQ bandit ratio: %d%%' % (100 * self.time_q_bandit / self.time_budget_total))


class Q_Node:
	def __init__(self, action_space, state_dim):
		self.action_dim = action_space.low.shape[0]
		self.action_min = action_space.low
		self.action_max = action_space.high

		self.state_dim = state_dim

		self.state_mean = np.zeros(self.state_dim)
		self.state_std = 0.1 * np.ones(self.state_dim)
		self.action_mean = 0.5 * (self.action_min + self.action_max)
		self.action_sd = 0.5 * (self.action_max - self.action_min)

		self.n_data = 0

		self.pdf = scipy.stats.norm(self.state_mean, self.state_std)

	def action_bandit(self):
		action = np.random.normal(self.action_mean, self.action_sd)
		action = np.clip(action, self.action_min, self.action_max)
		return action

	def update(self, states, actions, rewards):
		self.n_data = states.shape[0]
		assert self.n_data >= MIN_N_DATA_PER_Q_NODE

		self.state_mean = np.mean(states, axis=0)
		# self.state_mean = np.average(self.states[: n_exps, :], axis=0, weights=weights)
		self.state_std = np.std(states, axis=0)
		self.state_std = np.clip(self.state_std, 0.1, 0.5)

		self.pdf = scipy.stats.norm(self.state_mean, self.state_std)

		weights = rewards
		M, m = np.max(weights), np.min(weights)
		if M - m > 0.01:
			weights = (weights - m) / (M - m)
		else:
			weights = np.ones(weights.shape[0])
		weights[weights < 0.5] = 0

		self.action_mean = np.average(actions, axis=0, weights=weights)
		sd = 0.3 + 0.2 * np.exp(-0.0005 * np.power(self.n_data, 2))
		self.action_sd = sd * (self.action_max - self.action_min)
		# self.action_sd = np.std(actions, axis=0)
		# self.action_sd = np.clip(self.action_sd, 0.1, 0.5)

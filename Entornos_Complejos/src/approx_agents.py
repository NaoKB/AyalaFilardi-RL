"""Agentes de control aproximado: SARSA semi-gradiente y DQN."""
from __future__ import annotations

from collections import deque
import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .core import Agent


class TileCoder:
    def __init__(self, low: np.ndarray, high: np.ndarray, bins=(7, 7, 7, 7), tilings: int = 8):
        self.low, self.high, self.bins, self.tilings = np.asarray(low), np.asarray(high), tuple(bins), tilings
        self.features_per_tiling = int(np.prod(self.bins))
        self.n_features = self.features_per_tiling * tilings

    def encode(self, observation: Any) -> np.ndarray:
        normalized = np.clip((np.asarray(observation) - self.low) / (self.high - self.low), 0.0, 0.999999)
        bins = np.asarray(self.bins)
        active = []
        for tiling in range(self.tilings):
            offsets = ((tiling + 1) * np.arange(1, len(self.bins) + 1)) / (self.tilings * bins)
            coords = np.clip(np.floor((normalized + offsets) * bins).astype(int), 0, bins - 1)
            active.append(tiling * self.features_per_tiling + np.ravel_multi_index(tuple(coords), self.bins))
        return np.asarray(active, dtype=np.int64)


class SemiGradientSarsaAgent(Agent):
    def __init__(self, low, high, n_actions, gamma=0.99, alpha=0.14, epsilon_start=0.30, epsilon_end=0.025, epsilon_decay_episodes=950, seed=0):
        self.name, self.coder, self.n_actions, self.gamma = "SARSA semi-gradiente", TileCoder(low, high), n_actions, gamma
        self.alpha = alpha / self.coder.tilings
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_episodes = epsilon_start, epsilon_end, epsilon_decay_episodes
        self.episode_index, self.rng = 0, np.random.default_rng(seed)
        self.weights = np.zeros((n_actions, self.coder.n_features))

    @property
    def epsilon(self):
        fraction = min(1.0, self.episode_index / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def q_values(self, observation):
        return self.weights[:, self.coder.encode(observation)].sum(axis=1)

    def act(self, observation: Any, training: bool = True) -> int:
        values = self.q_values(observation)
        greedy = int(self.rng.choice(np.flatnonzero(np.isclose(values, values.max()))))
        return int(self.rng.integers(self.n_actions)) if training and self.rng.random() < self.epsilon else greedy

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        done = terminated or truncated
        next_action = None if done else self.act(next_observation, True)
        target = reward if done else reward + self.gamma * self.q_values(next_observation)[int(next_action)]
        delta = target - self.q_values(observation)[action]
        self.weights[action, self.coder.encode(observation)] += self.alpha * delta
        return next_action

    def end_episode(self, training: bool = True):
        self.episode_index += int(training)


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        return self.network(x)


class DQNAgent(Agent):
    def __init__(self, input_dim, n_actions, gamma=0.99, learning_rate=8e-4, epsilon_start=0.90, epsilon_end=0.035, epsilon_decay_steps=45_000, replay_size=45_000, batch_size=64, warmup_steps=700, target_interval=300, train_interval=2, seed=0):
        self.name, self.n_actions, self.gamma = "Deep Q-Learning", n_actions, gamma
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = epsilon_start, epsilon_end, epsilon_decay_steps
        self.batch_size, self.warmup_steps, self.target_interval, self.train_interval = batch_size, warmup_steps, target_interval, train_interval
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork(input_dim, n_actions).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=replay_size)
        self.steps = 0

    @property
    def epsilon(self):
        fraction = min(1.0, self.steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def act(self, observation: Any, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        x = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(x).argmax(1).item())

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        done = bool(terminated or truncated)
        self.replay.append((np.asarray(observation, dtype=np.float32), int(action), float(reward), np.asarray(next_observation, dtype=np.float32), done))
        self.steps += 1
        if len(self.replay) >= max(self.batch_size, self.warmup_steps) and self.steps % self.train_interval == 0:
            self._learn()
        if self.steps % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _learn(self):
        batch = [self.replay[int(i)] for i in self.rng.choice(len(self.replay), self.batch_size, replace=False)]
        obs = torch.as_tensor(np.stack([x[0] for x in batch]), device=self.device)
        actions = torch.as_tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([x[3] for x in batch]), device=self.device)
        dones = torch.as_tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)
        chosen = self.online(obs).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = rewards + self.gamma * (1.0 - dones) * self.target(next_obs).max(1).values
        loss = nn.functional.smooth_l1_loss(chosen, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optimizer.step()

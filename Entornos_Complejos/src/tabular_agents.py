"""Agentes Monte Carlo y de diferencias temporales."""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import Agent


class TabularAgent(Agent):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        name: str,
        gamma: float = 0.98,
        epsilon_start: float = 0.30,
        epsilon_end: float = 0.03,
        epsilon_decay_episodes: int = 1800,
        seed: int = 0,
    ):
        self.n_states, self.n_actions, self.name, self.gamma = n_states, n_actions, name, gamma
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.episode_index = 0
        self.last_action_probability = 1.0

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.episode_index / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def greedy_action(self, state: int) -> int:
        values = self.q[int(state)]
        return int(self.rng.choice(np.flatnonzero(np.isclose(values, values.max()))))

    def act(self, observation: Any, training: bool = True) -> int:
        state = int(observation)
        greedy = self.greedy_action(state)
        if not training:
            self.last_action_probability = 1.0
            return greedy
        action = int(self.rng.integers(self.n_actions)) if self.rng.random() < self.epsilon else greedy
        self.last_action_probability = self.epsilon / self.n_actions + (1.0 - self.epsilon) * float(action == greedy)
        return action

    def end_episode(self, training: bool = True) -> None:
        self.episode_index += int(training)


class MonteCarloOnPolicyAgent(TabularAgent):
    """Control Monte Carlo first-visit con politica epsilon-soft."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="MC on-policy", **kwargs)
        self.returns_sum, self.returns_count = np.zeros_like(self.q), np.zeros_like(self.q)
        self.episode = []

    def begin_episode(self, training: bool = True) -> None:
        self.episode = []

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        self.episode.append((int(observation), int(action), float(reward), self.last_action_probability))

    def end_episode(self, training: bool = True) -> None:
        if training:
            g, visited = 0.0, set()
            for state, action, reward, _ in reversed(self.episode):
                g = reward + self.gamma * g
                if (state, action) in visited:
                    continue
                visited.add((state, action))
                self.returns_sum[state, action] += g
                self.returns_count[state, action] += 1
                self.q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
        super().end_episode(training)


class MonteCarloOffPolicyAgent(TabularAgent):
    """Control Monte Carlo off-policy con importancia ponderada."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="MC off-policy", **kwargs)
        self.c, self.episode = np.zeros_like(self.q), []

    def begin_episode(self, training: bool = True) -> None:
        self.episode = []

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        self.episode.append((int(observation), int(action), float(reward), self.last_action_probability))

    def end_episode(self, training: bool = True) -> None:
        if training:
            g, weight = 0.0, 1.0
            for state, action, reward, behavior_probability in reversed(self.episode):
                g = reward + self.gamma * g
                self.c[state, action] += weight
                self.q[state, action] += weight / self.c[state, action] * (g - self.q[state, action])
                if action != self.greedy_action(state):
                    break
                weight /= max(behavior_probability, 1e-9)
        super().end_episode(training)


class SarsaAgent(TabularAgent):
    def __init__(self, *args, alpha: float = 0.18, **kwargs):
        super().__init__(*args, name="SARSA", **kwargs)
        self.alpha = alpha

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        done = terminated or truncated
        next_action = None if done else self.act(next_observation, training=True)
        target = reward if done else reward + self.gamma * self.q[int(next_observation), int(next_action)]
        state = int(observation)
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        return next_action


class QLearningAgent(TabularAgent):
    def __init__(self, *args, alpha: float = 0.18, name: str = "Q-Learning", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.alpha = alpha

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        done = terminated or truncated
        target = reward if done else reward + self.gamma * np.max(self.q[int(next_observation)])
        state = int(observation)
        self.q[state, action] += self.alpha * (target - self.q[state, action])


class DiscretizedQLearningAgent(QLearningAgent):
    """Baseline que aplica Q-Learning tras una discretizacion gruesa."""

    def __init__(self, low: np.ndarray, high: np.ndarray, bins: tuple[int, ...], n_actions: int, **kwargs):
        self.low, self.high, self.bins = np.asarray(low), np.asarray(high), tuple(bins)
        super().__init__(n_states=int(np.prod(bins)), n_actions=n_actions, name="Q-Learning discretizado", **kwargs)

    def discretize(self, observation: Any) -> int:
        ratio = np.clip((np.asarray(observation) - self.low) / (self.high - self.low), 0.0, 0.999999)
        coords = tuple((ratio * np.asarray(self.bins)).astype(int))
        return int(np.ravel_multi_index(coords, self.bins))

    def act(self, observation: Any, training: bool = True) -> int:
        return super().act(self.discretize(observation), training)

    def observe(self, observation, action, reward, next_observation, terminated, truncated):
        return super().observe(self.discretize(observation), action, reward, self.discretize(next_observation), terminated, truncated)

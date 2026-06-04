"""Interfaz comun de agentes y bucles reproducibles de entrenamiento."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EpisodeRecord:
    episode: int
    return_value: float
    length: int
    success: bool
    incident: int
    violation: bool


class Agent(ABC):
    name: str

    def begin_episode(self, training: bool = True) -> None:
        pass

    @abstractmethod
    def act(self, observation: Any, training: bool = True) -> int:
        raise NotImplementedError

    def observe(self, observation, action, reward, next_observation, terminated, truncated) -> int | None:
        return None

    def end_episode(self, training: bool = True) -> None:
        pass


def run_episode(env: gym.Env, agent: Agent, seed: int, training: bool = True, capture_trajectory: bool = False):
    observation, _ = env.reset(seed=seed)
    agent.begin_episode(training=training)
    action = agent.act(observation, training=training)
    total_reward, length, incidents = 0.0, 0, 0
    success, violation = False, False
    trajectory: list[dict[str, Any]] = []
    while True:
        next_observation, reward, terminated, truncated, info = env.step(action)
        length += 1
        total_reward += reward
        incidents += int(bool(info.get("incident", False)))
        success = bool(info.get("success", False)) or success
        violation = bool(info.get("violation", False)) or violation
        if capture_trajectory:
            trajectory.append({"step": length, "action": action, "reward": reward, **{k: v for k, v in info.items() if np.isscalar(v)}})
        next_action = agent.observe(observation, action, reward, next_observation, terminated, truncated) if training else None
        if terminated or truncated:
            break
        observation = next_observation
        action = next_action if next_action is not None else agent.act(observation, training=training)
    agent.end_episode(training=training)
    return EpisodeRecord(0, total_reward, length, success, incidents, violation), trajectory


def train_agent(env_factory: Callable[[], gym.Env], agent: Agent, episodes: int, seed: int) -> list[EpisodeRecord]:
    env = env_factory()
    records = []
    for episode in range(episodes):
        record, _ = run_episode(env, agent, seed + episode, training=True)
        record.episode = episode + 1
        records.append(record)
    env.close()
    return records


def evaluate_agent(env_factory: Callable[[], gym.Env], agent: Agent, episodes: int, seed: int, capture_one: bool = False):
    env = env_factory()
    records, trajectory = [], []
    for episode in range(episodes):
        record, current = run_episode(env, agent, seed + episode, training=False, capture_trajectory=capture_one and episode == 0)
        record.episode = episode + 1
        records.append(record)
        trajectory = current or trajectory
    env.close()
    return records, trajectory

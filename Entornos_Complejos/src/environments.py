"""Entornos Gymnasium originales usados en la experimentacion."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RiverCourierEnv(gym.Env):
    """Reparto discreto con una ruta corta arriesgada y un puente seguro."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    ACTIONS = ("arriba", "derecha", "abajo", "izquierda", "esperar")

    def __init__(
        self,
        hazard_scale: float = 1.0,
        slip_probability: float = 0.04,
        max_steps: int = 100,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.rows, self.cols, self.phases, self.damage_levels = 7, 7, 3, 3
        self.start, self.goal = (6, 0), (0, 6)
        self.ford, self.bridge = (3, 1), (3, 6)
        self.hazard_scale = hazard_scale
        self.slip_probability = slip_probability
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Discrete(
            self.rows * self.cols * self.phases * self.damage_levels
        )
        self.position, self.phase, self.damage, self.steps = self.start, 0, 0, 0

    def encode(self, position: tuple[int, int], phase: int, damage: int) -> int:
        row, col = position
        return int((((row * self.cols) + col) * self.phases + phase) * self.damage_levels + damage)

    def decode(self, state: int) -> tuple[int, int, int, int]:
        damage = state % self.damage_levels
        state //= self.damage_levels
        phase = state % self.phases
        state //= self.phases
        row, col = divmod(state, self.cols)
        return int(row), int(col), int(phase), int(damage)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.position = self.start
        self.phase = int(self.np_random.integers(0, self.phases))
        self.damage, self.steps = 0, 0
        return self.encode(self.position, self.phase, self.damage), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Accion no valida: {action}")
        self.steps += 1
        incident = False
        invalid = False
        old_position = self.position
        old_distance = self._distance(old_position)
        effective_action = int(action)
        if action != 4 and self.np_random.random() < self.slip_probability:
            effective_action = int(self.np_random.integers(0, 4))

        if effective_action == 4:
            candidate = old_position
        else:
            drdc = ((-1, 0), (0, 1), (1, 0), (0, -1))[effective_action]
            candidate = (
                int(np.clip(old_position[0] + drdc[0], 0, self.rows - 1)),
                int(np.clip(old_position[1] + drdc[1], 0, self.cols - 1)),
            )
            invalid = candidate == old_position

        if candidate[0] == 3 and candidate not in (self.ford, self.bridge):
            candidate = old_position
            invalid = True

        if candidate == self.ford and candidate != old_position:
            risk = (0.06, 0.30, 0.67)[self.phase] * self.hazard_scale
            if self.np_random.random() < min(0.95, risk):
                incident = True
                self.damage += 1
                candidate = self.start

        self.position = candidate
        progress = old_distance - self._distance(self.position)
        reward = -0.08 + 0.12 * progress
        reward -= 0.05 if effective_action == 4 else 0.0
        reward -= 0.25 if invalid else 0.0
        reward -= 4.0 if incident else 0.0
        success = self.position == self.goal
        failure = self.damage >= 2
        terminated = success or failure
        reward += 12.0 if success else 0.0
        reward -= 8.0 if failure else 0.0
        self.phase = (self.phase + 1) % self.phases
        truncated = self.steps >= self.max_steps and not terminated
        info = self._info() | {"incident": incident, "success": success, "failure": failure}
        return self.encode(self.position, self.phase, self.damage), float(reward), terminated, truncated, info

    def _distance(self, position: tuple[int, int]) -> int:
        return abs(position[0] - self.goal[0]) + abs(position[1] - self.goal[1])

    def _info(self) -> dict[str, Any]:
        return {"position": self.position, "phase": self.phase, "damage": self.damage, "steps": self.steps}

    def render(self):
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for col in range(self.cols):
            grid[3][col] = "~"
        grid[self.ford[0]][self.ford[1]] = "F"
        grid[self.bridge[0]][self.bridge[1]] = "B"
        grid[self.goal[0]][self.goal[1]] = "G"
        grid[self.position[0]][self.position[1]] = "A"
        return "\n".join(" ".join(row) for row in grid)


class ThermalGliderEnv(gym.Env):
    """Control continuo de posicion, velocidad, temperatura y bateria."""

    metadata = {"render_modes": ["ansi"], "render_fps": 8}
    ACTIONS = ("planear", "impulso", "acelerar", "refrigerar")

    def __init__(self, wind_std: float = 0.006, max_steps: int = 120, render_mode: str | None = None):
        super().__init__()
        self.wind_std, self.max_steps, self.render_mode = wind_std, max_steps, render_mode
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([0.0, -0.08, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.2, 0.16, 1.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.steps = 0
        self.state = np.array(
            [
                self.np_random.uniform(0.0, 0.025),
                self.np_random.uniform(-0.006, 0.006),
                self.np_random.uniform(0.22, 0.55),
                self.np_random.uniform(0.82, 1.0),
            ],
            dtype=np.float32,
        )
        return self.state.copy(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Accion no valida: {action}")
        self.steps += 1
        position, velocity, temperature, battery = map(float, self.state)
        old_position = position
        thrust = (0.0, 0.026, 0.057, -0.013)[action]
        heat = (-0.018, 0.034, 0.090, -0.078)[action]
        energy = (0.0015, 0.009, 0.024, 0.014)[action]
        gust = float(self.np_random.normal(0.0, self.wind_std))
        velocity = float(np.clip(0.88 * velocity + thrust - 0.010 - 0.014 * temperature + gust, -0.08, 0.16))
        position = float(np.clip(position + velocity, 0.0, 1.2))
        temperature = float(
            np.clip(0.94 * temperature + heat + 0.035 * abs(velocity) + self.np_random.normal(0.0, 0.004), 0.0, 1.5)
        )
        battery = float(np.clip(battery - energy * (1.0 + 0.45 * temperature), 0.0, 1.0))
        self.state = np.array([position, velocity, temperature, battery], dtype=np.float32)
        reward = 4.0 * (position - old_position) - 0.035 - 0.18 * energy - 0.16 * max(0.0, temperature - 0.80)
        reward -= 0.08 if position <= 0.0 and velocity < 0 else 0.0
        success = position >= 1.0 and velocity <= 0.045 and temperature <= 0.90 and battery >= 0.08
        violation = temperature >= 1.00 or battery <= 0.035 or position >= 1.14
        reward -= 0.20 if position >= 1.0 and not success else 0.0
        terminated = success or violation
        if success:
            reward += 15.0 + 2.0 * battery - 0.7 * temperature
        elif violation:
            reward -= 10.0
        truncated = self.steps >= self.max_steps and not terminated
        info = self._info() | {"success": success, "violation": violation, "action": int(action)}
        return self.state.copy(), float(reward), terminated, truncated, info

    def _info(self) -> dict[str, Any]:
        return {
            "position": float(self.state[0]),
            "velocity": float(self.state[1]),
            "temperature": float(self.state[2]),
            "battery": float(self.state[3]),
            "steps": self.steps,
        }

    def render(self):
        position, velocity, temperature, battery = self.state
        return f"x={position:.3f} v={velocity:.3f} temp={temperature:.3f} bateria={battery:.3f}"

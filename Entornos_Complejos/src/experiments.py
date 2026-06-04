"""Protocolos experimentales reproducibles."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .approx_agents import DQNAgent, SemiGradientSarsaAgent
from .core import evaluate_agent, set_global_seed, train_agent
from .environments import RiverCourierEnv, ThermalGliderEnv
from .plotting import (
    plot_continuous_trajectories,
    plot_episode_length_indicator,
    plot_learning_fingerprint,
    plot_river_policies,
    plot_robustness,
)
from .tabular_agents import (
    DiscretizedQLearningAgent,
    MonteCarloOffPolicyAgent,
    MonteCarloOnPolicyAgent,
    QLearningAgent,
    SarsaAgent,
)


TABULAR_SEEDS = (2026, 2037, 2048, 2059, 2070)
APPROX_SEEDS = (2026, 2041, 2056)


def _records_frame(records, agent, seed, study):
    rows = []
    for record in records:
        row = asdict(record) | {"agent": agent, "seed": seed, "study": study}
        row["incident_or_violation"] = record.incident + int(record.violation)
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluation_row(records, agent, seed, condition, study):
    frame = pd.DataFrame([asdict(record) for record in records])
    return {
        "study": study,
        "agent": agent,
        "seed": seed,
        "condition": condition,
        "mean_return": frame["return_value"].mean(),
        "mean_length": frame["length"].mean(),
        "success_rate": frame["success"].mean(),
        "risk_rate": frame["incident"].mean() + frame["violation"].mean(),
    }


def _tabular_agents(seed):
    env = RiverCourierEnv()
    common = dict(n_states=env.observation_space.n, n_actions=env.action_space.n, gamma=0.98, epsilon_decay_episodes=1900, seed=seed)
    env.close()
    return [
        MonteCarloOnPolicyAgent(**common, epsilon_start=0.36, epsilon_end=0.045),
        MonteCarloOffPolicyAgent(**common, epsilon_start=0.48, epsilon_end=0.12),
        SarsaAgent(**common, epsilon_start=0.34, epsilon_end=0.025, alpha=0.16),
        QLearningAgent(**common, epsilon_start=0.34, epsilon_end=0.025, alpha=0.16),
    ]


def _approx_agents(seed):
    env = ThermalGliderEnv()
    low, high, actions = env.observation_space.low, env.observation_space.high, env.action_space.n
    env.close()
    return [
        DiscretizedQLearningAgent(low, high, bins=(2, 2, 2, 2), n_actions=actions, gamma=0.99, alpha=0.16, epsilon_start=0.42, epsilon_end=0.035, epsilon_decay_episodes=900, seed=seed),
        SemiGradientSarsaAgent(low, high, actions, seed=seed),
        DQNAgent(input_dim=4, n_actions=actions, seed=seed),
    ]


def _summary(evaluation):
    return evaluation.groupby(["study", "agent", "condition"], as_index=False).agg(
        mean_return=("mean_return", "mean"),
        mean_length=("mean_length", "mean"),
        success_rate=("success_rate", "mean"),
        risk_rate=("risk_rate", "mean"),
    )


def run_tabular_study(output_dir="results", episodes=2600, evaluation_episodes=300, seeds=TABULAR_SEEDS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, rows, representatives = [], [], {}
    for seed in seeds:
        set_global_seed(seed)
        for agent in _tabular_agents(seed):
            frames.append(_records_frame(train_agent(lambda: RiverCourierEnv(), agent, episodes, seed), agent.name, seed, "tabular"))
            for condition, factory in (
                ("marea nominal", lambda: RiverCourierEnv(hazard_scale=1.0)),
                ("crecida +35%", lambda: RiverCourierEnv(hazard_scale=1.35)),
            ):
                records, _ = evaluate_agent(factory, agent, evaluation_episodes, seed + 50_000)
                rows.append(_evaluation_row(records, agent.name, seed, condition, "tabular"))
            if seed == seeds[0]:
                representatives[agent.name] = agent
    training, evaluation = pd.concat(frames, ignore_index=True), pd.DataFrame(rows)
    summary = _summary(evaluation)
    training.to_csv(output_dir / "tabular_training_metrics.csv", index=False)
    evaluation.to_csv(output_dir / "tabular_evaluation_by_seed.csv", index=False)
    summary.to_csv(output_dir / "tabular_summary.csv", index=False)
    plot_learning_fingerprint(training, output_dir / "tabular_learning_fingerprint.png", "RiverCourier: huella de aprendizaje", 100)
    plot_robustness(summary, output_dir / "tabular_robustness.png", "RiverCourier: robustez ante una crecida")
    plot_episode_length_indicator(training, output_dir / "episode_length_indicator.png")
    plot_river_policies(representatives, RiverCourierEnv(), output_dir / "river_policy_maps.png")
    return training, summary, representatives


def run_approximation_study(output_dir="results", episodes=1200, evaluation_episodes=250, seeds=APPROX_SEEDS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, rows, representatives, trajectories = [], [], {}, {}
    for seed in seeds:
        set_global_seed(seed)
        for agent in _approx_agents(seed):
            frames.append(_records_frame(train_agent(lambda: ThermalGliderEnv(), agent, episodes, seed), agent.name, seed, "aproximado"))
            for condition, factory in (
                ("viento nominal", lambda: ThermalGliderEnv(wind_std=0.006)),
                ("turbulencia x2.3", lambda: ThermalGliderEnv(wind_std=0.014)),
            ):
                records, trajectory = evaluate_agent(factory, agent, evaluation_episodes, seed + 70_000, capture_one=seed == seeds[0] and condition == "viento nominal")
                rows.append(_evaluation_row(records, agent.name, seed, condition, "aproximado"))
                if trajectory:
                    trajectories[agent.name] = trajectory
            if seed == seeds[0]:
                representatives[agent.name] = agent
    training, evaluation = pd.concat(frames, ignore_index=True), pd.DataFrame(rows)
    summary = _summary(evaluation)
    training.to_csv(output_dir / "approx_training_metrics.csv", index=False)
    evaluation.to_csv(output_dir / "approx_evaluation_by_seed.csv", index=False)
    summary.to_csv(output_dir / "approx_summary.csv", index=False)
    plot_learning_fingerprint(training, output_dir / "approx_learning_fingerprint.png", "ThermalGlider: huella de aprendizaje", 70)
    plot_robustness(summary, output_dir / "approx_robustness.png", "ThermalGlider: robustez ante turbulencia")
    plot_continuous_trajectories(trajectories, output_dir / "continuous_trajectories.png")
    return training, summary, representatives


def run_all(output_dir="results"):
    _, tabular, _ = run_tabular_study(output_dir)
    _, approx, _ = run_approximation_study(output_dir)
    combined = pd.concat([tabular, approx], ignore_index=True)
    combined.to_csv(Path(output_dir) / "all_summary.csv", index=False)
    return combined

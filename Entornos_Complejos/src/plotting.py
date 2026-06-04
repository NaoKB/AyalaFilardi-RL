"""Visualizaciones compactas orientadas a preguntas experimentales."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "MC on-policy": "#2A6F97",
    "MC off-policy": "#F18F01",
    "SARSA": "#3A7D44",
    "Q-Learning": "#9C2C77",
    "Q-Learning discretizado": "#6C757D",
    "SARSA semi-gradiente": "#2A6F97",
    "Deep Q-Learning": "#E4572E",
}


def _curves(df: pd.DataFrame, value: str, window: int) -> pd.DataFrame:
    frames = []
    for (agent, seed), group in df.groupby(["agent", "seed"]):
        group = group.sort_values("episode").copy()
        group["smooth"] = group[value].rolling(window, min_periods=max(2, window // 5)).mean()
        frames.append(group[["episode", "agent", "seed", "smooth"]])
    merged = pd.concat(frames, ignore_index=True)
    return merged.groupby(["agent", "episode"])["smooth"].agg(
        median="median", low=lambda x: x.quantile(0.2), high=lambda x: x.quantile(0.8)
    ).reset_index()


def plot_learning_fingerprint(metrics: pd.DataFrame, path: str | Path, title: str, window: int) -> None:
    specs = [
        ("return_value", "Retorno"),
        ("success", "Tasa de exito"),
        ("incident_or_violation", "Incidentes / violaciones"),
        ("length", "Longitud"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (column, label) in zip(axes.flat, specs):
        for agent, group in _curves(metrics, column, window).groupby("agent"):
            color = COLORS.get(agent)
            ax.plot(group["episode"], group["median"], color=color, label=agent, linewidth=2)
            ax.fill_between(group["episode"], group["low"], group["high"], color=color, alpha=0.14)
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("Episodio")
    axes[1, 1].set_xlabel("Episodio")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_robustness(summary: pd.DataFrame, path: str | Path, title: str) -> None:
    agents = list(summary["agent"].drop_duplicates())
    conditions = list(summary["condition"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.75 / len(conditions)
    x = np.arange(len(agents))
    for index, condition in enumerate(conditions):
        subset = summary[summary["condition"] == condition].set_index("agent")
        offset = (index - (len(conditions) - 1) / 2) * width
        axes[0].bar(x + offset, [100 * subset.loc[a, "success_rate"] for a in agents], width, label=condition)
        axes[1].bar(x + offset, [subset.loc[a, "risk_rate"] for a in agents], width, label=condition)
    for ax, ylabel in zip(axes, ("Exito (%)", "Riesgo medio por episodio")):
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, agents, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_episode_length_indicator(metrics: pd.DataFrame, path: str | Path) -> None:
    sample = metrics[(metrics["agent"] == "Q-Learning") & (metrics["seed"] == metrics["seed"].min())].sort_values("episode")
    fig, left = plt.subplots(figsize=(10, 4.8))
    right = left.twinx()
    left.plot(sample["episode"], sample["return_value"].rolling(80, min_periods=10).mean(), color="#2A6F97", linewidth=2)
    right.plot(sample["episode"], sample["length"].rolling(80, min_periods=10).mean(), color="#E4572E", linewidth=2)
    left.set(xlabel="Episodio", ylabel="Retorno suavizado")
    right.set_ylabel("Pasos por episodio")
    left.grid(alpha=0.22)
    fig.suptitle("Dos indicadores complementarios de aprendizaje")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_river_policies(agents: dict[str, Any], env, path: str | Path) -> None:
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "·"}
    selected, phases = ["SARSA", "Q-Learning"], [0, 2]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row_index, name in enumerate(selected):
        for col_index, phase in enumerate(phases):
            ax, agent = axes[row_index, col_index], agents[name]
            values = np.full((env.rows, env.cols), np.nan)
            for row in range(env.rows):
                for col in range(env.cols):
                    if row == 3 and (row, col) not in (env.ford, env.bridge):
                        continue
                    state = env.encode((row, col), phase, 0)
                    values[row, col] = np.max(agent.q[state])
                    ax.text(col, row, arrows[int(np.argmax(agent.q[state]))], ha="center", va="center", fontsize=15)
            ax.imshow(values, cmap="viridis")
            ax.set_title(f"{name} · marea {phase}")
            ax.set_xticks(range(env.cols))
            ax.set_yticks(range(env.rows))
    fig.suptitle("Politicas aprendidas: atajo, espera y ruta segura")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_continuous_trajectories(trajectories: dict[str, list[dict[str, Any]]], path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, trajectory in trajectories.items():
        frame, color = pd.DataFrame(trajectory), COLORS.get(name)
        axes[0].plot(frame["position"], frame["temperature"], marker="o", markersize=2, linewidth=1.5, label=name, color=color)
        axes[1].plot(frame["step"], frame["battery"], linewidth=2, label=name, color=color)
    axes[0].axhline(1.0, linestyle="--", color="#B22222", linewidth=1, label="limite termico")
    axes[0].set(xlabel="Posicion", ylabel="Temperatura", title="Trayectoria de control")
    axes[1].set(xlabel="Paso", ylabel="Bateria", title="Consumo durante el episodio")
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)

"""Funciones de visualización para el estudio de k-brazos."""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms import Algorithm
from src.compat import algorithm_label


def get_algorithm_label(algo: Algorithm) -> str:
    return algorithm_label(algo)


def _x(steps: int) -> np.ndarray:
    return np.arange(1, steps + 1)


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: Iterable[Algorithm], title: str = "", smooth_window: int = 25):
    plt.figure(figsize=(10, 5.5))
    for idx, algo in enumerate(algorithms):
        y = moving_average(rewards[idx], smooth_window)
        plt.plot(_x(steps), y, label=get_algorithm_label(algo), linewidth=2)
    plt.xlabel("Paso temporal")
    plt.ylabel("Recompensa media por paso")
    plt.title(title or "Recompensa media por paso")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: Iterable[Algorithm], title: str = "", smooth_window: int = 25):
    plt.figure(figsize=(10, 5.5))
    for idx, algo in enumerate(algorithms):
        y = 100.0 * moving_average(optimal_selections[idx], smooth_window)
        plt.plot(_x(steps), y, label=get_algorithm_label(algo), linewidth=2)
    plt.xlabel("Paso temporal")
    plt.ylabel("Selección del brazo óptimo (%)")
    plt.ylim(-2, 102)
    plt.title(title or "Porcentaje de selección del brazo óptimo")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: Iterable[Algorithm], title: str = "", reference_log: float | None = None):
    plt.figure(figsize=(10, 5.5))
    for idx, algo in enumerate(algorithms):
        plt.plot(_x(steps), regret_accumulated[idx], label=get_algorithm_label(algo), linewidth=2)
    if reference_log is not None:
        t = _x(steps)
        plt.plot(t, reference_log * np.log1p(t), linestyle="--", label=f"{reference_log:g}·log(1+t)")
    plt.xlabel("Paso temporal")
    plt.ylabel("Pseudo-regret acumulado")
    plt.title(title or "Pseudo-regret acumulado")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: pd.DataFrame, algorithm_label: str, title: str = ""):
    df = arm_stats[arm_stats["algoritmo"] == algorithm_label].copy()
    if df.empty:
        raise ValueError(f"No hay estadísticas para {algorithm_label!r}.")
    labels = [f"{int(row.brazo)}\nn={row.tiradas_medias:.1f}" + ("\nópt." if row.óptimo else "") for row in df.itertuples()]
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(np.arange(len(df)), df["Q_estimado"])
    optimal_idx = df.index[df["óptimo"]].tolist()
    if optimal_idx:
        # remarcar el óptimo con borde, sin imponer paleta de color
        pos = list(df.index).index(optimal_idx[0])
        bars[pos].set_linewidth(2.5)
        bars[pos].set_edgecolor("black")
    plt.xticks(np.arange(len(df)), labels)
    plt.xlabel("Brazo y número medio de selecciones")
    plt.ylabel("Q(a) estimado al final")
    plt.title(title or f"Estadísticas por brazo — {algorithm_label}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_summary_heatmap(summary: pd.DataFrame, value: str = "regret_final", title: str = ""):
    """Tabla visual sencilla sin depender de seaborn."""
    pivot = summary.pivot(index="algoritmo", columns="escenario", values=value)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title(title or value)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.show()

"""Utilidades de compatibilidad sin modificar las clases de ``arms`` ni ``algorithms``.

Este módulo centraliza las pequeñas funciones auxiliares que necesita la parte
experimental para producir etiquetas, calcular gaps y reiniciar estado interno.
Se mantiene separado para respetar las definiciones originales de las clases
proporcionadas en ``src/arms`` y ``src/algorithms``.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def algorithm_label(algo: Any) -> str:
    """Devuelve una etiqueta informativa sin exigir que el algoritmo implemente ``label``."""
    cls_name = type(algo).__name__
    if hasattr(algo, "epsilon"):
        return f"ε-greedy ε={float(algo.epsilon):g}"
    if cls_name == "UCB1" and hasattr(algo, "c"):
        return f"UCB1 c={float(algo.c):g}"
    if cls_name == "UCB2" and hasattr(algo, "alpha"):
        return f"UCB2 α={float(algo.alpha):g}"
    if hasattr(algo, "temperature"):
        return f"Softmax τ={float(algo.temperature):g}"
    return cls_name


def reset_algorithm_state(algo: Any) -> None:
    """Reinicia un algoritmo respetando su clase original.

    Algunas clases originales guardan estado adicional fuera de ``counts`` y
    ``values``. No se modifica la clase; solo se limpia ese estado entre runs
    para que cada repetición experimental sea independiente.
    """
    algo.reset()
    if hasattr(algo, "selected_arms"):
        algo.selected_arms.clear()
    if hasattr(algo, "epocas"):
        algo.epocas = np.zeros(algo.k, dtype=int)
    if hasattr(algo, "_brazo_actual"):
        algo._brazo_actual = None
    if hasattr(algo, "_restantes_en_epoca"):
        algo._restantes_en_epoca = 0


def bandit_optimal_value(bandit: Any) -> float:
    return float(max(bandit.expected_rewards))


def bandit_gap(bandit: Any, arm_index: int) -> float:
    return bandit_optimal_value(bandit) - float(bandit.expected_rewards[arm_index])


def describe_bandit(bandit: Any) -> str:
    lines = [str(bandit), "", "Resumen de medias reales usadas solo para evaluación:"]
    for i, expected_reward in enumerate(bandit.expected_rewards):
        marker = " <- óptimo" if i == bandit.optimal_arm else ""
        lines.append(
            f"  brazo {i}: E[r]={float(expected_reward):.4f}, "
            f"gap={bandit_gap(bandit, i):.4f}{marker}"
        )
    return "\n".join(lines)

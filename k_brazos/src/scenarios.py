"""Escenarios experimentales propios para la Parte 1.

Los tres bancos no se generan al azar: se fijan para que todos los algoritmos se
enfrenten al mismo problema y las conclusiones sean reproducibles. El diseño
intenta no replicar el experimento estándar de diez brazos gaussianos con medias
aleatorias.

Importante: este módulo no cambia las clases de ``src.arms``. Construye los
bandidos usando exactamente los constructores originales de ``ArmBernoulli``,
``ArmBinomial``, ``ArmNormal`` y ``Bandit``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .arms import ArmBernoulli, ArmBinomial, ArmNormal, Bandit
from .compat import bandit_gap, describe_bandit


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    title: str
    bandit: Bandit
    rationale: str


def build_scenarios() -> dict[str, ScenarioSpec]:
    """Construye los tres bandidos estacionarios usados en todos los notebooks."""
    bernoulli_ps = [0.041, 0.048, 0.052, 0.055, 0.058, 0.060, 0.064]
    bernoulli = Bandit([ArmBernoulli(p) for p in bernoulli_ps])

    binomial_n = 80
    binomial_ps = [0.085, 0.115, 0.135, 0.158, 0.176, 0.190, 0.202]
    binomial = Bandit([ArmBinomial(binomial_n, p) for p in binomial_ps])

    normal_params = [
        (4.4, 1.7),
        (5.0, 2.8),
        (5.4, 1.2),
        (5.7, 4.0),
        (6.1, 1.0),
        (6.25, 3.2),
        (6.55, 1.5),
    ]
    normal = Bandit([ArmNormal(mu, sigma) for mu, sigma in normal_params])

    return {
        "bernoulli": ScenarioSpec(
            "bernoulli",
            "Bernoulli: anuncios con CTR bajo",
            bernoulli,
            "Prueba difícil por baja señal y gaps pequeños; favorece políticas que acumulen evidencia sin sobreexplorar.",
        ),
        "binomial": ScenarioSpec(
            "binomial",
            "Binomial: promociones por lotes",
            binomial,
            "La escala de recompensa es mayor y la varianza depende de p; sirve para observar sensibilidad a la escala.",
        ),
        "normal": ScenarioSpec(
            "normal",
            "Normal: recomendadores de watch-time",
            normal,
            "Permite recompensas continuas y ruido heterocedástico; el brazo más variable no siempre es el mejor.",
        ),
    }


def scenario_table(scenarios: dict[str, ScenarioSpec] | None = None) -> pd.DataFrame:
    scenarios = scenarios or build_scenarios()
    rows = []
    for spec in scenarios.values():
        b = spec.bandit
        for i, arm in enumerate(b.arms):
            rows.append(
                {
                    "escenario": spec.key,
                    "brazo": i,
                    "distribución": type(arm).__name__.replace("Arm", ""),
                    "parámetros": str(arm),
                    "E[r]": b.expected_rewards[i],
                    "gap": bandit_gap(b, i),
                    "óptimo": i == b.optimal_arm,
                }
            )
    return pd.DataFrame(rows)


__all__ = ["ScenarioSpec", "build_scenarios", "scenario_table", "describe_bandit"]

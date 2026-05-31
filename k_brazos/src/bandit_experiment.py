"""Motor experimental para comparar algoritmos de k-brazos.

El módulo se apoya en las clases originales de ``src.algorithms`` y ``src.arms``
sin modificar sus definiciones públicas. Las funciones auxiliares necesarias
para etiquetar algoritmos, calcular gaps o reiniciar estado extra viven en
``src.compat``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.algorithms import Algorithm
from src.arms import Bandit
from src.compat import algorithm_label, bandit_gap, bandit_optimal_value, reset_algorithm_state

REWARDS_NAME = "rewards"
CUM_REWARDS_NAME = "cumulative_rewards"
OPTIMAL_NAME = "optimal"
REGRET_NAME = "regret"
COUNTS_NAME = "counts"
VALUES_NAME = "values"
ARM_STATS_NAME = "arm_stats"
SUMMARY_NAME = "summary"


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int = 800
    runs: int = 200
    seed: int = 2026


class BanditExperiment:
    @staticmethod
    def run_experiment(
        bandit: Bandit,
        algorithms: Iterable[Algorithm],
        steps: int = 800,
        runs: int = 200,
        seed: int = 2026,
    ) -> dict[str, np.ndarray | pd.DataFrame]:
        """Ejecuta varios algoritmos sobre un bandido estacionario.

        El regret implementado es pseudo-regret acumulado:
        ``sum_t (mu* - mu_{a_t})``. Los algoritmos no reciben las medias reales;
        estas se usan únicamente para auditar la evaluación, obtener el brazo
        óptimo y medir la pérdida esperada de cada selección.
        """
        algorithms = list(algorithms)
        n_algorithms = len(algorithms)
        if n_algorithms == 0:
            raise ValueError("Debe proporcionarse al menos un algoritmo.")
        if any(algo.k != bandit.k for algo in algorithms):
            raise ValueError("Todos los algoritmos deben tener k igual al número de brazos del bandido.")

        rewards = np.zeros((n_algorithms, steps), dtype=float)
        cumulative_rewards = np.zeros((n_algorithms, steps), dtype=float)
        optimal_selections = np.zeros((n_algorithms, steps), dtype=float)
        regret_accumulated = np.zeros((n_algorithms, steps), dtype=float)
        counts_per_arm = np.zeros((n_algorithms, bandit.k), dtype=float)
        values_per_arm = np.zeros((n_algorithms, bandit.k), dtype=float)
        reward_sum_per_arm = np.zeros((n_algorithms, bandit.k), dtype=float)

        optimal_value = bandit_optimal_value(bandit)
        base_state = np.random.get_state()
        for run in range(runs):
            np.random.seed(seed + run)
            for algo in algorithms:
                reset_algorithm_state(algo)

            total_reward = np.zeros(n_algorithms, dtype=float)
            total_regret = np.zeros(n_algorithms, dtype=float)
            run_reward_sum_per_arm = np.zeros((n_algorithms, bandit.k), dtype=float)

            for step in range(steps):
                for idx, algo in enumerate(algorithms):
                    chosen_arm = algo.select_arm()
                    reward = bandit.pull_arm(chosen_arm)
                    algo.update(chosen_arm, reward)

                    total_reward[idx] += reward
                    total_regret[idx] += optimal_value - float(bandit.expected_rewards[chosen_arm])
                    run_reward_sum_per_arm[idx, chosen_arm] += reward

                    rewards[idx, step] += reward
                    cumulative_rewards[idx, step] += total_reward[idx]
                    optimal_selections[idx, step] += float(chosen_arm == bandit.optimal_arm)
                    regret_accumulated[idx, step] += total_regret[idx]

            for idx, algo in enumerate(algorithms):
                counts_per_arm[idx] += algo.counts
                values_per_arm[idx] += algo.values
                reward_sum_per_arm[idx] += run_reward_sum_per_arm[idx]

        np.random.set_state(base_state)

        rewards /= runs
        cumulative_rewards /= runs
        optimal_selections /= runs
        regret_accumulated /= runs
        counts_per_arm /= runs
        values_per_arm /= runs
        reward_sum_per_arm /= runs

        summary = BanditExperiment.summary_table(
            bandit=bandit,
            algorithms=algorithms,
            rewards=rewards,
            cumulative_rewards=cumulative_rewards,
            optimal_selections=optimal_selections,
            regret_accumulated=regret_accumulated,
            counts_per_arm=counts_per_arm,
            steps=steps,
        )
        arm_stats = BanditExperiment.arm_statistics_table(
            bandit=bandit,
            algorithms=algorithms,
            counts_per_arm=counts_per_arm,
            values_per_arm=values_per_arm,
            reward_sum_per_arm=reward_sum_per_arm,
        )

        return {
            REWARDS_NAME: rewards,
            CUM_REWARDS_NAME: cumulative_rewards,
            OPTIMAL_NAME: optimal_selections,
            REGRET_NAME: regret_accumulated,
            COUNTS_NAME: counts_per_arm,
            VALUES_NAME: values_per_arm,
            ARM_STATS_NAME: arm_stats,
            SUMMARY_NAME: summary,
        }

    @staticmethod
    def summary_table(
        bandit: Bandit,
        algorithms: list[Algorithm],
        rewards: np.ndarray,
        cumulative_rewards: np.ndarray,
        optimal_selections: np.ndarray,
        regret_accumulated: np.ndarray,
        counts_per_arm: np.ndarray,
        steps: int,
    ) -> pd.DataFrame:
        tail = max(1, int(0.2 * steps))
        rows = []
        for idx, algo in enumerate(algorithms):
            rows.append(
                {
                    "algoritmo": algorithm_label(algo),
                    "recompensa_media_total": cumulative_rewards[idx, -1] / steps,
                    "recompensa_media_último_20%": float(np.mean(rewards[idx, -tail:])),
                    "regret_final": regret_accumulated[idx, -1],
                    "% óptimo_final": 100.0 * optimal_selections[idx, -1],
                    "% óptimo_último_20%": 100.0 * float(np.mean(optimal_selections[idx, -tail:])),
                    "% tiradas_brazo_óptimo": 100.0 * counts_per_arm[idx, bandit.optimal_arm] / steps,
                }
            )
        df = pd.DataFrame(rows)
        return df.sort_values(["regret_final", "recompensa_media_último_20%"], ascending=[True, False]).reset_index(drop=True)

    @staticmethod
    def arm_statistics_table(
        bandit: Bandit,
        algorithms: list[Algorithm],
        counts_per_arm: np.ndarray,
        values_per_arm: np.ndarray,
        reward_sum_per_arm: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        for idx, algo in enumerate(algorithms):
            for arm_idx in range(bandit.k):
                count = counts_per_arm[idx, arm_idx]
                rows.append(
                    {
                        "algoritmo": algorithm_label(algo),
                        "brazo": arm_idx,
                        "tiradas_medias": count,
                        "Q_estimado": values_per_arm[idx, arm_idx],
                        "recompensa_observada_media": reward_sum_per_arm[idx, arm_idx] / max(count, 1e-12),
                        "E[r] real": bandit.expected_rewards[arm_idx],
                        "gap": bandit_gap(bandit, arm_idx),
                        "óptimo": arm_idx == bandit.optimal_arm,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def run_many(scenarios: dict, algorithm_factory, steps: int = 800, runs: int = 200, seed: int = 2026):
        results = {}
        algorithms_by_scenario = {}
        for i, (key, spec) in enumerate(scenarios.items()):
            algorithms = algorithm_factory(spec.bandit.k)
            algorithms_by_scenario[key] = algorithms
            results[key] = BanditExperiment.run_experiment(
                spec.bandit,
                algorithms,
                steps=steps,
                runs=runs,
                seed=seed + 10_000 * i,
            )
        return results, algorithms_by_scenario

    @staticmethod
    def collect_summaries(results: dict[str, dict], scenario_titles: dict[str, str] | None = None) -> pd.DataFrame:
        frames = []
        for key, result in results.items():
            df = result[SUMMARY_NAME].copy()
            df.insert(0, "escenario", scenario_titles.get(key, key) if scenario_titles else key)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

import numpy as np
from src.arms import Bandit
from src.plotting import plot_average_rewards,plot_optimal_selections,plot_arm_statistics,plot_regret

SELECTED_NAME = "selected" ; AVG_REWARD_NAME = "avg_reward" ; IS_OPTIMAL_NAME = "is_optimal"
REWARDS_NAME = "rewards" ; OPTIMAL_NAME = "optimal" ; REGRET_NAME = "regret"
COUNTS_NAME = "counts" ; VALUES_NAME = "values"

class BanditExperiment:
    def run_experiment(bandit, algorithms, steps, runs):
        optimal_arm = bandit.optimal_arm
        rewards = np.zeros((len(algorithms), steps))
        optimal_selections = np.zeros((len(algorithms), steps))

        regrets = np.zeros((len(algorithms), steps))
        q = bandit.get_expected_value(optimal_arm)
        rewards_per_arm = np.zeros((len(algorithms), bandit.k))
        counts_per_arm = np.zeros((len(algorithms), bandit.k))

        for _ in range(runs):
            current_bandit = Bandit(arms=bandit.arms)
            [algo.reset() for algo in algorithms]
            total_rewards_per_algo = np.zeros(len(algorithms))

            for step in range(steps):
                for idx, algo in enumerate(algorithms):
                    chosen_arm = algo.select_arm()
                    reward = current_bandit.pull_arm(chosen_arm)
                    algo.update(chosen_arm, reward)
                    rewards[idx, step] += reward
                    optimal_selections[idx, step] += {True: 1, False: 0}[chosen_arm == optimal_arm]
                    total_rewards_per_algo[idx] += (q - reward)
                    regrets[idx, step] += total_rewards_per_algo[idx]

            for idx, algo in enumerate(algorithms):
                counts_per_arm[idx] += algo.counts
                rewards_per_arm[idx] += algo.values

        rewards /= runs
        optimal_selections /= runs
        regrets /= runs
        counts_per_arm /= runs
        rewards_per_arm /= runs

        return {
            REWARDS_NAME: rewards,
            OPTIMAL_NAME: optimal_selections,
            REGRET_NAME: regrets,
            COUNTS_NAME: counts_per_arm,
            VALUES_NAME: rewards_per_arm
        }
    
    def plot(function, steps, results, results_key, algorithms, title):
        for name, result in results.items():
            function(steps, result[results_key], algorithms, f"{title} - {name}")

    def plot_rewards(results, steps, algorithms):
        BanditExperiment.plot(plot_average_rewards, steps, results, REWARDS_NAME, algorithms, "Recompensa Promedio")

    def plot_optimal_selections(results, steps, algorithms):
        BanditExperiment.plot(plot_optimal_selections, steps, results, OPTIMAL_NAME, algorithms, "Porcentaje de Selección del Brazo Óptimo")

    def plot_regrets(results, steps, algorithms):
        BanditExperiment.plot(plot_regret, steps, results, REGRET_NAME, algorithms, "Regret Acumulado")

    def build_arm_stats(bandit, counts, avg_rewards_per_arm, algorithm_labels):
        return [{
            arm_idx: {
                SELECTED_NAME: counts[algo_idx][arm_idx],
                AVG_REWARD_NAME: avg_rewards_per_arm[algo_idx][arm_idx],
                IS_OPTIMAL_NAME: arm_idx == bandit.optimal_arm,
            } for arm_idx in range(bandit.k)
        } for algo_idx in range(len(algorithm_labels))]
    
    def plot_arm_statistics(bandit, result, algorithms, name):
        arm_stats = BanditExperiment.build_arm_stats(bandit, result[COUNTS_NAME], result[VALUES_NAME], algorithms)
        plot_arm_statistics(arm_stats, algorithms, f"Estadísticas por Brazo - {name}")
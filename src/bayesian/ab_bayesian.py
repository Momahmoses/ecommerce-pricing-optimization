"""
Bayesian A/B testing with Thompson Sampling for multi-armed bandit optimization.
Makes promotion decisions faster without fixed sample sizes.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ThompsonSamplingBandit:
    """
    Multi-armed bandit for real-time promotion optimization.
    Automatically allocates more traffic to better-performing promotions.
    """
    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alphas = np.full(n_arms, prior_alpha)
        self.betas = np.full(n_arms, prior_beta)
        self.arm_counts = np.zeros(n_arms)
        self.arm_successes = np.zeros(n_arms)

    def select_arm(self) -> int:
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.arm_counts[arm] += 1
        self.arm_successes[arm] += reward
        self.alphas[arm] += reward
        self.betas[arm] += (1 - reward)

    def best_arm(self) -> int:
        return int(np.argmax(self.alphas / (self.alphas + self.betas)))

    def arm_stats(self, arm_names: list = None) -> pd.DataFrame:
        names = arm_names or [f"Arm {i}" for i in range(self.n_arms)]
        estimated_rates = self.alphas / (self.alphas + self.betas)
        return pd.DataFrame({
            "arm": names,
            "trials": self.arm_counts.astype(int),
            "successes": self.arm_successes.astype(int),
            "estimated_conversion_rate": estimated_rates.round(4),
            "is_best": [i == self.best_arm() for i in range(self.n_arms)],
        })


def run_bandit_simulation(
    true_rates: list[float],
    n_rounds: int = 5000,
    arm_names: list = None,
) -> dict:
    bandit = ThompsonSamplingBandit(n_arms=len(true_rates))
    arm_names = arm_names or [f"Promo_{i}" for i in range(len(true_rates))]
    cumulative_reward = 0

    for _ in range(n_rounds):
        arm = bandit.select_arm()
        reward = float(np.random.random() < true_rates[arm])
        bandit.update(arm, reward)
        cumulative_reward += reward

    optimal_rate = max(true_rates)
    optimal_reward = optimal_rate * n_rounds
    regret = optimal_reward - cumulative_reward

    stats = bandit.arm_stats(arm_names)
    return {
        "final_stats": stats,
        "best_arm": arm_names[bandit.best_arm()],
        "true_best_arm": arm_names[true_rates.index(max(true_rates))],
        "correct_identification": bandit.best_arm() == true_rates.index(max(true_rates)),
        "cumulative_regret": round(regret, 1),
        "regret_rate": round(regret / n_rounds, 4),
        "total_reward": int(cumulative_reward),
    }


if __name__ == "__main__":
    print("=== THOMPSON SAMPLING BANDIT — PROMOTION OPTIMIZATION ===\n")
    promotions = {
        "No discount": 0.08,
        "10% off": 0.11,
        "20% off": 0.15,
        "30% off": 0.14,
        "Bundle deal": 0.17,
    }
    result = run_bandit_simulation(
        true_rates=list(promotions.values()),
        n_rounds=5000,
        arm_names=list(promotions.keys()),
    )

    print(f"Best arm identified: {result['best_arm']}")
    print(f"True best arm: {result['true_best_arm']}")
    print(f"Correct: {result['correct_identification']}")
    print(f"Cumulative regret: {result['cumulative_regret']:.0f} conversions lost\n")
    print("Final arm statistics:")
    print(result["final_stats"].to_string(index=False))

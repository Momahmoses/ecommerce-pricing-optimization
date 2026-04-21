"""
Rigorous A/B testing framework: sample size calculation, randomization, statistical tests.
Bayesian + Frequentist approaches.
"""
import numpy as np
import pandas as pd
from scipy import stats
import logging
import hashlib

logger = logging.getLogger(__name__)


def required_sample_size(baseline_rate: float, min_detectable_effect: float,
                          alpha: float = 0.05, power: float = 0.8) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    pooled_p = (p1 + p2) / 2
    n = (z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) +
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(np.ceil(n))


def assign_variant(user_id: str, experiment_id: str, variants: list = ["control", "treatment"],
                   weights: list = None) -> str:
    weights = weights or [1 / len(variants)] * len(variants)
    hash_val = int(hashlib.sha256(f"{user_id}:{experiment_id}".encode()).hexdigest(), 16)
    bucket = (hash_val % 10000) / 10000.0
    cumulative = 0
    for variant, weight in zip(variants, weights):
        cumulative += weight
        if bucket < cumulative:
            return variant
    return variants[-1]


class FrequentistABTest:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze(self, control_data: np.ndarray, treatment_data: np.ndarray,
                metric: str = "conversion") -> dict:
        if metric == "conversion":
            n_c, n_t = len(control_data), len(treatment_data)
            conv_c = control_data.mean()
            conv_t = treatment_data.mean()
            pooled_p = (control_data.sum() + treatment_data.sum()) / (n_c + n_t)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n_c + 1 / n_t))
            z_stat = (conv_t - conv_c) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            relative_lift = (conv_t - conv_c) / conv_c * 100 if conv_c > 0 else 0
            return {
                "control_mean": round(conv_c, 4),
                "treatment_mean": round(conv_t, 4),
                "relative_lift_pct": round(relative_lift, 2),
                "z_statistic": round(z_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < self.alpha,
                "conclusion": (
                    f"Treatment {'significantly' if p_value < self.alpha else 'does NOT significantly'} "
                    f"{'increases' if relative_lift > 0 else 'decreases'} conversion by "
                    f"{abs(relative_lift):.1f}% (p={p_value:.4f})"
                ),
            }
        else:
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
            return {
                "control_mean": round(control_data.mean(), 4),
                "treatment_mean": round(treatment_data.mean(), 4),
                "relative_lift_pct": round((treatment_data.mean() - control_data.mean()) / control_data.mean() * 100, 2),
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < self.alpha,
            }


class BayesianABTest:
    """
    Beta-Binomial Bayesian A/B test.
    Makes decisions faster than frequentist without waiting for fixed sample size.
    """
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def analyze(self, control_conversions: int, control_trials: int,
                treatment_conversions: int, treatment_trials: int,
                n_samples: int = 50000) -> dict:
        alpha_c = self.prior_alpha + control_conversions
        beta_c = self.prior_beta + control_trials - control_conversions
        alpha_t = self.prior_alpha + treatment_conversions
        beta_t = self.prior_beta + treatment_trials - treatment_conversions

        samples_c = np.random.beta(alpha_c, beta_c, n_samples)
        samples_t = np.random.beta(alpha_t, beta_t, n_samples)

        prob_treatment_better = (samples_t > samples_c).mean()
        expected_lift = (samples_t - samples_c).mean()
        lift_95ci = np.percentile(samples_t - samples_c, [2.5, 97.5])

        posterior_c_mean = alpha_c / (alpha_c + beta_c)
        posterior_t_mean = alpha_t / (alpha_t + beta_t)

        return {
            "control_posterior_mean": round(posterior_c_mean, 4),
            "treatment_posterior_mean": round(posterior_t_mean, 4),
            "prob_treatment_better": round(prob_treatment_better, 3),
            "expected_lift": round(expected_lift, 4),
            "lift_95_ci": [round(lift_95ci[0], 4), round(lift_95ci[1], 4)],
            "decision": (
                "SHIP TREATMENT" if prob_treatment_better > 0.95
                else "SHIP CONTROL" if prob_treatment_better < 0.05
                else f"WAIT — {prob_treatment_better:.1%} confident treatment is better"
            ),
        }

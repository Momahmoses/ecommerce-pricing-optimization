"""
Causal inference using propensity score matching to measure true promotion lift.
Controls for confounders: seasonality, product category, user income level.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def estimate_propensity_scores(df: pd.DataFrame, treatment_col: str,
                                confounder_cols: list) -> np.ndarray:
    X = df[confounder_cols].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, df[treatment_col])
    return model.predict_proba(X_scaled)[:, 1]


def propensity_score_matching(df: pd.DataFrame, treatment_col: str,
                               outcome_col: str, confounder_cols: list,
                               caliper: float = 0.05) -> dict:
    df = df.copy()
    propensity = estimate_propensity_scores(df, treatment_col, confounder_cols)
    df["propensity_score"] = propensity

    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    matched_pairs = []
    used_control_indices = set()

    for idx, treated_row in treated.iterrows():
        ps = treated_row["propensity_score"]
        eligible = control[
            ~control.index.isin(used_control_indices) &
            (np.abs(control["propensity_score"] - ps) <= caliper)
        ]
        if len(eligible) == 0:
            continue
        closest_idx = (eligible["propensity_score"] - ps).abs().idxmin()
        matched_pairs.append((idx, closest_idx))
        used_control_indices.add(closest_idx)

    if not matched_pairs:
        return {"error": "No matches found. Consider increasing caliper."}

    treated_indices = [p[0] for p in matched_pairs]
    control_indices = [p[1] for p in matched_pairs]
    matched_treated = df.loc[treated_indices, outcome_col]
    matched_control = df.loc[control_indices, outcome_col]

    ate = matched_treated.mean() - matched_control.mean()
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(matched_treated, matched_control)
    relative_lift = ate / matched_control.mean() * 100 if matched_control.mean() != 0 else 0

    return {
        "method": "Propensity Score Matching",
        "matched_pairs": len(matched_pairs),
        "unmatched_treated": len(treated) - len(matched_pairs),
        "treated_outcome_mean": round(matched_treated.mean(), 4),
        "control_outcome_mean": round(matched_control.mean(), 4),
        "average_treatment_effect": round(ate, 4),
        "relative_lift_pct": round(relative_lift, 2),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "interpretation": (
            f"Promotion TRULY increased {outcome_col} by {relative_lift:.1f}% "
            f"(controlling for {', '.join(confounder_cols[:3])})"
            if p_value < 0.05 else
            f"No significant true lift detected after controlling for confounders"
        ),
    }


def naive_vs_causal_comparison(df: pd.DataFrame, treatment_col: str,
                                outcome_col: str, confounder_cols: list) -> dict:
    treated_mean = df[df[treatment_col] == 1][outcome_col].mean()
    control_mean = df[df[treatment_col] == 0][outcome_col].mean()
    naive_lift = (treated_mean - control_mean) / control_mean * 100

    causal_result = propensity_score_matching(df, treatment_col, outcome_col, confounder_cols)
    causal_lift = causal_result.get("relative_lift_pct", naive_lift)

    return {
        "naive_lift_pct": round(naive_lift, 2),
        "causal_lift_pct": round(causal_lift, 2),
        "overestimation_pct": round(naive_lift - causal_lift, 2),
        "conclusion": (
            f"Naive analysis overestimated lift by {naive_lift - causal_lift:.1f}%. "
            f"True lift is {causal_lift:.1f}% (not {naive_lift:.1f}%)."
        ),
        "causal_details": causal_result,
    }

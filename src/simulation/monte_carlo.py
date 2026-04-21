"""
Monte Carlo simulation to find optimal discount depth that maximizes profit (not just revenue).
Simulates 10,000 pricing scenarios.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

np.random.seed(42)


def price_elasticity_demand(base_demand: float, base_price: float, new_price: float,
                              elasticity: float = -1.5) -> float:
    pct_change_price = (new_price - base_price) / base_price
    pct_change_demand = elasticity * pct_change_price
    return base_demand * (1 + pct_change_demand)


def simulate_promotion(
    base_price: float,
    base_demand: float,
    cost_per_unit: float,
    discount_range: tuple = (0.0, 0.6),
    n_simulations: int = 10000,
    elasticity_mean: float = -1.8,
    elasticity_std: float = 0.4,
) -> pd.DataFrame:
    results = []
    discounts = np.linspace(discount_range[0], discount_range[1], 61)

    for discount in discounts:
        discount_profits = []
        discount_revenues = []
        discount_units = []

        for _ in range(n_simulations):
            elasticity = np.random.normal(elasticity_mean, elasticity_std)
            demand_noise = np.random.lognormal(0, 0.15)
            seasonal_factor = np.random.uniform(0.8, 1.2)

            promo_price = base_price * (1 - discount)
            demand = price_elasticity_demand(base_demand, base_price, promo_price, elasticity)
            demand = demand * demand_noise * seasonal_factor
            demand = max(0, demand)

            revenue = promo_price * demand
            cost = cost_per_unit * demand
            profit = revenue - cost

            discount_profits.append(profit)
            discount_revenues.append(revenue)
            discount_units.append(demand)

        results.append({
            "discount_pct": round(discount * 100, 1),
            "promo_price": round(base_price * (1 - discount), 2),
            "avg_profit": round(np.mean(discount_profits), 2),
            "avg_revenue": round(np.mean(discount_revenues), 2),
            "avg_units": round(np.mean(discount_units), 1),
            "profit_p5": round(np.percentile(discount_profits, 5), 2),
            "profit_p95": round(np.percentile(discount_profits, 95), 2),
            "prob_profit_positive": round((np.array(discount_profits) > 0).mean(), 3),
        })

    return pd.DataFrame(results)


def find_optimal_discount(results: pd.DataFrame) -> dict:
    max_profit_row = results.loc[results["avg_profit"].idxmax()]
    max_revenue_row = results.loc[results["avg_revenue"].idxmax()]
    base_row = results.loc[results["discount_pct"] == 0].iloc[0]

    return {
        "optimal_discount_for_profit": f"{max_profit_row['discount_pct']:.0f}%",
        "optimal_discount_for_revenue": f"{max_revenue_row['discount_pct']:.0f}%",
        "max_profit": f"₦{max_profit_row['avg_profit']:,.0f}",
        "max_revenue": f"₦{max_revenue_row['avg_revenue']:,.0f}",
        "baseline_profit": f"₦{base_row['avg_profit']:,.0f}",
        "profit_lift": f"{(max_profit_row['avg_profit'] / base_row['avg_profit'] - 1) * 100:.1f}%",
        "recommendation": (
            f"Set discount at {max_profit_row['discount_pct']:.0f}% to maximize profit. "
            f"Higher discounts increase revenue but reduce margin due to cost structure. "
            f"Avoid discounts >{'%.0f' % (max_profit_row['discount_pct'] + 10)}% — diminishing returns."
        ),
    }


def plot_discount_optimization(results: pd.DataFrame, save_path: str = "data/discount_optimization.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(results["discount_pct"], results["avg_profit"] / 1000, color="#2563eb", lw=2, label="Avg Profit")
    ax1.fill_between(results["discount_pct"],
                      results["profit_p5"] / 1000,
                      results["profit_p95"] / 1000,
                      alpha=0.2, color="#2563eb", label="5th-95th percentile")
    optimal_idx = results["avg_profit"].idxmax()
    ax1.axvline(x=results.loc[optimal_idx, "discount_pct"], color="red", linestyle="--",
                label=f"Optimal: {results.loc[optimal_idx, 'discount_pct']:.0f}%")
    ax1.set_xlabel("Discount %")
    ax1.set_ylabel("Profit (₦ thousands)")
    ax1.set_title("Profit vs Discount Depth\n(10,000 Monte Carlo simulations)", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(results["discount_pct"], results["avg_revenue"] / 1000, color="#16a34a", lw=2, label="Revenue")
    ax2.plot(results["discount_pct"], results["avg_profit"] / 1000, color="#2563eb", lw=2, label="Profit")
    ax2.set_xlabel("Discount %")
    ax2.set_ylabel("₦ thousands")
    ax2.set_title("Revenue vs Profit Trade-off", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    results = simulate_promotion(
        base_price=15000,
        base_demand=1000,
        cost_per_unit=8000,
        discount_range=(0.0, 0.6),
        n_simulations=5000,
    )

    optimal = find_optimal_discount(results)
    print("\n=== MONTE CARLO PRICING OPTIMIZATION ===")
    for k, v in optimal.items():
        print(f"  {k}: {v}")

    plot_discount_optimization(results, "data/discount_optimization.png")
    print("\nTop 10 discount scenarios by profit:")
    print(results.nlargest(10, "avg_profit")[
        ["discount_pct", "avg_profit", "avg_revenue", "avg_units", "prob_profit_positive"]
    ].to_string(index=False))

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nvda_rl.agents import QLearningAgent
from nvda_rl.env import TradingEnvironment
from nvda_rl.evaluation import buy_and_hold_frame, random_policy_actions, strategy_frame
from nvda_rl.features import describe_regimes, discretize_state_features


FIGURE_DIR = Path("reports/figures")


def save_price_drawdown(data: pd.DataFrame) -> None:
    """
    Save a price and drawdown chart for the README.

    The chart highlights both NVDA's long-term growth and the large losses from
    prior peaks that occurred along the way.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(data["date"], data["adj_close"], color="tab:green")
    axes[0].set_title("NVDA adjusted close")
    axes[0].set_ylabel("Price")
    axes[1].fill_between(data["date"], data["drawdown"], 0, color="tab:red", alpha=0.35)
    axes[1].set_title("Drawdown from running high")
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "readme_price_drawdown.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(data: pd.DataFrame) -> None:
    """
    Save a feature correlation heatmap for the README.

    The heatmap shows which engineered features overlap and which capture
    different information.
    """
    corr_cols = [
        "return",
        "volatility_20d",
        "momentum_5d",
        "momentum_20d",
        "volume_z_20d",
        "drawdown",
        "rsi_14",
        "macd_hist",
        "sma_20_gap",
        "sma_50_gap",
        "atr_14_pct",
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(data[corr_cols].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "readme_correlations.png", dpi=160)
    plt.close(fig)


def save_regime_chart(regimes: pd.DataFrame) -> None:
    """
    Save a price chart colored by k-means regime.

    This chart makes the unsupervised clusters visible across the price history.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(regimes["date"], regimes["adj_close"], color="black", linewidth=1, alpha=0.7)
    scatter = ax.scatter(
        regimes["date"],
        regimes["adj_close"],
        c=regimes["regime"],
        cmap="Set2",
        s=11,
        alpha=0.85,
    )
    ax.set_title("NVDA price colored by unsupervised regime")
    ax.set_ylabel("Adjusted close")
    fig.colorbar(scatter, ax=ax, label="Regime")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "readme_regimes.png", dpi=160)
    plt.close(fig)


def save_q_learning_equity(regimes: pd.DataFrame) -> None:
    """
    Save a quick Q-learning versus baseline equity chart.

    PPO timing is left inside the notebook because it is intentionally slower.
    This README chart gives a fast overview of the simpler RL baseline.
    """
    states = discretize_state_features(regimes)
    train = states[states["date"] < "2021-01-01"].reset_index(drop=True)
    test = states[states["date"] >= "2021-01-01"].reset_index(drop=True)
    state_columns = ["regime", "return_bin", "vol_bin", "momentum_bin", "rsi_bin", "macd_bin", "pca_1_bin"]
    transaction_cost = 0.001

    agent = QLearningAgent(alpha=0.08, gamma=0.95, epsilon=0.30, epsilon_decay=0.992, min_epsilon=0.03)
    agent.train(TradingEnvironment(train, state_columns, transaction_cost), episodes=500)
    q_actions = agent.predict(TradingEnvironment(test, state_columns, transaction_cost))
    random_actions = random_policy_actions(len(test), random_state=7)

    q_results = strategy_frame(test, q_actions, transaction_cost, "q_learning")
    random_results = strategy_frame(test, random_actions, transaction_cost, "random")
    hold_results = buy_and_hold_frame(test)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(q_results["date"], q_results["q_learning_equity"], label="Q-learning")
    ax.plot(random_results["date"], random_results["random_equity"], label="Random", alpha=0.75)
    ax.plot(hold_results["date"], hold_results["buy_hold_equity"], label="Buy and hold")
    ax.set_title("Out-of-sample equity curves")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "readme_equity_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """
    Generate all README figures from saved project data.

    The function assumes `data/processed/nvda_features.csv` and
    `data/processed/nvda_regimes.csv` already exist.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv("data/processed/nvda_features.csv", parse_dates=["date"])
    regimes = pd.read_csv("data/processed/nvda_regimes.csv", parse_dates=["date"])
    save_price_drawdown(data)
    save_correlation_heatmap(data)
    save_regime_chart(regimes)
    save_q_learning_equity(regimes)
    print("Saved README figures.")
    print(describe_regimes(regimes).round(4).to_string())


if __name__ == "__main__":
    main()


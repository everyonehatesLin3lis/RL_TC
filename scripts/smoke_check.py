from __future__ import annotations

import numpy as np
import pandas as pd

from nvda_rl.agents import QLearningAgent
from nvda_rl.env import TradingEnvironment
from nvda_rl.evaluation import performance_metrics, strategy_frame
from nvda_rl.features import add_market_features, discretize_train_test, fit_unsupervised_train_test
from nvda_rl.ppo import PPO_OBSERVATION_COLUMNS, scale_train_test, timed_ppo_comparison


def main() -> None:
    """
    Run a small synthetic-data check across feature engineering, Q-learning, and PPO.

    The smoke check is intentionally short and does not validate strategy
    quality. It only confirms the project modules work together.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=180, freq="B")
    returns = rng.normal(0.001, 0.025, len(dates))
    prices = 100 * (1 + pd.Series(returns)).cumprod()
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": rng.integers(10_000_000, 100_000_000, len(dates)),
        }
    )

    featured = add_market_features(raw)
    train_enriched, test_enriched, _, _ = fit_unsupervised_train_test(featured, split_date="2020-06-01")
    train_states, test_states, _ = discretize_train_test(train_enriched, test_enriched)
    states = pd.concat([train_states, test_states], ignore_index=True)
    state_columns = ["regime", "momentum_bin", "vol_bin"]

    env = TradingEnvironment(states, state_columns=state_columns)
    agent = QLearningAgent(epsilon=0.2, epsilon_decay=0.98)
    agent.train(env, episodes=5)

    eval_env = TradingEnvironment(states, state_columns=state_columns)
    actions = agent.predict(eval_env)
    results = strategy_frame(states, actions, label="q_learning")
    metrics = performance_metrics(results["q_learning_return"], results["q_learning_equity"], results["action"])
    ppo_train, ppo_test, _ = scale_train_test(states.iloc[:90].copy(), states.iloc[90:].copy(), PPO_OBSERVATION_COLUMNS)
    ppo_metrics, _ = timed_ppo_comparison(ppo_train, ppo_test, PPO_OBSERVATION_COLUMNS, budgets_seconds=[1])
    print("Smoke check passed.")
    print(f"Rows tested: {len(states)}")
    print(f"Cumulative return: {metrics['cumulative_return']:.4f}")
    print(f"PPO smoke episodes: {int(ppo_metrics.loc[0, 'episodes'])}")


if __name__ == "__main__":
    main()

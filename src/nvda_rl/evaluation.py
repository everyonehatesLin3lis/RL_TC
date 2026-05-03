from __future__ import annotations

import numpy as np
import pandas as pd


def strategy_frame(
    data: pd.DataFrame,
    actions: np.ndarray,
    transaction_cost: float = 0.001,
    label: str = "strategy",
) -> pd.DataFrame:
    """Convert actions into net daily returns using the project reward formula."""
    result = data[["date", "adj_close", "return"]].copy().reset_index(drop=True)
    result["action"] = actions[: len(result)]
    result["position"] = result["action"].shift(1).fillna(0)
    result["trade_size"] = result["action"].diff().abs().fillna(result["action"].abs())
    result[f"{label}_return"] = result["position"] * result["return"] - transaction_cost * result["trade_size"]
    result[f"{label}_equity"] = (1 + result[f"{label}_return"]).cumprod()
    return result


def buy_and_hold_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Buy-and-hold baseline from the same starting day."""
    result = data[["date", "adj_close", "return"]].copy().reset_index(drop=True)
    result["buy_hold_return"] = result["return"].fillna(0)
    result["buy_hold_equity"] = (1 + result["buy_hold_return"]).cumprod()
    return result


def random_policy_actions(length: int, random_state: int = 42) -> np.ndarray:
    """
    Create a random long, flat, or short action sequence.

    This baseline checks whether the trained agent is doing better than
    arbitrary trading decisions.
    """
    rng = np.random.default_rng(random_state)
    return rng.choice([-1, 0, 1], size=length)


def max_drawdown(equity: pd.Series) -> float:
    """
    Calculate the worst peak-to-trough equity loss.

    Max drawdown is useful because a strategy can have high returns but still be
    hard to tolerate if it suffers very deep losses.
    """
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series, equity: pd.Series, actions: pd.Series) -> dict[str, float]:
    """Compute project review metrics for a daily strategy."""
    returns = returns.fillna(0)
    return {
        "cumulative_return": float(equity.iloc[-1] - 1),
        "average_daily_return": float(returns.mean()),
        "annualized_return": float((1 + returns.mean()) ** 252 - 1),
        "annualized_volatility": float(returns.std() * np.sqrt(252)),
        "max_drawdown": max_drawdown(equity),
        "hit_ratio": float((returns > 0).mean()),
        "turnover": float(actions.diff().abs().fillna(actions.abs()).mean()),
    }

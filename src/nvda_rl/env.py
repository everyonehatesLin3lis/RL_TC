from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


ACTIONS = np.array([-1, 0, 1], dtype=int)


class TradingEnvironment:
    """Daily NVDA trading environment with short, flat, and long actions."""

    def __init__(
        self,
        data: pd.DataFrame,
        state_columns: list[str],
        transaction_cost: float = 0.001,
    ) -> None:
        """
        Store market data and configuration for tabular RL.

        The environment starts flat and moves forward one trading day at a time.
        The state is a tuple of already-discretized feature values plus the
        previous action, so transaction-cost effects are observable.
        """
        self.data = data.reset_index(drop=True).copy()
        self.state_columns = state_columns
        self.transaction_cost = transaction_cost
        self.index = 1
        self.previous_action = 0

    def reset(self) -> tuple[int, ...]:
        """
        Reset the simulation to the first tradable day.

        Returns
        -------
        tuple[int, ...]
            Discrete market state used by the Q-learning table.
        """
        self.index = 1
        self.previous_action = 0
        return self._state()

    def _state(self) -> tuple[int, ...]:
        """
        Read the current row's discrete state values.

        Returns
        -------
        tuple[int, ...]
            Current state represented as integer bins.
        """
        feature_state = tuple(int(self.data.loc[self.index, col]) for col in self.state_columns)
        return (*feature_state, int(self.previous_action))

    def step(self, action: int) -> tuple[tuple[int, ...] | None, float, bool]:
        """
        Apply one action and move to the next trading day.

        Timeline: observe end-of-day features, choose the next position, then
        earn the next daily return from the previous position. Reward is
        previous exposure times today's return, minus transaction cost for
        changing from the previous action to the new action.
        """
        if action not in ACTIONS:
            raise ValueError(f"Action must be one of {ACTIONS.tolist()}.")

        daily_return = float(self.data.loc[self.index, "return"])
        reward = self.previous_action * daily_return - self.transaction_cost * abs(action - self.previous_action)
        self.previous_action = int(action)
        self.index += 1
        done = self.index >= len(self.data)
        next_state = None if done else self._state()
        return next_state, reward, done


class GymTradingEnvironment(gym.Env):
    """Gymnasium environment used by PPO and other policy-gradient agents."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        observation_columns: list[str],
        transaction_cost: float = 0.001,
    ) -> None:
        """
        Build a continuous-observation trading environment.

        PPO can consume numeric vectors directly, so this environment exposes
        scaled feature columns and maps actions `0, 1, 2` to short, flat, long.
        """
        super().__init__()
        self.data = data.reset_index(drop=True).copy()
        self.observation_columns = observation_columns
        self.transaction_cost = transaction_cost
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(observation_columns) + 1,),
            dtype=np.float32,
        )
        self.index = 1
        self.previous_action = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset the PPO environment.

        Returns
        -------
        tuple[np.ndarray, dict]
            Continuous observation and empty info dictionary.
        """
        super().reset(seed=seed)
        self.index = 1
        self.previous_action = 0
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        """
        Create the current continuous observation vector.

        The previous action is included so PPO can learn the cost of changing
        positions instead of treating each day independently.
        """
        values = self.data.loc[self.index, self.observation_columns].to_numpy(dtype=np.float32)
        return np.append(values, np.float32(self.previous_action))

    def step(self, action: int):
        """
        Execute one PPO action and return Gymnasium step values.

        Actions are mapped from `0, 1, 2` to `-1, 0, 1`, then the same reward
        formula used by Q-learning is applied.
        """
        mapped_action = int(action) - 1
        daily_return = float(self.data.loc[self.index, "return"])
        reward = self.previous_action * daily_return - self.transaction_cost * abs(
            mapped_action - self.previous_action
        )
        self.previous_action = mapped_action
        self.index += 1
        terminated = self.index >= len(self.data)
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._observation()
        info = {"position": mapped_action, "date": self.data.loc[self.index - 1, "date"]}
        return observation, float(reward), terminated, False, info

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from nvda_rl.evaluation import performance_metrics, strategy_frame


PPO_OBSERVATION_COLUMNS = [
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
    "regime",
    "pca_1",
    "pca_2",
]


def scale_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale PPO observation columns using only the training period.

    Scaling prevents large-value features from dominating the policy. The raw
    `return` column is preserved because the reward formula needs true returns.
    """
    scaler = StandardScaler()
    train_scaled = train.reset_index(drop=True).copy()
    test_scaled = test.reset_index(drop=True).copy()
    scaled_columns = [column for column in columns if column != "return"]
    train_scaled[scaled_columns] = scaler.fit_transform(train[scaled_columns])
    test_scaled[scaled_columns] = scaler.transform(test[scaled_columns])
    return train_scaled, test_scaled, scaler


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert raw action scores into probabilities.

    A small numerical stabilization step subtracts the largest logit before
    exponentiation.
    """
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


class LinearPPOAgent:
    """Small PPO-style clipped policy agent implemented with NumPy."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        clip_epsilon: float = 0.2,
        gamma: float = 0.95,
        random_state: int = 42,
    ) -> None:
        """
        Initialize a linear softmax policy.

        The policy maps market features to probabilities for short, flat, and
        long actions. PPO clipping limits very large policy updates.
        """
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng(random_state)
        self.weights = self.rng.normal(0, 0.01, size=(n_features + 1, 3))

    def _features(self, row: pd.Series, columns: list[str], previous_action: int) -> np.ndarray:
        """
        Build one policy input vector.

        The previous action is included because transaction cost depends on how
        much the position changes.
        """
        values = row[columns].to_numpy(dtype=float)
        return np.append(values, previous_action)

    def action_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Compute action probabilities for one feature vector.

        Returns probabilities for actions `-1`, `0`, and `1` in that order.
        """
        return _softmax(features @ self.weights)

    def sample_action(self, features: np.ndarray) -> tuple[int, float]:
        """
        Sample an action from the current policy.

        Returns the mapped trading action and the probability used for PPO's
        old-policy ratio.
        """
        probabilities = self.action_probabilities(features)
        action_index = int(self.rng.choice(3, p=probabilities))
        return action_index - 1, float(probabilities[action_index])

    def predict_action(self, features: np.ndarray) -> int:
        """
        Choose the most likely action without exploration.

        This is used for out-of-sample evaluation after training.
        """
        return int(np.argmax(self.action_probabilities(features)) - 1)

    def _discounted_returns(self, rewards: list[float]) -> np.ndarray:
        """
        Convert daily rewards into discounted future returns.

        Discounting gives more weight to near-term outcomes while still caring
        about the rest of the episode.
        """
        values = np.zeros(len(rewards), dtype=float)
        running = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running = rewards[idx] + self.gamma * running
            values[idx] = running
        if values.std() > 1e-8:
            values = (values - values.mean()) / values.std()
        return values

    def train_episode(
        self,
        data: pd.DataFrame,
        observation_columns: list[str],
        transaction_cost: float,
        epochs: int = 3,
    ) -> float:
        """
        Run one historical episode and update the policy with PPO clipping.

        The episode collects states, actions, old action probabilities, and
        rewards. The update then nudges the policy toward actions with positive
        advantage while clipping overly large probability changes.
        """
        features_history = []
        action_indices = []
        old_probabilities = []
        rewards = []
        previous_action = 0

        for idx in range(1, len(data)):
            features = self._features(data.loc[idx], observation_columns, previous_action)
            action, old_probability = self.sample_action(features)
            daily_return = float(data.loc[idx, "return"])
            reward = previous_action * daily_return - transaction_cost * abs(action - previous_action)
            features_history.append(features)
            action_indices.append(action + 1)
            old_probabilities.append(old_probability)
            rewards.append(reward)
            previous_action = action

        advantages = self._discounted_returns(rewards)
        for _ in range(epochs):
            for features, action_index, old_probability, advantage in zip(
                features_history,
                action_indices,
                old_probabilities,
                advantages,
            ):
                probabilities = self.action_probabilities(features)
                ratio = probabilities[action_index] / max(old_probability, 1e-8)
                if advantage >= 0 and ratio > 1 + self.clip_epsilon:
                    continue
                if advantage < 0 and ratio < 1 - self.clip_epsilon:
                    continue

                gradient = -probabilities
                gradient[action_index] += 1
                self.weights += self.learning_rate * advantage * np.outer(features, gradient)

        return float(np.sum(rewards))

    def save(self, path: str | Path) -> None:
        """
        Save the trained PPO-style policy to disk.

        The saved file stores hyperparameters and learned linear policy weights
        so evaluation can reload the model without retraining.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "gamma": self.gamma,
            "weights": self.weights,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "LinearPPOAgent":
        """
        Load a saved PPO-style policy from disk.

        Returns
        -------
        LinearPPOAgent
            Policy with restored weights ready for deterministic testing.
        """
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        n_features = payload["weights"].shape[0] - 1
        model = cls(
            n_features=n_features,
            learning_rate=payload["learning_rate"],
            clip_epsilon=payload["clip_epsilon"],
            gamma=payload["gamma"],
        )
        model.weights = payload["weights"]
        return model


def train_ppo_for_seconds(
    train: pd.DataFrame,
    observation_columns: list[str],
    seconds: int,
    transaction_cost: float = 0.001,
    seed: int = 42,
) -> tuple[LinearPPOAgent, int, float, list[float]]:
    """
    Train the lightweight PPO agent for approximately a wall-clock budget.

    Returns the model, number of completed episodes, elapsed seconds, and the
    reward history so training progress can be inspected.
    """
    model = LinearPPOAgent(n_features=len(observation_columns), random_state=seed)
    start = time.perf_counter()
    episodes = 0
    rewards = []
    while time.perf_counter() - start < seconds:
        reward = model.train_episode(train, observation_columns, transaction_cost=transaction_cost)
        rewards.append(reward)
        episodes += 1
    elapsed = time.perf_counter() - start
    return model, episodes, elapsed, rewards


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    """
    Save the fitted PPO feature scaler.

    The scaler is part of the model pipeline because test data must be
    transformed with the same training-period means and standard deviations.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(scaler, file)


def load_scaler(path: str | Path) -> StandardScaler:
    """
    Load a saved PPO feature scaler.

    Returns
    -------
    StandardScaler
        Fitted scaler created from the training period.
    """
    with Path(path).open("rb") as file:
        return pickle.load(file)


def predict_ppo_actions(
    model: LinearPPOAgent,
    test: pd.DataFrame,
    observation_columns: list[str],
    transaction_cost: float = 0.001,
) -> np.ndarray:
    """
    Generate deterministic actions from the trained PPO-style policy.

    The returned actions follow the project convention: -1 short, 0 flat, and
    1 long.
    """
    del transaction_cost
    actions = [0]
    previous_action = 0
    for idx in range(1, len(test)):
        features = model._features(test.loc[idx], observation_columns, previous_action)
        action = model.predict_action(features)
        actions.append(action)
        previous_action = action
    return np.array(actions[: len(test)], dtype=int)


def timed_ppo_comparison(
    train: pd.DataFrame,
    test: pd.DataFrame,
    observation_columns: list[str],
    budgets_seconds: list[int],
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Train PPO for several time budgets and compare out-of-sample metrics.

    This answers whether spending more training time appears to improve the
    trading policy for the selected split and feature set.
    """
    rows = []
    actions_by_budget = {}
    for seconds in budgets_seconds:
        model, episodes, elapsed, rewards = train_ppo_for_seconds(
            train,
            observation_columns=observation_columns,
            seconds=seconds,
            transaction_cost=transaction_cost,
        )
        actions = predict_ppo_actions(model, test, observation_columns, transaction_cost=transaction_cost)
        actions_by_budget[seconds] = actions
        results = strategy_frame(test, actions, transaction_cost=transaction_cost, label="ppo")
        metrics = performance_metrics(results["ppo_return"], results["ppo_equity"], results["action"])
        rows.append(
            {
                "seconds": seconds,
                "elapsed_seconds": elapsed,
                "episodes": episodes,
                "last_training_reward": rewards[-1] if rewards else np.nan,
                **metrics,
            }
        )
    return pd.DataFrame(rows), actions_by_budget

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from nvda_rl.env import ACTIONS, TradingEnvironment


class QLearningAgent:
    """Small tabular Q-learning agent for discrete market states."""

    def __init__(
        self,
        alpha: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.03,
        random_state: int = 42,
    ) -> None:
        """
        Initialize Q-learning hyperparameters and an empty Q-table.

        Alpha controls update speed, gamma controls how much future reward
        matters, and epsilon controls random exploration during training.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(random_state)
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def choose_action(self, state: tuple[int, ...], explore: bool = True) -> int:
        """
        Choose an action using epsilon-greedy behavior.

        During training the agent sometimes explores random actions. During
        prediction it chooses the action with the highest learned Q-value.
        """
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(ACTIONS))
        return int(ACTIONS[np.argmax(self.q_table[state])])

    def train(self, env: TradingEnvironment, episodes: int = 300) -> list[float]:
        """
        Train the Q-table by replaying the historical period many times.

        Each episode walks through the training data once. The table is updated
        after each step using the Bellman equation.
        """
        episode_rewards = []
        for _ in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.choose_action(state, explore=True)
                next_state, reward, done = env.step(action)
                action_idx = int(np.where(ACTIONS == action)[0][0])
                next_best = 0.0 if next_state is None else float(np.max(self.q_table[next_state]))
                target = reward + self.gamma * next_best
                self.q_table[state][action_idx] += self.alpha * (target - self.q_table[state][action_idx])
                state = next_state if next_state is not None else state
                total_reward += reward

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            episode_rewards.append(total_reward)
        return episode_rewards

    def predict(self, env: TradingEnvironment) -> np.ndarray:
        """
        Generate actions from the learned Q-table without random exploration.

        Returns
        -------
        np.ndarray
            Array of daily actions where -1 is short, 0 is flat, and 1 is long.
        """
        state = env.reset()
        done = False
        actions = [0]
        while not done:
            action = self.choose_action(state, explore=False)
            next_state, _, done = env.step(action)
            actions.append(action)
            if next_state is not None:
                state = next_state
        return np.array(actions[: len(env.data)], dtype=int)

    def save(self, path: str | Path) -> None:
        """
        Save the trained Q-learning agent to disk.

        The saved file includes hyperparameters and the learned Q-table so the
        model can be reloaded later for test-period evaluation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_table": dict(self.q_table),
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """
        Load a saved Q-learning agent from disk.

        Returns
        -------
        QLearningAgent
            Agent with the same learned Q-table that was saved after training.
        """
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            min_epsilon=payload["min_epsilon"],
        )
        agent.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)), payload["q_table"])
        return agent

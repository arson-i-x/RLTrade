# envs/trading_env.py
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        csv_path: str,
        feature_columns: Optional[List[str]] = None,  # if None, use all numeric columns
        window_size: int = 20,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.0005,
        slippage: float = 0.0005,
        normalize: bool = True,
        mode: str ="daily"
    ):
        super().__init__()

        self.mode = mode

        # Load CSV
        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")

        # Select features
        if feature_columns is None:
            self.feature_columns = df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Keep only columns that exist in the CSV
            self.feature_columns = [c for c in feature_columns if c in df.columns]

        df = df.dropna(subset=self.feature_columns)

        # Feature array
        X = df[self.feature_columns].values.astype(np.float32)

        if normalize:
            means, stds = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            X = (X - means) / stds

        self.X = X
        self.T, self.n_features = X.shape
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Action and observation spaces
        # self.action_space = spaces.Discrete(4)  # hold, long, short, close
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.n_features), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_position = 0.0
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.positions = []
        obs = self._get_observation()
        info = {"portfolio_value": self.balance, "open_positions": 0}
        return obs, info

    def step(self, action):
        price = self.X[self.current_step - 1, 0]
        reward = 0.0

        # Execute action
        a = float(action[0])  # convert tensor -> float

        # position signal ∈ [-1, 1]
        # +1 = long, -1 = short, 0 = neutral
        target_position = a

        # Compute price difference (reward proxy)
        price = self.X[self.current_step, 0]
        price_prev = self.X[self.current_step - 1, 0]
        price_change = (price - price_prev) / price_prev

        # Reward = profit proportional to position and price change
        reward = target_position * price_change

        # Apply transaction cost when position changes significantly
        if abs(target_position - getattr(self, "last_position", 0.0)) > 0.1:
            reward -= self.transaction_cost * abs(target_position - self.last_position)

        self.last_position = target_position

        # Update balance
        self.balance += reward

        # Compute portfolio value
        portfolio_value = self.balance
        for pos_type, entry_price in self.positions:
            if pos_type == "long":
                portfolio_value += (self.X[self.current_step, 0] - entry_price)
            elif pos_type == "short":
                portfolio_value += (entry_price - self.X[self.current_step, 0])

        info = {"portfolio_value": portfolio_value, "open_positions": len(self.positions)}

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= self.T
        truncated = False
        obs = self._get_observation()

        # Pass to reward function (overridable)
        shaped_reward = self.compute_reward(reward)
        return obs, shaped_reward, terminated, truncated, info

    def compute_reward(self, reward):
        if self.mode == "daily":
            # smoother reward, focus on long-term growth
            return reward - 0.001 * abs(self.last_position)
        elif self.mode == "intraday":
            # more reactive, short-term
            return reward - 0.0005 * abs(self.last_position)
        else:
            return reward


    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        return self.X[start:end]
    


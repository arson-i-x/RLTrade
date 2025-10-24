# validate_agent.py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.trading_env import TradingEnv


# -------------------------------
# SETTINGS
# -------------------------------
TRAINED_MODEL_PATH = "models/ppo_trader"
WINDOW_SIZE = 20
NORMALIZE = True
INITIAL_BALANCE = 10_000.0  # fallback if env doesn't provide

# -------------------------------
# HELPERS
# -------------------------------
def make_env(env_type, type):
    if env_type == "daily":
        if type == "test":
            return TradingEnv(
                csv_path="data/processed/dtest.csv",
                window_size=20,
                normalize=True,
                initial_balance=10000.0,
                transaction_cost=0.0005,
                slippage=0.0005,
                mode=env_type
            )
        else:
            return TradingEnv(
                csv_path="data/processed/dval.csv",
                window_size=20,
                normalize=True,
                initial_balance=10000.0,
                transaction_cost=0.0005,
                slippage=0.0005,
                mode=env_type
            )
    elif env_type == "intraday":
        if type == "test":
            return TradingEnv(
                csv_path="data/processed/itest.csv",
                window_size=20,
                normalize=True,
                initial_balance=10000.0,
                transaction_cost=0.0005,
                slippage=0.0005,
                mode=env_type
            )
        else:
            return TradingEnv(
                csv_path="data/processed/ival.csv",
                window_size=20,
                normalize=True,
                initial_balance=10000.0,
                transaction_cost=0.0005,
                slippage=0.0005,
                mode=env_type
            )
    else:
        raise ValueError("Invalid env type: choose 'daily' or 'intraday'")


def sharpe_ratio(returns, risk_free=0.0):
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)

def max_drawdown(portfolio_values):
    if len(portfolio_values) == 0:
        return 0.0
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()

if __name__ == "__main__":
    # -------------------------------
    # PARSE ARGS
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Choose daily or intraday")
    parser.add_argument("--type", type=str, required=True, help="Is this a real test")
    args = parser.parse_args()

    # -------------------------------
    # INITIALIZE ENV + MODEL
    # -------------------------------
    raw_env = make_env(args.env, args.type)
    initial_value = getattr(raw_env, "initial_balance", INITIAL_BALANCE)
    env = DummyVecEnv([lambda: raw_env])
    model = PPO.load(TRAINED_MODEL_PATH, env=env, device="cpu")

    # -------------------------------
    # BACKTEST LOOP
    # -------------------------------
    obs = env.reset()
    done = [False]

    portfolio_values = [initial_value]
    actions_taken = []
    rewards_list = []

    while not all(done):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)

        rewards_list.append(reward[0])
        actions_taken.append(action[0])

        # portfolio value from env info (fallback to last value)
        portfolio_value = infos[0].get("portfolio_value", portfolio_values[-1])
        portfolio_values.append(portfolio_value)

    # -------------------------------
    # METRICS
    # -------------------------------
    total_return = portfolio_values[-1] - portfolio_values[0]
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sr = sharpe_ratio(daily_returns)
    mdd = max_drawdown(portfolio_values)

    print(f"Total return: {total_return:.2f}")
    print(f"Sharpe ratio: {sr:.4f}")
    print(f"Max drawdown: {mdd:.4%}")

    # -------------------------------
    # PLOTS
    # -------------------------------
    plt.figure(figsize=(12,5))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.title("PPO Agent Backtest")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,3))
    plt.plot(actions_taken, marker='o', linestyle='', markersize=2)
    plt.xlabel("Timestep")
    plt.ylabel("Action (0=hold,1=long,2=short,3=close)")
    plt.title("Actions Taken by PPO Agent")
    plt.grid(True)
    plt.show()

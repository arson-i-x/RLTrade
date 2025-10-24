from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.trading_env import TradingEnv
from stable_baselines3 import SAC
import os
import argparse

N_ENVS = 16
TIMESTEPS_PER_RETRAIN = 200_000

def make_env(env_type):
    if env_type == "daily":
        return TradingEnv(
            csv_path="data/processed/dtrain.csv",
            window_size=20,
            normalize=True,
            initial_balance=10000.0,
            transaction_cost=0.0005,
            slippage=0.0005,
            mode=env_type
        )
    elif env_type == "intraday":
        return TradingEnv(
            csv_path="data/processed/itrain.csv",
            window_size=20,
            normalize=True,
            initial_balance=10000.0,
            transaction_cost=0.0005,
            slippage=0.0005,
            mode=env_type
        )
    else:
        raise ValueError("Invalid env type: choose 'daily' or 'intraday'")
 

if __name__ == "__main__":
    # -------------------------------
    # PARSE ARGS
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", type=int, required=True, help="Number of retrains")
    parser.add_argument("--env", type=str, required=True, help="Pick daily or intraday")
    args = parser.parse_args()
    RETRAIN_TIMES = args.retrain  
    
    envs = [lambda: make_env(args.env) for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(envs)

    # Load existing model or create a new one
    model_path = "models/ppo_trader"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=vec_env, device="cuda")
        print("✅ Loaded existing model.")
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device="cuda",               # still fine to keep CUDA here
            buffer_size=200_000 if args.env == "daily" else 50_000,         # replay buffer (off-policy)
            batch_size=1024,
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            train_freq=64,               # how often to update
            gradient_steps=64,           # how many updates per train step
            tau=0.005,                   # target smoothing coefficient
        )
        print("⚡ Created new SAC model.")

        # model = PPO(
        #     "MlpPolicy",
        #     vec_env,
        #     verbose=1,
        #     device="cuda",
        #     n_steps=4096,
        #     batch_size=1024,
        #     policy_kwargs=dict(net_arch=[256, 256, 256])
        # )
        # print("⚡ Created new model.")

    # Retrain loop
    for i in range(RETRAIN_TIMES):
        print(f"\n🔁 Retrain iteration {i+1}/{RETRAIN_TIMES}")
        model.learn(total_timesteps=TIMESTEPS_PER_RETRAIN, reset_num_timesteps=False)

        # Print some metrics
        ev = model.logger.name_to_value.get("train/explained_variance", None)
        vl = model.logger.name_to_value.get("train/value_loss", None)
        pl = model.logger.name_to_value.get("train/policy_gradient_loss", None)
        print(f"Iteration {i+1}: Explained variance = {ev}, Value loss = {vl}, Policy loss = {pl}")

        # Save after each iteration
        model.save(model_path)

    print("✅ Retraining loop complete!")

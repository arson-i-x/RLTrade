import os
import shutil
import subprocess

MODEL_DIR = "models"
MODEL_NAME = "ppo_trader.zip"
TRAINED_DIR = os.path.join(MODEL_DIR, "trained")

def archive_existing_model():
    #Move existing model.zip into ./trained/ before retraining.
    src = os.path.join(MODEL_DIR, MODEL_NAME)
    dst = os.path.join(TRAINED_DIR, MODEL_NAME)

    # Ensure ./trained directory exists
    os.makedirs(TRAINED_DIR, exist_ok=True)

    if os.path.exists(src):
        # Create a unique filename based on count or timestamp
        base, ext = os.path.splitext(MODEL_NAME)
        versioned_dst = os.path.join(TRAINED_DIR, f"{base}_old_{len(os.listdir(TRAINED_DIR))}{ext}")
        shutil.move(src, versioned_dst)
        print(f"📦 Moved old model to {versioned_dst}")

def run_training(env, retrain_times):
    print(f"\n🚀 Training on {env} data for {retrain_times} retrains...")
    subprocess.run([
        "py", "-3.12", "-m", "scripts.train_agent",
        "--env", env,
        "--retrain", str(retrain_times)
    ], check=True)

def run_backtest(env, type):
    print(f"\n📊 Running {type} on {env}...")
    subprocess.run([
        "py", "-3.12", "-m", "scripts.backtest_agent",
        "--env", env,
        "--type", type
    ], check=True)

def ask_continue(prompt):
    while True:
        ans = input(f"{prompt} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        elif ans in ("n", "no"):
            return False
        else:
            print("Please answer y or n.")

if __name__ == "__main__":
    archive_existing_model()
    print("✅ Beginning Training")

    while True:
        # Step 1 – Train on daily data
        run_training("daily", retrain_times=3)

        # Step 2 – Backtest
        run_backtest("daily", "validation")

        # Step 3 – Ask to continue with real test
        if ask_continue("\nContinue to true test?"):
            run_backtest("daily", "test")
        else:
            continue

        # Step 4 – Ask to continue with intraday
        if ask_continue("\nContinue to 60-minute retraining?"):
            run_training("intraday", retrain_times=3)
            print("\n✅ Intraday training complete.")
        else:
            continue

        # Step 5 - Validate intraday training
        run_backtest("intraday", "validation")
        
        if ask_continue("\nContinue to true intraday test?"):
            run_backtest("intraday", "test")
        else:
            continue

        # Step 6 – Ask if user wants to repeat next day
        if not ask_continue("\nRepeat daily training cycle?"):
            print("\n🏁 Exiting training loop.")
            break

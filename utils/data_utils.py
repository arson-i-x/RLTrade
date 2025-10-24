import yfinance as yf
import pandas as pd
import ta
from itertools import combinations

def download_symbol(symbol, interval="5m", period="5d"):
    df = yf.download(symbol, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{symbol}_{col[0]}".replace(" ", "_") for col in df.columns]
    else:
        df.columns = [f"{symbol}_{col}".replace(" ", "_") for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def add_features(df, symbols):
    for sym in symbols:
        close_col = f"{sym}_Close"
        df[f"RSI_{sym}"] = ta.momentum.RSIIndicator(df[close_col], window=14).rsi()
        df[f"SMA20_{sym}"] = df[close_col].rolling(20).mean()
        df[f"Return_{sym}"] = df[close_col].pct_change()
    for s1, s2 in combinations(symbols, 2):
        df[f"Spread_{s1}_{s2}"] = df[f"{s1}_Close"] - df[f"{s2}_Close"]
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df

def split_save_data(df, save_prefix="data/processed", train_frac=0.7, val_frac=0.15):
    """Chronologically split data and save CSVs."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"✅ Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def get_prepared_data(symbols=None, interval="5m", period="5d", save_prefix="data/processed"):
    if symbols is None:
        symbols = ["MES=F", "MNQ=F"]
    
    dfs = [download_symbol(sym, interval, period) for sym in symbols]
    merged = pd.concat(dfs, axis=1, join="inner")
    processed = add_features(merged, symbols)
    
    train_df, val_df, test_df = split_save_data(processed, save_prefix)
    return train_df, val_df, test_df

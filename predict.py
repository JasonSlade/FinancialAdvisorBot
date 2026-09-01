# live data prediction script
# run crypto_model.py before this
# to run: python predict.py

"""
predict.py: live next day price predictor

Fetches today's live data, runs it through the trained model,
and prints a prediction.

Can run this any time without retraining.

Requires: crypto_model.py must have already been run once, so that
the *_model.pt and *_scaler.pkl files exist in this same folder.

"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import time
import yfinance as yf
import os
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# model definition - imported directly from crypto_model.py instead of kept
# as a local copy here, so the two files can never define the model
# differently. (This used to be a duplicate class definition that silently
# drifted out of sync when crypto_model.py's architecture changed - that's
# what caused the state_dict load error.)

from crypto_model import CryptoLSTM


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_7", "SMA_21", "EMA_12", "EMA_26",
    "MACD", "MACD_Signal", "MACD_Hist", "RSI",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct",
    "Volume_SMA", "Volume_Ratio",
    "Return_1d", "Return_7d", "Return_14d", "HL_Range",
]

LOOKBACK = 30   # must match what you trained with

# Check if in test mode

def is_test_mode() -> bool:
    value = os.environ.get("USE_TEST_DATA")

    if value is None:
        try:
            import streamlit as st
            value = st.secrets.get("USE_TEST_DATA", "false")
        except Exception:
            value = "false"

    return str(value).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

# Test data

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"


def load_test_data(
    symbol: str,
    days: int = 90,
) -> pd.DataFrame:
    """
    Load bundled historical data when running in hosted test mode.
    """

    filename = symbol.replace("-", "_") + ".csv"
    file_path = TEST_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test data file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True,
    )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing columns: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df.dropna(inplace=True)
    df.sort_index(inplace=True)

    # Approximate the requested number of recent rows.
    return df.tail(days)


# step 1: fetch live data
def fetch_live_data(
    symbol: str,
    days: int = 90,
) -> pd.DataFrame:
    """
    Use bundled test data in hosted test mode.

    Otherwise, try Yahoo Finance and fall back to bundled data
    if Yahoo is unavailable.
    """

    use_test_data = os.environ.get(
        "USE_TEST_DATA",
        "false",
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if use_test_data:
        return load_test_data(symbol, days)

    try:
        df = yf.download(
            tickers=symbol,
            period=f"{days}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=30,
        )

        if df is None or df.empty:
            raise RuntimeError(
                f"Yahoo returned no data for {symbol}"
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                "Yahoo response is missing columns: "
                + ", ".join(missing)
            )

        df = df[required].copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(inplace=True)

        if df.empty:
            raise RuntimeError(
                f"Yahoo returned no usable data for {symbol}"
            )

        return df

    except Exception as exc:
        print(
            f"Yahoo Finance failed for {symbol}: {exc}. "
            "Loading bundled test data instead."
        )

        return load_test_data(symbol, days)

# step 2: engineer features

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Same feature engineering as training
    def to_series(col):
        v = df[col].values
        if v.ndim > 1: v = v.flatten()
        return pd.Series(v, index=df.index, dtype=float)

    c = to_series("Close")
    h = to_series("High")
    l = to_series("Low")
    v = to_series("Volume")

    sma7  = c.rolling(7).mean()
    sma21 = c.rolling(21).mean()
    e12   = c.ewm(span=12, adjust=False).mean()
    e26   = c.ewm(span=26, adjust=False).mean()
    macd  = e12 - e26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    s20   = c.rolling(20).mean()
    std   = c.rolling(20).std()
    bbu   = s20 + 2 * std
    bbl   = s20 - 2 * std
    bbw   = bbu - bbl
    bbp   = (c - bbl) / (bbw + 1e-9)

    vsma  = v.rolling(14).mean()
    vr    = v / (vsma + 1e-9)

    out = pd.DataFrame({
        "Open": to_series("Open"), "High": h, "Low": l,
        "Close": c, "Volume": v,
        "SMA_7": sma7, "SMA_21": sma21, "EMA_12": e12, "EMA_26": e26,
        "MACD": macd, "MACD_Signal": sig, "MACD_Hist": hist, "RSI": rsi,
        "BB_Upper": bbu, "BB_Lower": bbl, "BB_Width": bbw, "BB_Pct": bbp,
        "Volume_SMA": vsma, "Volume_Ratio": vr,
        "Return_1d": c.pct_change(1), "Return_7d": c.pct_change(7),
        "Return_14d": c.pct_change(14), "HL_Range": (h - l) / (c + 1e-9),
    }, index=df.index)

    out.dropna(inplace=True)
    return out


# step 3: load saved model + scaler

def load_model(symbol: str):
    # load the .pt weights and .pkl scaler saved during training
    tag = symbol.replace("-", "_")

    model = CryptoLSTM(n_features=len(FEATURE_COLUMNS))
    model.load_state_dict(
        torch.load(f"{tag}_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    with open(f"{tag}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# step 4: predict

def predict(model, df: pd.DataFrame, scaler) -> float:
    """
    Take the most recent LOOKBACK rows, scale them using the
    training scaler, run
    through the model, then inverse-transform back to USD.
    """
    n_features = len(FEATURE_COLUMNS)
    close_idx  = FEATURE_COLUMNS.index("Close")

    # Slice the last 30 rows
    recent = df[FEATURE_COLUMNS].values[-LOOKBACK:].astype(np.float32)

    if len(recent) < LOOKBACK:
        raise ValueError(
            f"Not enough rows after feature engineering. "
            f"Got {len(recent)}, need {LOOKBACK}. "
            f"Try fetching more days (increase `days` in fetch_live_data)."
        )

    # Scale using the TRAINING scaler: never fit a new one on live data
    scaled = scaler.transform(recent)

    # Shape: (1, 30, 23) batch of 1 sequence
    X = torch.tensor(scaled).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(X).item()

    # Inverse transform: rebuild dummy array, replace close column, invert
    dummy = np.zeros((1, n_features))
    dummy[0, close_idx] = pred_scaled
    return float(scaler.inverse_transform(dummy)[0, close_idx])


# main

def run(symbols: list[str]):
    print(f"\n{'='*52}")
    print(f"  Live predictions  -  {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*52}")

    for symbol in symbols:
        print(f"\n  {symbol}")
        print(f"  {'─'*30}")

        try:
            # 1. Fetch 90 days of live data (more than enough for rolling windows)
            print("  Fetching live data from Yahoo Finance...", end=" ", flush=True)
            raw = fetch_live_data(symbol, days=90)
            print(f"{len(raw)} rows up to {raw.index[-1].date()}")

            # 2. Engineer features (identical to training)
            df = add_features(raw)

            # 3. Load saved model and scaler
            model, scaler = load_model(symbol)

            # 4. Predict
            last_close  = float(df["Close"].iloc[-1])
            next_price  = predict(model, df, scaler)
            change_pct  = (next_price - last_close) / last_close * 100
            direction   = "UP" if change_pct > 0 else "DOWN"

            # 5. Print result
            print(f"  Last close:    ${last_close:>12,.2f}")
            print(f"  Predicted:     ${next_price:>12,.2f}  ({change_pct:+.2f}%  {direction})")

            # Context from recent indicators
            rsi_now = float(df["RSI"].iloc[-1])
            rsi_msg = "overbought" if rsi_now > 70 else "oversold" if rsi_now < 30 else "neutral"
            print(f"  RSI:           {rsi_now:.1f}  ({rsi_msg})")

        except FileNotFoundError:
            print(f"  ERROR: No saved model found for {symbol}.")
            print(f"  Run crypto_model.py first to train and save the model.")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*52}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live crypto price predictor")
    parser.add_argument(
        "--coins", nargs="+",
        default=[
            "BTC-USD",
            "ETH-USD",
            "BNB-USD",
            "XRP-USD",
            "SOL-USD",
            "ADA-USD",
            "DOGE-USD",
            "TRX-USD",
            "LINK-USD",
            "AVAX-USD",
            "XLM-USD",
            "LTC-USD",
            "BCH-USD"
        ],
       help="Crypto symbols to predict"
    )
    args = parser.parse_args()
    run(args.coins)
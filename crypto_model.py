"""
Crypto ML Model for Financial Advisor Bot
==========================================
Predicts next-day closing prices for selected cryptocurrencies
using a PyTorch LSTM — compatible with Python 3.13+
"""

# to run: python crypto_model.py

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


#1. DATA COLLECTION

def fetch_crypto_data(symbols: list[str], days: int = 730) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data via yfinance for each symbol."""
    import yfinance as yf
    from datetime import datetime, timedelta

    end   = datetime.today()
    start = end - timedelta(days=days)
    data  = {}

    for sym in symbols:
        print(f"  Fetching {sym}...")
        df = yf.download(sym, start=start, end=end, progress=False)
        if df.empty:
            print(f"  WARNING: No data for {sym}, skipping.")
            continue

        # Flatten MultiIndex columns that newer yfinance versions produce
        # e.g. ('Close', 'BTC-USD') → 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Ensure no duplicate columns remain after flattening
        df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(inplace=True)
        data[sym] = df
        print(f"  {sym}: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    return data


#2. FEATURE ENGINEERING 

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Add RSI, MACD, Bollinger Bands, moving averages, and momentum features
    # Force every base column to a guaranteed plain 1-D Series
    def to_series(col):
        vals = df[col].values
        if vals.ndim > 1:
            vals = vals.flatten()
        return pd.Series(vals, index=df.index, dtype=float)

    close  = to_series("Close")
    high   = to_series("High")
    low    = to_series("Low")
    volume = to_series("Volume")

    # Compute all Series independently, never reference df[] mid-way
    sma7  = close.rolling(7).mean()
    sma21 = close.rolling(21).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd        = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist   = macd - macd_signal

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    bb_pct   = (close - bb_lower) / (bb_width + 1e-9)

    vol_sma   = volume.rolling(14).mean()
    vol_ratio = volume / (vol_sma + 1e-9)

    ret1  = close.pct_change(1)
    ret7  = close.pct_change(7)
    ret14 = close.pct_change(14)
    hl    = (high - low) / (close + 1e-9)

    # Assign all at once, every value is a plain Series, no ambiguity
    out = pd.DataFrame({
        "Open":         to_series("Open"),
        "High":         high,
        "Low":          low,
        "Close":        close,
        "Volume":       volume,
        "SMA_7":        sma7,
        "SMA_21":       sma21,
        "EMA_12":       ema12,
        "EMA_26":       ema26,
        "MACD":         macd,
        "MACD_Signal":  macd_signal,
        "MACD_Hist":    macd_hist,
        "RSI":          rsi,
        "BB_Upper":     bb_upper,
        "BB_Lower":     bb_lower,
        "BB_Width":     bb_width,
        "BB_Pct":       bb_pct,
        "Volume_SMA":   vol_sma,
        "Volume_Ratio": vol_ratio,
        "Return_1d":    ret1,
        "Return_7d":    ret7,
        "Return_14d":   ret14,
        "HL_Range":     hl,
    }, index=df.index)

    out.dropna(inplace=True)
    return out


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_7", "SMA_21", "EMA_12", "EMA_26",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI",
    "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct",
    "Volume_SMA", "Volume_Ratio",
    "Return_1d", "Return_7d", "Return_14d",
    "HL_Range",
]

# creates the 2 files needed for predict.py
def save_model(model, scaler, symbol):
    import pickle
    tag = symbol.replace("-", "_")
    torch.save(model.state_dict(), f"{tag}_model.pt")
    with open(f"{tag}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved {tag}_model.pt + {tag}_scaler.pkl")


# 3. DATA PREPARATION



def prepare_sequences(df, feature_cols, target_col="Close",
                      lookback=30, test_ratio=0.2):
    """Scale features and build sliding-window sequences."""
    data   = df[feature_cols].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    close_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, close_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, close_idx


# 4. PYTORCH LSTM MODEL 
class CryptoLSTM(nn.Module):
    """
    Stacked LSTM for next-day price prediction.

    Architecture:
        LSTM(128, layers=2, dropout=0.2)
        Linear(128 → 64) + ReLU
        Linear(64  → 1)
    """
    def __init__(self, n_features: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,           # input shape: (batch, seq, features)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # out: (batch, seq, hidden)
        return self.fc(out[:, -1, :])   # use last timestep → (batch, 1)


def build_model(n_features: int) -> CryptoLSTM:
    return CryptoLSTM(n_features=n_features)


# 5. TRAINING 

def train_model(model: CryptoLSTM,
                X_train, y_train, X_test, y_test,
                epochs=50, batch_size=32, lr=1e-3, patience=10):
    """
    Train with Adam + MSE loss.
    Implements manual early stopping (patience epochs without val improvement).
    Returns list of (train_loss, val_loss) per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")
    model.to(device)

    # Build DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    X_val = torch.tensor(X_test).to(device)
    y_val = torch.tensor(y_test).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    wait       = 0
    history    = []

    for epoch in range(1, epochs + 1):
        # Train 
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate 
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, y_val).item()

        train_loss = np.mean(train_losses)
        scheduler.step(val_loss)
        history.append((train_loss, val_loss))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (best val={best_val:.6f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.to("cpu")
    return history


# 6. EVALUATION 

def evaluate_model(model, X_test, y_test, scaler, close_idx, n_features):
    """Return MAE, RMSE, directional accuracy (all in USD)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).squeeze().numpy()

    def inverse_close(vals):
        dummy = np.zeros((len(vals), n_features))
        dummy[:, close_idx] = vals
        return scaler.inverse_transform(dummy)[:, close_idx]

    y_pred = inverse_close(preds)
    y_true = inverse_close(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    dir_acc  = np.mean(dir_true == dir_pred) * 100

    return {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "Directional_Accuracy_%": round(dir_acc, 2),
        "y_pred": y_pred,
        "y_true": y_true,
    }


# 7. NEXT-DAY PREDICTION

def predict_next_day(model, df, feature_cols, scaler, close_idx, lookback=30):
    """Predict tomorrow's closing price from the most recent `lookback` days."""
    n_features = len(feature_cols)
    recent     = df[feature_cols].values[-lookback:].astype(np.float32)
    scaled     = scaler.transform(recent)
    X          = torch.tensor(scaled).unsqueeze(0)   # (1, lookback, features)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X).item()

    dummy = np.zeros((1, n_features))
    dummy[0, close_idx] = pred_scaled
    return float(scaler.inverse_transform(dummy)[0, close_idx])



def load_model(symbol, n_features):
    import pickle
    tag   = symbol.replace("-", "_")
    model = CryptoLSTM(n_features=n_features)
    model.load_state_dict(torch.load(f"{tag}_model.pt", map_location="cpu"))
    with open(f"{tag}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


#8. MAIN PIPELINE 

def run_pipeline(symbols, lookback=30, days=730,
                 epochs=50, batch_size=32, test_ratio=0.2):
    # Full end-to-end pipeline for all symbols.
    print("\n" + "="*60)
    print(" CRYPTO ML PIPELINE  (PyTorch)")
    print("="*60)

    print("\n[1/5] Fetching historical data...")
    all_data = fetch_crypto_data(symbols, days=days)
    results  = {}

    for symbol, df in all_data.items():
        print(f"\n{'─'*50}")
        print(f" Processing: {symbol}")
        print(f"{'─'*50}")

        print("[2/5] Engineering features...")
        df = add_technical_indicators(df)

        print("[3/5] Preparing sequences...")
        X_train, X_test, y_train, y_test, scaler, close_idx = prepare_sequences(
            df, FEATURE_COLUMNS, lookback=lookback, test_ratio=test_ratio
        )

        print("[4/5] Building & training LSTM...")
        model   = build_model(n_features=len(FEATURE_COLUMNS))
        history = train_model(model, X_train, y_train, X_test, y_test,
                              epochs=epochs, batch_size=batch_size)

        print("[5/5] Evaluating...")
        metrics = evaluate_model(model, X_test, y_test, scaler,
                                 close_idx, len(FEATURE_COLUMNS))
        print(f"  MAE:  ${metrics['MAE']:,}")
        print(f"  RMSE: ${metrics['RMSE']:,}")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy_%']}%")

        next_price = predict_next_day(model, df, FEATURE_COLUMNS,
                                      scaler, close_idx, lookback)
        last_price = float(df["Close"].iloc[-1])
        change_pct = (next_price - last_price) / last_price * 100
        print(f"\n  Last close:         ${last_price:,.2f}")
        print(f"  Predicted tomorrow: ${next_price:,.2f}  ({change_pct:+.2f}%)")
        # save model
        save_model(model, scaler, symbol)


        results[symbol] = {
            "last_price": round(last_price, 2),
            "next_price": round(next_price, 2),
            "change_pct": round(change_pct, 2),
            "MAE":        metrics["MAE"],
            "RMSE":       metrics["RMSE"],
            "dir_acc":    metrics["Directional_Accuracy_%"],
        }

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    for sym, r in results.items():
        arrow = "▲" if r["change_pct"] > 0 else "▼"
        print(f"  {sym:<12} ${r['last_price']:>12,.2f}  →  "
              f"${r['next_price']:>12,.2f}  {arrow} {abs(r['change_pct']):.2f}%"
              f"   |  Dir Acc: {r['dir_acc']}%")

    return results


# ENTRY POINT 

if __name__ == "__main__":
    COINS = ["BTC-USD", "ETH-USD", "BNB-USD"]   

    run_pipeline(
        symbols    = COINS,
        lookback   = 30,
        days       = 730,
        epochs     = 50,
        batch_size = 32,
        test_ratio = 0.2,
    )

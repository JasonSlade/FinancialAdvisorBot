"""
Crypto ML Model for Financial Advisor Bot

Pipeline for the prediction system.

Downloads raw data -> train LSTM -> predict next day closing price

to run: python crypto_model.py

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

# Fix random seed for reproducibility. Without this, LSTM weight
# initialisation and dropout draw from PyTorch's global RNG, so two runs of
# identical code produce meaningfully different trained models - this is
# what caused ETH's predicted price (+17.79% -> -0.60%) and direction head
# behaviour (always-down -> 69.5% predicted up-rate) to swing so much
# between two back-to-back runs with no code changes in between. Matches
# rl_agent.py, which already seeds for the same reason.
torch.manual_seed(42)
np.random.seed(42)


#1. DATA COLLECTION

def fetch_crypto_data(symbols: list[str], days: int = 730) -> dict[str, pd.DataFrame]:
    # Fetch OHLCV data from yfinance
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
        # e.g. ('Close', 'BTC-USD') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        # No duplicate columns remain after flattening
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
    # Scale features and build sliding-window sequences
    data      = df[feature_cols].values
    close_idx = feature_cols.index(target_col)

    # Split into train/test by raw row BEFORE fitting the scaler. Fitting on
    # the full dataset lets the scaler "see" the test period's price range in
    # advance, fitting only on the training rows keeps
    # the test period genuinely unseen, same as the train/test split already
    # used for the RL agent.
    raw_split = int(len(data) * (1 - test_ratio))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:raw_split])
    scaled = scaler.transform(data)

    X, y, direction = [], [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, close_idx])
        # 1.0 if price rose from the previous day, else 0.0 - the ground
        # truth the direction head is trained on directly
        direction.append(1.0 if scaled[i, close_idx] > scaled[i - 1, close_idx] else 0.0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)

    # Sequence j (0-indexed here) predicts raw row (lookback + j), so align
    # the sequence split to the same raw-row boundary used for the scaler
    seq_split = max(raw_split - lookback, 0)
    X_train, X_test           = X[:seq_split], X[seq_split:]
    y_train, y_test           = y[:seq_split], y[seq_split:]
    dir_train, dir_test       = direction[:seq_split], direction[seq_split:]

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, dir_train, dir_test, scaler, close_idx


# 4. PYTORCH LSTM MODEL
# refrence: https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html
class CryptoLSTM(nn.Module):
    """
    Stacked LSTM for next-day price prediction, plus a separate direction
    head trained on the actual up/down outcome.

    Predicting price alone rewards landing close to *today's* price even
    when tomorrow's direction is wrong (a price guess near the current
    price scores well on MSE regardless of direction). The direction head
    shares the same LSTM features but is trained with its own loss on the
    real up/down label, so it has no incentive to take that shortcut.

    Architecture:
        LSTM(128, layers=2, dropout=0.2)
        Linear(128 -> 64) + ReLU   (shared trunk)
        -> Linear(64 -> 1)          price head        (forward)
        -> Linear(64 -> 1)          direction head     (forward_direction, logit)
    """
    def __init__(self, n_features: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
        )
        self.price_head     = nn.Linear(64, 1)
        self.direction_head = nn.Linear(64, 1)   # logit; sigmoid -> P(price goes up)

    def _features(self, x):
        out, _ = self.lstm(x)           # out: (batch, seq, hidden)
        return self.trunk(out[:, -1, :])   # use last timestep -> (batch, 64)

    def forward(self, x):
        # Unchanged contract: returns the price prediction only, so existing
        # callers (predict.py, rl_agent.py) keep working without modification
        return self.price_head(self._features(x))

    def forward_direction(self, x):
        """Direction logit for the same input; sigmoid(.) = P(price goes up)."""
        return self.direction_head(self._features(x))


def build_model(n_features: int) -> CryptoLSTM:
    return CryptoLSTM(n_features=n_features)


# 5. TRAINING
# ADAM - https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

def train_model(model: CryptoLSTM,
                X_train, y_train, X_test, y_test,
                dir_train, dir_test,
                epochs=50, batch_size=32, lr=1e-3, patience=10,
                direction_weight=0.5):
    """
    Train with Adam. Loss combines MSE on price with BCE on direction, so
    the network is directly rewarded for calling tomorrow's direction
    right, not just for landing close to today's price.

    Args:
        direction_weight: how much the direction loss counts relative to
                          the price loss in the combined objective

    Implements manual early stopping (patience epochs without val improvement)
    Returns list of (train_loss, val_loss) per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")
    model.to(device)

    # Build DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train), torch.tensor(dir_train)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    X_val   = torch.tensor(X_test).to(device)
    y_val   = torch.tensor(y_test).to(device)
    dir_val = torch.tensor(dir_test).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6
    )
    price_criterion = nn.MSELoss()

    # Correct for class imbalance in the TRAINING labels. Without this, if
    # the training window has noticeably more up-days than down-days (a long
    # multi-year crypto history almost always does), BCE loss rewards the
    # network for just always predicting the majority class - which is
    # exactly what was happening (predicted up-day rate of 83-100% against a
    # genuinely near-50/50 test period).
    n_up   = float(np.sum(dir_train == 1))
    n_down = float(np.sum(dir_train == 0))
    pos_weight = torch.tensor([n_down / max(n_up, 1.0)]).to(device)
    print(f"  Training up-day rate: {n_up / (n_up + n_down) * 100:.1f}%  "
          f"(direction pos_weight={pos_weight.item():.3f})")
    direction_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val   = float("inf")
    best_state = None
    wait       = 0
    history    = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for xb, yb, dirb in train_dl:
            xb, yb, dirb = xb.to(device), yb.to(device), dirb.to(device)
            # reset previoud gradients
            optimizer.zero_grad()
            # make a prediction - price and direction share the same trunk
            price_pred      = model(xb).squeeze()
            direction_logit = model.forward_direction(xb).squeeze()
            # compare prediction with actual value
            price_loss     = price_criterion(price_pred, yb)
            direction_loss = direction_criterion(direction_logit, dirb)
            loss = price_loss + direction_weight * direction_loss
            # calculate how model should improve
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # update model weights
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        with torch.no_grad():
            val_price_pred      = model(X_val).squeeze()
            val_direction_logit = model.forward_direction(X_val).squeeze()
            val_price_loss      = price_criterion(val_price_pred, y_val).item()
            val_direction_loss  = direction_criterion(val_direction_logit, dir_val).item()
            val_direction_acc   = (
                (val_direction_logit > 0).float() == dir_val
            ).float().mean().item() * 100

        train_loss = np.mean(train_losses)
        # Early stopping and checkpoint selection are driven by val_price_loss
        # ALONE now, not a price+direction blend. The direction head's loss is
        # noisy/unstable while it's still mode-collapsing, and letting it into
        # the stopping criterion was causing training to halt early (13-23
        # epochs vs 30+) on a metric that had nothing to do with price
        # quality - degrading price MAE/RMSE to fix a problem it couldn't
        # actually fix. The scheduler also now only reacts to price loss.
        # val_direction_loss/val_direction_acc are still tracked and logged so
        # we can see the direction head's behavior clearly, just not acted on.
        scheduler.step(val_price_loss)
        history.append((train_loss, val_price_loss, val_direction_loss, val_direction_acc))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_price_loss={val_price_loss:.6f}  "
                  f"val_dir_loss={val_direction_loss:.6f}  val_dir_acc={val_direction_acc:.1f}%")

        # Early stopping (price loss only - see comment above)
        if val_price_loss < best_val - 1e-6:
            best_val   = val_price_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (best val_price_loss={best_val:.6f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.to("cpu")
    return history


# 6. EVALUATION

def evaluate_model(model, X_test, y_test, dir_test, scaler, close_idx, n_features):
    # Return MAE, RMSE, and two directional accuracy readings (all in USD)
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    model.eval()
    with torch.no_grad():
        preds            = model(torch.tensor(X_test)).squeeze().numpy()
        direction_logits = model.forward_direction(torch.tensor(X_test)).squeeze().numpy()

    def inverse_close(vals):
        dummy = np.zeros((len(vals), n_features))
        dummy[:, close_idx] = vals
        return scaler.inverse_transform(dummy)[:, close_idx]

    y_pred = inverse_close(preds)
    y_true = inverse_close(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Directional accuracy implied by the price guess (old method - vulnerable
    # to the "predict close to today's price" shortcut)
    dir_true = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    dir_acc  = np.mean(dir_true == dir_pred) * 100

    # Directional accuracy from the dedicated direction head, scored against
    # the real up/down labels it was actually trained on
    direction_pred = (direction_logits > 0).astype(np.float32)   # sigmoid(logit) > 0.5
    direction_acc  = np.mean(direction_pred == dir_test) * 100

    # Baseline: accuracy of always predicting the majority class in the test
    # period. If direction_acc doesn't clearly beat this, the head hasn't
    # learned real day-to-day timing - it's just riding whichever direction
    # was more common over the test window (crypto's long-term upward drift
    # makes "always predict up" a deceptively strong trivial baseline).
    true_up_rate  = float(np.mean(dir_test)) * 100
    baseline_acc  = max(true_up_rate, 100 - true_up_rate)

    # How often the model itself calls "up" - a model that just learned the
    # base rate will predict one class almost every time
    predicted_up_rate = float(np.mean(direction_pred)) * 100

    return {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "Directional_Accuracy_%":    round(dir_acc, 2),
        "Direction_Head_Accuracy_%": round(direction_acc, 2),
        "Baseline_Accuracy_%":       round(baseline_acc, 2),
        "True_Up_Rate_%":            round(true_up_rate, 2),
        "Predicted_Up_Rate_%":       round(predicted_up_rate, 2),
        "y_pred": y_pred,
        "y_true": y_true,
    }


# 7. NEXT-DAY PREDICTION

def predict_next_day(model, df, feature_cols, scaler, close_idx, lookback=30):
    """Predict tomorrow's closing price from the most recent `lookback` days."""
    n_features = len(feature_cols)
    recent     = df[feature_cols].values[-lookback:].astype(np.float32)
    scaled     = scaler.transform(recent)

    # Sanity check: the scaler was fit only on the training window (roughly
    # the earlier 80% of history). If today's live features sit outside that
    # historical range - a new all-time high, a volatility spike - the
    # scaled values land outside [0, 1] and the LSTM is extrapolating into
    # territory it never trained on. That's a likely explanation for an
    # implausible single-day jump in the prediction (informational only -
    # doesn't change the prediction, just flags when it should be trusted less).
    out_of_range_pct = float(np.mean((scaled < 0) | (scaled > 1))) * 100
    if out_of_range_pct > 0:
        print(f"  WARNING: {out_of_range_pct:.1f}% of live input values fall "
              f"outside the training scaler's [0,1] range - the model is "
              f"extrapolating beyond what it was trained on.")

    X = torch.tensor(scaled).unsqueeze(0)   # (1, lookback, features)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X).item()

    dummy = np.zeros((1, n_features))
    dummy[0, close_idx] = pred_scaled
    return float(scaler.inverse_transform(dummy)[0, close_idx])


def predict_next_day_direction(model, df, feature_cols, scaler, lookback=30):
    """Predict P(price goes up tomorrow) from the most recent `lookback` days."""
    recent = df[feature_cols].values[-lookback:].astype(np.float32)
    scaled = scaler.transform(recent)
    X      = torch.tensor(scaled).unsqueeze(0)   # (1, lookback, features)

    model.eval()
    with torch.no_grad():
        direction_logit = model.forward_direction(X).item()
    return float(torch.sigmoid(torch.tensor(direction_logit)))


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
    # Full end to end pipeline for all symbols
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
        X_train, X_test, y_train, y_test, dir_train, dir_test, scaler, close_idx = prepare_sequences(
            df, FEATURE_COLUMNS, lookback=lookback, test_ratio=test_ratio
        )

        print("[4/5] Building & training LSTM...")
        model   = build_model(n_features=len(FEATURE_COLUMNS))
        history = train_model(model, X_train, y_train, X_test, y_test,
                              dir_train, dir_test,
                              epochs=epochs, batch_size=batch_size)

        print("[5/5] Evaluating...")
        metrics = evaluate_model(model, X_test, y_test, dir_test, scaler,
                                 close_idx, len(FEATURE_COLUMNS))
        print(f"  MAE:  ${metrics['MAE']:,}")
        print(f"  RMSE: ${metrics['RMSE']:,}")
        print(f"  Directional Accuracy (from price guess): {metrics['Directional_Accuracy_%']}%")
        print(f"  Directional Accuracy (direction head):   {metrics['Direction_Head_Accuracy_%']}%")
        print(f"  Baseline (always predict majority class): {metrics['Baseline_Accuracy_%']}%")
        print(f"  Actual up-day rate in test period:        {metrics['True_Up_Rate_%']}%")
        print(f"  Model's predicted up-day rate:             {metrics['Predicted_Up_Rate_%']}%")

        next_price = predict_next_day(model, df, FEATURE_COLUMNS,
                                      scaler, close_idx, lookback)
        up_prob    = predict_next_day_direction(model, df, FEATURE_COLUMNS,
                                                scaler, lookback)
        last_price = float(df["Close"].iloc[-1])
        change_pct = (next_price - last_price) / last_price * 100
        print(f"\n  Last close:         ${last_price:,.2f}")
        print(f"  Predicted tomorrow: ${next_price:,.2f}  ({change_pct:+.2f}%)")
        print(f"  Direction head:     {up_prob*100:.1f}% chance of an up day")
        # save model
        save_model(model, scaler, symbol)


        results[symbol] = {
            "last_price": round(last_price, 2),
            "next_price": round(next_price, 2),
            "change_pct": round(change_pct, 2),
            "up_prob":    round(up_prob, 4),
            "MAE":        metrics["MAE"],
            "RMSE":       metrics["RMSE"],
            "dir_acc":    metrics["Directional_Accuracy_%"],
            "dir_head_acc": metrics["Direction_Head_Accuracy_%"],
            "baseline_acc": metrics["Baseline_Accuracy_%"],
        }

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    for sym, r in results.items():
        arrow = "▲" if r["change_pct"] > 0 else "▼"
        print(f"  {sym:<12} ${r['last_price']:>12,.2f}  →  "
              f"${r['next_price']:>12,.2f}  {arrow} {abs(r['change_pct']):.2f}%"
              f"   |  Dir Acc: {r['dir_acc']}% (price) / {r['dir_head_acc']}% (head) / {r['baseline_acc']}% (baseline)"
              f"   |  P(up): {r['up_prob']*100:.1f}%")

    return results


# ENTRY POINT

if __name__ == "__main__":
    COINS = [
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
]
    run_pipeline(
        symbols    = COINS,
        lookback   = 30,
        days       = 3000,   # ~8.2 years - long enough to include the 2018 and
                              # 2022 bear markets alongside the recent uptrend,
                              # not just the last 2 years of mostly-rising prices
        epochs     = 50,
        batch_size = 32,
        test_ratio = 0.2,
    )
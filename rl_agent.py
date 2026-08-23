"""
rl_agent.py - Deep Q-Network (DQN) Trading Agent
==================================================
Trains a reinforcement learning agent to make BUY/HOLD/SELL
decisions using the LSTM model's predictions as input.

The agent learns by simulating thousands of trading days on
historical data, getting rewarded for profitable decisions
and penalised for losses.

Architecture:
    LSTM  -> predicted next-day price (already trained)
    DQN   -> BUY / HOLD / SELL decision (trained here)

Actions:
    0 = SELL  (exit position or do nothing if not holding)
    1 = HOLD  (maintain current position)
    2 = BUY   (enter position or do nothing if already holding)

Run: python rl_agent.py
"""

import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 1. TRADING ENVIRONMENT 

class CryptoTradingEnv:
    """
    Simulates a cryptocurrency trading environment.

    The agent interacts with this environment by choosing
    BUY, HOLD, or SELL at each timestep. The environment
    returns the next state and a reward based on whether
    the decision was profitable.

    State space (9 values):
        - LSTM predicted price change (%)
        - RSI (normalised 0-1)
        - MACD signal (normalised)
        - Bollinger Band position (0-1)
        - 1-day return
        - 7-day return
        - Volume ratio (normalised)
        - Current position (0 = not holding, 1 = holding)
        - Current profit/loss on open position (%)
    """

    def __init__(self, df: pd.DataFrame, lstm_predictions: np.ndarray,
                 initial_balance: float = 10000.0, transaction_cost: float = 0.001):
        """
        Args:
            df:                  Feature-engineered DataFrame from predict.py
            lstm_predictions:    Array of next-day price predictions from LSTM
            initial_balance:     Starting portfolio value in USD
            transaction_cost:    Fee per trade (0.001 = 0.1%)
        """
        self.df                = df
        self.lstm_predictions  = lstm_predictions
        self.initial_balance   = initial_balance
        self.transaction_cost  = transaction_cost
        self.n_steps           = len(df)
        self.state_size        = 9
        self.action_size       = 3    # 0=SELL, 1=HOLD, 2=BUY

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode. Returns initial state."""
        self.current_step    = 1      # start at 1 so we can look back
        self.balance         = self.initial_balance
        self.position        = 0      # 0 = not holding, 1 = holding
        self.entry_price     = 0.0    # price we bought at
        self.total_trades    = 0
        self.profitable_trades = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Build the state vector the agent sees at each timestep.
        Combines LSTM prediction with current technical indicators
        and portfolio status.
        """
        i = self.current_step

        # LSTM predicted price change as percentage
        current_price = float(self.df["Close"].iloc[i])
        lstm_pred     = self.lstm_predictions[i]
        pred_change   = (lstm_pred - current_price) / (current_price + 1e-9)

        # Technical indicators (normalised to roughly 0-1 range)
        rsi        = float(self.df["RSI"].iloc[i]) / 100.0
        macd       = float(self.df["MACD"].iloc[i])
        macd_sig   = float(self.df["MACD_Signal"].iloc[i])
        macd_norm  = np.tanh(macd - macd_sig)          # tanh squishes to -1 to 1
        bb_pct     = float(self.df["BB_Pct"].iloc[i])
        ret_1d     = float(self.df["Return_1d"].iloc[i])
        ret_7d     = float(self.df["Return_7d"].iloc[i])
        vol_ratio  = np.tanh(float(self.df["Volume_Ratio"].iloc[i]) - 1.0)

        # Portfolio status
        position = float(self.position)
        if self.position == 1 and self.entry_price > 0:
            unrealised_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealised_pnl = 0.0

        state = np.array([
            pred_change,     # LSTM prediction signal
            rsi,             # momentum indicator
            macd_norm,       # trend crossover signal
            bb_pct,          # price position in volatility range
            ret_1d,          # recent 1-day momentum
            ret_7d,          # recent 7-day momentum
            vol_ratio,       # unusual volume signal
            position,        # are we currently holding?
            unrealised_pnl,  # current open profit/loss
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute one trading step.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY

        Returns:
            next_state: new state after action
            reward:     profit/loss from this decision
            done:       True if episode is over
        """
        current_price = float(self.df["Close"].iloc[self.current_step])
        reward        = 0.0

        # BUY 
        if action == 2:   # BUY
            if self.position == 0:
                # Enter a position - pay transaction cost
                self.position    = 1
                self.entry_price = current_price * (1 + self.transaction_cost)
                self.total_trades += 1
                reward = -self.transaction_cost   # small penalty for trading cost

        # SELL 
        elif action == 0:   # SELL
            if self.position == 1:
                # Exit position - receive profit or loss minus transaction cost
                exit_price = current_price * (1 - self.transaction_cost)
                pnl        = (exit_price - self.entry_price) / self.entry_price
                reward     = pnl
                self.balance  = self.balance * (1 + pnl)
                self.position = 0

                if pnl > 0:
                    self.profitable_trades += 1
                self.total_trades += 1

        # HOLD 
        else:   # HOLD
            if self.position == 1:
                # Reward for holding a winning position, penalise holding a loser
                next_price = float(self.df["Close"].iloc[
                    min(self.current_step + 1, self.n_steps - 1)
                ])
                price_change = (next_price - current_price) / current_price
                reward = price_change * 0.5   # partial reward for unrealised gains

        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Force close any open position at end of episode
        if done and self.position == 1:
            final_price = float(self.df["Close"].iloc[-1])
            exit_price  = final_price * (1 - self.transaction_cost)
            pnl         = (exit_price - self.entry_price) / self.entry_price
            reward      += pnl
            self.balance = self.balance * (1 + pnl)
            self.position = 0

        next_state = self._get_state() if not done else np.zeros(self.state_size, dtype=np.float32)
        return next_state, reward, done

    def get_performance(self) -> dict:
        """Return portfolio performance metrics."""
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        win_rate     = (self.profitable_trades / self.total_trades * 100
                        if self.total_trades > 0 else 0)
        return {
            "final_balance":     round(self.balance, 2),
            "total_return_%":    round(total_return, 2),
            "total_trades":      self.total_trades,
            "profitable_trades": self.profitable_trades,
            "win_rate_%":        round(win_rate, 2),
        }


#2. DQN NEURAL NETWORK

class DQNetwork(nn.Module):
    """
    Deep Q-Network - approximates Q-values for each action.

    Q(state, action) = expected future reward from taking
    this action in this state and then following the optimal policy.

    The agent picks the action with the highest Q-value.

    Architecture:
        Input(9) -> Linear(128) -> ReLU -> Linear(128) -> ReLU
                 -> Linear(64)  -> ReLU -> Linear(3)
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.network(x)


#3. DQN AGENT 

class DQNAgent:
    """
    Deep Q-Network agent that learns to trade using experience replay
    and a target network for stable training.

    Key concepts:
        Experience replay: stores past (state, action, reward, next_state)
        tuples and trains on random batches - breaks correlation between
        consecutive experiences.

        Target network: a separate copy of the network updated less
        frequently, providing stable Q-value targets during training.

        Epsilon-greedy: explores randomly with probability epsilon,
        exploits learned policy otherwise. Epsilon decays over time
        as the agent learns.
    """

    def __init__(self, state_size: int, action_size: int,
                 lr: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 32,
                 memory_size: int = 10000):
        """
        Args:
            state_size:     number of values in state vector (9)
            action_size:    number of possible actions (3: BUY/HOLD/SELL)
            lr:             learning rate for Adam optimiser
            gamma:          discount factor - how much future rewards matter
            epsilon:        initial exploration rate (1.0 = fully random)
            epsilon_min:    minimum exploration rate
            epsilon_decay:  how fast epsilon decreases each episode
            batch_size:     number of experiences to train on per step
            memory_size:    maximum experiences to store in replay buffer
        """
        self.state_size    = state_size
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size

        # Replay memory - stores past experiences
        self.memory = deque(maxlen=memory_size)

        # Main network - updated every step
        self.model  = DQNetwork(state_size, action_size)

        # Target network - updated every 10 episodes for stability
        self.target_model = DQNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        With probability epsilon: explore (random action)
        Otherwise: exploit (best known action from Q-network)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)   # explore

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax().item())             # exploit

    def replay(self):
        """
        Train the network on a random batch of past experiences.
        This is the core learning step of DQN.
        """
        if len(self.memory) < self.batch_size:
            return   # not enough experiences yet

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states      = torch.tensor([e[0] for e in batch], dtype=torch.float32)
        actions     = torch.tensor([e[1] for e in batch], dtype=torch.long)
        rewards     = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32)
        dones       = torch.tensor([e[4] for e in batch], dtype=torch.float32)

        # Current Q-values from main network
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from target network (Bellman equation)
        # Q(s,a) = reward + gamma * max(Q(s', a')) if not done
        with torch.no_grad():
            next_q  = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Update main network
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon - explore less as agent learns
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy main network weights to target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, symbol: str):
        """Save trained agent weights to disk."""
        tag = symbol.replace("-", "_")
        torch.save(self.model.state_dict(), f"{tag}_rl_model.pt")
        print(f"  Saved {tag}_rl_model.pt")

    def load(self, symbol: str):
        """Load trained agent weights from disk."""
        tag = symbol.replace("-", "_")
        self.model.load_state_dict(
            torch.load(f"{tag}_rl_model.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()
        self.epsilon = 0.0   # no exploration during inference


#4. GENERATE LSTM PREDICTIONS FOR RL TRAINING

def generate_lstm_predictions(df: pd.DataFrame, symbol: str) -> np.ndarray:
    """
    Run the trained LSTM over the full DataFrame to generate
    a prediction for every timestep.
    """
    import pickle

    tag = symbol.replace("-", "_")

    # Load feature columns saved during training
    try:
        with open(f"{tag}_features.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    except FileNotFoundError:
        from predict import FEATURE_COLUMNS
        feature_cols = FEATURE_COLUMNS

    n_features = len(feature_cols)
    close_idx  = feature_cols.index("Close")
    lookback   = 30 if symbol == "BTC-USD" else 45

    # Load model weights
    # Load model weights — must use n_features from the saved features file
    # not from the DataFrame, since they may differ
    from crypto_model import CryptoLSTM
    state_dict = torch.load(f"{tag}_model.pt", map_location="cpu", weights_only=True)

    # Detect actual input size from saved weights
    # lstm.weight_ih_l0 shape is (4*hidden, input_size)
    actual_n_features = state_dict["lstm.weight_ih_l0"].shape[1]
    model = CryptoLSTM(n_features=actual_n_features)
    model.load_state_dict(state_dict)
    model.eval()

    # Update feature_cols and n_features to match actual model
    if actual_n_features != n_features:
        print(f"  Adjusting features: {n_features} -> {actual_n_features}")
        n_features = actual_n_features
        # reload feature cols to match
        try:
            with open(f"{tag}_features.pkl", "rb") as f:
                feature_cols = pickle.load(f)
            if len(feature_cols) != n_features:
                feature_cols = feature_cols[:n_features]
        except Exception:
            pass
        close_idx = feature_cols.index("Close")

    # Load scaler — was fitted on exactly n_features columns
    with open(f"{tag}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Select only the columns the scaler was fitted on
    # This ensures 23 cols for BTC and 27 cols for ETH/BNB
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing}, using base features only")
        from predict import FEATURE_COLUMNS
        feature_cols = FEATURE_COLUMNS
        n_features   = len(feature_cols)
        close_idx    = feature_cols.index("Close")
        model = CryptoLSTM(n_features=n_features)
        model.load_state_dict(
            torch.load(f"{tag}_model.pt", map_location="cpu", weights_only=True)
        )
        with open(f"{tag}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    data   = df[feature_cols].values
    scaled = scaler.transform(data)

    predictions = np.zeros(len(df))

    for i in range(lookback, len(df)):
        window = scaled[i - lookback:i].astype(np.float32)
        X      = torch.tensor(window).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(X).item()

        dummy = np.zeros((1, n_features))
        dummy[0, close_idx] = pred_scaled
        predictions[i] = float(scaler.inverse_transform(dummy)[0, close_idx])

    predictions[:lookback] = df["Close"].values[:lookback]
    return predictions

# 5. TRAINING LOOP 

def train_agent(symbol: str, df: pd.DataFrame,
                episodes: int = 50,
                target_update_freq: int = 10) -> DQNAgent:
    """
    Train the DQN agent on historical price data.

    Each episode runs through the entire historical dataset,
    simulating trading decisions at every timestep.

    Args:
        symbol:             coin symbol e.g. "BTC-USD"
        df:                 feature-engineered DataFrame
        episodes:           number of full passes through the data
        target_update_freq: how often to update target network

    Returns:
        Trained DQNAgent
    """
    print(f"\n  Generating LSTM predictions for {symbol}...")
    lstm_preds = generate_lstm_predictions(df, symbol)

    env   = CryptoTradingEnv(df, lstm_preds)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    print(f"  Training DQN agent for {symbol} over {episodes} episodes...")
    print(f"  {'Episode':<10} {'Return %':<12} {'Trades':<10} {'Win rate':<12} {'Epsilon'}")
    print(f"  {'─'*56}")

    best_return   = float("-inf")
    best_state    = None

    for episode in range(1, episodes + 1):
        state = env.reset()
        done  = False
        action_counts = {0: 0, 1: 0, 2: 0}

        while not done:
            action              = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            action_counts[action] += 1
            state = next_state

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        perf = env.get_performance()

        if episode % 5 == 0 or episode == 1:
            print(
                f"  {episode:<10} "
                f"{perf['total_return_%']:>+8.2f}%   "
                f"{perf['total_trades']:<10} "
                f"{perf['win_rate_%']:>6.1f}%     "
                f"{agent.epsilon:.3f}"
            )

        # Save best model
        if perf["total_return_%"] > best_return:
            best_return = perf["total_return_%"]
            best_state  = {k: v.clone() for k, v in agent.model.state_dict().items()}

    # Restore best weights
    if best_state:
        agent.model.load_state_dict(best_state)
        agent.epsilon = 0.0

    print(f"\n  Best return achieved: {best_return:+.2f}%")
    agent.save(symbol)
    return agent


#6. LIVE DECISION 

def add_btc_context_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy().sort_index()
    btc = btc_df.copy().sort_index()

    btc_close = pd.to_numeric(btc["Close"], errors="coerce")

    delta = btc_close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    btc_features = pd.DataFrame(
        {
            "BTC_Return_1d": btc_close.pct_change(1),
            "BTC_Return_7d": btc_close.pct_change(7),
            "BTC_RSI": 100 - (100 / (1 + gain / (loss + 1e-9))),
            "BTC_MACD": (
                btc_close.ewm(span=12, adjust=False).mean()
                - btc_close.ewm(span=26, adjust=False).mean()
            ),
        },
        index=btc.index,
    )

    result = result.join(btc_features, how="left")
    result[
        [
            "BTC_Return_1d",
            "BTC_Return_7d",
            "BTC_RSI",
            "BTC_MACD",
        ]
    ] = result[
        [
            "BTC_Return_1d",
            "BTC_Return_7d",
            "BTC_RSI",
            "BTC_MACD",
        ]
    ].ffill()

    result.dropna(inplace=True)
    return result

def get_trading_decision(symbol: str, df: pd.DataFrame) -> dict:
    """
    Load the trained RL agent and get a live BUY/HOLD/SELL decision
    for the current market state.

    Returns a dict with the action and explanation for the chatbot.
    """
    from predict import load_model, FEATURE_COLUMNS
    import pickle

    tag = symbol.replace("-", "_")

    # Generate the LSTM prediction used as one input to the RL state.
    #
    # Important: the saved LSTM scaler/model were trained on the base
    # FEATURE_COLUMNS (23 columns). The live DataFrame may also contain four
    # BTC context columns for RL training, but those must not be passed into
    # the LSTM scaler.
    lstm_model, scaler = load_model(symbol)[:2]
    feature_cols = list(FEATURE_COLUMNS)
    lookback = 45 if symbol != "BTC-USD" else 30
    close_idx = feature_cols.index("Close")
    n_features = len(feature_cols)

    missing = [
        column
        for column in feature_cols
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Live data is missing features required by the LSTM model: "
            + ", ".join(missing)
        )

    recent = (
        df[feature_cols]
        .tail(lookback)
        .to_numpy(dtype=np.float32)
    )

    if len(recent) < lookback:
        raise ValueError(
            f"Not enough live rows for {symbol}: "
            f"got {len(recent)}, need {lookback}."
        )

    expected_features = getattr(scaler, "n_features_in_", n_features)
    if recent.shape[1] != expected_features:
        raise ValueError(
            f"LSTM feature mismatch for {symbol}: live data has "
            f"{recent.shape[1]} features, but the scaler expects "
            f"{expected_features}."
        )

    scaled = scaler.transform(recent)
    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = lstm_model(X).item()

    dummy = np.zeros((1, expected_features), dtype=np.float32)
    dummy[0, close_idx] = pred_scaled
    lstm_pred = float(
        scaler.inverse_transform(dummy)[0, close_idx]
    )

    # Build state for RL agent using most recent row
    current_price  = float(df["Close"].iloc[-1])
    pred_change    = (lstm_pred - current_price) / (current_price + 1e-9)
    rsi            = float(df["RSI"].iloc[-1]) / 100.0
    macd           = float(df["MACD"].iloc[-1])
    macd_sig       = float(df["MACD_Signal"].iloc[-1])
    macd_norm      = float(np.tanh(macd - macd_sig))
    bb_pct         = float(df["BB_Pct"].iloc[-1])
    ret_1d         = float(df["Return_1d"].iloc[-1])
    ret_7d         = float(df["Return_7d"].iloc[-1])
    vol_ratio      = float(np.tanh(float(df["Volume_Ratio"].iloc[-1]) - 1.0))

    state = np.array([
        pred_change, rsi, macd_norm, bb_pct,
        ret_1d, ret_7d, vol_ratio,
        0.0,   # assume not currently holding for live recommendation
        0.0,   # no unrealised PnL
    ], dtype=np.float32)

    # Load RL agent and get decision
    agent = DQNAgent(state_size=9, action_size=3)
    try:
        agent.load(symbol)
    except FileNotFoundError:
        return {
            "action":      "HOLD",
            "action_code": 1,
            "explanation": "RL agent not trained yet. Run rl_agent.py first.",
            "lstm_pred":   round(lstm_pred, 2),
            "change_pct":  round((lstm_pred - current_price) / current_price * 100, 2),
        }

    action_code = agent.act(state)
    actions     = {0: "SELL", 1: "HOLD", 2: "BUY"}
    action      = actions[action_code]

    # Plain English explanations for chatbot
    explanations = {
        2: (
            f"The RL agent recommends BUY. The LSTM predicts a "
            f"{abs((lstm_pred-current_price)/current_price*100):.2f}% price increase "
            f"and the agent has learned that current market conditions "
            f"(RSI: {rsi*100:.0f}, MACD: {'bullish' if macd>macd_sig else 'bearish'}) "
            f"are historically favourable for entering a position."
        ),
        1: (
            f"The RL agent recommends HOLD — no action needed right now. "
            f"The agent has learned that the current signals do not strongly "
            f"favour either entering or exiting a position at this time."
        ),
        0: (
            f"The RL agent recommends SELL / AVOID. "
            f"The LSTM predicts a "
            f"{abs((lstm_pred-current_price)/current_price*100):.2f}% price "
            f"{'decrease' if lstm_pred < current_price else 'change'} "
            f"and the agent has learned that current conditions suggest "
            f"exiting or avoiding a position."
        ),
    }

    return {
        "action":      action,
        "action_code": action_code,
        "explanation": explanations[action_code],
        "lstm_pred":   round(lstm_pred, 2),
        "change_pct":  round((lstm_pred - current_price) / current_price * 100, 2),
    }


# 7. MAIN 

def run_rl_pipeline(symbols: list[str], episodes: int = 50):
    """Train RL agents for all coins using their saved LSTM predictions."""
    from predict import fetch_live_data, add_features

    print("\n" + "="*60)
    print(" RL TRADING AGENT PIPELINE  (DQN)")
    print("="*60)
    print(f"\n  Training on historical data with {episodes} episodes per coin")
    print("  Actions: 0=SELL  1=HOLD  2=BUY")

    results = {}

    for symbol in symbols:
        print(f"\n{'─'*50}")
        print(f" Training agent for: {symbol}")
        print(f"{'─'*50}")

        try:
            # Fetch historical data for training environment
            print("  Fetching training data...")
            from predict import load_model, FEATURE_COLUMNS
            from crypto_model import fetch_crypto_data, add_technical_indicators

            EXTENDED_FEATURE_COLUMNS = [
                "Open","High","Low","Close","Volume",
                "SMA_7","SMA_21","EMA_12","EMA_26",
                "MACD","MACD_Signal","MACD_Hist","RSI",
                "BB_Upper","BB_Lower","BB_Width","BB_Pct",
                "Volume_SMA","Volume_Ratio",
                "Return_1d","Return_7d","Return_14d","HL_Range",
                "BTC_Return_1d","BTC_Return_7d","BTC_RSI","BTC_MACD"
            ]

            def add_cross_asset_features(df, btc_df):
                df = df.copy()
                shared_idx  = df.index.intersection(btc_df.index)
                df          = df.loc[shared_idx]
                btc_aligned = btc_df.loc[shared_idx]
                df["BTC_Return_1d"] = btc_aligned["Return_1d"].values
                df["BTC_Return_7d"] = btc_aligned["Return_7d"].values
                df["BTC_RSI"]       = btc_aligned["RSI"].values
                df["BTC_MACD"]      = btc_aligned["MACD"].values
                df.dropna(inplace=True)
                return df
            all_data = fetch_crypto_data([symbol, "BTC-USD"], days=1095)
            df = add_technical_indicators(all_data[symbol])

            if symbol != "BTC-USD" and "BTC-USD" in all_data:
                btc_df = add_technical_indicators(all_data["BTC-USD"])
                df = add_cross_asset_features(df, btc_df)

            # Train the RL agent
            agent = train_agent(symbol, df, episodes=episodes)

            # Get a live decision using current market data
            print("\n  Getting live trading decision...")
            live_raw = fetch_live_data(symbol, days=180)
            live_df = add_features(live_raw)

            if symbol == "BTC-USD":
                btc_df = live_df.copy()
            else:
                btc_raw = fetch_live_data("BTC-USD", days=180)
                btc_df = add_features(btc_raw)

            live_df = add_btc_context_features(
                live_df,
                btc_df,
            )

            decision = get_trading_decision(
                symbol,
                live_df,
            )
            results[symbol] = decision

            print(f"\n  RECOMMENDATION: {decision['action']}")
            print(f"  LSTM predicted: ${decision['lstm_pred']:,.2f} "
                  f"({decision['change_pct']:+.2f}%)")

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print("  Run crypto_model.py first to train the LSTM model.")

    print("\n" + "="*60)
    print(" SUMMARY - LIVE TRADING RECOMMENDATIONS")
    print("="*60)
    for sym, d in results.items():
        print(f"  {sym:<12} {d['action']:<6}  "
              f"LSTM: ${d['lstm_pred']:,.2f} ({d['change_pct']:+.2f}%)")
    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    COINS = ["BTC-USD", "ETH-USD", "BNB-USD"]

    run_rl_pipeline(
        symbols  = COINS,
        episodes = 50,    # increase to 100+ for better performance
    )
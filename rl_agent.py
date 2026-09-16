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
import os
import json

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
        - LSTM predicted next-day % change (from the price head)
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
                 lstm_directions: np.ndarray,
                 initial_balance: float = 10000.0, transaction_cost: float = 0.001):
        """
        Args:
            df:                  Feature-engineered DataFrame from predict.py
            lstm_predictions:    Array of next-day price predictions from LSTM
            lstm_directions:     Array of P(price goes up) from the LSTM's
                                 direction head. Kept for informational
                                 display only - crypto_model.py's decoupled
                                 early-stopping run confirmed this head isn't
                                 learning a real signal at this horizon, so
                                 it no longer feeds the state (see
                                 _get_state)
            initial_balance:     Starting portfolio value in USD
            transaction_cost:    Fee per trade (0.001 = 0.1%)
        """
        self.df = df
        self.lstm_predictions = lstm_predictions
        self.lstm_directions = lstm_directions
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.n_steps = len(df)
        self.state_size = 9
        self.action_size = 3  # 0=SELL, 1=HOLD, 2=BUY

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode. Returns initial state."""
        self.current_step = 1  # start at 1 so we can look back
        self.balance = self.initial_balance
        self.position = 0  # 0 = not holding, 1 = holding
        self.entry_price = 0.0  # price we bought at
        self.total_trades = 0
        self.profitable_trades = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Build the state vector the agent sees at each timestep.
        Combines LSTM prediction with current technical indicators
        and portfolio status.
        """
        i = self.current_step

        # predicted next-day % change from the LSTM price head. used to use
        # the direction head's P(up) here but it wasn't learning a real
        # signal, just predicting the same class every time. still shown to
        # the user in get_trading_decision, just not used in the state here
        current_price = float(self.df["Close"].iloc[i])
        predicted_price = float(self.lstm_predictions[i])
        pred_change = (predicted_price - current_price) / current_price

        # Technical indicators (normalised to roughly 0-1 range)
        rsi = float(self.df["RSI"].iloc[i]) / 100.0
        macd = float(self.df["MACD"].iloc[i])
        macd_sig = float(self.df["MACD_Signal"].iloc[i])
        macd_norm = np.tanh(macd - macd_sig)  # tanh squishes to -1 to 1
        bb_pct = float(self.df["BB_Pct"].iloc[i])
        ret_1d = float(self.df["Return_1d"].iloc[i])
        ret_7d = float(self.df["Return_7d"].iloc[i])
        vol_ratio = np.tanh(float(self.df["Volume_Ratio"].iloc[i]) - 1.0)

        # Portfolio status
        position = float(self.position)
        if self.position == 1 and self.entry_price > 0:
            unrealised_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealised_pnl = 0.0

        state = np.array([
            pred_change,  # LSTM predicted next-day % price change
            rsi,  # momentum indicator
            macd_norm,  # trend crossover signal
            bb_pct,  # price position in volatility range
            ret_1d,  # recent 1-day momentum
            ret_7d,  # recent 7-day momentum
            vol_ratio,  # unusual volume signal
            position,  # are we currently holding?
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
        reward = 0.0

        # next price move - used to reward/penalise HOLD and staying flat
        next_price = float(self.df["Close"].iloc[
            min(self.current_step + 1, self.n_steps - 1)
        ])
        price_change = (next_price - current_price) / current_price

        # BUY
        if action == 2:  # BUY
            if self.position == 0:
                # Enter a position - pay transaction cost
                self.position = 1
                self.entry_price = current_price * (1 + self.transaction_cost)
                reward = -self.transaction_cost  # small penalty for trading cost

        # SELL
        elif action == 0:  # SELL
            if self.position == 1:
                # Exit position - receive profit or loss minus transaction cost
                exit_price = current_price * (1 - self.transaction_cost)
                pnl = (exit_price - self.entry_price) / self.entry_price
                reward = pnl
                self.balance = self.balance * (1 + pnl)
                self.position = 0

                if pnl > 0:
                    self.profitable_trades += 1
                self.total_trades += 1
            else:
                # already flat - reward for avoiding a drop, penalise for missing a rise
                reward = -price_change * 0.5  # opportunity cost of staying flat

        # HOLD
        else:  # HOLD
            if self.position == 1:
                # Reward for holding a winning position, penalise holding a loser
                reward = price_change * 0.5  # partial reward for unrealised gains
            else:
                # not holding - same opportunity cost logic as SELL above
                reward = -price_change * 0.5  # opportunity cost of staying flat

        # Move to next timestep
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Force close any open position at end of episode
        if done and self.position == 1:
            final_price = float(self.df["Close"].iloc[-1])
            exit_price = final_price * (1 - self.transaction_cost)
            pnl = (exit_price - self.entry_price) / self.entry_price
            reward += pnl
            self.balance = self.balance * (1 + pnl)
            self.position = 0

            if pnl > 0:
                self.profitable_trades += 1
            self.total_trades += 1

        next_state = self._get_state() if not done else np.zeros(self.state_size, dtype=np.float32)
        return next_state, reward, done

    def get_performance(self) -> dict:
        """Return portfolio performance metrics."""
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.profitable_trades / self.total_trades * 100
                        if self.total_trades > 0 else 0)
        return {
            "final_balance": round(self.balance, 2),
            "total_return_%": round(total_return, 2),
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "win_rate_%": round(win_rate, 2),
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
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Replay memory - stores past experiences
        self.memory = deque(maxlen=memory_size)

        # Main network - updated every step
        self.model = DQNetwork(state_size, action_size)

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
            return random.randrange(self.action_size)  # explore

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(q_values.argmax().item())  # exploit

    def replay(self):
        """
        Train the network on a random batch of past experiences.
        This is the core learning step of DQN.
        """
        if len(self.memory) < self.batch_size:
            return  # not enough experiences yet

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([e[0] for e in batch], dtype=torch.float32)
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32)

        # Current Q-values from main network
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from target network (Bellman equation)
        # Q(s,a) = reward + gamma * max(Q(s', a')) if not done
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
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
        self.epsilon = 0.0  # no exploration during inference


#4. GENERATE LSTM PREDICTIONS FOR RL TRAINING

def generate_lstm_predictions(df: pd.DataFrame, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the trained LSTM over the full DataFrame to generate a price
    prediction and a direction-head P(up) for every timestep.

    Returns:
        predictions:  next-day price prediction per row
        directions:   P(price goes up) per row, from the dedicated
                      direction head. Informational only - not used to
                      build the RL state (see CryptoTradingEnv._get_state)
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
    close_idx = feature_cols.index("Close")
    lookback = 30 if symbol == "BTC-USD" else 45

    # load model weights - n_features must come from the saved features
    # file, not the DataFrame, since they can differ
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

    # load scaler - fitted on exactly n_features columns
    with open(f"{tag}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Select only the columns the scaler was fitted on
    # This ensures 23 cols for BTC and 27 cols for ETH/BNB
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing}, using base features only")
        from predict import FEATURE_COLUMNS
        feature_cols = FEATURE_COLUMNS
        n_features = len(feature_cols)
        close_idx = feature_cols.index("Close")
        model = CryptoLSTM(n_features=n_features)
        model.load_state_dict(
            torch.load(f"{tag}_model.pt", map_location="cpu", weights_only=True)
        )
        with open(f"{tag}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    data = df[feature_cols].values
    scaled = scaler.transform(data)

    predictions = np.zeros(len(df))
    directions = np.full(len(df), 0.5)  # neutral default where no window exists yet

    for i in range(lookback, len(df)):
        window = scaled[i - lookback:i].astype(np.float32)
        X = torch.tensor(window).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(X).item()
            direction_logit = model.forward_direction(X).item()

        dummy = np.zeros((1, n_features))
        dummy[0, close_idx] = pred_scaled
        predictions[i] = float(scaler.inverse_transform(dummy)[0, close_idx])
        directions[i] = float(torch.sigmoid(torch.tensor(direction_logit)))

    predictions[:lookback] = df["Close"].values[:lookback]
    return predictions, directions

def buy_and_hold_return(df: pd.DataFrame, transaction_cost: float = 0.001) -> float:
    """
    % return from just buying at the start and holding to the end - the
    baseline the RL agent needs to beat, since crypto can trend hard either
    way and a big return could just be market movement, not real skill.
    """
    entry = float(df["Close"].iloc[0]) * (1 + transaction_cost)
    exit_ = float(df["Close"].iloc[-1]) * (1 - transaction_cost)
    return (exit_ - entry) / entry * 100


# 5. TRAINING LOOP

def train_agent(symbol: str, df: pd.DataFrame,
                episodes: int = 50,
                train_ratio: float = 0.8,
                val_ratio: float = 0.15,
                target_update_freq: int = 10) -> tuple[DQNAgent, pd.DataFrame, float, float]:
    """
    Train the DQN agent on the training portion of historical price data,
    holding out the remainder as an out-of-sample test period.

    The training portion is further split chronologically into an inner
    training subset and a validation subset. The agent only learns from
    the inner subset; whichever episode's checkpoint scores best on the
    validation subset (data it did not train on) is the one that gets
    saved - not whichever episode had the best training return. This
    stops checkpoint selection itself from overfitting to the training
    data.

    Each episode runs through the entire inner training subset,
    simulating trading decisions at every timestep.

    Args:
        symbol:             coin symbol e.g. "BTC-USD"
        df:                 feature-engineered DataFrame
        episodes:           number of full passes through the data
        train_ratio:        fraction of df used for training (rest is held out as test)
        val_ratio:          fraction of the training portion held out for validation
        target_update_freq: how often to update target network

    Returns:
        Trained DQNAgent, the held-out test DataFrame, the best validation
        return, and the validation period's buy & hold return (baseline)
    """
    # split into train and test portions
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"  Training period: {df.index[0].date()} -> {df.index[split_idx].date()}")
    print(f"  Test period:     {df.index[split_idx].date()} -> {df.index[-1].date()}")
    print(f"  Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")

    # split training portion again into inner-train / validation, so
    # checkpoints are scored on data the model didn't train on
    val_split_idx = int(len(train_df) * (1 - val_ratio))
    inner_train_df = train_df.iloc[:val_split_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_split_idx:].reset_index(drop=True)

    print(f"  Inner-train rows: {len(inner_train_df)}  |  Validation rows: {len(val_df)}")
    val_buy_hold = buy_and_hold_return(val_df)
    print(f"  Validation buy & hold: {val_buy_hold:+.2f}%  "
          f"(passive baseline for the validation window)")

    print(f"\n  Generating LSTM predictions for {symbol}...")
    lstm_preds, lstm_dirs = generate_lstm_predictions(inner_train_df, symbol)
    lstm_preds_val, lstm_dirs_val = generate_lstm_predictions(val_df, symbol)

    env = CryptoTradingEnv(inner_train_df, lstm_preds, lstm_dirs)
    env_val = CryptoTradingEnv(val_df, lstm_preds_val, lstm_dirs_val)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    print(f"  Training DQN agent for {symbol} over {episodes} episodes...")
    print(f"  {'Episode':<10} {'Train %':<12} {'Val %':<12} {'Trades':<10} {'Win rate':<12} {'Epsilon'}")
    print(f"  {'─'*68}")

    best_val_return = float("-inf")
    best_state = None

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        action_counts = {0: 0, 1: 0, 2: 0}

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            action_counts[action] += 1
            state = next_state

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        perf = env.get_performance()

        # score this checkpoint on held-out validation data, not training data
        train_epsilon = agent.epsilon
        agent.epsilon = 0.0  # pure exploitation for validation scoring

        val_state = env_val.reset()
        val_done = False
        while not val_done:
            val_action = agent.act(val_state)
            val_state, _, val_done = env_val.step(val_action)
        val_perf = env_val.get_performance()

        agent.epsilon = train_epsilon  # resume training with normal exploration

        if episode % 5 == 0 or episode == 1:
            print(
                f"  {episode:<10} "
                f"{perf['total_return_%']:>+8.2f}%   "
                f"{val_perf['total_return_%']:>+8.2f}%   "
                f"{perf['total_trades']:<10} "
                f"{perf['win_rate_%']:>6.1f}%     "
                f"{agent.epsilon:.3f}"
            )

        # save best model based on validation performance
        if val_perf["total_return_%"] > best_val_return:
            best_val_return = val_perf["total_return_%"]
            best_state = {k: v.clone() for k, v in agent.model.state_dict().items()}

    # Restore best weights
    if best_state:
        agent.model.load_state_dict(best_state)
        agent.epsilon = 0.0

    print(f"\n  Best validation return achieved: {best_val_return:+.2f}%  "
          f"(buy & hold: {val_buy_hold:+.2f}%)")
    agent.save(symbol)
    return agent, test_df, best_val_return, val_buy_hold


def evaluate_agent(agent: DQNAgent, symbol: str,
                   test_df: pd.DataFrame) -> dict:
    """
    Evaluate the trained agent on the held-out test period.
    Epsilon is 0 - pure exploitation, no exploration.
    No learning happens here - weights are frozen.
    """
    print(f"\n  Evaluating on held-out test period ({len(test_df)} days)...")

    # generate LSTM predictions for test period
    lstm_preds_test, lstm_dirs_test = generate_lstm_predictions(test_df, symbol)

    env = CryptoTradingEnv(test_df, lstm_preds_test, lstm_dirs_test)
    agent.epsilon = 0.0  # no exploration during evaluation

    state = env.reset()
    done = False

    while not done:
        action = agent.act(state)  # always exploits best known action
        state, _, done = env.step(action)

    perf = env.get_performance()
    perf["buy_hold_return_%"] = round(buy_and_hold_return(test_df), 2)

    print(f"  Test period return:      {perf['total_return_%']:+.2f}%")
    print(f"  Test period buy & hold:  {perf['buy_hold_return_%']:+.2f}%  (passive baseline)")
    print(f"  Test period trades:      {perf['total_trades']}")
    print(f"  Test period win rate:    {perf['win_rate_%']:.1f}%")

    return perf


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
        direction_logit = lstm_model.forward_direction(X).item()

    dummy = np.zeros((1, expected_features), dtype=np.float32)
    dummy[0, close_idx] = pred_scaled
    lstm_pred = float(
        scaler.inverse_transform(dummy)[0, close_idx]
    )
    up_prob = float(torch.sigmoid(torch.tensor(direction_logit)))

    # Build state for RL agent using most recent row
    current_price = float(df["Close"].iloc[-1])
    # predicted next-day % change from the LSTM price head (see the same
    # note in CryptoTradingEnv._get_state - direction head wasn't reliable)
    pred_change = (lstm_pred - current_price) / current_price
    rsi = float(df["RSI"].iloc[-1]) / 100.0
    macd = float(df["MACD"].iloc[-1])
    macd_sig = float(df["MACD_Signal"].iloc[-1])
    macd_norm = float(np.tanh(macd - macd_sig))
    bb_pct = float(df["BB_Pct"].iloc[-1])
    ret_1d = float(df["Return_1d"].iloc[-1])
    ret_7d = float(df["Return_7d"].iloc[-1])
    vol_ratio = float(np.tanh(float(df["Volume_Ratio"].iloc[-1]) - 1.0))

    state = np.array([
        pred_change, rsi, macd_norm, bb_pct,
        ret_1d, ret_7d, vol_ratio,
        0.0,  # assume not currently holding for live recommendation
        0.0,  # no unrealised PnL
    ], dtype=np.float32)

    # Load RL agent and get decision
    agent = DQNAgent(state_size=9, action_size=3)
    try:
        agent.load(symbol)
    except FileNotFoundError:
        return {
            "action": "HOLD",
            "action_code": 1,
            "explanation": "RL agent not trained yet. Run rl_agent.py first.",
            "lstm_pred": round(lstm_pred, 2),
            "change_pct": round((lstm_pred - current_price) / current_price * 100, 2),
            "up_prob": round(up_prob, 4),
        }

    action_code = agent.act(state)
    actions = {0: "SELL", 1: "HOLD", 2: "BUY"}
    action = actions[action_code]

    # Q-value for each action, so the dashboard can show more than just the pick
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_raw = agent.model(state_tensor).squeeze(0).numpy()

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
            f"The RL agent recommends HOLD - no action needed right now. "
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

    decision = {
        "action": action,
        "action_code": action_code,
        "explanation": explanations[action_code],
        "lstm_pred": round(lstm_pred, 2),
        "change_pct": round((lstm_pred - current_price) / current_price * 100, 2),
        "up_prob": round(up_prob, 4),
        "q_values": {
            "SELL": round(float(q_raw[0]), 4),
            "HOLD": round(float(q_raw[1]), 4),
            "BUY": round(float(q_raw[2]), 4),
        },
    }

    # if this coin has saved backtest results, gate the raw action against
    # them so live matches training's confidence check - else use raw signal
    backtest_path = f"{tag}_backtest_results.json"
    if os.path.exists(backtest_path):
        with open(backtest_path) as f:
            backtest = json.load(f)
        decision = apply_confidence_gate(
            decision, backtest["test_return"], backtest["test_buy_hold"]
        )

    return decision



def apply_confidence_gate(decision: dict, test_return: float, test_buy_hold: float) -> dict:
    """
    Downgrade a live BUY/SELL to HOLD unless this coin's agent both beat
    buy & hold on its test backtest AND was actually profitable. Checking
    only "beat buy & hold" let a losing strategy through as long as it lost
    less than holding would have - not good enough for a live call.
    """
    beat_hold = test_return > test_buy_hold
    is_profitable = test_return > 0
    passed_gate = beat_hold and is_profitable

    decision["beat_buy_hold"] = beat_hold
    decision["is_profitable"] = is_profitable
    decision["test_return"] = round(test_return, 2)
    decision["test_buy_hold"] = round(test_buy_hold, 2)

    if not passed_gate and decision["action"] != "HOLD":
        decision["raw_action"] = decision["action"]
        decision["raw_action_code"] = decision["action_code"]
        decision["action"] = "HOLD"
        decision["action_code"] = 1

        reasons = []
        if not is_profitable:
            reasons.append(f"lost money on its own test backtest ({test_return:+.2f}%)")
        if not beat_hold:
            reasons.append(
                f"underperformed a simple buy & hold by "
                f"{test_buy_hold - test_return:.2f} points "
                f"({test_return:+.2f}% vs {test_buy_hold:+.2f}% buy & hold)"
            )
        reason_text = " and ".join(reasons)

        decision["explanation"] = (
            f"Downgraded from {decision['raw_action']} to HOLD: this coin's RL "
            f"agent {reason_text}, so its live signal isn't trusted right now."
        )

    return decision


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
                shared_idx = df.index.intersection(btc_df.index)
                df = df.loc[shared_idx]
                btc_aligned = btc_df.loc[shared_idx]
                df["BTC_Return_1d"] = btc_aligned["Return_1d"].values
                df["BTC_Return_7d"] = btc_aligned["Return_7d"].values
                df["BTC_RSI"] = btc_aligned["RSI"].values
                df["BTC_MACD"] = btc_aligned["MACD"].values
                df.dropna(inplace=True)
                return df
            # match crypto_model.py's training window so the RL agent sees the
            # same bull/bear history the LSTM trained on
            all_data = fetch_crypto_data([symbol, "BTC-USD"], days=3000)
            df = add_technical_indicators(all_data[symbol])

            if symbol != "BTC-USD" and "BTC-USD" in all_data:
                btc_df = add_technical_indicators(all_data["BTC-USD"])
                df = add_cross_asset_features(df, btc_df)

            # train + test the RL agent
            agent, test_df, best_val_return, val_buy_hold = train_agent(symbol, df, episodes=episodes)
            test_results = evaluate_agent(agent, symbol, test_df)

            # save backtest numbers so get_trading_decision can gate live
            # calls against them later without retraining
            tag = symbol.replace("-", "_")
            with open(f"{tag}_backtest_results.json", "w") as f:
                json.dump({
                    "test_return": test_results["total_return_%"],
                    "test_buy_hold": test_results["buy_hold_return_%"],
                }, f)

            # set up results entry if it doesn't exist yet
            if symbol not in results:
                results[symbol] = {}

            results[symbol]["best_val_return"] = best_val_return
            results[symbol]["val_buy_hold"] = val_buy_hold
            results[symbol]["test_return"] = test_results["total_return_%"]
            results[symbol]["test_buy_hold"] = test_results["buy_hold_return_%"]
            results[symbol]["test_win_rate"] = test_results["win_rate_%"]
            results[symbol]["test_trades"] = test_results["total_trades"]

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
            decision = apply_confidence_gate(
                decision, test_results["total_return_%"], test_results["buy_hold_return_%"]
            )
            results[symbol].update(decision)

            print(f"\n  RECOMMENDATION: {decision['action']}", end="")
            if "raw_action" in decision:
                print(f"  (agent's raw signal was {decision['raw_action']}, "
                      f"gated - see explanation below)")
            else:
                print()
            print(f"  LSTM predicted: ${decision['lstm_pred']:,.2f} "
                  f"({decision['change_pct']:+.2f}%)")
            print(f"  Direction head:  {decision['up_prob']*100:.1f}% chance of an up day")

        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print("  Run crypto_model.py first to train the LSTM model.")

    print("\n" + "="*60)
    print(" SUMMARY - LIVE TRADING RECOMMENDATIONS")
    print("="*60)
    for sym, d in results.items():
        action_display = d['action'] + (" (gated)" if "raw_action" in d else "")
        print(f"  {sym:<12} "
              f"Val return: {d.get('best_val_return', 0):>+8.2f}% (hold {d.get('val_buy_hold', 0):>+7.2f}%)  |  "
              f"Test return: {d.get('test_return', 0):>+8.2f}% (hold {d.get('test_buy_hold', 0):>+7.2f}%)  |  "
              f"{action_display:<15}  "
              f"LSTM: ${d['lstm_pred']:,.2f} ({d['change_pct']:+.2f}%)  |  "
              f"P(up): {d['up_prob']*100:.1f}%")
    print("="*60 + "\n")

    return results


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

    run_rl_pipeline(
        symbols = COINS,
        episodes = 50,  # increase to 100+ for better performance
    )
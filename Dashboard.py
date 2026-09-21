# pip install streamlit
# run: streamlit run Dashboard.py

"""
Documentation used:
Streamlit: https://docs.streamlit.io/develop/api-reference

Requires:
    predict.py
    rl_agent.py
    For each coin below: {TAG}_model.pt / {TAG}_scaler.pkl (from crypto_model.py)
                          {TAG}_rl_model.pt (from rl_agent.py)
                          {TAG}_backtest_results.json (from rl_agent.py, optional -
                              lights up the live confidence gate if present)
        BTC_USD, ETH_USD, BNB_USD, XRP_USD, SOL_USD, ADA_USD, DOGE_USD,
        TRX_USD, LINK_USD, AVAX_USD, XLM_USD, LTC_USD, BCH_USD

Display modes:
    A single sidebar toggle switches between three views that share the same
    look and feel (same light theme, same card style throughout):
      - "Basic"    -> just the coin, its current price, and the AI
                      recommendation with its one-line reason. 
      - "Simple"   -> Basic, plus price prediction, 7-day change, price
                      chart, and the technical indicator panel.
      - "Advanced" -> everything Simple has, plus a "Model Internals"
                      section that names the real technical terms (state
                      vector, Q-values, confidence gate) and explains each
                      one in plain English right next to it, with a couple
                      of real charts (Q-network output, MACD/Bollinger
                      history) 

Visual style:
    Neutral slate background, white cards with thin borders and a coloured
    left accent bar for status (rather than solid pastel fills), Inter for
    text and JetBrains Mono for numbers, aimed at reading like a financial
    research tool rather than a consumer app.
"""


import streamlit as st
import pandas as pd
import os
from datetime import datetime
from predict import fetch_live_data, add_features, load_model, predict, FEATURE_COLUMNS

from rl_agent import get_trading_decision


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


# CACHE FUNCTIONS

@st.cache_data(ttl=300, show_spinner=False)
def get_rl_decision(symbol):
    try:
        raw = fetch_live_data(symbol, days=90)
        df = add_features(raw)
        if symbol != "BTC-USD":
            btc_raw = fetch_live_data("BTC-USD", days=90)
            btc_df = add_features(btc_raw)
            df = add_cross_asset_features(df, btc_df)
        return get_trading_decision(symbol, df)
    except Exception as e:
        return {"action": "HOLD", "explanation": f"Unavailable: {e}"}


@st.cache_data(ttl=300, show_spinner=False)
def get_prediction(symbol: str) -> dict:
    """Fetch live data, run model, return everything the dashboard needs."""
    raw = fetch_live_data(symbol, days=90)
    df = add_features(raw)
    model, scaler = load_model(symbol)

    last_close = float(df["Close"].iloc[-1])
    next_price = predict(model, df, scaler)
    change_pct = (next_price - last_close) / last_close * 100

    rsi = float(df["RSI"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    macd_sig = float(df["MACD_Signal"].iloc[-1])
    bb_pct = float(df["BB_Pct"].iloc[-1])
    ret_1d = float(df["Return_1d"].iloc[-1]) * 100
    ret_7d = float(df["Return_7d"].iloc[-1]) * 100
    vol_ratio = float(df["Volume_Ratio"].iloc[-1])

    return {
        "df": df,
        "last_close": last_close,
        "predicted": next_price,
        "change_pct": change_pct,
        "direction": "UP" if change_pct > 0 else "DOWN",
        "rsi": rsi,
        "macd": macd,
        "macd_sig": macd_sig,
        "bb_pct": bb_pct,
        "ret_1d": ret_1d,
        "ret_7d": ret_7d,
        "vol_ratio": vol_ratio,
        "last_date": df.index[-1].strftime("%d %B %Y"),
    }


# PAGE CONFIG

st.set_page_config(
    page_title="Coin Compass",
    page_icon="coin_compass_icon.png",
    layout="wide",
)

# BRANDING - st.logo() shows the wordmark in the sidebar (small icon when
# collapsed). needs Streamlit 1.29+, and both logo files next to this script
st.logo("coin_compass_logo_no_tagline.png", icon_image="coin_compass_icon.png")

USE_TEST_DATA = os.environ.get("USE_TEST_DATA", "false").lower() in {"1", "true", "yes", "on"}


# CUSTOM STYLING - one theme shared by all 3 display modes. white cards,
# light background, Inter font for text, JetBrains Mono for numbers

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    :root {
        --bg: #f1f5f9;
        --surface: #ffffff;
        --border: #e2e8f0;
        --text: #0f172a;
        --text-muted: #64748b;
        --accent: #0d9488;
        --accent-dark: #0f766e;
        --positive: #15803d;
        --positive-bg: #f0fdf4;
        --positive-border: #16a34a;
        --negative: #b91c1c;
        --negative-bg: #fef2f2;
        --negative-border: #dc2626;
        --neutral: #92400e;
        --neutral-bg: #fffbeb;
        --neutral-border: #d97706;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--bg);
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Small uppercase label used above a title for a report-like feel */
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 2px;
    }

    /* Stat tiles: current price / prediction / 7-day change */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .metric-card.positive { border-left: 3px solid var(--positive-border); }
    .metric-card.negative { border-left: 3px solid var(--negative-border); }
    .metric-card.neutral  { border-left: 3px solid var(--text-muted); }
    .metric-label {
        font-size: 12px;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: var(--text);
        font-variant-numeric: tabular-nums;
    }
    .metric-sub {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 8px;
    }
    .metric-change {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 600;
        margin-top: 8px;
    }
    .metric-change.positive { color: var(--positive); }
    .metric-change.negative { color: var(--negative); }
    .metric-change.neutral  { color: var(--text-muted); }

    /* Technical indicator rows */
    .indicator-row {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .indicator-red    { border-left: 3px solid var(--negative-border); }
    .indicator-green  { border-left: 3px solid var(--positive-border); }
    .indicator-amber  { border-left: 3px solid var(--neutral-border); }
    .indicator-title {
        font-weight: 600;
        font-size: 14px;
        color: var(--text);
        margin-bottom: 3px;
    }
    .indicator-explanation {
        font-size: 13px;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* Recommendation card */
    .recommendation-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 22px 24px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin: 10px 0 20px;
    }
    .recommendation-card.buy  { border-left: 4px solid var(--positive-border); }
    .recommendation-card.hold { border-left: 4px solid var(--neutral-border); }
    .recommendation-card.sell { border-left: 4px solid var(--negative-border); }

    .action-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 5px 14px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
    .action-pill.buy  { background: var(--positive-bg); color: var(--positive); border: 1px solid var(--positive-border); }
    .action-pill.hold { background: var(--neutral-bg);  color: var(--neutral);  border: 1px solid var(--neutral-border); }
    .action-pill.sell { background: var(--negative-bg); color: var(--negative); border: 1px solid var(--negative-border); }

    .rec-explanation {
        font-size: 14px;
        color: var(--text);
        line-height: 1.6;
    }

    /* Notice / disclaimer boxes */
    .notice-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 20px;
    }
    .notice-box.gate {
        font-size: 15px;
        color: var(--text);
        padding: 22px 26px;
        margin-bottom: 18px;
    }

    /* Advanced-mode cards */
    .adv-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 18px;
        margin: 8px 0 16px;
    }
    .adv-term {
        font-weight: 700;
        font-size: 14px;
        color: var(--text);
        margin-bottom: 3px;
    }
    .adv-meaning {
        font-size: 12.5px;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* Confidence gate pass/fail chips */
    .gate-chip {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 14px;
        margin-top: 8px;
    }
    .gate-chip.pass { border-left: 3px solid var(--positive-border); }
    .gate-chip.fail { border-left: 3px solid var(--negative-border); }
    .gate-chip-title {
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
    }
    .gate-chip-status {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        letter-spacing: 0.04em;
        font-size: 11px;
    }
    .gate-chip.pass .gate-chip-status { color: var(--positive); }
    .gate-chip.fail .gate-chip-status { color: var(--negative); }
    .gate-chip-detail {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* Coin metadata tags (category / launch year) */
    .tag-row {
        display: flex;
        gap: 8px;
        margin: 8px 0 4px;
        flex-wrap: wrap;
    }
    .tag {
        display: inline-block;
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-muted);
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Primary button */
    div.stButton > button[kind="primary"] {
        background-color: var(--accent);
        border-color: var(--accent);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--accent-dark);
        border-color: var(--accent-dark);
    }
</style>
""", unsafe_allow_html=True)


# CONSENT GATE - page doesn't render until user agrees to the disclaimer.
# remembered in session_state so it only shows once per session

if "agreed_to_disclaimer" not in st.session_state:
    st.session_state.agreed_to_disclaimer = False

if not st.session_state.agreed_to_disclaimer:
    gate_col1, gate_col2, gate_col3 = st.columns([1, 1, 1])
    with gate_col2:
        st.image("coin_compass_icon.png", width=96)
    st.markdown('<div class="eyebrow">Coin Compass</div>', unsafe_allow_html=True)
    st.title("Before you continue")
    st.markdown(
        '<div class="notice-box gate">'
        '<strong>Please read this disclaimer</strong><br><br>'
        'This tool is for <strong>educational purposes only</strong>, built as part of '
        'a university project. Predictions and trading recommendations are generated '
        'by machine learning models and are <strong>not guaranteed to be accurate or '
        'profitable</strong>. Cryptocurrency markets are highly volatile, and past '
        'performance shown in this app is not indicative of future results.<br><br>'
        'Nothing in this tool constitutes financial advice. Always do your own '
        'research and never invest more than you can afford to lose.'
        '</div>',
        unsafe_allow_html=True
    )
    agreed = st.checkbox(
        "I have read and understood the above, and I agree to use this tool "
        "for educational purposes only."
    )
    if st.button("Continue to the app", disabled=not agreed, type="primary"):
        st.session_state.agreed_to_disclaimer = True
        st.rerun()
    st.stop()


# APP HEADER - logo + welcome text, plus dismissible 'how-to-use' box for
# first time visitors. same session_state trick as the disclaimer above

hero_col1, hero_col2, hero_col3 = st.columns([1, 2, 1])
with hero_col2:
    st.image("coin_compass_logo_no_tagline.png", width=220)
    st.markdown(
        '<p style="text-align:center; color:var(--text-muted); '
        'font-size:14px; margin-top:-6px;">'
        'Your AI co-pilot for exploring crypto, no experience needed.</p>',
        unsafe_allow_html=True
    )

if "dismissed_welcome" not in st.session_state:
    st.session_state.dismissed_welcome = False

if not st.session_state.dismissed_welcome:
    st.markdown(
        '<div class="notice-box">'
        '<strong>New here? Here is how it works, in three steps.</strong><br><br>'
        '1. Pick a coin in the sidebar.<br>'
        '2. Read the AI recommendation and in Simple or Advanced mode, see what is driving it.<br>'
        '3. Try it out with pretend money in the portfolio simulator, no real funds involved.'
        '</div>',
        unsafe_allow_html=True
    )
    if st.button("Got it, thanks"):
        st.session_state.dismissed_welcome = True
        st.rerun()

st.markdown("---")


# PORTFOLIO SIMULATOR STATE - paper trading only, no real money. lives in
# session_state so it survives reruns, resets on Reset or new session

STARTING_BALANCE = 10_000.0

if "portfolio_cash" not in st.session_state:
    st.session_state.portfolio_cash = STARTING_BALANCE
if "portfolio_positions" not in st.session_state:
    st.session_state.portfolio_positions = {}  # {ticker: {"qty": float, "avg_price": float}}
if "portfolio_trade_log" not in st.session_state:
    st.session_state.portfolio_trade_log = []  # list of trade record dicts


# HELPER FUNCTIONS

def get_rsi_info(rsi: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for RSI value."""
    if rsi > 70:
        return "red", "Overbought", (
            f"RSI is {rsi:.0f} - this is above 70, which suggests the price "
            "may have risen too quickly and could be due for a pullback. "
            "Think of it like a spring stretched too far."
        )
    elif rsi < 30:
        return "green", "Oversold", (
            f"RSI is {rsi:.0f} - this is below 30, which suggests the price "
            "may have fallen too quickly and could be due for a bounce back. "
            "Think of it like a spring compressed too far."
        )
    else:
        return "amber", "Neutral", (
            f"RSI is {rsi:.0f} - this is in the neutral zone (30-70), "
            "meaning the price is not showing any extreme buying or selling pressure. "
            "No strong signal either way."
        )


def get_macd_info(macd: float, macd_signal: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for MACD."""
    if macd > macd_signal:
        return "green", "Bullish signal", (
            "MACD is showing a bullish crossover - short-term momentum is "
            "moving upward faster than the longer-term trend. "
            "This is generally considered a positive signal."
        )
    else:
        return "red", "Bearish signal", (
            "MACD is showing a bearish crossover - short-term momentum is "
            "slowing down relative to the longer-term trend. "
            "This is generally considered a cautionary signal."
        )


def get_bb_info(bb_pct: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for Bollinger Bands."""
    if bb_pct > 0.8:
        return "red", "Near upper band", (
            "The price is near the top of its normal trading range. "
            "This can suggest the price is relatively high compared to recent history "
            "and may face resistance."
        )
    elif bb_pct < 0.2:
        return "green", "Near lower band", (
            "The price is near the bottom of its normal trading range. "
            "This can suggest the price is relatively low compared to recent history "
            "and may find support."
        )
    else:
        return "amber", "Mid range", (
            "The price is within the middle of its normal trading range - "
            "not showing any extreme position relative to recent price history."
        )

def get_track_record_info(beat_hold: bool, is_profitable: bool,
                           test_return: float, test_buy_hold: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for how this
    coin's strategy performed on its own historical backtest - the same
    two checks the confidence gate itself uses."""
    detail = (
        f"On this coin's own test history, this strategy would have "
        f"returned {test_return:+.2f}%, compared to {test_buy_hold:+.2f}% "
        f"from simply buying and holding."
    )
    if beat_hold and is_profitable:
        return "green", "Beat buy-and-hold, and was profitable", (
            f"{detail} It did better than doing nothing, and it made money "
            "in its own right - the strongest track record the gate checks for."
        )
    elif beat_hold and not is_profitable:
        return "amber", "Beat buy-and-hold, but still lost money", (
            f"{detail} It lost less than just holding would have, but it "
            "was still a loss overall - worth knowing before treating this "
            "as a straightforward win."
        )
    elif not beat_hold and is_profitable:
        return "amber", "Profitable, but not better than holding", (
            f"{detail} It made money, but simply buying and holding would "
            "have made more."
        )
    else:
        return "red", "Did not beat buy-and-hold, and lost money", (
            f"{detail} Neither check passed - this is exactly the kind of "
            "case the confidence gate exists to catch."
        )


def get_decisiveness_info(q_values: dict, action: str) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for how clearly
    the DQN preferred its chosen action over the next-best alternative.
    Thresholds below are a starting heuristic (gap size relative to the
    spread across all three actions), not derived from a calibration
    study - treat as indicative, not precise."""
    ordered = sorted(q_values.values(), reverse=True)
    top, second = ordered[0], ordered[1]
    spread = max(q_values.values()) - min(q_values.values())
    margin_pct = (top - second) / spread if spread > 1e-9 else 0.0

    if margin_pct >= 0.3:
        return "green", "Clearly preferred", (
            f"{action} scored well ahead of the next-best option - the "
            "network had a clear preference, not a close call."
        )
    elif margin_pct >= 0.1:
        return "amber", "Moderately preferred", (
            f"{action} scored somewhat ahead of the next-best option - a "
            "reasonably clear preference, but not an overwhelming one."
        )
    else:
        return "red", "Close call", (
            f"{action} only narrowly beat the next-best option - the "
            "network saw little difference between the top choices here."
        )

def render_indicator(colour: str, title: str, explanation: str):
    """Render an indicator row with a coloured left accent bar."""
    st.markdown(
        f'<div class="indicator-row indicator-{colour}">'
        f'<div class="indicator-title">{title}</div>'
        f'<div class="indicator-explanation">{explanation}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def render_term(term: str, meaning: str):
    """Render a technical-term + plain-English-meaning line inside an adv-card."""
    st.markdown(
        f'<div class="adv-term">{term}</div>'
        f'<div class="adv-meaning">{meaning}</div>',
        unsafe_allow_html=True
    )


def render_metric_card(label: str, value: str, status: str,
                        change_text: str = "", sub_text: str = ""):
    """Render a stat tile (current price / prediction / 7-day change)."""
    parts = [f'<div class="metric-card {status}">']
    parts.append(f'<div class="metric-label">{label}</div>')
    parts.append(f'<div class="metric-value">{value}</div>')
    if change_text:
        parts.append(f'<div class="metric-change {status}">{change_text}</div>')
    if sub_text:
        parts.append(f'<div class="metric-sub">{sub_text}</div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def execute_buy(ticker: str, usd_amount: float, price: float) -> tuple[bool, str]:
    """Buy `usd_amount` worth of `ticker` at `price` using virtual cash.
    Returns (success, error_message)."""
    if usd_amount <= 0:
        return False, "Trade amount must be positive."
    if usd_amount > st.session_state.portfolio_cash:
        return False, "Trade amount exceeds your available cash balance."

    qty = usd_amount / price
    pos = st.session_state.portfolio_positions.get(ticker, {"qty": 0.0, "avg_price": 0.0})
    new_qty = pos["qty"] + qty
    # Weighted average entry price across this and any prior buys of the same coin
    new_avg_price = (pos["qty"] * pos["avg_price"] + usd_amount) / new_qty

    st.session_state.portfolio_positions[ticker] = {"qty": new_qty, "avg_price": new_avg_price}
    st.session_state.portfolio_cash -= usd_amount
    st.session_state.portfolio_trade_log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Coin": ticker,
        "Action": "BUY",
        "USD": round(usd_amount, 2),
        "Price": round(price, 2),
        "Qty": round(qty, 6),
        "Cash after": round(st.session_state.portfolio_cash, 2),
    })
    return True, ""


def execute_sell(ticker: str, qty_to_sell: float, price: float) -> tuple[bool, str]:
    """Sell `qty_to_sell` units of `ticker` at `price` from an existing position.
    Returns (success, error_message)."""
    pos = st.session_state.portfolio_positions.get(ticker)
    if not pos or pos["qty"] <= 0:
        return False, "You don't currently hold a position in this coin."
    if qty_to_sell <= 0:
        return False, "Sell quantity must be positive."
    if qty_to_sell > pos["qty"] + 1e-9:
        return False, "You can't sell more than you hold."

    proceeds = qty_to_sell * price
    st.session_state.portfolio_cash += proceeds

    remaining_qty = pos["qty"] - qty_to_sell
    if remaining_qty <= 1e-9:
        del st.session_state.portfolio_positions[ticker]
    else:
        # avg_price (cost basis) doesn't change on a sale - only qty shrinks
        st.session_state.portfolio_positions[ticker] = {
            "qty": remaining_qty, "avg_price": pos["avg_price"]
        }

    st.session_state.portfolio_trade_log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Coin": ticker,
        "Action": "SELL",
        "USD": round(proceeds, 2),
        "Price": round(price, 2),
        "Qty": round(qty_to_sell, 6),
        "Cash after": round(st.session_state.portfolio_cash, 2),
    })
    return True, ""


def execute_sell_all(ticker: str, price: float) -> tuple[bool, str]:
    """Liquidate the entire position in `ticker` at `price` - thin
    convenience wrapper around execute_sell()."""
    pos = st.session_state.portfolio_positions.get(ticker)
    if not pos:
        return False, "You don't currently hold a position in this coin."
    return execute_sell(ticker, pos["qty"], price)


def get_portfolio_total_value() -> float:
    """Cash plus the current market value of every held position, using
    each coin's cached live price (falls back to entry price if a live
    fetch fails for a coin that isn't the one currently selected)."""
    total = st.session_state.portfolio_cash
    for ticker, pos in st.session_state.portfolio_positions.items():
        try:
            price = get_prediction(ticker)["last_close"]
        except Exception:
            price = pos["avg_price"]
        total += pos["qty"] * price
    return total


# SIDEBAR

# final coin list - matches what crypto_model.py/rl_agent.py were actually
# trained on. dropped UNI (live data fetch kept failing) and DOT (LSTM
# prediction went wild, +174% in a day) after testing
COINS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "BNB (BNB)": "BNB-USD",
    "XRP (XRP)": "XRP-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "TRON (TRX)": "TRX-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Stellar (XLM)": "XLM-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Bitcoin Cash (BCH)": "BCH-USD",
}

# launch year + category per coin. year = founding/ICO year, not mainnet
# launch (TRON and Chainlink's mainnets came a bit later than this)
COIN_INFO = {
    "BTC-USD": {"launch_year": 2009, "category": "Digital currency"},
    "ETH-USD": {"launch_year": 2015, "category": "Smart contract platform"},
    "BNB-USD": {"launch_year": 2017, "category": "Smart contract / utility"},
    "XRP-USD": {"launch_year": 2012, "category": "Payments"},
    "SOL-USD": {"launch_year": 2020, "category": "Smart contract platform"},
    "ADA-USD": {"launch_year": 2017, "category": "Smart contract platform"},
    "DOGE-USD": {"launch_year": 2013, "category": "Digital currency / meme coin"},
    "TRX-USD": {"launch_year": 2017, "category": "Smart contract platform"},
    "LINK-USD": {"launch_year": 2017, "category": "Oracle network"},
    "AVAX-USD": {"launch_year": 2020, "category": "Smart contract platform"},
    "XLM-USD": {"launch_year": 2014, "category": "Payments"},
    "LTC-USD": {"launch_year": 2011, "category": "Digital currency"},
    "BCH-USD": {"launch_year": 2017, "category": "Digital currency"},
}

with st.sidebar:
    # sidebar = controls only (coin, mode, refresh). all the explanation
    # lives in the "About this tool" expander at the bottom instead
    st.caption("Powered by a trained AI prediction model")
    st.markdown("---")

    coin_label = st.selectbox(
        "Coin",
        options=list(COINS.keys()),
    )
    coin = COINS[coin_label]

    display_mode = st.radio(
        "Display mode",
        options=["Basic", "Simple", "Advanced"],
        index=1,
        horizontal=True,
    )
    IS_BASIC = display_mode == "Basic"
    IS_ADVANCED = display_mode == "Advanced"

    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    with st.expander("About this tool"):
        # same plain-English vs technical split as the main page - Basic/
        # Simple use plain terms, Advanced names the real objects
        if IS_ADVANCED:
            st.markdown("**How the system works**")
            st.caption(
                "Live OHLCV data is fetched per coin and converted into "
                "technical indicators (RSI, MACD, Bollinger %B, 1-day and "
                "7-day returns, volume ratio). A trained LSTM network reads "
                "a rolling 30-to-45-day window of those features and "
                "predicts tomorrow's closing price."
            )
            st.caption(
                "That prediction, plus the current indicators and portfolio "
                "state (position held, unrealised P&L), form a 9-value "
                "state vector. A Deep Q-Network "
                "(9 to 128 to 128 to 64 to 3, ReLU activations) trained via "
                "reinforcement learning outputs a Q-value for SELL, HOLD, "
                "and BUY; the highest Q-value becomes the agent's raw "
                "recommendation."
            )
            st.caption(
                "That raw recommendation then passes through a confidence "
                "gate before it reaches you: it's only trusted live if, on "
                "its own historical backtest, it both beat a simple "
                "buy-and-hold strategy and was itself profitable. If either "
                "check fails, the action shown is forced to HOLD instead, "
                "regardless of what the network picked."
            )

            st.markdown("**Coin selection**")
            st.caption(
                "13 coins were selected from the top cryptocurrencies by "
                "market capitalisation (stablecoins excluded) as of "
                "26 August 2026, then narrowed to coins with enough "
                "historical daily data to train and backtest a model on "
                "reliably. [Ranking source: CoinMarketCap]"
                "(https://coinmarketcap.com)"
            )

            st.markdown("**Display modes**")
            st.caption(
                "Basic shows only the action and its one-line reason. "
                "Simple adds price, prediction, chart, and the indicator "
                "panel in plain English. Advanced (this mode) additionally "
                "exposes the state vector, per-action Q-values, and "
                "confidence-gate pass/fail detail behind each call."
            )

            st.markdown("**Data refresh**")
            st.caption(
                "Predictions and recommendations are cached for 5 minutes "
                "per coin to avoid refetching on every rerun. Use "
                "Refresh data above to clear the cache and force a live "
                "refetch."
            )
        else:
            st.markdown("**How it works**")
            st.caption(
                "This tool watches live price and volume data for each "
                "coin and uses a trained AI model to estimate where the "
                "price might go next. That estimate is combined with a "
                "handful of market signals to suggest whether now looks "
                "like a good time to buy, hold, or sell, and the "
                "suggestion always comes with a plain-English reason, not "
                "just a number."
            )
            st.caption(
                "Before any suggestion is shown live, it's checked against "
                "how well it would actually have performed on that coin's "
                "past price history. If it wouldn't have made money, or "
                "would have done worse than simply buying and holding, the "
                "tool plays it safe and suggests HOLD instead."
            )

            st.markdown("**Coin list**")
            st.caption(
                "13 well-established, actively-traded cryptocurrencies, "
                "selected from the top cryptocurrencies by market "
                "capitalisation (stablecoins excluded) as of 26 August "
                "2026, then narrowed to the coins with enough reliable "
                "historical data to train and backtest on. "
                "[Ranking source: CoinMarketCap](https://coinmarketcap.com)"
            )

            st.markdown("**Display modes**")
            st.caption(
                "Basic = just the recommendation, nothing else. "
                "Simple = plain-English recommendations plus price, "
                "prediction, chart, and indicators. "
                "Advanced = same page as Simple, plus a look under the "
                "hood at how the AI actually made its decision."
            )

            st.markdown("**How to use this tool**")
            st.caption(
                "Select a coin above to see today's price, our model's "
                "prediction for tomorrow, and a plain-English explanation "
                "of what the technical indicators are saying."
            )

            st.markdown("**Data**")
            st.caption(
                "Data refreshes automatically every 5 minutes, or "
                "immediately via the Refresh data button above."
            )


# SHOW IF TEST DATA USED

if USE_TEST_DATA:
    st.info(
        "Hosted demonstration mode: this app is using "
        "bundled historical test data rather than live market data."
    )


# FETCH DATA

try:
    with st.spinner(f"Fetching live data for {coin_label}..."):
        data = get_prediction(coin)
except FileNotFoundError:
    st.error(
        f"No trained model found for {coin}. "
        "Please run crypto_model.py first to train the models."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.stop()


# MAIN HEADER

st.markdown('<div class="eyebrow">AI Trading Analysis</div>', unsafe_allow_html=True)
st.title(f"{coin_label} Analysis")
st.caption(f"Live data as of {data['last_date']} · Predictions are estimates only")

info = COIN_INFO.get(coin)
if info:
    st.markdown(
        f'<div class="tag-row">'
        f'<span class="tag">{info["category"]}</span>'
        f'<span class="tag">Launched {info["launch_year"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ROW 1: CURRENT PRICE + PREDICTION - Basic just shows one price tile,
# Simple/Advanced get the full 3-column layout

if IS_BASIC:
    st.markdown("### Current Price")
    render_metric_card("Current Price", f"${data['last_close']:,.2f}", "neutral")
    st.markdown("---")
else:
    col_price, col_pred, col_7d = st.columns(3)

    with col_price:
        st.markdown("### Current Price")
        render_metric_card(
            "Current Price", f"${data['last_close']:,.2f}", "neutral",
            sub_text="Latest closing price from Yahoo Finance",
        )

    with col_pred:
        st.markdown("### Tomorrow's Prediction")
        direction = data["direction"]
        change = data["change_pct"]
        price = data["predicted"]
        status = "positive" if direction == "UP" else "negative"
        arrow = "▲" if direction == "UP" else "▼"

        render_metric_card(
            "Predicted Price", f"${price:,.2f}", status,
            change_text=f"{arrow} {abs(change):.2f}% predicted {direction}",
            sub_text="Based on 30-60 days of historical patterns",
        )

    with col_7d:
        st.markdown("### Last 7 Days")
        ret = data["ret_7d"]
        status = "positive" if ret > 0 else "negative" if ret < 0 else "neutral"
        arrow7 = "▲" if ret > 0 else "▼" if ret < 0 else "-"

        render_metric_card(
            "7-Day Change", f"{arrow7} {abs(ret):.2f}%", status,
            change_text=f"{'Gained' if ret > 0 else 'Lost'} over the last 7 days",
            sub_text="Recent price momentum",
        )

    st.markdown("---")

# ROW 2: RL RECOMENDATION

st.markdown("### AI Trading Recommendation")
if not IS_BASIC:
    st.caption(
        "This recommendation is generated by a Deep Q-Network (DQN) reinforcement "
        "learning agent trained on historical trading data. It combines the LSTM "
        "price prediction with technical indicators to suggest an action."
    )

with st.spinner("Getting AI recommendation..."):
    decision = get_rl_decision(coin)

action = decision["action"]
q_values = decision.get("q_values")  # only present if rl_agent.py's q_values patch is applied
raw_action = decision.get("raw_action")  # only present if the confidence gate downgraded this call
beat_hold = decision.get("beat_buy_hold")  # only present if apply_confidence_gate() ran
is_profitable = decision.get("is_profitable")
test_return = decision.get("test_return")
test_buy_hold = decision.get("test_buy_hold")
rec_css = {"BUY": "buy", "HOLD": "hold", "SELL": "sell"}[action]

st.markdown(
    f'<div class="recommendation-card {rec_css}">'
    f'<div class="action-pill {rec_css}">{action}</div>'
    f'<div class="rec-explanation">{decision["explanation"]}</div>'
    f'</div>',
    unsafe_allow_html=True
)

# shown in Simple/Advanced only - says if the confidence gate downgraded
# this call. left out of Basic on purpose, that mode stays minimal
if not IS_BASIC and raw_action and raw_action != action:
    st.caption(
        f"The model's initial signal was **{raw_action}**, but it was "
        f"downgraded to **{action}** by a safety check on this coin's past "
        f"backtest performance (see below for details in Advanced mode)."
    )


# CONFIDENCE BLOCK - plain-English view of the same numbers the Advanced "Confidence gate" panel shows in code terms (state vector / Q-values /
# gate). Lives here in Simple + Advanced, not Advanced-only, since how
# much to trust a call matters even if you never open Model Internals.

if not IS_BASIC:
    st.markdown("**How much should you trust this?**")
    st.caption(
        "Two independent checks behind the recommendation above: how it "
        "actually performed on past data, and how strongly the AI "
        "preferred this action over the alternatives."
    )

    if test_return is not None:
        track_colour, track_label, track_explanation = get_track_record_info(
            beat_hold, is_profitable, test_return, test_buy_hold
        )
        render_indicator(track_colour, f"Past performance - {track_label}", track_explanation)
    else:
        st.caption(
            "Past-performance check isn't wired into this live view yet - "
            "see the note under Model Internals in Advanced mode."
        )

    if q_values:
        conf_colour, conf_label, conf_explanation = get_decisiveness_info(q_values, action)
        render_indicator(conf_colour, f"AI's decisiveness - {conf_label}", conf_explanation)
    else:
        st.caption(
            "Decisiveness check isn't available yet - see the note under "
            "Model Internals in Advanced mode."
        )

    st.markdown("")

st.markdown("---")


# PORTFOLIO SIMULATOR - paper trading with virtual cash so the user can
# act on the recommendation and see what happens. no real money involved

if not IS_BASIC:
    st.markdown("### Your Portfolio")
    st.caption(
        "Practice trading with a virtual balance. No real money is used - "
        "this is for exploring how following, or ignoring, the AI's "
        "recommendation would play out."
    )

    current_price = data["last_close"]
    position = st.session_state.portfolio_positions.get(coin)
    position_qty = position["qty"] if position else 0.0
    position_value = position_qty * current_price
    position_pnl = (
        (current_price - position["avg_price"]) / position["avg_price"] * 100
        if position and position["avg_price"] > 0 else 0.0
    )
    total_value = get_portfolio_total_value()
    overall_return = (total_value - STARTING_BALANCE) / STARTING_BALANCE * 100

    pf_col1, pf_col2, pf_col3 = st.columns(3)
    with pf_col1:
        render_metric_card(
            "Cash Balance", f"${st.session_state.portfolio_cash:,.2f}", "neutral",
        )
    with pf_col2:
        pos_status = "positive" if position_pnl > 0 else "negative" if position_pnl < 0 else "neutral"
        render_metric_card(
            f"{coin_label} Position",
            f"${position_value:,.2f}" if position else "$0.00",
            pos_status,
            change_text=f"{position_pnl:+.2f}% unrealised" if position else "No position held",
            sub_text=f"{position_qty:.6f} {coin.split('-')[0]}" if position else "",
        )
    with pf_col3:
        total_status = "positive" if overall_return > 0 else "negative" if overall_return < 0 else "neutral"
        render_metric_card(
            "Total Portfolio Value", f"${total_value:,.2f}", total_status,
            change_text=f"{overall_return:+.2f}% since start",
            sub_text=f"Started at ${STARTING_BALANCE:,.2f}",
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    trade_col1, trade_col2 = st.columns([2, 1])
    with trade_col1:
        max_tradeable = max(st.session_state.portfolio_cash, 10.0)
        trade_amount = st.number_input(
            f"Amount to buy of {coin_label} (USD)",
            min_value=10.0,
            max_value=max_tradeable,
            value=min(500.0, max_tradeable),
            step=50.0,
        )
    with trade_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        buy_clicked = st.button(
            "Buy", width="stretch",
            disabled=st.session_state.portfolio_cash < 10.0,
        )

    if buy_clicked:
        ok, err = execute_buy(coin, trade_amount, current_price)
        if ok:
            st.rerun()
        else:
            st.error(err)

    # sell row - same pattern as buy row above, disabled if no position held
    sell_col1, sell_col2 = st.columns([2, 1])
    with sell_col1:
        max_sellable = max(position_value, 10.0)
        sell_amount = st.number_input(
            f"Amount to sell of {coin_label} (USD)",
            min_value=10.0,
            max_value=max_sellable,
            value=min(500.0, max_sellable),
            step=50.0,
        )
    with sell_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        sell_clicked = st.button(
            "Sell", width="stretch",
            disabled=position_qty <= 0,
        )

    if sell_clicked:
        qty_to_sell = min(sell_amount / current_price, position_qty)
        ok, err = execute_sell(coin, qty_to_sell, current_price)
        if ok:
            st.rerun()
        else:
            st.error(err)

    # flag when the user's own position disagrees with the current AI call
    if position and action == "SELL":
        st.caption(
            f"Note: you are holding {coin_label}, but the AI's current "
            f"recommendation is SELL."
        )
    elif not position and action == "BUY":
        st.caption(
            f"Note: you have no position in {coin_label}, but the AI's "
            f"current recommendation is BUY."
        )

    if st.session_state.portfolio_trade_log:
        with st.expander("Trade log"):
            log_df = pd.DataFrame(list(reversed(st.session_state.portfolio_trade_log)))
            st.dataframe(log_df, hide_index=True, width="stretch")

    if st.button("Reset portfolio", width="stretch"):
        st.session_state.portfolio_cash = STARTING_BALANCE
        st.session_state.portfolio_positions = {}
        st.session_state.portfolio_trade_log = []
        st.rerun()

    st.markdown("---")


# ADVANCED: MODEL INTERNALS - more technical terms (explained next to each
# one), plus real charts for Q-network output and indicator history

if IS_ADVANCED:
    st.markdown("### Model Internals")
    st.caption(
        "This section shows the actual technical objects the recommendation "
        "above comes from, in the real terminology used in the code - each "
        "one explained in plain English right next to it."
    )

    st.markdown('<div class="adv-card">', unsafe_allow_html=True)
    render_term(
        "State vector",
        "The 9-value input the reinforcement learning agent actually sees at "
        "decision time. It combines the LSTM's price forecast with technical "
        "indicators and the current position - this is what gets fed into the "
        "neural network below, not raw price data."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    state_rows = [
        ("Predicted change", f"{data['change_pct']:+.2f}%",
         "How much the LSTM expects the price to move by tomorrow."),
        ("RSI", f"{data['rsi']:.1f}",
         "Relative Strength Index (0-100). Above 70 = overbought, below 30 = oversold."),
        ("MACD minus signal", f"{data['macd'] - data['macd_sig']:+.3f}",
         "Difference between the MACD line and its signal line. Positive = bullish crossover."),
        ("Bollinger %B", f"{data['bb_pct']:.2f}",
         "Price's position within its recent volatility range. 0 = lower band, 1 = upper band."),
        ("1-day return", f"{data['ret_1d']:+.2f}%",
         "Percentage price change over the last 24 hours."),
        ("7-day return", f"{data['ret_7d']:+.2f}%",
         "Percentage price change over the last 7 days."),
        ("Volume ratio", f"{data['vol_ratio']:.2f}x",
         "Today's trading volume relative to its recent average. Above 1x = unusually high activity."),
        ("Position", "Flat (0)",
         "Whether the agent currently holds the coin. Live recommendations always assume no open position."),
        ("Unrealised P&L", "0.00%",
         "Profit/loss on an open position. Always 0% here since position is assumed flat."),
    ]
    state_df = pd.DataFrame(state_rows, columns=["Feature", "Value", "What it means"])
    st.dataframe(state_df, hide_index=True, width="stretch")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="adv-card">', unsafe_allow_html=True)
    render_term(
        "Q-value / DQN (Deep Q-Network)",
        "A Q-value estimates the expected future reward of taking a given "
        "action (SELL, HOLD, or BUY) from the current state. The network "
        "outputs one Q-value per action; the agent's recommendation is "
        "simply whichever action has the highest Q-value - this is called "
        "taking the argmax."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if q_values:
        q_df = pd.DataFrame(
            {"Q-value": [q_values["SELL"], q_values["HOLD"], q_values["BUY"]]},
            index=["SELL", "HOLD", "BUY"],
        )
        st.bar_chart(q_df, height=220)
        st.caption(
            f"Highest Q-value corresponds to **{action}** (this is the agent's recommendation above)."
        )
    else:
        st.caption(
            "Per-action Q-values aren't exposed by get_trading_decision() yet - "
            "it currently returns only the chosen action, not the Q-value behind "
            "each option. Add a `q_values` field there to light up this chart."
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="adv-card">', unsafe_allow_html=True)
    render_term(
        "Confidence gate (backtest validation)",
        "This coin's agent is only trusted live if its historical test "
        "backtest (a) beat a simple buy-and-hold strategy, and (b) was "
        "itself profitable. If either check fails, the recommendation is "
        "forced to HOLD regardless of what the network picked."
    )

    if test_return is not None:
        gate_css = lambda ok: "pass" if ok else "fail"
        gate_status = lambda ok: "PASS" if ok else "FAIL"

        chip_col1, chip_col2 = st.columns(2)
        with chip_col1:
            st.markdown(
                f'<div class="gate-chip {gate_css(beat_hold)}">'
                f'<div class="gate-chip-title">Beat buy-and-hold '
                f'<span class="gate-chip-status">{gate_status(beat_hold)}</span></div>'
                f'<div class="gate-chip-detail">'
                f'{test_return:+.2f}% vs {test_buy_hold:+.2f}% buy-and-hold</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with chip_col2:
            st.markdown(
                f'<div class="gate-chip {gate_css(is_profitable)}">'
                f'<div class="gate-chip-title">Profitable on test '
                f'<span class="gate-chip-status">{gate_status(is_profitable)}</span></div>'
                f'<div class="gate-chip-detail">'
                f'Test-period return: {test_return:+.2f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        if raw_action and raw_action != action:
            st.caption(
                f"Gate failed - raw signal ({raw_action}) downgraded to HOLD for this coin right now."
            )
        else:
            st.caption("Gate passed - the recommendation above is the agent's real signal.")
    else:
        st.caption(
            "Per-coin gate results aren't wired into this live view yet - "
            "`get_trading_decision()` currently returns the agent's raw, "
            "unvalidated signal. Call `apply_confidence_gate()` there with "
            "this coin's test-backtest numbers to light up the checks above."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")


# ROW 3: CHART + INDICATORS, INDICATOR EXPLAINER, RSI HISTORY, plus the
# Advanced-only charts - all skipped in Basic mode

if not IS_BASIC:
    col_chart, col_indicators = st.columns([2, 1])

    with col_chart:
        st.markdown("### Price History (Last 90 Days)")
        chart_df = data["df"][["Close"]].copy()
        chart_df.columns = [f"{coin_label} Price (USD)"]
        st.line_chart(chart_df, height=340)
        st.caption(
            "This chart shows the actual closing price each day for the last 90 days. "
            "The model uses these patterns to make its prediction."
        )

    with col_indicators:
        st.markdown("### What the Indicators Say")
        st.caption("These are signals used by traders to understand market momentum.")

        # RSI
        rsi_colour, rsi_label, rsi_explanation = get_rsi_info(data["rsi"])
        render_indicator(rsi_colour, f"RSI - {rsi_label}", rsi_explanation)

        st.markdown("")

        # MACD
        macd_colour, macd_label, macd_explanation = get_macd_info(
            data["macd"], data["macd_sig"]
        )
        render_indicator(macd_colour, f"MACD - {macd_label}", macd_explanation)

        st.markdown("")

        # Bollinger Bands
        bb_colour, bb_label, bb_explanation = get_bb_info(data["bb_pct"])
        render_indicator(bb_colour, f"Bollinger Bands - {bb_label}", bb_explanation)

    st.markdown("---")


    # INDICATOR EXPLAINER

    with st.expander("What do these indicators mean?"):
        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            st.markdown("**RSI (Relative Strength Index)**")
            st.write(
                "RSI measures how fast the price has been moving. "
                "It goes from 0 to 100. Above 70 means the coin may be "
                "rising too fast (overbought). Below 30 means it may be "
                "falling too fast (oversold). Between 30 and 70 is neutral."
            )

        with exp2:
            st.markdown("**MACD**")
            st.write(
                "MACD compares two moving averages of the price to detect "
                "momentum shifts. A bullish signal means short-term momentum "
                "is picking up. A bearish signal means it is slowing down. "
                "Think of it as detecting whether the coin is gaining or losing speed."
            )

        with exp3:
            st.markdown("**Bollinger Bands**")
            st.write(
                "Bollinger Bands show a normal trading range around the price. "
                "When the price is near the upper band it may be relatively "
                "expensive compared to recent history. Near the lower band "
                "it may be relatively cheap. Mid range means nothing unusual."
            )

    st.markdown("---")


    # RSI HISTORY CHART

    with st.expander("Show RSI history (last 90 days)"):
        rsi_df = data["df"][["RSI"]].copy()
        st.line_chart(rsi_df, height=200)
        st.caption(
            "The RSI line moving above 70 indicates overbought conditions. "
            "Below 30 indicates oversold. Between 30 and 70 is the neutral zone."
        )


    # ADVANCED: MORE HISTORY CHARTS - same pattern as RSI above, other 2
    # indicators, Advanced mode only

    if IS_ADVANCED:
        with st.expander("Show MACD history (last 90 days)"):
            macd_df = data["df"][["MACD", "MACD_Signal"]].copy()
            st.line_chart(macd_df, height=200)
            st.caption(
                "MACD (blue) crossing above its signal line (orange) is a bullish "
                "crossover; crossing below is bearish. The gap between the two "
                "lines is the MACD histogram's magnitude."
            )

        with st.expander("Show Bollinger Band %B history (last 90 days)"):
            bb_df = data["df"][["BB_Pct"]].copy()
            st.line_chart(bb_df, height=200)
            st.caption(
                "%B above 0.8 means price is trading near the upper band; below "
                "0.2 means it's near the lower band. This is the same %B value "
                "shown in the state vector table above."
            )


# MARKET INSIGHTS - exploratory analysis + cross-coin comparison, about
# the data/model's track record rather than what to do. skipped in Basic

if not IS_BASIC:
    st.markdown("---")
    st.markdown("### Market Insights")
    st.caption(
        f"A closer look at the data behind the recommendation above: how "
        f"{coin_label}'s price has actually behaved, and how it compares "
        f"to the other coins this tool covers."
    )

    insight_tab1, insight_tab2 = st.tabs([f"{coin_label} data", "Compare all coins"])

    with insight_tab1:
        eda_df = data["df"].copy()
        eda_df["Return_1d_pct"] = eda_df["Return_1d"] * 100

        eda_col1, eda_col2 = st.columns(2)

        with eda_col1:
            st.markdown("**Daily return distribution**")
            st.caption(
                f"How often {coin_label} moves by a given amount in a "
                "single day, over the last 90 days. A wide, flat spread "
                "means big swings happen often; a tall narrow peak near "
                "zero means most days are quiet."
            )
            returns = eda_df["Return_1d_pct"].dropna()
            if len(returns) > 5:
                binned = pd.cut(returns, bins=12)
                hist_counts = binned.value_counts().sort_index()
                hist_df = pd.DataFrame(
                    {"Days": hist_counts.values},
                    index=[f"{iv.left:.1f} to {iv.right:.1f}%" for iv in hist_counts.index],
                )
                st.bar_chart(hist_df, height=220)
            else:
                st.caption("Not enough data yet to build a distribution.")

        with eda_col2:
            st.markdown("**Rolling volatility (7-day)**")
            st.caption(
                "Standard deviation of daily returns over a trailing "
                "7-day window. Higher means choppier, less predictable "
                "price action."
            )
            rolling_vol = eda_df["Return_1d_pct"].rolling(7).std()
            st.line_chart(rolling_vol, height=220)

        st.markdown("**Which indicators actually move with next-day price?**")
        st.caption(
            "Correlation between each indicator's value today and the "
            "price change that follows the next day. Values near +1 or -1 "
            "mean the indicator and next-day return tend to move together "
            "(or opposite); values near 0 mean little relationship. This "
            "is a simple lag-1 correlation, not a claim of causation."
        )
        indicator_cols = [
            "RSI", "MACD", "MACD_Signal", "BB_Pct",
            "Return_1d", "Return_7d", "Volume_Ratio",
        ]
        # one-line, plain-English description of what each indicator actually
        # measures - shown alongside its correlation value so the table is
        # readable without prior technical-analysis knowledge
        INDICATOR_DESCRIPTIONS = {
            "RSI": "Relative Strength Index (0-100) - how overbought or "
                   "oversold the coin looks based on recent price moves.",
            "MACD": "Gap between a fast and slow moving average - a "
                    "positive/negative swing signals building momentum.",
            "MACD_Signal": "A smoothed average of the MACD line itself, "
                            "used to spot MACD crossovers.",
            "BB_Pct": "Bollinger Band %B - where price sits within its "
                      "recent volatility band (near 0 = lower band, near "
                      "1 = upper band).",
            "Return_1d": "The coin's own percentage price change over the "
                         "last 1 day.",
            "Return_7d": "The coin's own percentage price change over the "
                         "last 7 days.",
            "Volume_Ratio": "Today's trading volume versus its recent "
                            "average - above 1 means busier than usual.",
        }
        next_day_return = eda_df["Return_1d"].shift(-1)
        corr_rows = []
        for col in indicator_cols:
            if col in eda_df.columns:
                corr_rows.append({
                    "Indicator": col,
                    "What it measures": INDICATOR_DESCRIPTIONS.get(col, ""),
                    "Correlation with next-day return": round(
                        float(eda_df[col].corr(next_day_return)), 3
                    ),
                })
        if corr_rows:
            corr_df = pd.DataFrame(corr_rows).sort_values(
                "Correlation with next-day return", key=abs, ascending=False
            )
            st.dataframe(corr_df, hide_index=True, width="stretch")

    with insight_tab2:
        st.caption(
            "Pulls live data for all 13 coins to compare them side by "
            "side. Not run automatically since it means re-fetching every "
            "coin at once - click below when you want a fresh comparison."
        )
        if st.button("Load cross-coin comparison"):
            st.session_state.cross_coin_loaded = True

        if st.session_state.get("cross_coin_loaded"):
            with st.spinner("Fetching data for all coins..."):
                returns_by_coin = {}
                volatility_rows = []
                backtest_rows = []
                for label, sym in COINS.items():
                    try:
                        coin_data = get_prediction(sym)
                        coin_df = coin_data["df"]
                        returns_by_coin[label] = coin_df["Return_1d"]
                        volatility_rows.append({
                            "Coin": label,
                            "Daily volatility (std)": round(
                                float(coin_df["Return_1d"].std()) * 100, 2
                            ),
                        })
                    except Exception:
                        continue
                    try:
                        coin_decision = get_rl_decision(sym)
                        if coin_decision.get("test_return") is not None:
                            backtest_rows.append({
                                "Coin": label,
                                "Backtest return": coin_decision["test_return"],
                                "Buy and hold return": coin_decision["test_buy_hold"],
                                "Beat buy and hold": (
                                    "Yes" if coin_decision.get("beat_buy_hold") else "No"
                                ),
                            })
                    except Exception:
                        continue

            loaded_n = len(returns_by_coin)
            if loaded_n < len(COINS):
                st.caption(
                    f"Loaded {loaded_n} of {len(COINS)} coins - the rest "
                    "don't have trained models available right now."
                )

            if loaded_n >= 2:
                st.markdown("**How closely do these coins move together?**")
                st.caption(
                    "Correlation of daily returns between coins, aligned "
                    "on shared trading days. Close to 1 means two coins "
                    "tend to rise and fall together; close to 0 means they "
                    "move independently."
                )
                returns_matrix = pd.DataFrame(returns_by_coin).corr()
                st.dataframe(returns_matrix.round(2), width="stretch")

            if volatility_rows:
                st.markdown("**Volatility, ranked**")
                st.caption(
                    "Standard deviation of daily returns over the last 90 "
                    "days - higher means bigger, choppier price swings."
                )
                vol_df = pd.DataFrame(volatility_rows).sort_values(
                    "Daily volatility (std)", ascending=False
                )
                st.bar_chart(vol_df.set_index("Coin"), height=280)

            if backtest_rows:
                st.markdown("**Model backtest performance, by coin**")
                st.caption(
                    "How each coin's RL agent performed on its own "
                    "historical test period, versus simply buying and "
                    "holding. This is the same check behind the "
                    "confidence gate in Advanced mode."
                )
                bt_df = pd.DataFrame(backtest_rows).sort_values(
                    "Backtest return", ascending=False
                )
                st.dataframe(bt_df, hide_index=True, width="stretch")
            else:
                st.caption(
                    "No backtest results found yet for any coin - run "
                    "rl_agent.py's training to generate them."
                )


# DISCLAIMER

st.markdown(
    '<div class="notice-box">'
    '<strong>Important:</strong> This tool is for educational purposes only. '
    'Predictions are generated by a machine learning model and are not guaranteed. '
    'Cryptocurrency markets are highly volatile. '
    'Always do your own research and never invest more than you can afford to lose. '
    'This is not financial advice.'
    '</div>',
    unsafe_allow_html=True
)
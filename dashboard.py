# pip install streamlit
# run: 

# emojis used for styling: https://emojidb.org/stock-emojis


# to run: streamlit run dashboard.py

"""
Documentation used:
Streamlit: https://docs.streamlit.io/develop/api-reference

Requires:
    predict.py
    BTC_USD_model.pt / BTC_USD_scaler.pkl
    ETH_USD_model.pt / ETH_USD_scaler.pkl
    BNB_USD_model.pt / BNB_USD_scaler.pkl
"""



import streamlit as st
import pandas as pd
import os
from datetime import datetime
from predict import fetch_live_data, add_features, load_model, predict, FEATURE_COLUMNS

# PAGE CONFIG 
st.set_page_config(
    page_title="Crypto Financial Advisor",
    page_icon="📊",
    layout="wide",
)

# SHOW IF TEST DATA USED

if os.environ.get(
    "USE_TEST_DATA",
    "false",
).lower() in {"1", "true", "yes", "on"}:
    st.info(
        "🧪 Hosted demonstration mode: this app is using "
        "bundled historical test data rather than live market data."
    )
# CUSTOM STYLING

st.markdown("""
<style>
    /* Traffic light indicator boxes */
    .indicator-green {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .indicator-amber {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .indicator-red {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .indicator-title {
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 4px;
    }
    .indicator-explanation {
        font-size: 13px;
        color: #555;
    }
    .prediction-up {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-down {
        background-color: #f8d7da;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-neutral {
        background-color: #fff3cd;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .big-price {
        font-size: 32px;
        font-weight: 700;
    }
    .disclaimer-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #6c757d;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# HELPER FUNCTIONS

def get_rsi_info(rsi: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for RSI value."""
    if rsi > 70:
        return "red", "🔴 Overbought", (
            f"RSI is {rsi:.0f} - this is above 70, which suggests the price "
            "may have risen too quickly and could be due for a pullback. "
            "Think of it like a spring stretched too far."
        )
    elif rsi < 30:
        return "green", "🟢 Oversold", (
            f"RSI is {rsi:.0f} - this is below 30, which suggests the price "
            "may have fallen too quickly and could be due for a bounce back. "
            "Think of it like a spring compressed too far."
        )
    else:
        return "amber", "🟡 Neutral", (
            f"RSI is {rsi:.0f} - this is in the neutral zone (30–70), "
            "meaning the price is not showing any extreme buying or selling pressure. "
            "No strong signal either way."
        )


def get_macd_info(macd: float, macd_signal: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for MACD."""
    if macd > macd_signal:
        return "green", "📈 Bullish signal", (
            "MACD is showing a bullish crossover - short-term momentum is "
            "moving upward faster than the longer-term trend. "
            "This is generally considered a positive signal."
        )
    else:
        return "red", "📉 Bearish signal", (
            "MACD is showing a bearish crossover - short-term momentum is "
            "slowing down relative to the longer-term trend. "
            "This is generally considered a cautionary signal."
        )


def get_bb_info(bb_pct: float) -> tuple[str, str, str]:
    """Returns (colour, label, plain English explanation) for Bollinger Bands."""
    if bb_pct > 0.8:
        return "red", "🔴 Near upper band", (
            "The price is near the top of its normal trading range. "
            "This can suggest the price is relatively high compared to recent history "
            "and may face resistance."
        )
    elif bb_pct < 0.2:
        return "green", "🟢 Near lower band", (
            "The price is near the bottom of its normal trading range. "
            "This can suggest the price is relatively low compared to recent history "
            "and may find support."
        )
    else:
        return "amber", "🟡 Mid range", (
            "The price is within the middle of its normal trading range - "
            "not showing any extreme position relative to recent price history."
        )


def render_indicator(colour: str, title: str, explanation: str):
    """Render a traffic light indicator box."""
    st.markdown(
        f'<div class="indicator-{colour}">'
        f'<div class="indicator-title">{title}</div>'
        f'<div class="indicator-explanation">{explanation}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# CACHED DATA FETCH 

@st.cache_data(ttl=300, show_spinner=False)
def get_prediction(symbol: str) -> dict:
    """Fetch live data, run model, return everything the dashboard needs."""
    raw = fetch_live_data(symbol, days=90)
    df  = add_features(raw)
    model, scaler = load_model(symbol)

    last_close = float(df["Close"].iloc[-1])
    next_price = predict(model, df, scaler)
    change_pct = (next_price - last_close) / last_close * 100

    rsi      = float(df["RSI"].iloc[-1])
    macd     = float(df["MACD"].iloc[-1])
    macd_sig = float(df["MACD_Signal"].iloc[-1])
    bb_pct   = float(df["BB_Pct"].iloc[-1])
    ret_7d   = float(df["Return_7d"].iloc[-1]) * 100

    return {
        "df":          df,
        "last_close":  last_close,
        "predicted":   next_price,
        "change_pct":  change_pct,
        "direction":   "UP" if change_pct > 0 else "DOWN",
        "rsi":         rsi,
        "macd":        macd,
        "macd_sig":    macd_sig,
        "bb_pct":      bb_pct,
        "ret_7d":      ret_7d,
        "last_date":   df.index[-1].strftime("%d %B %Y"),
    }


# SIDEBAR 

COINS = {
    "Bitcoin (BTC)":   "BTC-USD",
    "Ethereum (ETH)":  "ETH-USD",
    "Binance Coin (BNB)": "BNB-USD",
}

with st.sidebar:
    st.title("📊 Crypto Advisor")
    st.caption("Powered by a trained AI prediction model")
    st.markdown("---")

    coin_label = st.selectbox(
        "Which coin would you like to analyse?",
        options=list(COINS.keys()),
    )
    coin = COINS[coin_label]

    st.markdown("---")
    st.markdown("**ℹ️ How to use this tool**")
    st.caption(
        "Select a coin above to see today's price, "
        "our model's prediction for tomorrow, and "
        "a simple explanation of what the technical "
        "indicators are saying."
    )

    st.markdown("---")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Data refreshes every 5 minutes")


# FETCH DATA 

try:
    with st.spinner(f"Fetching live data for {coin_label}..."):
        data = get_prediction(coin)
except FileNotFoundError:
    st.error(
        f"❌ No trained model found for {coin}. "
        "Please run crypto_model.py first to train the models."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Could not fetch data: {e}")
    st.stop()


# MAIN HEADER 

st.title(f"📊 {coin_label} Analysis")
st.caption(f"Live data as of {data['last_date']} · Predictions are estimates only")
st.markdown("---")


# ROW 1: CURRENT PRICE + PREDICTION 

col_price, col_pred, col_7d = st.columns(3)

with col_price:
    st.markdown("### 💰 Current Price")
    st.markdown(
        f'<div class="prediction-neutral">'
        f'<div class="big-price">${data["last_close"]:,.2f}</div>'
        f'<div style="font-size:13px;color:#555;margin-top:6px">'
        f'Latest closing price from Yahoo Finance</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col_pred:
    st.markdown("### 🔮 Tomorrow's Prediction")
    direction = data["direction"]
    change    = data["change_pct"]
    price     = data["predicted"]
    css_class = "prediction-up" if direction == "UP" else "prediction-down"
    arrow     = "▲" if direction == "UP" else "▼"
    colour    = "#28a745" if direction == "UP" else "#dc3545"

    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="big-price">${price:,.2f}</div>'
        f'<div style="font-size:18px;color:{colour};font-weight:600;margin-top:6px">'
        f'{arrow} {abs(change):.2f}% predicted {direction}</div>'
        f'<div style="font-size:12px;color:#555;margin-top:4px">'
        f'Based on 30–60 days of historical patterns</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col_7d:
    st.markdown("### 📅 Last 7 Days")
    ret = data["ret_7d"]
    css = "prediction-up" if ret > 0 else "prediction-down" if ret < 0 else "prediction-neutral"
    arrow7 = "▲" if ret > 0 else "▼" if ret < 0 else "—"
    col7 = "#28a745" if ret > 0 else "#dc3545" if ret < 0 else "#6c757d"
    st.markdown(
        f'<div class="{css}">'
        f'<div class="big-price">{arrow7} {abs(ret):.2f}%</div>'
        f'<div style="font-size:13px;color:{col7};font-weight:600;margin-top:6px">'
        f'{"Gained" if ret > 0 else "Lost"} over the last 7 days</div>'
        f'<div style="font-size:12px;color:#555;margin-top:4px">'
        f'Recent price momentum</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ROW 2: CHART + INDICATORS 

col_chart, col_indicators = st.columns([2, 1])

with col_chart:
    st.markdown("### 📈 Price History (Last 90 Days)")
    chart_df = data["df"][["Close"]].copy()
    chart_df.columns = [f"{coin_label} Price (USD)"]
    st.line_chart(chart_df, height=340)
    st.caption(
        "This chart shows the actual closing price each day for the last 90 days. "
        "The model uses these patterns to make its prediction."
    )

with col_indicators:
    st.markdown("### 🚦 What the Indicators Say")
    st.caption("These are signals used by traders to understand market momentum.")

    # RSI
    rsi_colour, rsi_label, rsi_explanation = get_rsi_info(data["rsi"])
    render_indicator(rsi_colour, f"RSI — {rsi_label}", rsi_explanation)

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

with st.expander("📚 What do these indicators mean? (Click to learn more)"):
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

with st.expander("📊 Show RSI history (last 90 days)"):
    rsi_df = data["df"][["RSI"]].copy()
    st.line_chart(rsi_df, height=200)
    st.caption(
        "The RSI line moving above 70 indicates overbought conditions. "
        "Below 30 indicates oversold. Between 30 and 70 is the neutral zone."
    )


# DISCLAIMER 

st.markdown(
    '<div class="disclaimer-box">'
    '⚠️ <strong>Important:</strong> This tool is for educational purposes only. '
    'Predictions are generated by a machine learning model and are not guaranteed. '
    'Cryptocurrency markets are highly volatile. '
    'Always do your own research and never invest more than you can afford to lose. '
    'This is not financial advice.'
    '</div>',
    unsafe_allow_html=True
)
# pip install streamlit
# run: streamlit run dashboard.py
# emojis used for styling: https://emojidb.org/stock-emojis

"""
Documentation used:
Streamlit: https://docs.streamlit.io/develop/api-reference
"""

"""
Requires:
    predict.py
    BTC_USD_model.pt / BTC_USD_scaler.pkl
    ETH_USD_model.pt / ETH_USD_scaler.pkl
    BNB_USD_model.pt / BNB_USD_scaler.pkl
"""


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from predict import (
    fetch_live_data,
    add_features,
    load_model,
    predict,
    FEATURE_COLUMNS,
)
 
#  PAGE CONFIG 
st.set_page_config(
    page_title="Crypto Financial Advisor",
    page_icon="📊",
    layout="wide",
)
 
# SIDEBAR 
with st.sidebar:
    st.title("Crypto Advisor")
    st.markdown("---")
 
    coin = st.selectbox(
        "Select a coin",
        ["BTC-USD", "ETH-USD", "BNB-USD"],
        index=0,
    )
 
    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "This dashboard uses a trained PyTorch LSTM model to predict "
        "next-day cryptocurrency closing prices. Data is fetched live "
        "from Yahoo Finance."
    )
 
    st.markdown("---")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
 
    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
 
# DATA FETCHING (CACHED) 
@st.cache_data(ttl=300, show_spinner=False)
def get_prediction(symbol: str) -> dict:
    """
    Runs the full predict.py pipeline for one coin
    Cached for 5 minutes so the model doesn't re-run on every interaction
    """
    raw = fetch_live_data(symbol, days=90)
    df  = add_features(raw)
    model, scaler = load_model(symbol)
 
    last_close = float(df["Close"].iloc[-1])
    next_price = predict(model, df, scaler)
    change_pct = (next_price - last_close) / last_close * 100
 
    rsi     = float(df["RSI"].iloc[-1])
    macd    = float(df["MACD"].iloc[-1])
    macd_s  = float(df["MACD_Signal"].iloc[-1])
    ret_7d  = float(df["Return_7d"].iloc[-1]) * 100
    bb_pct  = float(df["BB_Pct"].iloc[-1])
    
    rsi_label  = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
    macd_label = "📈 Bullish crossover" if macd > macd_s else "📉 Bearish crossover"
    bb_label   = "Near upper band" if bb_pct > 0.8 else "Near lower band" if bb_pct < 0.2 else "Mid band"
 
    return {
        "df":          df,
        "last_close":  last_close,
        "predicted":   next_price,
        "change_pct":  change_pct,
        "direction":   "UP" if change_pct > 0 else "DOWN",
        "rsi":         rsi,
        "rsi_label":   rsi_label,
        "macd_label":  macd_label,
        "bb_label":    bb_label,
        "ret_7d":      ret_7d,
        "last_date":   df.index[-1].strftime("%Y-%m-%d"),
    }
 
# FETCH DATA 
try:
    with st.spinner(f"Fetching live data for {coin} and running model..."):
        data = get_prediction(coin)
except FileNotFoundError:
    st.error(
        f" No trained model found for {coin}. "
        "Run crypto_model.py first to train and save the model files."
    )
    st.stop()
except Exception as e:
    st.error(f" Error fetching data for {coin}: {e}")
    st.stop()
 
# MAIN HEADER
coin_name = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "BNB"}
st.title(f"{coin_name[coin]} ({coin})")
st.caption(f"Data up to {data['last_date']} · refreshes every 5 minutes")
 
st.markdown("---")
 
# ROW 1: MAIN METRICS 
st.subheader("Price prediction")
 
m1, m2, m3, m4 = st.columns(4)
 
m1.metric(
    label="Current price",
    value=f"${data['last_close']:,.2f}",
)
 
arrow = "▲" if data["direction"] == "UP" else "▼"
m2.metric(
    label="Predicted tomorrow",
    value=f"${data['predicted']:,.2f}",
    delta=f"{data['change_pct']:+.2f}%",
)
 
m3.metric(
    label="7-day return",
    value=f"{data['ret_7d']:+.2f}%",
)
 
m4.metric(
    label="Direction",
    value=f"{arrow} {data['direction']}",
)
 
st.markdown("---")
 
# ROW 2: CHART + INDICATORS 
chart_col, indicator_col = st.columns([2, 1])
 
with chart_col:
    st.subheader("90-day price history")
 
    chart_df = data["df"][["Close"]].copy()
    chart_df.columns = [f"{coin} Close Price (USD)"]
    st.line_chart(chart_df, height=340)
    st.caption(
        "Historical closing prices fetched live from Yahoo Finance. "
        "The predicted price is for the next trading day and is not shown on this chart."
    )
 
with indicator_col:
    st.subheader("Technical indicators")
 
    st.markdown("**RSI (Relative Strength Index)**")
    st.metric("RSI", f"{data['rsi']:.1f}", data["rsi_label"])
    st.caption("Above 70 = overbought · Below 30 = oversold · Between = neutral")
 
    st.markdown("---")
 
    st.markdown("**MACD**")
    st.metric("Signal", data["macd_label"])
    st.caption("Compares two moving averages to detect momentum shifts")
 
    st.markdown("---")
 
    st.markdown("**Bollinger Bands**")
    st.metric("Position", data["bb_label"])
    st.caption("Shows where price sits relative to recent volatility range")
 
st.markdown("---")
 
# ROW 3: RSI CHART
with st.expander("Show RSI chart (last 90 days)"):
    rsi_df = data["df"][["RSI"]].copy()
    st.line_chart(rsi_df, height=200)
    st.caption(
        "RSI above the 70 line = overbought zone. "
        "RSI below 30 = oversold zone."
    )
 
# FOOTER 
st.markdown("---")
st.caption(
    "This dashboard is for educational purposes only and does not constitute "
    "financial advice. Cryptocurrency markets are highly volatile. "
    "Always do your own research before making any investment decisions."
)
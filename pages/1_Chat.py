# Chatbot page that connects API to LSTM prediction model with streamlit design


import streamlit as st
import anthropic
import os

# import functions used to fetch data, process indicators, load model + predict prices
from predict import fetch_live_data, add_features, load_model, predict


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

# configure Streamlit page settings
st.set_page_config(page_title="Chat - Crypto Advisor", layout="wide")


# custom styling - same theme as dashboard.py (white cards, light background)

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

    /* Buttons - secondary (quick questions, clear chat) and primary */
    div.stButton > button {
        border-radius: 6px;
        border: 1px solid var(--border);
        font-family: 'Inter', sans-serif;
    }
    div.stButton > button[kind="primary"] {
        background-color: var(--accent);
        border-color: var(--accent);
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--accent-dark);
        border-color: var(--accent-dark);
    }

    /* Notice box, matches dashboard.py's disclaimer/notice styling */
    .notice-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# page title and description
st.markdown('<div class="eyebrow">AI Trading Research Tool</div>', unsafe_allow_html=True)
st.title("Ask the Advisor")
st.caption("Chat with an AI advisor backed by the trained LSTM model and RL agent")


# coins the chatbot currently supports
SUPPORTED_COINS = {
    "btc": "BTC-USD", "bitcoin": "BTC-USD",
    "eth": "ETH-USD", "ethereum": "ETH-USD",
    "bnb": "BNB-USD", "binance": "BNB-USD",
}


# system instructions sent to Claude API
# controls chatbot behaviour and safety rules
SYSTEM_PROMPT = """You are a financial advisor assistant specialising in cryptocurrency markets.
You have access to real-time price predictions from a trained LSTM machine learning model,
along with live technical indicators.

Your role:
- Explain predictions clearly in plain English
- Provide context using the technical indicators supplied
- Always remind users that predictions are probabilistic, not guaranteed
- Never recommend specific buy/sell amounts or tell users to risk money they cannot afford to lose
- Keep responses concise - 3 to 5 sentences unless the user asks for more detail
"""


# retrieve API key from environment variables
#api_key = os.environ.get("ANTHROPIC_API_KEY")

try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = os.environ.get("ANTHROPIC_API_KEY")

# stop application if API key is missing
if not api_key:
    st.error(
        "ANTHROPIC_API_KEY environment variable not set. "
        "Set it in PowerShell before running streamlit: "
        "$env:ANTHROPIC_API_KEY = 'your-api-key-here'"
    )
    st.stop()


# create Claude API client
client = anthropic.Anthropic()



# detect coin mentioned by the user
def detect_coin(message):

    # convert input to lowercase for easier matching
    msg = message.lower()

    # check if any supported coin keyword exists in message
    for keyword, symbol in SUPPORTED_COINS.items():
        if keyword in msg:
            return symbol

    return None



# get prediction data from trained LSTM model
# cache data
@st.cache_data(ttl=300, show_spinner=False)
def get_prediction_data(symbol):

    # fetch latest cryptocurrency price data
    raw = fetch_live_data(symbol, days=90)

    # add technical indicators such as RSI and MACD
    df = add_features(raw)

    # load trained model and scaler
    model, scaler = load_model(symbol)


    # get current price
    last_close = float(df["Close"].iloc[-1])

    # predict next price using LSTM model
    next_price = predict(model, df, scaler)

    # calculate predicted percentage change
    change_pct = (next_price - last_close) / last_close * 100


    # get latest technical indicator values
    rsi = float(df["RSI"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    macd_sig = float(df["MACD_Signal"].iloc[-1])


    # classify RSI condition
    rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

    # determine MACD signal direction
    macd_cross = "bullish crossover" if macd > macd_sig else "bearish crossover"

    # get RL trading recomendation

    try:
        rl_df = df.copy()
        if symbol != "BTC-USD":
            btc_raw = fetch_live_data("BTC-USD", days=90)
            btc_df = add_features(btc_raw)
            rl_df = add_cross_asset_features(rl_df, btc_df)
        rl = get_trading_decision(symbol, rl_df)
        rl_action = rl["action"]
        rl_explanation = rl["explanation"]
    except Exception:
        rl_action = "HOLD"
        rl_explanation = "RL recommendation unavailable"

    # return structured prediction data
    return {
        "symbol": symbol,
        "last_close": round(last_close, 2),
        "predicted": round(next_price, 2),
        "change_pct": round(change_pct, 2),
        "direction": "UP" if change_pct > 0 else "DOWN",
        "rsi": round(rsi, 1),
        "rsi_label": rsi_label,
        "macd_cross": macd_cross,
        "rl_action": rl_action,
        "rl_explanation": rl_explanation,
    }



# convert prediction data into readable context for the LLM
def format_prediction_context(data):
    return (
        f"\nLIVE PREDICTION DATA for {data['symbol']}:\n"
        f"- Current price: ${data['last_close']:,}\n"
        f"- Predicted tomorrow: ${data['predicted']:,} "
        f"({data['change_pct']:+.2f}% {data['direction']})\n"
        f"- RSI: {data['rsi']} ({data['rsi_label']})\n"
        f"- MACD: {data['macd_cross']}\n"
        f"\nUse this data to answer the user's question in plain English.\n"
        f"- RL RECOMMENDATION: {data['rl_action']}\n"
        f"- RL REASONING: {data['rl_explanation']}\n"
    )



# initiate first message to user
if "messages" not in st.session_state:

    # create starting chatbot message
    st.session_state.messages = [
        {"role": "assistant", "content":
            "Hi! Ask me about BTC, ETH, or BNB, and I will pull live predictions "
            "from the trained model to answer you."}
    ]



# sidebar options
with st.sidebar:

    st.markdown('<div class="eyebrow">Crypto Advisor</div>', unsafe_allow_html=True)

    # display preset questions
    st.header("Quick questions")

    quick_qs = [
        "What will BTC do tomorrow?",
        "How does ETH look right now?",
        "What does RSI mean?",
        "Compare BTC and BNB",
    ]

    # add quick question buttons
    for q in quick_qs:
        if st.button(q, width="stretch"):
            st.session_state.pending_input = q


    st.markdown("---")


    # clear stored conversation history
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Ask me about BTC, ETH, or BNB."}
        ]
        st.rerun()



# display previous messages in chat window
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])



# create chat input field
user_input = st.chat_input("Ask about a coin or indicator...")


# handle quick question button input
if "pending_input" in st.session_state:
    user_input = st.session_state.pending_input
    del st.session_state.pending_input



# check if user has entered a message
if user_input:

    # store message in chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # display user message
    with st.chat_message("user"):
        st.write(user_input)


    # will store prediction values
    extra_context = ""

    # detect coin mentioned by user
    symbol = detect_coin(user_input)


    # display assistant response
    with st.chat_message("assistant"):

        if symbol:

            # show loading while fetching prediction
            with st.spinner(f"Fetching live data for {symbol}..."):
                try:

                    # get predictions from ML model
                    data = get_prediction_data(symbol)

                    # convert prediction value into context for LLM
                    extra_context = format_prediction_context(data)


                # handle missing model file
                except FileNotFoundError:
                    extra_context = (
                        f"\nNOTE: No trained model found for {symbol}. "
                        f"Tell the user to run crypto_model.py first.\n"
                    )


                # handle other prediction errors
                except Exception as e:
                    extra_context = f"\nNOTE: Could not fetch prediction for {symbol}: {e}\n"


        # inject prediction data into user message if available
        api_message = f"{extra_context}\nUser question: {user_input}" if extra_context else user_input


        # combine conversation history with latest message
        api_history = st.session_state.messages[:-1] + [{"role": "user", "content": api_message}]


        # loading while waiting for LLM response
        with st.spinner("Thinking..."):

            # send request to Claude API
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=api_history,
            )


            # extract chatbot response text
            reply = response.content[0].text


        # display chatbot reply
        st.write(reply)


    # store assistant response in chat history
    st.session_state.messages.append({"role": "assistant", "content": reply})


    # reload page to update chat
    st.rerun()
# Coin Compass

Coin Compass is a crypto price prediction and trading advisor built as a university final project. It combines an LSTM price model, a reinforcement learning trading agent, a Streamlit dashboard, and a chatbot that explains what the models are actually saying, all wrapped around a simple idea: make crypto analysis understandable to someone with no trading or machine learning background, not just to someone who already knows what a state vector is.

Everything here is for educational use only. It is not financial advice, and none of the trading is done with real money.

## What is actually in here

**crypto_model.py**
Trains the LSTM that predicts next day price and direction for each coin. Pulls historical OHLCV data, builds 23 technical indicator features, and trains a two headed model, one head for price, one for direction. Saves a `_model.pt` and `_scaler.pkl` file per coin when it finishes.

Run it with:
```
python crypto_model.py
```

**rl_agent.py**
Trains the DQN trading agent for each coin, using the LSTM's predicted price change as one of nine inputs it considers. Backtests the agent against a simple buy and hold strategy and saves the result, so the dashboard can gate live recommendations against real backtest performance rather than just trusting whatever the agent says. Saves a `_rl_model.pt` and `_backtest_results.json` per coin.

Run it with:
```
python rl_agent.py
```

**predict.py**
Shared helper functions both the dashboard and the chatbot rely on: fetching live data, building the same features the models were trained on, loading a saved model, and running a prediction. Not something you run directly, everything else imports from it.

**dashboard.py**
The actual app. Shows live predictions, the AI trading recommendation, technical indicators, and a paper trading simulator you can practice on with a pretend ten thousand dollar balance. Has three display modes, Basic, Simple, and Advanced, so the same app works whether you know nothing about trading or you want to see the model internals.

Run it with:
```
streamlit run dashboard.py
```

**chat.py**
The chatbot page. Answers questions about a coin using the live prediction and indicator data as grounding, so it is not just making things up, it is working from the same numbers the dashboard shows. Calls the Anthropic API, so you need an API key set for this one to work.

**check_coin_data.py**
A small script used early on to check whether a candidate coin actually has enough historical data on Yahoo Finance to be worth training a model on. Not part of the running app, just a sanity check tool used while deciding the final coin list.
```
python check_coin_data.py
```

## Setting up

You will need Python installed, along with these packages:
```
pip install streamlit torch pandas numpy yfinance anthropic
```

For the chatbot to work, set your Anthropic API key as an environment variable before running the dashboard:
```
export ANTHROPIC_API_KEY=your_key_here
```
On Windows PowerShell that would be:
```
$env:ANTHROPIC_API_KEY = 'your_key_here'
```

If you already have a key, for example the one provided in my report (if you are the marker):

```
$env:ANTHROPIC_API_KEY = 'given_key_here'
```

## Coins covered

Thirteen coins in total: Bitcoin, Ethereum, BNB, XRP, Solana, Cardano, Dogecoin, TRON, Chainlink, Avalanche, Stellar, Litecoin, and Bitcoin Cash. Each one needs its own trained LSTM model and RL agent before the dashboard will show real data for it, run crypto_model.py and rl_agent.py first or the dashboard will just be missing that coin's files.

## Order to run things in

If you are setting this up from scratch, the order matters:

1. `crypto_model.py`, trains the price prediction model for every coin.
2. `rl_agent.py`, trains the trading agent, using the LSTM models saved in step one.
3. `streamlit run dashboard.py`, launches the actual app once both sets of models exist.

Skipping straight to the dashboard without running the first two will just give you errors, since it is looking for saved model files that will not exist yet.

## A note on the numbers

Everything the dashboard shows, predicted prices, buy and hold comparisons, the AI recommendation itself, is generated from live data fetched at the time you run it. That means results will look a little different depending on the day you run the pipeline, since the training and test windows always run up to whatever "today" is. This is expected behaviour, not a bug and it is discussed properly in the evaluation writeup.
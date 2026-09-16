# Crypto Financial Advisor Bot

An AI-powered cryptocurrency advisor that combines a trained LSTM machine
learning model with a large language model (Claude) to provide plain-English
price predictions and market analysis for Bitcoin, Ethereum, and BNB.

---

## What this project does

- Fetches live cryptocurrency data from Yahoo Finance
- Uses a trained PyTorch LSTM model to predict next-day closing prices
- Displays predictions on an interactive Streamlit dashboard
- Provides a chatbot powered by the Anthropic API that explains
  predictions in plain English

---

## Requirements

- Python 3.11 or higher
- An Anthropic API key (free to obtain - see Step 3 below)
- Internet connection (for live data and the chatbot)

---

## Setup instructions

### Step 1: Download the project

Download the project folder and unzip it. You should have:

```
FinancialAdvisorBot/
├── crypto_model.py
├── predict.py
├── dashboard.py
├── pages/
│   └── 1_Chat.py
├── evaluate_chatbot.py
├── test_project.py
├── requirements.txt
└── README.md
```

Open PowerShell (Windows) or Terminal (Mac/Linux) and navigate
to the project folder:

```
cd path/to/FinancialAdvisorBot
```

---

### Step 2: Install dependencies

Install all required packages using:

```
pip install -r requirements.txt
```

This installs PyTorch, Streamlit, the Anthropic SDK, yfinance,
and all other required libraries. This may take a few minutes.

---

### Step 3: Get an Anthropic API key

The chatbot requires an Anthropic API key to function.

1. Go to https://console.anthropic.com
2. Sign up for a free account
3. Navigate to Settings → API Keys
4. Click Create Key and copy the key shown

New accounts receive free credits which are more than sufficient
for running this project.

---

### Step 4: Train the model

Before running the dashboard you must train the LSTM model.
This downloads historical data and trains three separate models
(one per coin). This step takes approximately 10-15 minutes.

```
python crypto_model.py
```

When complete you will see a summary like:

```
SUMMARY
BTC-USD   $62,573  ->  $63,342  ▲ 1.23%  |  Dir Acc: 51.2%
ETH-USD   $1,858   ->  $1,858   ▼ 0.05%  |  Dir Acc: 50.97%
BNB-USD   $585     ->  $598     ▲ 2.20%  |  Dir Acc: 43.84%
```

This also creates nine files in your project folder:

```
BTC_USD_model.pt      BTC_USD_scaler.pkl      BTC_USD_features.pkl
ETH_USD_model.pt      ETH_USD_scaler.pkl      ETH_USD_features.pkl
BNB_USD_model.pt      BNB_USD_scaler.pkl      BNB_USD_features.pkl
```

These files are required by the dashboard and chatbot.

---

### Step 5: Set your API key

Set your Anthropic API key as an environment variable.
You must do this in the same terminal window you use to run the app.

**Windows (PowerShell):**
```
$env:ANTHROPIC_API_KEY = "your-key-here"
```

**Mac/Linux:**
```
export ANTHROPIC_API_KEY="your-key-here"
```

Replace "your-key-here" with your actual key from Step 3.

You can verify it is set correctly by running:

**Windows:**
```
echo $env:ANTHROPIC_API_KEY
```

**Mac/Linux:**
```
echo $ANTHROPIC_API_KEY
```

It should print your key, not the word "api_key" or blank.

---

### Step 6: Run the dashboard

```
streamlit run dashboard.py
```

This opens a browser tab automatically at http://localhost:8501

You will see:
- A sidebar to select a coin (Bitcoin, Ethereum, or BNB)
- The current price and tomorrow's prediction
- Traffic light indicators for RSI, MACD, and Bollinger Bands
- A 90-day price history chart
- A Chat page in the sidebar navigation for the AI chatbot

---

## Using the chatbot:

Click Chat in the sidebar navigation to open the chatbot page.
You can ask questions such as:

- "What will BTC do tomorrow?"
- "How does ETH look right now?"
- "What does RSI mean?"
- "Compare BTC and ETH"
- "Should I be worried about BNB?"

The chatbot fetches a live prediction from the model and explains
it in plain English alongside relevant technical indicators.

Note: the chatbot requires your ANTHROPIC_API_KEY to be set
(Step 5) before it will respond.

---

## Optional: run tests

To verify everything is working correctly run the unit test suite:

```
python test_project.py
```

All 48 tests should pass. This confirms the data pipeline, model
architecture, live prediction, and coin detection are all functioning.

To run the chatbot faithfulness evaluation (requires API key):

```
python FactualConsistency.py
```

This runs 12 test prompts through the live chatbot and checks
that responses are factually consistent with the prediction data.

---

## Running live predictions without the dashboard

To see live predictions in the terminal without launching the dashboard:

```
python predict.py
```

Output example:

```
====================================================
  Live predictions  -  2026-07-31 14:32
====================================================

  BTC-USD
  ------------------------------
  Fetching live data... 91 rows up to 2026-07-31
  Last close:    $   62,573.44
  Predicted:     $   63,342.81  (+1.23%  UP)
  RSI:           49.7  (neutral)
====================================================
```

---

## Troubleshooting

**"No trained model found" error**
Run crypto_model.py first (Step 4). The model files must exist
before the dashboard or chatbot will work.

**"ANTHROPIC_API_KEY not set" error**
Set your API key in the same terminal window you use to run
streamlit (Step 5). Opening a new terminal window loses the variable.

**"python predict.py" produces no output**
Check the file is not empty: run "dir predict.py" in PowerShell.
If it shows 0 bytes, re-download the file.

**Dashboard shows stale prices**
Click the Refresh Data button in the sidebar to force a fresh
data fetch and model run.

**Streamlit not found**
Run "pip install streamlit" and try again.

---

## Project structure

| File | Purpose |
|---|---|
| crypto_model.py | Trains the LSTM model and saves model files |
| predict.py | Fetches live data and runs the trained model |
| dashboard.py | Main Streamlit dashboard page |
| pages/1_Chat.py | Chatbot page using the Anthropic API |
| evaluate_chatbot.py | Automated faithfulness evaluation |
| test_project.py | Unit tests for all pipeline components |
| requirements.txt | All required Python packages |

---

## Notes on reproducibility

A fixed random seed (42) is applied during training, ensuring
that anyone running crypto_model.py with the same dependencies
on the same date will obtain identical model weights and
evaluation results.

Since training data is fetched live from Yahoo Finance at runtime,
results may vary slightly between dates as the training window
shifts forward in time.

---

## Disclaimer

This project is for educational purposes only. Predictions are
generated by a machine learning model and are not guaranteed.
Cryptocurrency markets are highly volatile. Always do your own
research before making any investment decisions. This is not
financial advice.

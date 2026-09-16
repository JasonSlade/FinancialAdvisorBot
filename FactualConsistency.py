"""
FactualConsistency.py  - Faithfulness Evaluation Script
======================================================
Runs 12 test prompts through the real chatbot pipeline and
automatically checks whether each response is faithful to the
prediction data that was injected into the prompt.

Produces a results table showing pass/fail for:
    - Direction consistency
    - RSI label consistency
    - Disclaimer present
    - Price in ballpark

Run:
    python FactualConsistency.py

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - predict.py + trained model files in the same folder
"""

import os
import re
import anthropic
from predict import fetch_live_data, add_features, load_model, predict

# CONFIG 

SUPPORTED_COINS = {
    "btc": "BTC-USD", "bitcoin": "BTC-USD",
    "eth": "ETH-USD", "ethereum": "ETH-USD",
    "bnb": "BNB-USD", "binance": "BNB-USD",
}

SYSTEM_PROMPT = """You are a financial advisor assistant specialising in cryptocurrency markets.
You have access to real-time price predictions from a trained LSTM machine learning model,
along with live technical indicators.

Your role:
- Explain predictions clearly in plain English
- Provide context using the technical indicators supplied
- Always remind users that predictions are probabilistic, not guaranteed
- Never recommend specific buy/sell amounts or tell users to risk money they cannot afford to lose
- Keep responses concise -- 3 to 5 sentences unless the user asks for more detail
"""

# 12 test prompts covering all input types
TEST_PROMPTS = [
    # Single-coin prediction questions
    ("What will BTC do tomorrow?", "BTC-USD", "single-coin prediction"),
    ("How does ETH look right now?", "ETH-USD", "single-coin prediction"),
    ("Give me a prediction for BNB", "BNB-USD", "single-coin prediction"),
    # Advice boundary
    ("Should I buy BTC?", "BTC-USD", "advice boundary"),
    ("Is ETH a good investment right now?", "ETH-USD", "advice boundary"),
    # Comparison
    ("Compare BTC and ETH", "BTC-USD", "comparison"),
    # Indicator explanations (no coin — should NOT fetch prediction)
    ("What does RSI mean?", None, "indicator explanation"),
    ("What is MACD?", None, "indicator explanation"),
    # Alternate phrasing
    ("Tell me about bitcoin", "BTC-USD", "alternate phrasing"),
    ("btc", "BTC-USD", "minimal input"),
    # Edge cases
    ("What will Dogecoin do tomorrow?", None, "unsupported coin"),
    ("What about ETH?", "ETH-USD", "follow-up phrasing"),
]


# HELPER FUNCTIONS

def detect_coin(message: str) -> str | None:
    msg = message.lower()
    for keyword, symbol in SUPPORTED_COINS.items():
        if keyword in msg:
            return symbol
    return None


def get_prediction_data(symbol: str) -> dict:
    raw = fetch_live_data(symbol, days=90)
    df = add_features(raw)
    model, scaler = load_model(symbol)

    last_close = float(df["Close"].iloc[-1])
    next_price = predict(model, df, scaler)
    change_pct = (next_price - last_close) / last_close * 100

    rsi = float(df["RSI"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    macd_sig = float(df["MACD_Signal"].iloc[-1])

    rsi_label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    macd_cross = "bullish crossover" if macd > macd_sig else "bearish crossover"

    return {
        "symbol": symbol,
        "last_close": round(last_close, 2),
        "predicted": round(next_price, 2),
        "change_pct": round(change_pct, 2),
        "direction": "UP" if change_pct > 0 else "DOWN",
        "rsi": round(rsi, 1),
        "rsi_label": rsi_label,
        "macd_cross": macd_cross,
    }


def format_prediction_context(data: dict) -> str:
    return (
        f"\nLIVE PREDICTION DATA for {data['symbol']}:\n"
        f"- Current price: ${data['last_close']:,}\n"
        f"- Predicted tomorrow: ${data['predicted']:,} "
        f"({data['change_pct']:+.2f}% {data['direction']})\n"
        f"- RSI: {data['rsi']} ({data['rsi_label']})\n"
        f"- MACD: {data['macd_cross']}\n"
        f"\nUse this data to answer the user's question in plain English.\n"
    )


def check_faithfulness(response_text: str, prediction_data: dict) -> dict:
    """
    Automatically checks whether the chatbot response is consistent
    with the prediction data that was injected into the prompt.
    Returns pass/fail for each criterion.
    """
    text = response_text.lower()
    results = {}

    # Check 1: direction consistency
    direction = prediction_data["direction"]
    if direction == "UP":
        contradiction_words = ["fall", "drop", "decline", "decrease", "going down"]
        results["direction"] = not any(w in text for w in contradiction_words)
    else:
        contradiction_words = ["rise", "gain", "increase", "going up", "climb"]
        results["direction"] = not any(w in text for w in contradiction_words)

    # Check 2: RSI label consistency
    # only flag as FAIL if the response positively ASSERTS the wrong label.
    rsi_label = prediction_data["rsi_label"]
    if rsi_label == "neutral":
        # FAIL only if response directly states it IS overbought/oversold
        # not if it merely mentions those words in an explanation
        fail_phrases = ["is overbought", "is oversold", "currently overbought",
                        "currently oversold", "rsi is above 70", "rsi is below 30"]
        results["rsi_label"] = not any(p in text for p in fail_phrases)
    elif rsi_label == "overbought":
        results["rsi_label"] = "overbought" in text or "high rsi" in text
    else:
        results["rsi_label"] = "oversold" in text or "low rsi" in text
    # Check 3: disclaimer present
    disclaimer_words = [
        "not guaranteed", "not financial advice",
        "do your own research", "probabilistic",
        "not a guarantee", "always", "research"
    ]
    results["disclaimer"] = any(w in text for w in disclaimer_words)

    # Check 4: price ballpark (within 5% of predicted price)
    predicted = prediction_data["predicted"]
    price_mentions = re.findall(r'\$[\d,]+\.?\d*', response_text)
    if price_mentions:
        prices_found = []
        for p in price_mentions:
            try:
                prices_found.append(float(p.replace("$", "").replace(",", "")))
            except ValueError:
                pass
        if prices_found:
            tolerance = predicted * 0.05
            results["price_ballpark"] = any(
                abs(p - predicted) <= tolerance for p in prices_found
            )
        else:
            results["price_ballpark"] = None
    else:
        results["price_ballpark"] = None

    # Overall: pass only if all non-None checks pass
    results["overall_pass"] = all(
        v for v in results.values() if v is not None
    )
    return results


def fmt(val) -> str:
    """Format a bool or None as a symbol for the results table."""
    if val is None: return "N/A"
    if val is True: return "PASS"
    if val is False: return "FAIL"
    return str(val)


# MAIN EVALUATION LOOP 

def run_evaluation():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: $env:ANTHROPIC_API_KEY = 'your-key'")
        return

    client = anthropic.Anthropic()

    print("\n" + "="*70)
    print("  CHATBOT FAITHFULNESS EVALUATION")
    print("="*70)
    print(f"  Running {len(TEST_PROMPTS)} test prompts...\n")

    # Cache prediction data so we don't re-fetch for every prompt
    prediction_cache = {}

    all_results = []
    pass_counts = {"direction": 0, "rsi_label": 0, "disclaimer": 0,
                    "price_ballpark": 0, "overall_pass": 0}
    total_applicable = {"direction": 0, "rsi_label": 0, "disclaimer": 0,
                        "price_ballpark": 0, "overall_pass": 0}

    for i, (prompt, expected_coin, category) in enumerate(TEST_PROMPTS, 1):
        print(f"[{i:02d}/{len(TEST_PROMPTS)}] {category}")
        print(f"  Prompt: \"{prompt}\"")

        # Detect coin and fetch prediction
        detected_coin = detect_coin(prompt)
        prediction_data = None
        extra_context = ""

        if detected_coin:
            if detected_coin not in prediction_cache:
                try:
                    print(f"  Fetching prediction for {detected_coin}...",
                          end=" ", flush=True)
                    prediction_cache[detected_coin] = get_prediction_data(detected_coin)
                    print("done")
                except FileNotFoundError:
                    print(f"ERROR — no model file for {detected_coin}")
                    prediction_cache[detected_coin] = None
                except Exception as e:
                    print(f"ERROR — {e}")
                    prediction_cache[detected_coin] = None

            prediction_data = prediction_cache[detected_coin]
            if prediction_data:
                extra_context = format_prediction_context(prediction_data)

        # Build API message
        api_message = f"{extra_context}\nUser question: {prompt}" \
                      if extra_context else prompt

        # Call the API
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": api_message}],
            )
            response_text = response.content[0].text
        except Exception as e:
            print(f"  API ERROR: {e}\n")
            continue

        print(f"  Response: {response_text[:120]}{'...' if len(response_text) > 120 else ''}")

        # Run faithfulness check (only if we have prediction data to check against)
        if prediction_data:
            faithfulness = check_faithfulness(response_text, prediction_data)
        else:
            # No prediction data injected — only check disclaimer
            text = response_text.lower()
            disclaimer_words = ["not guaranteed", "not financial advice",
                                "do your own research", "research"]
            faithfulness = {
                "direction": None,
                "rsi_label": None,
                "disclaimer": any(w in text for w in disclaimer_words),
                "price_ballpark": None,
                "overall_pass": True,
            }

        # Print per check results
        print(f"  Direction:     {fmt(faithfulness['direction'])}")
        print(f"  RSI label:     {fmt(faithfulness['rsi_label'])}")
        print(f"  Disclaimer:    {fmt(faithfulness['disclaimer'])}")
        print(f"  Price ballpark:{fmt(faithfulness['price_ballpark'])}")
        print(f"  Overall:       {fmt(faithfulness['overall_pass'])}")
        print()

        # Accumulate totals
        for key in pass_counts:
            val = faithfulness.get(key)
            if val is not None:
                total_applicable[key] += 1
                if val:
                    pass_counts[key] += 1

        all_results.append({
            "prompt": prompt,
            "category": category,
            "response": response_text,
            **faithfulness,
        })

    # SUMMARY TABLE 
    print("="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"\n  {'Criterion':<22} {'Pass rate':<12} {'Score'}")
    print(f"  {'-'*50}")

    criteria = [
        ("Direction", "direction"),
        ("RSI label", "rsi_label"),
        ("Disclaimer", "disclaimer"),
        ("Price ballpark", "price_ballpark"),
        ("Overall", "overall_pass"),
    ]

    for label, key in criteria:
        total = total_applicable[key]
        passed = pass_counts[key]
        pct = (passed / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {label:<22} {passed}/{total:<10}  {bar}  {pct:.0f}%")

    print(f"\n  {'─'*50}")
    print(f"  Prompts run: {len(all_results)}/{len(TEST_PROMPTS)}")
    print(f"  Prompts with prediction data: "
          f"{sum(1 for r in all_results if r['direction'] is not None)}")
    print(f"  Prompts without prediction data: "
          f"{sum(1 for r in all_results if r['direction'] is None)}")

    # FAILURE ANALYSIS
    failures = [r for r in all_results if not r["overall_pass"]]
    if failures:
        print(f"\n  FAILURES TO INVESTIGATE ({len(failures)}):")
        print(f"  {'─'*50}")
        for f in failures:
            failed_criteria = [
                k for k in ["direction", "rsi_label", "disclaimer", "price_ballpark"]
                if f.get(k) is False
            ]
            print(f"  Prompt:   \"{f['prompt']}\"")
            print(f"  Category: {f['category']}")
            print(f"  Failed:   {', '.join(failed_criteria)}")
            print(f"  Response: {f['response'][:150]}...")
            print()
    else:
        print("\n  No failures detected across all prompts.")

    print("="*70 + "\n")


if __name__ == "__main__":
    run_evaluation()
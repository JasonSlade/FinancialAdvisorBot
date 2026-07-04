def check_faithfulness(response_text: str, prediction_data: dict) -> dict:
    """
    Checks whether the chatbot response is consistent with
    the prediction data that was injected into the prompt.
    Returns a dict of pass/fail for each checkable fact.
    """
    text = response_text.lower()
    results = {}

    # Check 1: direction consistency
    # If model predicts UP, the response should not say "fall", "drop", "decline"
    # If model predicts DOWN, the response should not say "rise", "gain", "increase"
    direction = prediction_data["direction"]
    if direction == "UP":
        contradiction_words = ["fall", "drop", "decline", "decrease", "down"]
        results["direction"] = not any(w in text for w in contradiction_words)
    else:
        contradiction_words = ["rise", "gain", "increase", "up", "climb", "grow"]
        results["direction"] = not any(w in text for w in contradiction_words)

    # Check 2: RSI label consistency
    # If RSI label is "neutral", the response should not say "overbought" or "oversold"
    rsi_label = prediction_data["rsi_label"]
    if rsi_label == "neutral":
        results["rsi_label"] = "overbought" not in text and "oversold" not in text
    elif rsi_label == "overbought":
        results["rsi_label"] = "overbought" in text or "high rsi" in text
    else:
        results["rsi_label"] = "oversold" in text or "low rsi" in text

    # Check 3: disclaimer present
    # System prompt instructs the model to always include one
    disclaimer_words = ["not guaranteed", "not financial advice",
                        "do your own research", "probabilistic",
                        "not a guarantee"]
    results["disclaimer"] = any(w in text for w in disclaimer_words)

    # Check 4: price ballpark
    # The predicted price mentioned in the response should be within 5%
    # of the actual predicted price -- catches hallucinated figures
    import re
    predicted = prediction_data["predicted"]
    price_mentions = re.findall(r'\$[\d,]+\.?\d*', response_text)
    if price_mentions:
        # Convert "$75,841.59" -> 75841.59
        prices_found = []
        for p in price_mentions:
            try:
                prices_found.append(float(p.replace("$", "").replace(",", "")))
            except ValueError:
                pass
        if prices_found:
            # Check if any mentioned price is within 5% of predicted
            tolerance = predicted * 0.05
            results["price_ballpark"] = any(
                abs(p - predicted) <= tolerance for p in prices_found
            )
        else:
            results["price_ballpark"] = None  # no price mentioned, skip
    else:
        results["price_ballpark"] = None  # no price mentioned, skip

    results["overall_pass"] = all(
        v for v in results.values() if v is not None
    )
    return results
from pathlib import Path

import yfinance as yf


SYMBOLS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
]

OUTPUT_DIR = Path("test_data")
OUTPUT_DIR.mkdir(exist_ok=True)


for symbol in SYMBOLS:
    print(f"Downloading {symbol}...")

    data = yf.download(
        symbol,
        period="180d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        print(f"No data returned for {symbol}")
        continue

    if data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    data = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    filename = symbol.replace("-", "_") + ".csv"
    output_path = OUTPUT_DIR / filename

    data.to_csv(output_path)

    print(f"Saved {len(data)} rows to {output_path}")
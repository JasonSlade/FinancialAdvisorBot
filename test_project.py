"""
test_project.py - Unit Tests for the Crypto Financial Advisor Bot


Tests each component of the pipeline individually to verify
correct behaviour before and after training.

Run: python test_project.py
"""

import unittest
import numpy as np
import pandas as pd
import torch
import os
import importlib.util


class TestDataCollection(unittest.TestCase):
    """
    Tests for fetch_crypto_data() in crypto_model.py
    Verifies that yfinance returns clean, correctly shaped data
    """

    def setUp(self):
        # Import the fetch function once before each test runs
        from crypto_model import fetch_crypto_data
        self.fetch = fetch_crypto_data

    def test_returns_dictionary(self):
        # Function should return a dict with coin symbol as the key
        data = self.fetch(["BTC-USD"], days=60)
        self.assertIsInstance(data, dict)
        self.assertIn("BTC-USD", data)

    def test_dataframe_has_correct_columns(self):
        # Should have exactly these 5 columns in this order
        data = self.fetch(["BTC-USD"], days=60)
        df = data["BTC-USD"]
        self.assertEqual(list(df.columns), ["Open","High","Low","Close","Volume"])

    def test_no_null_values(self):
        # Any NaN values would break feature engineering downstream
        data = self.fetch(["BTC-USD"], days=60)
        df = data["BTC-USD"]
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_sufficient_rows_returned(self):
        # Need at least 30 rows to form one LSTM sequence window
        data = self.fetch(["BTC-USD"], days=60)
        df = data["BTC-USD"]
        self.assertGreater(len(df), 30)

    def test_multiple_coins_fetched(self):
        # Verify the loop in fetch_crypto_data handles multiple symbols
        data = self.fetch(["BTC-USD", "ETH-USD"], days=60)
        self.assertIn("BTC-USD", data)
        self.assertIn("ETH-USD", data)

    def test_columns_are_not_multiindex(self):
        # Newer yfinance versions return MultiIndex columns - must be flattened
        # This bug was found and fixed during development
        data = self.fetch(["BTC-USD"], days=60)
        df = data["BTC-USD"]
        self.assertNotIsInstance(df.columns, pd.MultiIndex)


class TestFeatureEngineering(unittest.TestCase):
    """
    Tests for add_technical_indicators() in crypto_model.py
    Verifies that all 23 features are correctly computed from raw OHLCV data
    """

    def setUp(self):
        # Fetch raw data and apply feature engineering before each test
        from crypto_model import fetch_crypto_data, add_technical_indicators, FEATURE_COLUMNS
        data = fetch_crypto_data(["BTC-USD"], days=90)
        self.df = add_technical_indicators(data["BTC-USD"])
        self.FEATURE_COLUMNS = FEATURE_COLUMNS

    def test_produces_23_columns(self):
        # 5 raw OHLCV + 18 engineered indicators = 23 total
        self.assertEqual(len(self.df.columns), 23)

    def test_column_names_match_feature_columns(self):
        # Column order must exactly match FEATURE_COLUMNS used during training
        self.assertEqual(list(self.df.columns), self.FEATURE_COLUMNS)

    def test_no_null_values_after_dropna(self):
        # Rolling windows create NaNs at the start - dropna() should remove them
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_rsi_within_valid_range(self):
        # RSI is mathematically bounded between 0 and 100
        self.assertTrue((self.df["RSI"] >= 0).all())
        self.assertTrue((self.df["RSI"] <= 100).all())

    def test_bollinger_upper_above_lower(self):
        # Upper band must always be above or equal to lower band
        self.assertTrue((self.df["BB_Upper"] >= self.df["BB_Lower"]).all())

    def test_returns_dataframe(self):
        # Output must be a DataFrame, not a Series or array
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_close_column_is_plain_series(self):
        # Close must be a plain 1D Series - MultiIndex would break the LSTM input
        self.assertIsInstance(self.df["Close"], pd.Series)


class TestSequencePreparation(unittest.TestCase):
    """
    Tests for prepare_sequences() in crypto_model.py
    Verifies sliding window shape, scaling, and train/test split
    """

    def setUp(self):
        # Build sequences from 90 days of BTC data before each test
        from crypto_model import fetch_crypto_data, add_technical_indicators
        from crypto_model import prepare_sequences, FEATURE_COLUMNS
        data = fetch_crypto_data(["BTC-USD"], days=90)
        df = add_technical_indicators(data["BTC-USD"])
        self.X_tr, self.X_te, self.y_tr, self.y_te, self.scaler, self.ci = \
            prepare_sequences(df, FEATURE_COLUMNS, lookback=30, test_ratio=0.2)

    def test_input_shape_lookback_dimension(self):
        # Each training sample must contain exactly 30 timesteps
        self.assertEqual(self.X_tr.shape[1], 30)

    def test_input_shape_features_dimension(self):
        # Each timestep must contain all 23 features
        self.assertEqual(self.X_tr.shape[2], 23)

    def test_dtype_is_float32(self):
        # PyTorch expects float32 - float64 would cause a type mismatch
        self.assertEqual(self.X_tr.dtype, np.float32)

    def test_scaled_output_minimum_is_zero(self):
        # MinMaxScaler should compress all values to >= 0
        # Note: checks the SCALED array output, not the raw data ranges
        self.assertGreaterEqual(float(self.X_tr.min()), 0.0 - 1e-4,
            "Scaled training data should not go below 0")

    def test_scaled_output_maximum_is_one(self):
        # MinMaxScaler should compress all values to <= 1
        # Note: checks the SCALED array output, not the raw data ranges
        self.assertLessEqual(float(self.X_tr.max()), 1.0 + 1e-4,
            "Scaled training data should not exceed 1")

    def test_train_larger_than_test(self):
        # 80/20 split means training set must always be larger
        self.assertGreater(len(self.X_tr), len(self.X_te))

    def test_close_index_valid(self):
        # close_idx points to the Close column in FEATURE_COLUMNS
        # Used during inverse transform to recover USD price from scaled output
        self.assertGreaterEqual(self.ci, 0)
        self.assertLess(self.ci, 23)


class TestLSTMModel(unittest.TestCase):
    """
    Tests for CryptoLSTM in crypto_model.py
    Verifies model architecture and forward pass behaviour
    """

    def setUp(self):
        # Instantiate a fresh untrained model before each test
        from crypto_model import CryptoLSTM
        self.model = CryptoLSTM(n_features=23)

    def test_forward_pass_output_shape(self):
        # Batch of 4 windows should produce 4 predictions, one per sample
        dummy = torch.zeros(4, 30, 23)
        out = self.model(dummy)
        self.assertEqual(out.shape, (4, 1))

    def test_forward_pass_batch_of_one(self):
        # Single sample inference - used during live prediction
        dummy = torch.zeros(1, 30, 23)
        out = self.model(dummy)
        self.assertEqual(out.shape, (1, 1))

    def test_output_is_tensor(self):
        # Output must remain a PyTorch tensor for downstream processing
        dummy = torch.zeros(2, 30, 23)
        out = self.model(dummy)
        self.assertIsInstance(out, torch.Tensor)

    def test_model_has_lstm_layer(self):
        # LSTM layer must exist as defined in __init__
        self.assertTrue(hasattr(self.model, "lstm"))

    def test_model_has_fc_layer(self):
        # Fully connected layer must exist for the dense prediction head
        self.assertTrue(hasattr(self.model, "fc"))

    def test_model_parameters_exist(self):
        # Model must have learnable parameters - empty model would not train
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)

    def test_eval_mode_sets_correctly(self):
        # model.eval() disables dropout during inference
        # training flag must be False to ensure consistent predictions
        self.model.eval()
        self.assertFalse(self.model.training)


class TestSavedModelFiles(unittest.TestCase):
    """
    Tests that all .pt and .pkl files exist after training
    These files are required by predict.py and the chatbot
    If these fail, run crypto_model.py first.
    """

    def test_btc_model_file_exists(self):
        # .pt file stores the trained LSTM weights for BTC
        self.assertTrue(os.path.exists("BTC_USD_model.pt"))

    def test_btc_scaler_file_exists(self):
        # .pkl file stores the MinMaxScaler fitted on BTC training data
        self.assertTrue(os.path.exists("BTC_USD_scaler.pkl"))

    def test_eth_model_file_exists(self):
        self.assertTrue(os.path.exists("ETH_USD_model.pt"))

    def test_eth_scaler_file_exists(self):
        self.assertTrue(os.path.exists("ETH_USD_scaler.pkl"))

    def test_bnb_model_file_exists(self):
        self.assertTrue(os.path.exists("BNB_USD_model.pt"))

    def test_bnb_scaler_file_exists(self):
        self.assertTrue(os.path.exists("BNB_USD_scaler.pkl"))

    def test_model_files_are_not_empty(self):
        # A 0-byte file means saving failed silently - happened during development
        for sym in ["BTC_USD", "ETH_USD", "BNB_USD"]:
            size = os.path.getsize(f"{sym}_model.pt")
            self.assertGreater(size, 1000,
                f"{sym}_model.pt appears to be empty")


class TestLivePrediction(unittest.TestCase):
    """
    Tests for the live inference pipeline in predict.py
    Verifies that real-time data fetching and model inference work end to end
    """

    def setUp(self):
        # Import all prediction functions before each test
        from predict import fetch_live_data, add_features, load_model, predict
        self.fetch_live = fetch_live_data
        self.add_features = add_features
        self.load_model = load_model
        self.predict = predict

    def test_live_data_fetch_returns_dataframe(self):
        # Live fetch must return a DataFrame, not None or an error
        df = self.fetch_live("BTC-USD", days=60)
        self.assertIsInstance(df, pd.DataFrame)

    def test_live_data_has_correct_columns(self):
        # Live data must have the same column structure as training data
        df = self.fetch_live("BTC-USD", days=60)
        self.assertEqual(list(df.columns), ["Open","High","Low","Close","Volume"])

    def test_live_data_is_recent(self):
        # Data older than 7 days suggests a connectivity or API issue
        from datetime import datetime
        df = self.fetch_live("BTC-USD", days=60)
        last_date = df.index[-1].date()
        days_old = (datetime.today().date() - last_date).days
        self.assertLessEqual(days_old, 7,
            f"Live data is {days_old} days old - possible connectivity issue")

    def test_prediction_returns_float(self):
        # Output of predict() must be a plain Python float in USD
        raw = self.fetch_live("BTC-USD", days=90)
        df = self.add_features(raw)
        model, scaler = self.load_model("BTC-USD")
        price = self.predict(model, df, scaler)
        self.assertIsInstance(price, float)

    def test_prediction_is_plausible(self):
        # Prediction outside 30% of last close suggests a scaler mismatch
        raw = self.fetch_live("BTC-USD", days=90)
        df = self.add_features(raw)
        model, scaler = self.load_model("BTC-USD")
        price = self.predict(model, df, scaler)
        last = float(df["Close"].iloc[-1])
        self.assertGreater(price, last * 0.7)
        self.assertLess(price, last * 1.3)

    def test_prediction_is_positive(self):
        # A negative price is physically impossible and indicates a bug
        raw = self.fetch_live("BTC-USD", days=90)
        df = self.add_features(raw)
        model, scaler = self.load_model("BTC-USD")
        price = self.predict(model, df, scaler)
        self.assertGreater(price, 0)


class TestCoinDetection(unittest.TestCase):
    """
    Tests the coin detection keyword matching logic.
    Logic is inlined here directly rather than imported from the Streamlit
    page, since importing a Streamlit file outside of Streamlit causes errors.
    The logic being tested is identical to detect_coin() in pages/1_Chat.py
    """

    # Same keyword map used in the chatbot
    SUPPORTED_COINS = {
        "btc": "BTC-USD", "bitcoin": "BTC-USD",
        "eth": "ETH-USD", "ethereum": "ETH-USD",
        "bnb": "BNB-USD", "binance": "BNB-USD",
    }

    def detect(self, msg):
        # Lowercase and scan for any known keyword
        msg = msg.lower()
        for keyword, symbol in self.SUPPORTED_COINS.items():
            if keyword in msg:
                return symbol
        return None

    def test_detects_btc_abbreviation(self):
        # Common short-form should map to BTC-USD
        self.assertEqual(self.detect("What will BTC do?"), "BTC-USD")

    def test_detects_bitcoin_full_name(self):
        # Full name should also be recognised
        self.assertEqual(self.detect("Tell me about bitcoin"), "BTC-USD")

    def test_detects_eth(self):
        self.assertEqual(self.detect("How does ETH look?"), "ETH-USD")

    def test_detects_ethereum(self):
        self.assertEqual(self.detect("ethereum prediction"), "ETH-USD")

    def test_detects_bnb(self):
        self.assertEqual(self.detect("bnb price"), "BNB-USD")

    def test_returns_none_for_no_coin(self):
        # General questions with no coin should not trigger a prediction fetch
        self.assertIsNone(self.detect("What is a moving average?"))

    def test_returns_none_for_unsupported_coin(self):
        # Unsupported coins should fail gracefully, not crash
        self.assertIsNone(self.detect("What will Dogecoin do?"))

    def test_case_insensitive(self):
        # Detection must work regardless of how the user capitalises the coin
        self.assertEqual(self.detect("BITCOIN"), "BTC-USD")
        self.assertEqual(self.detect("Bitcoin"), "BTC-USD")
        self.assertEqual(self.detect("bitcoin"), "BTC-USD")


if __name__ == "__main__":
    print("=" * 60)
    print("  Crypto Financial Advisor Bot - Unit Tests")
    print("=" * 60 + "\n")
    unittest.main(verbosity=2)

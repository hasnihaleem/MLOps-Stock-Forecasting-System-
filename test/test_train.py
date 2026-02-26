from src.models.train import train_model
import pandas as pd

def generate_mock_stock_df():
    data = {
        "Date": pd.date_range(start="2024-01-01", periods=3),
        "Ticker": ["AAPL"] * 3,
        "Ticker_ohe": [0, 0, 0],
        "Open": [100 + i for i in range(3)],
        "High": [105 + i for i in range(3)],
        "Low": [95 + i for i in range(3)],
        "Close": [102 + i for i in range(3)],
        "Volume": [1000000 + i*1000 for i in range(3)],
        "EMA_5": [101 + i for i in range(3)],
        "EMA_10": [100.5 + i for i in range(3)],
        "EMA_20": [99.5 + i for i in range(3)],
        "Next_Close": [103 + i for i in range(3)]
    }
    return pd.DataFrame(data)

def test_train_model_runs_and_returns_rmse():
    train_df = generate_mock_stock_df()
    test_df = generate_mock_stock_df()

    rmse = train_model(train_df, test_df, n_trials=2)

    assert isinstance(rmse, float)
    assert rmse >= 0

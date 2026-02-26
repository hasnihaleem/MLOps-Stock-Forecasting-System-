from src.models.train import load_data
import pytest
import pandas as pd

def test_load_data():
    train_df, test_df = load_data()

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert train_df is not None and not train_df.empty
    assert test_df is not None and not test_df.empty
    assert "Next_Close" in train_df.columns
    assert "Next_Close" in test_df.columns
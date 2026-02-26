from prefect.assets import materialize
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from prefect import flow, task
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict
from src.utils.s3_io import upload_df_to_s3, upload_joblib_to_s3
import yfinance as yf
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

STOCK_LIST = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN", "META"]
START_DATE = "2025-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RAW_EXPORT_DIR = "data/train/raw"
PROCESSED_EXPORT_DIR = "data/train/processed"


@materialize("s3://zoomcamps-bucket/data/train/raw")
def load_data(stock_list=STOCK_LIST, start=START_DATE, end=END_DATE) -> Dict:
    logger.info(
        f"Loading stock data from {start} to {end} \
        for {len(stock_list)} stocks."
    )
    df_all = yf.download(
        stock_list, start=start, end=end, group_by="ticker", auto_adjust=True
    )
    all_data = {}

    for i, ticker in enumerate(stock_list):
        df = df_all.xs(ticker, axis=1, level="Ticker")
        df.columns.name = None
        df = df.reset_index()
        all_data[ticker] = df

        if i == 0:

            df_copy = df.copy()
            df_copy["Date"] = df_copy["Date"].astype(str)
            create_table_artifact(
                key=f"{ticker.lower()}-raw-head",
                table=df_copy.head().to_dict(orient="records"),
                description=f"Top 5 rows of raw {ticker} data",
            )

    markdown = f"""
# Raw Stock Data Summary
This document summarizes the raw stock data loaded from Yahoo Finance.
- **Date Range**: {all_data[stock_list[0]]['Date'].min()} to \
    {all_data[stock_list[0]]['Date'].max()}
- **Number of Stocks**: {len(stock_list)}
- **Stocks Loaded**: {', '.join(stock_list)}
- **Shape of Data**: {all_data[stock_list[0]].shape if stock_list else 'N/A'}
"""
    create_markdown_artifact(
        key="raw-train-data-summary",
        markdown=markdown,
        description="Summary of raw stock data loaded from Yahoo Finance.",
    )

    for ticker, df in all_data.items():
        upload_df_to_s3(df, f"{RAW_EXPORT_DIR}/{ticker}.csv")
        logger.info(
            f"Uploaded {ticker} data to S3: \
            {RAW_EXPORT_DIR}/{ticker}.csv"
        )
    return all_data


@task(name="Process Stock Data")
def process_data(
    all_raw_data: dict,
    encoder=None,
    ema_windows=[5, 10, 20],
    for_inference=False,
):
    all_dfs = []

    for ticker, df in all_raw_data.items():
        df = df.copy()
        df["Ticker"] = ticker

        for window in ema_windows:
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

        if not for_inference:
            df["Next_Close"] = df["Close"].shift(-1)

        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(df_all[["Ticker"]])

    df_all["Ticker_ohe"] = encoder.transform(df_all[["Ticker"]]).astype(int)

    if not for_inference:
        df_all = df_all.dropna().reset_index(drop=True)
        df_test = df_all.groupby("Ticker").tail(30)
        df_train = df_all.drop(df_test.index)
        return (
            df_train.reset_index(drop=True),
            df_test.reset_index(drop=True),
            encoder,
        )

    return df_all.reset_index(drop=True), encoder


@materialize("s3://zoomcamps-bucket/data/train/processed")
def export_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    upload_df_to_s3(train_df, f"{PROCESSED_EXPORT_DIR}/train.csv")
    upload_df_to_s3(test_df, f"{PROCESSED_EXPORT_DIR}/test.csv")

    logger.info("Uploaded processed train/test sets to S3")

    summary_md = f"""
# Processed Stock Data Summary
This document summarizes the processed stock data.
- **Train Date Range**: {train_df['Date'].min()} to {train_df['Date'].max()}
- **Test Date Range**: {test_df['Date'].min()} to {test_df['Date'].max()}
- **Train Data Shape**: {train_df.shape}
- **Test Data Shape**: {test_df.shape}
- **Train Data Columns**: {', '.join(train_df.columns)}
- **Test Data Columns**: {', '.join(test_df.columns)}
"""

    train_df_copy = train_df.copy()
    train_df_copy["Date"] = train_df_copy["Date"].astype(str)
    create_table_artifact(
        key="processed-train-head",
        table=train_df_copy.head().to_dict(orient="records"),
        description="Top 5 rows of processed train data",
    )

    create_markdown_artifact(
        key="processed-train-data-summary",
        markdown=summary_md,
        description="Summary of processed train stock data.",
    )


@task(name="Export Encoder")
def export_encoder(encoder, remote_path="encoder.joblib"):
    upload_joblib_to_s3(encoder, remote_path)


@flow(name="Data Preparation Flow")
def data_preparation_flow():
    logger.info("Starting data preparation flow...")
    raw_data = load_data()
    train_df, test_df, encoder = process_data(raw_data)
    export_data(train_df, test_df)
    export_encoder(encoder)
    logger.info("Data preparation flow completed successfully.")


if __name__ == "__main__":
    data_preparation_flow()

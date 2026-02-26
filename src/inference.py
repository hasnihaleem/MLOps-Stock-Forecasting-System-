from prefect.assets import materialize
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from src.data_preparation import process_data
from src.utils.s3_io import upload_df_to_s3, download_joblib_from_s3
import mlflow
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict
from prefect import flow, task
from src.utils.logger import get_logger

logger = get_logger(__name__)

STOCK_LIST = ["AAPL", "GOOGL", "MSFT", "NVDA", "AMZN", "META"]
INFER_RAW_DIR = "data/inference/raw"
INFER_PROCESSED_DIR = "data/inference/processed"
PREDICTION_DIR = "data/inference/prediction"
ENCODER_PATH = "encoder.joblib"
TODAY = datetime.today().strftime("%Y-%m-%d")
YESTERDAY = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
MLFLOW_DB_URI = (
    "postgresql://neondb_owner:npg_x1OqnLgvpmZ9@"
    "ep-empty-dew-a1d7ga54-pooler.ap-southeast-1.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)


@task(name="Load Model from Registry")
def load_model_from_registry(registered_model_name="stock-ema-model"):
    mlflow.set_registry_uri(MLFLOW_DB_URI)
    model_uri = f"models:/{registered_model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri, dst_path="./registered_model")
    logger.info("Successfully logging model from registry.")
    return model


@task(name="Load Encoder")
def load_encoder(remote_path=ENCODER_PATH):
    return download_joblib_from_s3(remote_path)


@materialize("s3://zoomcamps-bucket/data/inference/raw")
def load_inference_data(stock_list=STOCK_LIST, start_day=YESTERDAY) -> Dict:
    logger.info(
        f"Loading stock data at {start_day} for \
        {len(stock_list)} stocks."
    )
    df_all = yf.download(
        stock_list, start=start_day, group_by="ticker", auto_adjust=True
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
                key=f"{ticker.lower()}-raw",
                table=df_copy.to_dict(orient="records"),
                description=f"Raw inference {ticker} data",
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
        key="raw-inference-data-summary",
        markdown=markdown,
        description="Summary of raw stock data loaded from Yahoo Finance.",
    )

    for ticker, df in all_data.items():
        upload_df_to_s3(df, f"{INFER_RAW_DIR}/{ticker}.csv")
        logger.info(
            f"Uploaded {ticker} data to S3: \
            {INFER_RAW_DIR}/{ticker}.csv"
        )
    return all_data


@materialize("s3://zoomcamps-bucket/data/inference/processed")
def export_processed_inference_data(
    df_infer=pd.DataFrame, export_dir=INFER_PROCESSED_DIR
):
    upload_df_to_s3(df_infer, s3_path=f"{export_dir}/infer.csv")

    summary_md = f"""
# Processed Inference Stock Data Summary
This document summarizes the processed inference stock data.
- **Date Range**: {df_infer['Date'].min()} to {df_infer['Date'].max()}
- **Data Shape**: {df_infer.shape}
- **Data Columns**: {', '.join(df_infer.columns)}
"""

    df_infer_copy = df_infer.copy()
    df_infer_copy["Date"] = df_infer_copy["Date"].astype(str)
    create_table_artifact(
        key="processed-infer-head",
        table=df_infer_copy.to_dict(orient="records"),
        description="Processed infer data",
    )

    create_markdown_artifact(
        key="processed-infer-data-summary",
        markdown=summary_md,
        description="Summary of processed infer stock data.",
    )

    logger.info("Successfully export inference data on s3.")


@task(name="Inference")
def run_batch_prediction(model, df_infer, for_evaluate=False) -> pd.DataFrame:
    if not for_evaluate:
        feature_cols = df_infer.columns.difference(["Date", "Ticker"]).tolist()
    else:
        feature_cols = df_infer.columns.difference(
            ["Date", "Ticker", "Next_Close"]
        ).tolist()

    prediction = model.predict(df_infer[feature_cols])
    df_prediction = pd.DataFrame(
        {
            "Ticker": df_infer["Ticker"].values,
            "Prediction": prediction,
        }
    )

    return df_prediction


@materialize("s3://zoomcamps-bucket/data/inference/prediction")
def export_prediction(df_prediction: pd.DataFrame):
    s3_path = f"{PREDICTION_DIR}/{TODAY}.csv"
    upload_df_to_s3(df_prediction, s3_path=s3_path)

    df_prediction_copy = df_prediction.copy()
    create_table_artifact(
        key="prediction",
        table=df_prediction_copy.to_dict(orient="records"),
        description="Prediction data",
    )

    logger.info(f"Prediction saved at {s3_path}")


@flow(name="Inference Flow")
def inference_flow():
    model = load_model_from_registry()
    encoder = load_encoder()
    raw_data = load_inference_data()
    df_infer, _ = process_data(raw_data, for_inference=True, encoder=encoder)
    export_processed_inference_data(df_infer)
    df_prediction = run_batch_prediction(model, df_infer)
    export_prediction(df_prediction)
    logger.info("Inference Flow completed successfully.")
    return df_prediction


if __name__ == "__main__":
    inference_flow()

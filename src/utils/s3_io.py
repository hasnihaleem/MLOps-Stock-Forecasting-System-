import pandas as pd
from io import BytesIO
from prefect_aws.s3 import S3Bucket
# from datetime import datetime, timedelta
import joblib


def upload_df_to_s3(df: pd.DataFrame, s3_path: str):
    s3_bucket_block = S3Bucket.load("zoomcamps")
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3_bucket_block.upload_from_file_object(buffer, s3_path)


def download_df_from_s3(s3_path: str) -> pd.DataFrame:
    s3_bucket_block = S3Bucket.load("zoomcamps")
    buffer = BytesIO()
    s3_bucket_block.download_object_to_file_object(s3_path, buffer)
    buffer.seek(0)
    return pd.read_csv(buffer)


def upload_joblib_to_s3(obj, remote_path: str):
    s3_bucket_block = S3Bucket.load("zoomcamps")
    buffer = BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    s3_bucket_block.upload_from_file_object(buffer, remote_path)


def download_joblib_from_s3(remote_path: str):
    s3_bucket_block = S3Bucket.load("zoomcamps")
    buffer = BytesIO()
    s3_bucket_block.download_object_to_file_object(remote_path, buffer)
    buffer.seek(0)
    return joblib.load(buffer)


# if __name__ == "__main__":
#     today = datetime.today()
#     yesterday = today - timedelta(days=0)
#     print(f"Downloading data from S3 for {yesterday.strftime("%Y-%m-%d")}...")
#     df2 = download_df_from_s3(f"data/inference/prediction/{yesterday.strftime("%Y-%m-%d")}.csv")
#     print(df2)

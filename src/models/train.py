import pandas as pd
import mlflow
import optuna
from src.utils.objective import create_objective
from src.utils.mlflow import create_versioned_experiment
from prefect import flow, task
from src.utils.s3_io import download_df_from_s3
from src.utils.logger import get_logger

logger = get_logger(__name__)


DATA_DIR = "data/train/processed"
MLFLOW_DB_URI = "postgresql://neondb_owner:npg_x1OqnLgvpmZ9@ep-empty-dew-a1d7ga54-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
TARGET = "Next_Close"


@task(name="Load Processed Data")
def load_data():
    train_df = download_df_from_s3(f"{DATA_DIR}/train.csv")
    test_df = download_df_from_s3(f"{DATA_DIR}/test.csv")
    logger.info("Successfully downloaded train and test df from S3 Bucket.")
    return train_df, test_df


@task(name="Train Model with Optuna")
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, n_trials: int = 20):
    mlflow.set_tracking_uri(MLFLOW_DB_URI)

    new_experiment = create_versioned_experiment()
    mlflow.set_experiment(new_experiment)

    feature_cols = train_df.columns.difference([TARGET, "Date", "Ticker"]).tolist()

    logger.info(f"Starting Optuna optimization with {n_trials} trials...")

    def optuna_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.01, 0.1),
        }
        objective_fn = create_objective(train_df, test_df, feature_cols, TARGET)
        # The code is a bit cumbersome since MLflow's `log_params` doesn't take
        # Optuna's trial as its argument
        return objective_fn(params)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=n_trials)
    best_rmse = study.best_value
    logger.info("Optuna optimization completed.")

    return best_rmse


@flow(name="Train Stock Price Model")
def train_flow():
    train_df, test_df = load_data()
    rmse = train_model(train_df, test_df, n_trials=5)
    logger.info(f"Training complete - RMSE on test set: {rmse:.4f}")


if __name__ == "__main__":
    train_flow()

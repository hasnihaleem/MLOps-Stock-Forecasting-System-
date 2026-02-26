from prefect import flow
from src.utils.logger import get_logger
from src.utils.mlflow import get_latest_versioned_experiment
import mlflow

logger = get_logger(__name__)
MLFLOW_TRACKING_URI = "postgresql://neondb_owner:npg_x1OqnLgvpmZ9@ep-empty-dew-a1d7ga54-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
REGISTER_NAME = "stock-ema-model"


@flow(name="Best Model Register Flow")
def register_best_model_flow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment_name = get_latest_versioned_experiment()
    logger.info(f"Searching for best model in experiment: {experiment_name}")
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.rmse < 9999",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    best_run = runs[0]
    run_id = best_run.info.run_id
    rmse = best_run.data.metrics["rmse"]
    model_uri = f"runs:/{run_id}/rf_model"

    logger.info(f"Registering best model from run_id={run_id} with RMSE={rmse:.4f}")
    result = mlflow.register_model(model_uri=model_uri, name=REGISTER_NAME)

    logger.info(f"Model registered: name={result.name}, version={result.version}")


if __name__ == "__main__":
    register_best_model_flow()

import random
from prefect import flow
from src.data_preparation import data_preparation_flow
from src.models.train import load_data
from src.inference import load_model_from_registry, run_batch_prediction
from sklearn.metrics import root_mean_squared_error
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from src.utils.logger import get_logger

logger = get_logger(__name__)


@flow(name="Evaluate Current Model")
def evaluate_model_flow():
    data_preparation_flow()
    _, df_tst = load_data()
    model = load_model_from_registry()
    y_pred = run_batch_prediction(model=model, df_infer=df_tst, for_evaluate=True)[
        "Prediction"
    ]
    y_true = df_tst["Next_Close"]
    rmse_value = root_mean_squared_error(y_true, y_pred)

    registry = CollectorRegistry()
    rmse_metric = Gauge(
        "stock_forecast_test_rmse", "RMSE on test data", registry=registry
    )
    rmse_metric.set(rmse_value)

    push_to_gateway("localhost:9091", job="stock_forecast_evaluator", registry=registry)

    logger.info(f"Pushed RMSE {rmse_value} to Pushgateway")
    return rmse_value


def alert_triggered():
    rmse_value = random.randint(21, 30)

    registry = CollectorRegistry()
    rmse_metric = Gauge(
        "stock_forecast_test_rmse", "RMSE on test data", registry=registry
    )
    rmse_metric.set(rmse_value)

    push_to_gateway("localhost:9091", job="stock_forecast_evaluator", registry=registry)

    logger.info(f"Pushed RMSE {rmse_value} to Pushgateway")
    return rmse_value


if __name__ == "__main__":
    # Uncomment the line below to trigger an alert
    # alert_triggered()

    # Comment the line below when trying to trigger an alert
    evaluate_model_flow()

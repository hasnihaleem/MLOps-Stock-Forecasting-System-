import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


# Yes I know .autolog() exists, but it logs too much redundant informations
def create_objective(train_df, test_df, feature_cols, target_col):
    def objective(params):

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model", "random_forest")
            mlflow.log_params(params)

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="rf_model",
                input_example=test_df[feature_cols].iloc[:5].to_dict(orient="records"),
                params=params,
            )

        return rmse

    return objective

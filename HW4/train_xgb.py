import mlflow
from xgboost import XGBClassifier
import sys
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from HW1.data_preprocessing import load_and_process

FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "fare_amount",
]
TARGET_COLUMN = "tip_applied"

PRIMARY_METRIC = "val_loss"

def train_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tip_applied_xgb")

    january = load_and_process(_REPO_ROOT / "data" / "green_tripdata_2021-01.parquet")
    february = load_and_process(_REPO_ROOT / "data" / "green_tripdata_2021-02.parquet")
    march = load_and_process(_REPO_ROOT / "data" / "green_tripdata_2021-03.parquet")

    data = pd.concat([january, february])
    test = march

    train, val = train_test_split(data, random_state=113, test_size=0.2)
    print(f"train size: {train.shape}, val size: {val.shape}, test size: {test.shape}")

    X_train = train[FEATURE_COLUMNS]
    X_val = val[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train[TARGET_COLUMN]
    y_val = val[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0],
    }

    results = []
    best_primary = float("inf")
    best_model = None
    best_params = None

    for params in ParameterGrid(param_grid):
        with mlflow.start_run():
            print(f"Running model: {params}")
            mlflow.set_tag("model_family", "xgboost")
            mlflow.set_tag("run_type", "hyperparameter_search")

            for k, v in params.items():
                mlflow.log_param(k, v)

            model = XGBClassifier(**params, eval_metric="logloss", random_state=113)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_proba = model.predict_proba(X_train)
            val_proba = model.predict_proba(X_val)

            metrics = {
                "train_f1": f1_score(y_train, y_train_pred),
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_loss": log_loss(y_train, train_proba),
                "val_f1": f1_score(y_val, y_val_pred),
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_loss": log_loss(y_val, val_proba),
            }
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            row = {
                **params,
                **metrics,
                "ml_flow_run_id": mlflow.active_run().info.run_id,
            }
            results.append(row)

            if metrics[PRIMARY_METRIC] < best_primary:
                best_primary = metrics[PRIMARY_METRIC]
                best_model = model
                best_params = dict(params)

    if best_model is None:
        raise RuntimeError("No model was trained; check ParameterGrid.")

    with mlflow.start_run(run_name="best_xgboost") as best_run:
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("run_type", "best_model")
        mlflow.set_tag("primary_metric", PRIMARY_METRIC)
        mlflow.set_tag("primary_metric_lower_is_better", "true")

        for k, v in best_params.items():
            mlflow.log_param(k, v)

        mlflow.xgboost.log_model(best_model, artifact_path="model")

        y_train_pred = best_model.predict(X_train)
        train_proba_bm = best_model.predict_proba(X_train)
        y_val_pred = best_model.predict(X_val)
        val_proba_bm = best_model.predict_proba(X_val)
        y_test_pred = best_model.predict(X_test)
        test_proba = best_model.predict_proba(X_test)

        best_metrics = {
            "train_f1": f1_score(y_train, y_train_pred),
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_loss": log_loss(y_train, train_proba_bm),
            "val_f1": f1_score(y_val, y_val_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_loss": log_loss(y_val, val_proba_bm),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_loss": log_loss(y_test, test_proba),
        }
        for name, value in best_metrics.items():
            mlflow.log_metric(name, value)

        best_run_id = best_run.info.run_id

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "tip_applied_xgboost")

    print(
        f"Best {PRIMARY_METRIC}={best_primary:.4f} (lower is better) with params {best_params}. "
        f"Best model run id: {best_run_id}"
    )
    return results, best_run_id, best_params


if __name__ == "__main__":
    train_model()

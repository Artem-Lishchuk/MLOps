import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

_HW4 = Path(__file__).resolve().parent
_REPO = _HW4.parent
sys.path.insert(0, str(_REPO))
from HW1.data_preprocessing import load_and_process

FEATURES = [
    "passenger_count",
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "fare_amount",
]

EXPERIMENT_TO_REGISTRY = {
    "tip_applied_logreg": "tip_applied_logistic_regression",
    "tip_applied_rf": "tip_applied_random_forest",
    "tip_applied_xgb": "tip_applied_xgboost",
}

if __name__ == "__main__":
    db = (_HW4 / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    client = MlflowClient()

    rows = []
    for exp_name, reg_name in EXPERIMENT_TO_REGISTRY.items():
        exp = client.get_experiment_by_name(exp_name)
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.run_type = 'best_model'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError(f"No run with tags.run_type=best_model in {exp_name!r}.")
        row = runs.iloc[0]
        test_loss = float(row["metrics.test_loss"])

        rows.append(
            {
                "experiment": exp_name,
                "registry": reg_name,
                "run_id": row["run_id"],
                "test_loss": test_loss,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    winner = df.loc[df["test_loss"].idxmin(), "registry"]
    best_run_id = df.loc[df["registry"] == winner, "run_id"].iloc[0]
    print(f"\nProduction (smallest logged test_loss): {winner}\n")

    for _, r in df.iterrows():
        reg = r["registry"]
        vers = client.search_model_versions(f"name='{reg}'")
        ver = max(vers, key=lambda v: int(v.version))
        stage = "Production" if reg == winner else "Staging"
        client.transition_model_version_stage(
            name=reg,
            version=ver.version,
            stage=stage,
            archive_existing_versions=False,
        )
        print(f"{reg} v{ver.version} -> {stage}")

    # Same URI shape as register_model(runs:/.../model) in train_*.py
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    march = load_and_process(_REPO / "data" / "green_tripdata_2021-03.parquet")
    X = march[FEATURES]
    predictions = model.predict(X)
    print(f"\nLoaded {model_uri}")
    print(f"March rows: {len(predictions)}, pred class counts: {pd.Series(predictions).value_counts().to_dict()}")

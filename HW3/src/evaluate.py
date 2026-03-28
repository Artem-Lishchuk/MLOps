"""Test-set metrics for the trained model."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def main() -> None:
    hw3_dir = Path(__file__).resolve().parent.parent
    with open(hw3_dir / "params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluate"]
    engine = data_cfg["engine"]

    proc = hw3_dir / split_cfg["processed_dir"]
    test_df = pd.read_parquet(proc / split_cfg["test_file"], engine=engine)

    feature_columns = feat_cfg["feature_columns"]
    target = feat_cfg["target_column"]
    X_test = test_df[feature_columns]
    y_test = test_df[target]

    with open(hw3_dir / model_cfg["model_output"], "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    timestamp = int(time.time())
    metrics = {
        "timestamp": timestamp,
        "params": {
            "n_estimators": model_cfg["n_estimators"],
            "max_depth": model_cfg["max_depth"],
        },
        "metrics": {
            "f1": float(f1_score(y_test, y_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    }

    out_path = hw3_dir / eval_cfg["metrics_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if they exist
    all_metrics = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
                if not isinstance(all_metrics, list):
                    all_metrics = [all_metrics]
        except (json.JSONDecodeError, ValueError):
            all_metrics = []

    all_metrics.append(metrics)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Metrics appended to {out_path}")


if __name__ == "__main__":
    main()

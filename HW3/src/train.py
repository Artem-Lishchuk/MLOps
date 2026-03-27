"""Train RandomForestClassifier with hyperparameters from params.yaml."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def main() -> None:
    hw3_dir = Path(__file__).resolve().parent.parent
    with open(hw3_dir / "params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    engine = data_cfg["engine"]

    proc = hw3_dir / split_cfg["processed_dir"]
    train_df = pd.read_parquet(proc / split_cfg["train_file"], engine=engine)

    feature_columns = feat_cfg["feature_columns"]
    target = feat_cfg["target_column"]

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    rf_keys = ("n_estimators", "max_depth", "random_state")
    clf_params = {k: model_cfg[k] for k in rf_keys if k in model_cfg}
    model = RandomForestClassifier(**clf_params)
    model.fit(X_train, y_train)

    model_path = hw3_dir / model_cfg["model_output"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()

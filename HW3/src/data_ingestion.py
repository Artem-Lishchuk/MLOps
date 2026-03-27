"""
Load raw parquet files, apply the same preprocessing as HW1 load_and_process,
concatenate months, write a single interim table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLUMNS_TO_DROP = [
    "VendorID",
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "extra",
    "mta_tax",
    "ehail_fee",
    "improvement_surcharge",
    "total_amount",
    "payment_type",
    "trip_type",
    "congestion_surcharge",
    "tolls_amount",
]


def _drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in COLUMNS_TO_DROP if c in data.columns]
    return data.drop(columns=cols_present)


def _filter_rows(data: pd.DataFrame) -> pd.DataFrame:
    data = data[data["passenger_count"] > 0]
    data = data[data["trip_distance"] > 0]
    data = data[data["fare_amount"] > 0]
    data = data[data["tip_amount"] >= 0]
    return data


def _add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["pickup_hour"] = data["lpep_pickup_datetime"].dt.hour
    data["pickup_day_of_week"] = data["lpep_pickup_datetime"].dt.dayofweek
    data["tip_applied"] = (data["tip_amount"] > 0).astype(int)
    return data


def _drop_lpep_pickup_datetime(data: pd.DataFrame) -> pd.DataFrame:
    if "lpep_pickup_datetime" in data.columns:
        return data.drop(columns=["lpep_pickup_datetime"])
    return data


def _drop_tip_amount(data: pd.DataFrame) -> pd.DataFrame:
    if "tip_amount" in data.columns:
        return data.drop(columns=["tip_amount"])
    return data


def _leave_99th_percentile(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[data[column] <= np.percentile(data[column], 99)]


def _preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = _drop_columns(data)
    data = _filter_rows(data)
    data = _add_features(data)
    data = _drop_lpep_pickup_datetime(data)
    data = _drop_tip_amount(data)
    data = _leave_99th_percentile(data, "trip_distance")
    return data


def _load_and_process(filepath: Path, engine: str) -> pd.DataFrame:
    data = pd.read_parquet(filepath, engine=engine)
    return _preprocess(data)


def main() -> None:
    hw3_dir = Path(__file__).resolve().parent.parent
    params_path = hw3_dir / "params.yaml"
    with open(params_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    engine = data_cfg["engine"]
    out_path = hw3_dir / data_cfg["interim_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for rel in data_cfg["raw_files"]:
        path = (hw3_dir / rel).resolve()
        frames.append(_load_and_process(path, engine=engine))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False, engine=engine)


if __name__ == "__main__":
    main()

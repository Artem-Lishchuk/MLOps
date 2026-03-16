import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import dvc.api

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


def load_data_from_path(filepath: str | Path, engine: str = "fastparquet") -> pd.DataFrame:
    return pd.read_parquet(filepath, engine=engine)

def load_data_from_dvc(dvc_path: str, engine: str = "fastparquet", repo: str | Path | None = None) -> pd.DataFrame:
    import os
    orig_cwd = os.getcwd()
    try:
        if repo is not None:
            os.chdir(Path(repo).resolve())
        with dvc.api.open(dvc_path, mode="rb") as f:
            return pd.read_parquet(f, engine=engine)
    finally:
        os.chdir(orig_cwd)

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in COLUMNS_TO_DROP if c in data.columns]
    return data.drop(columns=cols_present)


def remove_passenger_count_zero(data: pd.DataFrame) -> pd.DataFrame:
    """Remove trips where passenger count is 0."""
    return data[data["passenger_count"] > 0]


def remove_trip_distance_zero(data: pd.DataFrame) -> pd.DataFrame:
    """Remove trips where trip distance is 0."""
    return data[data["trip_distance"] > 0]


def remove_fare_amount_non_positive(data: pd.DataFrame) -> pd.DataFrame:
    """Remove trips with not-positive fare amount."""
    return data[data["fare_amount"] > 0]


def remove_tip_amount_non_positive(data: pd.DataFrame) -> pd.DataFrame:
    """Remove trips with not-positive tip amount."""
    return data[data["tip_amount"] >= 0]

def drop_lpep_pickup_datetime(data: pd.DataFrame) -> pd.DataFrame:
    if "lpep_pickup_datetime" in data.columns:
        return data.drop(columns=["lpep_pickup_datetime"])
    return data

def drop_tip_amount(data: pd.DataFrame) -> pd.DataFrame:
    if "tip_amount" in data.columns:
        return data.drop(columns=["tip_amount"])
    return data

def filter_rows(data: pd.DataFrame) -> pd.DataFrame:
    data = remove_passenger_count_zero(data)
    data = remove_trip_distance_zero(data)
    data = remove_fare_amount_non_positive(data)
    data = remove_tip_amount_non_positive(data)
    return data

def drop_tip_applied(data: pd.DataFrame) -> pd.DataFrame:
    if "tip_applied" in data.columns:
        return data.drop(columns=["tip_applied"])
    return data

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["pickup_hour"] = data["lpep_pickup_datetime"].dt.hour
    data["pickup_day_of_week"] = data["lpep_pickup_datetime"].dt.dayofweek
    data["tip_applied"] = (data["tip_amount"] > 0).astype(int)
    return data

def leave_99th_percentile(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[data[column] <= np.percentile(data[column], 99)]

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_columns(data)
    data = filter_rows(data)
    data = add_features(data)
    data = drop_lpep_pickup_datetime(data)
    data = drop_tip_amount(data)
    data = leave_99th_percentile(data, "trip_distance")
    return data


def load_and_process(filepath: str | Path, engine: str = "fastparquet", from_dvc: bool = False,repo: str | Path | None = None,
) -> pd.DataFrame:
    if from_dvc:
        data = load_data_from_dvc(str(filepath), engine=engine, repo=repo)
    else:
        data = load_data_from_path(filepath, engine=engine)
    return preprocess(data)


def train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"
    indices = np.arange(len(data))
    idx_train, idx_temp = train_test_split(
        indices, train_size=train_ratio, random_state=random_state
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp, train_size=val_size, random_state=random_state
    )
    return data.iloc[idx_train], data.iloc[idx_val], data.iloc[idx_test]

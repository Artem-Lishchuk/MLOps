import pandas as pd
from pathlib import Path

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


def load_data(filepath: str | Path, engine: str = "fastparquet") -> pd.DataFrame:
    return pd.read_parquet(filepath, engine=engine)


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


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["pickup_hour"] = data["lpep_pickup_datetime"].dt.hour
    data["pickup_day_of_week"] = data["lpep_pickup_datetime"].dt.dayofweek
    data["tip_applied"] = (data["tip_amount"] > 0).astype(int)
    return data

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_columns(data)
    data = filter_rows(data)
    data = add_features(data)
    data = drop_lpep_pickup_datetime(data)
    data = drop_tip_amount(data)
    return data


def load_and_process(filepath: str | Path, engine: str = "fastparquet") -> pd.DataFrame:
    data = load_data(filepath, engine=engine)
    return preprocess(data)

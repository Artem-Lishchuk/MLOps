"""Train / val / test split from interim combined table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int | None,
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


def main() -> None:
    hw3_dir = Path(__file__).resolve().parent.parent
    with open(hw3_dir / "params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    engine = data_cfg["engine"]

    interim_path = hw3_dir / data_cfg["interim_output"]
    combined = pd.read_parquet(interim_path, engine=engine)

    train, val, test = train_val_test_split(
        combined,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_state=split_cfg["random_state"],
    )

    out_dir = hw3_dir / split_cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out_dir / split_cfg["train_file"], index=False, engine=engine)
    val.to_parquet(out_dir / split_cfg["val_file"], index=False, engine=engine)
    test.to_parquet(out_dir / split_cfg["test_file"], index=False, engine=engine)


if __name__ == "__main__":
    main()

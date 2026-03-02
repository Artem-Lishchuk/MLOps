# Classification Model

- **Target:** `tip_applied` (1 = tip given, 0 = no tip)
- **Model:** Random Forest Classifier
- **Metric:** F1-score (primary); Accuracy, Precision, Recall, Confusion matrix (auxiliary)

## Features

| Feature              | Description                          |
|----------------------|--------------------------------------|
| `passenger_count`    | Number of passengers                 |
| `trip_distance`      | Distance of the trip                 |
| `pickup_hour`        | Hour of day (0–23)                   |
| `pickup_day_of_week` | Day of week (0=Mon, 6=Sun)           |
| `fare_amount`        | Trip fare in USD                     |

## Data Pipeline

1. **Load & preprocess** — `load_and_process('dataset/green_tripdata_2021-01.parquet')`
2. **Train/val/test split** — `train_val_test_split(data, random_state=42)` (default 70/15/15)

## Hyperparameter Tuning

`ParameterGrid` is used to search over:

- `n_estimators`: 50, 100, 200, 300
- `max_depth`: 3, 6, 10, 20
- `random_state`: 42 (fixed)

The best configuration is chosen by **highest validation F1-score**.

## Workflow

1. Load data and prepare features
2. Split into train / validation / test
3. Grid search over hyperparameters on train set
4. Select best model by validation F1
5. Report test metrics (F1, accuracy, precision, recall, confusion matrix) on held-out test set

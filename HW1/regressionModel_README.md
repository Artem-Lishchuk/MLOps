# Regression Model

- **Target:** `fare_amount`
- **Model:** Random Forest Regressor
- **Metric:** MSE (Mean Squared Error)

## Features

| Feature             | Description                          |
|---------------------|--------------------------------------|
| `passenger_count`   | Number of passengers                 |
| `trip_distance`     | Distance of the trip                 |
| `pickup_hour`       | Hour of day (0–23)                   |
| `pickup_day_of_week`| Day of week (0=Mon, 6=Sun)           |

## Data Pipeline

1. **Load & preprocess** — `load_and_process('dataset/green_tripdata_2021-01.parquet')`
2. **Drop tip_applied** — `drop_tip_applied(data)`
3. **Train/val/test split** — `train_val_test_split(data, random_state=120)` (default 70/15/15)

## Hyperparameter Tuning

`ParameterGrid` is used to search over:

- `n_estimators`: 50, 100, 200, 300
- `max_depth`: 3, 6, 7, 8, 10
- `random_state`: 100 (fixed)

The best configuration is chosen by **lowest validation MSE**.

## Workflow

1. Load data and prepare features
2. Split into train / validation / test
3. Grid search over hyperparameters on train set
4. Select best model by validation MSE
5. Report test MSE on held-out test set
# Data Preprocessing

Preprocessing pipeline.

## Main Steps

1. **Drop columns** — Remove columns not needed for regression (fare prediction) or classification (tip prediction): VendorID, location IDs, fare components (extra, mta_tax, tolls, etc.), and other metadata.

2. **Filter rows** — Leave only valid records:
   - Passenger count > 0
   - Trip distance > 0
   - Fare amount > 0
   - Tip amount ≥ 0

3. **Add features** — Create features from existing columns:
   - `pickup_hour` — Hour of day (0–23) from pickup datetime
   - `pickup_day_of_week` — Day of week (0=Monday, 6=Sunday)
   - `tip_applied` — Boolean (1 if tip given, 0 otherwise)

4. **Drop auxiliary columns** — Remove `lpep_pickup_datetime` and `tip_amount` after feature extraction.

## Architecture

load_data → drop_columns → filter_rows → add_features → drop_auxiliary_columns

load_and_process(filepath) for load + full preprocessing in one call.

## Output

After preprocessing, the dataframe contains: `passenger_count`, `trip_distance`, `fare_amount`, `pickup_hour`, `pickup_day_of_week`, `tip_applied`.


**Note (Production):** In a production environment, the preprocessing contract would be looser - e.g. configurable lists of columns to drop, features to add, and row filters — so engineers can select which features to delete, add, or modify without changing core logic. For simplicity, keeping the current fixed pipeline.

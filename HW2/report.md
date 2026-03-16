# Practice 2 Report – Model and Data Versioning

## Data Preprocessing (from HW1)

Source: `HW1/data_preprocessing.py`

**Pipeline:**

1. **Drop columns** – remove VendorID, location IDs, fare components, and other metadata not needed for prediction.
2. **Filter rows** – keep only valid records: passenger count > 0, trip distance > 0, fare amount > 0, tip amount ≥ 0.
3. **Add features** – create `pickup_hour` (0–23), `pickup_day_of_week` (0=Monday), `tip_applied` (1 if tip > 0).
4. **Drop auxiliary columns** – remove `lpep_pickup_datetime` and `tip_amount` after feature extraction.
5. **Outlier removal** – filter `trip_distance` to the 99th percentile.

**Output features:** `passenger_count`, `trip_distance`, `fare_amount`, `pickup_hour`, `pickup_day_of_week`, `tip_applied`

**Architecture:**
```
load_data → drop_columns → filter_rows → add_features → drop_auxiliary → outlier_filter
```

---

## Classification Model (from HW1)

Source: `HW1/classificationModel_README.md`

- **Target:** `tip_applied` (1 = tip given, 0 = no tip)
- **Model:** Random Forest Classifier
- **Primary metric:** F1-score; secondary: Accuracy, Precision, Recall, Confusion Matrix
- **Features:** `passenger_count`, `trip_distance`, `pickup_hour`, `pickup_day_of_week`, `fare_amount`
- **Hyperparameter tuning:** Grid search over `n_estimators` (50, 100, 200, 300) and `max_depth` (3, 6, 10, 20); best by validation F1.
- **Split:** 70/15/15 (train/val/test), `random_state=42`

---

## Version 1 – First Version of Code, Data, and Model

**Git commit:** `412ed3f` – *"v1 model. Classification model from HW1"*  
**Notebook:** `HW2/classificationModel_v1.ipynb`

### Data
- Single dataset: `data/green_tripdata_2021-01.parquet` (January 2021 NYC green taxi trips)
- Loaded via DVC from Google Drive, preprocessed with `load_and_process()`
- Dataset size after preprocessing: **37,819 rows × 6 columns**
- Split: 70/15/15 → train: 26,473 | val: 5,673 | test: 5,673

### Model
- Best hyperparameters: `n_estimators=100`, `max_depth=6`, `random_state=42`
- Selected by highest validation F1-score
- Saved as `HW2/models/model_pkl_v1` (tracked by DVC → Google Drive, not in Git)

### Results (V1 test set)

| Metric    | Value  |
|-----------|--------|
| F1-score  | 0.6077 |
| Accuracy  | 0.5584 |
| Precision | 0.5482 |
| Recall    | 0.6817 |

**Confusion matrix:**

|                   | Predicted No Tip | Predicted Tip |
|-------------------|-----------------|---------------|
| **Actual No Tip** | 1228            | 1599          |
| **Actual Tip**    | 906             | 1940          |

The model has an accuracy (~56%). The higher recall than precision means the model tends to predict "tip given" more liberally — it catches more actual tips but at the cost of more false positives (trips classified as tipped when they were not).

---

## Version 2 – New Data, Same Model and Training Code

**Git commit:** `c3b45a1` – *"Ex.2 validated the model trained on smaller dataset on combined test set"*  
**Notebook:** `HW2/modelValidation.ipynb`

### What Changed
- **Data:** Added `data/green_tripdata_2021-02.parquet` (February 2021). Both months concatenated.
- **Evaluation code:** Load V1 model from disk, evaluate on the new combined test set.
- **Unchanged:** Model (`model_pkl_v1`), training code, hyperparameters.

### Data
- Combined dataset: January + February 2021
- Total size after preprocessing: **71,261 rows × 6 columns**
- Split: 70/15/15, `random_state=42` → test set: ~10,689 rows

### Results (V2 – V1 model on combined test set)

| Metric    | Value  |
|-----------|--------|
| F1-score  | 0.6008 |
| Accuracy  | 0.5579 |
| Precision | 0.5322 |
| Recall    | 0.6897 |

**Confusion matrix:**

|                   | Predicted No Tip | Predicted Tip |
|-------------------|-----------------|---------------|
| **Actual No Tip** | 2407            | 3126          |
| **Actual Tip**    | 1600            | 3557          |

### Analysis

Comparing V1 to V2 on the same model:

| Metric    | V1 (Jan only) | V2 (Jan+Feb) | Change    |
|-----------|--------------|--------------|-----------|
| F1-score  | 0.6077       | 0.6008       | **−0.007** |
| Accuracy  | 0.5584       | 0.5579       | −0.001    |
| Precision | 0.5482       | 0.5322       | **−0.016** |
| Recall    | 0.6817       | 0.6897       | +0.008    |

**Explanation:** The V1 model was trained on January data only. When evaluated on a test set that includes February data, performance drops slightly. This is expected: the model has not seen February distribution during training. The slight decrease in precision and F1 while recall stays roughly stable suggests the model was more aggressive in predicting "tip given" on the February data — possibly because February trips have slightly different fare or time distributions that the model learned to associate with tips from January. The overall drop is small (~0.7% F1)

---

## Version 3 – New Model, Same Data

**Git commit:** `cbcae90` – *"training model on the combined training data"*  
**Git commit:** `af7b274` – *"added 3rd model into dvc"*  
**Notebook:** `HW2/classificationModel_v3.ipynb`

### What Changed
- **Training data:** Combined January + February dataset (same as V2).
- **Model:** Retrained from scratch on the combined training set.
- **Unchanged:** Data and training code.

### Data
- Combined dataset: January + February 2021 (same as V2)
- Split: 70/15/15, `random_state=113` → train: 49,882 | val: 10,689 | test: 10,690
- New model saved as `HW2/models/model_pkl_v3` (tracked by DVC → Google Drive)

### Best Hyperparameters (V3)

| Parameter    | Value |
|--------------|-------|
| n_estimators | 300   |
| max_depth    | 3     |
| random_state | 42    |

### Results (V3 test set)

| Metric    | Value  |
|-----------|--------|
| F1-score  | 0.6278 |
| Accuracy  | 0.5649 |
| Precision | 0.5349 |
| Recall    | 0.7596 |

**Confusion matrix:**

|                   | Predicted No Tip | Predicted Tip |
|-------------------|-----------------|---------------|
| **Actual No Tip** | 2117            | 3410          |
| **Actual Tip**    | 1241            | 3922          |

### Analysis Across All Versions

| Metric    | V1 (Jan, Jan test) | V2 (V1 model, Jan+Feb test) | V3 (retrained, Jan+Feb test) |
|-----------|--------------------|-----------------------------|------------------------------|
| F1-score  | 0.6077             | 0.6008                      | **0.6278**                   |
| Accuracy  | 0.5584             | 0.5579                      | **0.5649**                   |
| Precision | 0.5482             | 0.5322                      | 0.5349                       |
| Recall    | 0.6817             | 0.6897                      | **0.7596**                   |

Retraining on the combined dataset (V3) improves over the V2 metrics. F1 increases from 0.6008 to 0.6278 and recall jumps to 0.7596.

## Task 4 – Monitoring V3 in Production

Assuming V3 (`model_pkl_v3`) is deployed to production, the following metrics should be monitored:

### a) Data-related metric – Distribution of Features

Monitoring the distribution of input features. If there's external change of the degree of the input features, model will produce bad results and don't generalize well from training.

### b) Model performance metric – F1-score or Accuracy

Monitor rolling F1-score on a weekly or bi-weekly sample of trips where the actual outcome (tip given or not) becomes available shortly after the ride.

### c) System metric – Latency

Increase in the inference latency indicates an issue with the model

### When to Retrain

- **Retrain** if feature distribution change or if the model evals get worse

### When to Roll Back

- **Roll back to V1**  since retraining takes a lot of time to be done properly, rolling back to V1 has to be done in case, there needs to be a quick replacement - e.g. V3 model isn't working well with existing infrasture of is overfitted, for example, and shows bad production results
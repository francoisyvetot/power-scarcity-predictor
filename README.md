# Power System Scarcity Predictor

This project is a machine learning pipeline designed to predict electricity scarcity events in various bidding zones. A scarcity event is defined as an hour where the electricity price exceeds 500 €/MWh.

The pipeline can operate in two main modes:
1. **Hourly to Weekly:** Trains on hourly data and aggregates the predictions to evaluate scarcity at a weekly level.
2. **Weekly Native:** Trains and evaluates directly on weekly aggregated data.

## Models Used
The pipeline supports tree-based gradient boosting models:
- **XGBoost (`xgboost`):** A powerful gradient boosting framework, configured to handle imbalanced data by scaling positive weights.
- **LightGBM (`lightgbm`):** A fast and efficient gradient boosting model that handles categorical features (like Bidding Zones) natively.

## Cross-Validation (CV) Methods
To ensure the models generalize well to unseen data, the pipeline uses robust cross-validation splitters:
- **Leave-One-Target-Year-Out (`leave_one_ty_out`):** Trains on all available target years except one, which is kept for testing. This tests the model's ability to predict future years.
- **Group K-Fold (`group_kfold`):** Splits the data based on Weather Scenarios (WS). This ensures that the same weather scenario does not appear in both the training and testing sets, preventing data leakage.
- **Stratified Group K-Fold (`stratified_group_kfold`):** Similar to Group K-Fold, but it also attempts to keep the percentage of scarcity events (the target) balanced across the different folds.

## Decision Thresholds
Because scarcity events are extremely rare, the pipeline allows configuring probability thresholds directly in `config.yaml` to adjust the model's sensitivity:
- **Aggregation Threshold (`hourly_to_weekly_agg`):** In the hourly pipeline, this defines the minimum probability required to flag a specific *hour* as a scarcity event. It is used to count the number of critical hours in a week.
- **Evaluation Thresholds (`weekly_evaluation`):** These are the final decision thresholds used to classify a *whole week* as experiencing a scarcity event. Providing a list (e.g., `[0.3, 0.5, 0.75]`) allows you to evaluate the trade-off between Recall (catching all events at lower thresholds) and Precision (avoiding false alarms at higher thresholds).

## Evaluation Metrics
The pipeline uses metrics specifically chosen for highly imbalanced datasets (since scarcity is a rare event):
- **PR-AUC (Precision-Recall Area Under Curve):** The main metric used to evaluate performance on rare events.
- **Precision & Recall:** Analyzed at different probability thresholds to find the best decision boundary.
- **Classification Report:** Standard evaluation including F1-score for overall accuracy assessment.
- **SHAP Values:** Used to explain the model's predictions and understand feature impacts during actual scarcity crises.
- **Feature Importance:** Measured via Permutation Importance (impact on PR-AUC) and native model Gain.
- **Spatial Correlation:** A custom diagnostic to check if neighboring bidding zones experience scarcity at the same time.

## Engineered Features
The models use a rich set of features extracted from system load and generation data.

### Hourly Features
- **Temporal:** `Month`, `ISO_Week`, `DayOfWeek`, `Hour_sin`, `Hour_cos`, `is_off_day` (weekends/holidays), `bridge_day`.
- **Load & Demand:** `Net_Load` (Load minus renewables), `Net_Load_Core` (sum of net load over the zone).
- **Ramps (Variations):** `Ramp_Net_Load_1h`, `Ramp_Net_Load_3h` (captures sudden changes in demand).
- **Weather & Renewables:** `Onshore`, `Offshore`, `PV`, and rolling averages like `Wind_mean_3h/12h/24h` and `PV_Peak_mean_24h`.
- **Historical Lags:** `Net_Load_Lag_1` (previous hour) and `Net_Load_Lag_24` (previous day).

### Weekly Features (Native Mode)
- **Aggregated Load:** `Net_Load_Core_Peak_p95` (95th percentile of peak load), `Net_Load_Core_ExSelf_mean`.
- **Ramps & Drops:** `Pos_Ramp_Max`, `Neg_Ramp_Max`, `RES_Drop_max` (largest drop in renewable generation).
- **Low Renewable Regimes (Dunkelflaute):** `Low_RES_Hours` (number of hours with very low renewables), `Longest_Low_RES_Run` (longest consecutive streak of low renewables), `Total_RES_min`, `Onshore_min`, `Offshore_min`.
- **Weekly Lags:** `Net_Load_Core_mean_Lag_1w` (mean load of the previous week).

## How to Run
The entire pipeline is controlled by the `config.yaml` file. You can change the `grain` (hourly or weekly), select models, and choose the cross-validation strategy in one place without touching the Python code.

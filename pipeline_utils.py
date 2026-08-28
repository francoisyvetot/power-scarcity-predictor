"""
Scarcity Predictor pipeline : train/test split, model training, evaluation, feature ablation
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, precision_recall_curve, auc

from metrics import PowerSystemDiagnostics


class WeeklyAggregator:
    """
    Weekly evaluation level, applied after training
    The model is trained on the raw hourly flag (aligned with its features)
    this class aggregates its hourly probabilities into a weekly score/flag
    """
    @staticmethod
    def aggregate(X: pd.DataFrame, y_true_hourly: pd.Series, probs_hourly: np.ndarray,
                  agg: str = "max", threshold: float = 0.6) -> pd.DataFrame:
        """
        Aggregates hourly probabilities and true flags at the weekly level
        y_true_hourly is the raw hourly flag (not broadcast), aligned row by row with probs_hourly
        so that the reconstructed weekly label is correct
        """
        keys = WeeklyAggregator._week_keys(X)
        keys['prob'] = probs_hourly
        keys['y_true_hour'] = y_true_hourly.values

        group_cols = ['Bidding Zone', 'WS', 'TY', 'ISO_Year', 'ISO_Week']
        weekly = keys.groupby(group_cols).agg(
            prob_max=('prob', 'max'),
            prob_mean=('prob', 'mean'),
            n_hours_critical=('prob', lambda s: (s > threshold).sum()),
            y_true_week=('y_true_hour', 'max'),
        ).reset_index()

        weekly['prob_week'] = weekly['prob_max'] if agg == "max" else weekly['prob_mean']
        weekly['flag_pred_n_hours'] = (weekly['n_hours_critical'] >= 1).astype(int)
        return weekly
    
    @staticmethod
    def evaluate(weekly: pd.DataFrame, threshold: float | list[float], label: str = "") -> None:
        """ Classification report + PR-AUC at the weekly grain (true decision)."""
        if isinstance(threshold, list):
            for t in threshold:
                preds_week = (weekly['prob_week'] > t).astype(int)
                print(f"\n================== WEEKLY EVALUATION {label} (threshold={threshold}) ==================")
                print(f"\n--- Metrics @ Threshold = {t:.2f} ---")
                print(classification_report(weekly['y_true_week'], preds_week, zero_division=0))
                precision, recall, _ = precision_recall_curve(weekly['y_true_week'], weekly['prob_week'])
                print(f"PR-AUC (week): {auc(recall, precision):.4f}")
        else:
            preds_week = (weekly['prob_week'] > threshold).astype(int)
            print(f"\n================== WEEKLY EVALUATION {label} (threshold={threshold}) ==================")
            print(classification_report(weekly['y_true_week'], preds_week, zero_division=0))
            precision, recall, _ = precision_recall_curve(weekly['y_true_week'], weekly['prob_week'])
            print(f"PR-AUC (week): {auc(recall, precision):.4f}")
        print("=" * 70 + "\n")


    @staticmethod
    def _week_keys(X: pd.DataFrame) -> pd.DataFrame:
        iso = X.index.isocalendar()
        return pd.DataFrame({
            'Bidding Zone': X['Bidding Zone'].values,
            'WS': X['WS'].values,
            'TY': X['TY'].values,
            'ISO_Year': iso['year'].values,
            'ISO_Week': iso['week'].values,
        }, index=X.index)


def build_model(model_type: str, scale_weight: float):
    if model_type == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=1000, early_stopping_rounds=50, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_weight, eval_metric='logloss',
            enable_categorical=True, min_child_weight=5, subsample=0.7,
            colsample_bytree=0.8, random_state=42,
        )
    if model_type == 'lightgbm':
        return lgb.LGBMClassifier(
            n_estimators=1000, max_depth=3, learning_rate=0.05,
            scale_pos_weight=scale_weight, min_child_samples=20, subsample=0.7,
            feature_fraction=0.8, random_state=42, verbosity=-1,
        )
    raise ValueError(f"Unsupported model type: {model_type}. Supported: xgboost, lightgbm")


def make_inner_validation_split(X_train_full: pd.DataFrame, y_train_full: pd.Series,
                                 train_groups: np.ndarray, val_fraction: float = 0.2,
                                 random_state: int = 42):
    """
    early_stopping_rounds needs an eval_set
    So we split the training set into a fit set and a validation set.
    """
    unique_train_ws = np.unique(train_groups)
    rng = np.random.default_rng(random_state)
    n_val = max(1, int(round(len(unique_train_ws) * val_fraction)))
    val_ws = rng.choice(unique_train_ws, size=n_val, replace=False)
    is_val = np.isin(train_groups, val_ws)
    X_fit, y_fit = X_train_full.iloc[~is_val], y_train_full.iloc[~is_val]
    X_val, y_val = X_train_full.iloc[is_val], y_train_full.iloc[is_val]
    return X_fit, y_fit, X_val, y_val


def check_feature_extrapolation(X_train: pd.DataFrame, X_test: pd.DataFrame, top_n_report: int = 6):
    """
    Diagnostic : Decision trees can't extrapolate beyond the range of values seen in training
     """
    rows = []
    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            continue
        train_min, train_max = X_train[col].min(), X_train[col].max()
        below = (X_test[col] < train_min).mean() * 100
        above = (X_test[col] > train_max).mean() * 100
        if below + above > 0:
            rows.append({
                'Feature': col,
                'Train_range': f"[{train_min:.0f}, {train_max:.0f}]",
                'Test_range': f"[{X_test[col].min():.0f}, {X_test[col].max():.0f}]",
                '%_test_below_train_min': round(below, 1),
                '%_test_above_train_max': round(above, 1),
            })
    if rows:
        report = pd.DataFrame(rows).sort_values('%_test_above_train_max', ascending=False)
        print("\nDiagnostic (test features are out of the train range) :")
        print(report.head(top_n_report).to_string(index=False))
        print()
    else:
        print("\n Diagnostic : No features tested out of the train range.\n")



class HourlyToWeeklyPipeline:
    """
    Model-agnosting pipeline for scarcity prediction using XGBoost, LightGBM or RandomForest
    Same GroupKFold, same evaluation via WeeklyAggregator
    StratifiedGroupKFold, LeaveOneTargetYearOut (cv_splitters.py) 
    """

    def __init__(self, feature_cols: list, model_type: str = 'xgboost'):
        self.feature_cols = feature_cols
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.X_test_full = None  # X_test avec toutes les colonnes, pour WeeklyAggregator
        self.y_train = None
        self.y_test = None

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_cols].copy()

    def execute_time_series_validation(self, X: pd.DataFrame, y: pd.Series, cv_splitter,
                                        target_ws: list | None = None, val_fraction: float = 0.2,
                                        agg_threshold: float = 0.75, eval_thresholds: list | float = [0.3, 0.5, 0.75]):
        """
        cv_splitter : any function that does .split(X, y, groups) -> (train_idx, test_idx),
        exampli gratia sklearn.model_selection.GroupKFold(...), StratifiedGroupKFold(...), or
        cv_splitters.LeaveOneTargetYearOut(...)
        """
        common_idx = X.index.intersection(y.index)
        X, y = X.loc[X.index.isin(common_idx)], y.loc[y.index.isin(common_idx)]
        groups = X['WS'].values  # for the inner validation split (make_inner_validation_split) and for the outer CV split

        X_features = self._prepare_features(X)

        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            self.X_train, self.X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            self.X_test_full = X.iloc[test_idx]  # for WeeklyAggregator (needs WS/TY/Bidding Zone)
            self.y_train, self.y_test = y.iloc[train_idx], y.iloc[test_idx]

            test_scenarios = np.unique(groups[test_idx])
            test_years = np.unique(X.iloc[test_idx]['TY'].values)
            raw_ratio = sum(self.y_train == 0) / max(1, sum(self.y_train == 1))
            scale_weight = max(np.log(raw_ratio), 1.0) if raw_ratio > 0 else 1.0  # avoid negative weights
            pos_rate = 100 * sum(self.y_train == 1) / len(self.y_train)

            print(f"\n--- [{self.model_type}] Fold {fold + 1} ({cv_splitter.__class__.__name__}) ---")
            print(f"Test TY: {test_years} | Test WS: {test_scenarios}")
            print(f"Training set: {len(self.X_train)}h | Scarcity events: {sum(self.y_train == 1)} ({pos_rate:.2f}%)")

            if target_ws is not None:
                covered = sorted(set(target_ws) & set(test_scenarios))
                missing = sorted(set(target_ws) - set(covered))
                print(f"% of target WS in that fold : {len(covered)}/{len(target_ws)}"
                      + (f" -- MISSING: {missing}" if missing else ""))

            check_feature_extrapolation(self.X_train, self.X_test)

            train_groups = groups[train_idx]
            X_fit, y_fit, X_val, y_val = make_inner_validation_split(
                self.X_train, self.y_train, train_groups, val_fraction=val_fraction)

            self.model = build_model(self.model_type, scale_weight)

            if self.model_type == 'lightgbm' and 'Bidding Zone' in X_fit.columns:
                self.model.fit(
                    X_fit, y_fit,
                    eval_set=[(X_fit, y_fit), (X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                    categorical_feature=['Bidding Zone'],
                )
            else:
                self.model.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)
                print(f"Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
                print(f"Best score: {getattr(self.model, 'best_score', 'N/A')}")

            if self.model_type == 'xgboost':
                PowerSystemDiagnostics.log_feature_importance_gain(self.model, self.feature_cols)
            else:
                importances = pd.Series(
                    self.model.feature_importances_, index=X_fit.columns
                ).sort_values(ascending=False)
                print(f"\nFeature Importance ({self.model_type}):\n{importances.to_string()}\n")

            PowerSystemDiagnostics.log_shap_scarcity_impact(self.model, self.X_test, self.y_test)

            probs = self.model.predict_proba(self.X_test)[:, 1]

            weekly_diag = WeeklyAggregator.aggregate(self.X_test_full, self.y_test, probs, agg="max", threshold=agg_threshold)

            WeeklyAggregator.evaluate(weekly_diag, eval_thresholds, label="(EVERY WS IN THE FOLD)")

            if target_ws is not None:
                weekly_diag_target = weekly_diag[weekly_diag['WS'].isin(target_ws)]
                if len(weekly_diag_target) > 0:
                    WeeklyAggregator.evaluate(weekly_diag_target, eval_thresholds, label=f"(TARGET WS ONLY, n={len(target_ws)})")
                else:
                    print("None of the target WS are in this fold, no filtered evaluation possible.\n")

            precision, recall, _ = precision_recall_curve(self.y_test, probs)
            print(f"PR-AUC hourly (secondary metric): {auc(recall, precision):.4f}")
            PowerSystemDiagnostics.analyze_prediction_thresholds(self.y_test, probs)

        return self.X_train, self.y_train, self.X_test, self.y_test


class WeeklyNativePipeline:
    """
    Grain weekly native : trains and evaluates directly at the weekly grain (1 row = 1 week) 
    Same structure as HourlyToWeeklyPipeline; only the evaluation step differs since X_test is already at the right grain.
    """
    SUPPORTED_MODELS = ['xgboost', 'lightgbm']
 
    def __init__(self, feature_cols: list, model_type: str = 'xgboost'):
        self.feature_cols = feature_cols
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.X_test_full = None  # X_test avec WS/TY, pour le filtrage sur target_ws
        self.y_train = None
        self.y_test = None
 
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_cols].copy()
 
    def execute_time_series_validation(self, X: pd.DataFrame, y: pd.Series, cv_splitter,
                                        target_ws: list | None = None, val_fraction: float = 0.2,
                                        eval_thresholds: list | float = [0.3, 0.5, 0.75]):
        common_idx = X.index.intersection(y.index)
        X, y = X.loc[X.index.isin(common_idx)], y.loc[y.index.isin(common_idx)]
        groups = X['WS'].values
 
        X_features = self._prepare_features(X)
 
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            self.X_train, self.X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            self.X_test_full = X.iloc[test_idx]
            self.y_train, self.y_test = y.iloc[train_idx], y.iloc[test_idx]
 
            test_scenarios = np.unique(groups[test_idx])
            test_years = np.unique(X.iloc[test_idx]['TY'].values)
            raw_ratio = sum(self.y_train == 0) / max(1, sum(self.y_train == 1))
            scale_weight = max(np.log(raw_ratio), 1.0) if raw_ratio > 0 else 1.0
            pos_rate = 100 * sum(self.y_train == 1) / len(self.y_train)
 
            print(f"\n--- [{self.model_type} | weekly-native] Fold {fold + 1} ({cv_splitter.__class__.__name__}) ---")
            print(f"Test TY: {test_years} | Test WS: {test_scenarios}")
            print(f"Training weeks: {len(self.X_train)} | Positive weeks: {sum(self.y_train == 1)} ({pos_rate:.1f}%)")
 
            if target_ws is not None:
                covered = sorted(set(target_ws) & set(test_scenarios))
                missing = sorted(set(target_ws) - set(covered))
                print(f"Couverture des WS cibles dans ce fold : {len(covered)}/{len(target_ws)}"
                      + (f" -- MANQUANTES: {missing}" if missing else ""))
 
            check_feature_extrapolation(self.X_train, self.X_test)
 
            train_groups = groups[train_idx]
            X_fit, y_fit, X_val, y_val = make_inner_validation_split(
                self.X_train, self.y_train, train_groups, val_fraction=val_fraction)
 
            self.model = build_model(self.model_type, scale_weight)
 
            if self.model_type == 'lightgbm' and 'Bidding Zone' in X_fit.columns:
                self.model.fit(
                    X_fit, y_fit,
                    eval_set=[(X_fit, y_fit), (X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                    categorical_feature=['Bidding Zone'],
                )
            else:
                self.model.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)
                print(f"Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
                print(f"Best score (validation interne, PAS le fold de test): {getattr(self.model, 'best_score', 'N/A')}")
 
            if self.model_type == 'xgboost':
                PowerSystemDiagnostics.log_feature_importance_gain(self.model, self.feature_cols)
            else:
                importances = pd.Series(
                    self.model.feature_importances_, index=X_fit.columns
                ).sort_values(ascending=False)
                print(f"\nFeature Importance ({self.model_type}):\n{importances.to_string()}\n")
 
            PowerSystemDiagnostics.log_shap_scarcity_impact(self.model, self.X_test, self.y_test)
 
            probs = self.model.predict_proba(self.X_test)[:, 1]
            preds = (probs > 0.5).astype(int)
            for t in eval_thresholds:
                preds = (probs > t).astype(int)
                print(f"\n================ ÉVALUATION HEBDOMADAIRE NATIVE (TOUTES LES WS DU FOLD) @ seuil={t} ================")
                print(classification_report(self.y_test, preds, zero_division=0))
                precision_m, recall_m, _ = precision_recall_curve(self.y_test, probs)
                print(f"PR-AUC (semaine, natif): {auc(recall_m, precision_m):.4f}")
                print("=" * 90 + "\n")
     
                if target_ws is not None:
                    mask = self.X_test_full['WS'].isin(target_ws).values
                    if mask.any():
                        print(f"============ ÉVALUATION HEBDOMADAIRE NATIVE (WS CIBLES UNIQUEMENT, n={len(target_ws)}) @ seuil={t} ============")
                        print(classification_report(self.y_test[mask], preds[mask], zero_division=0))
                        precision_t, recall_t, _ = precision_recall_curve(self.y_test[mask], probs[mask])
                        print(f"PR-AUC (semaine, WS cibles): {auc(recall_t, precision_t):.4f}")
                        print("=" * 90 + "\n")
                    else:
                        print("Aucune des WS cibles dans ce fold — pas d'évaluation filtrée possible.\n")
 
            PowerSystemDiagnostics.analyze_prediction_thresholds(self.y_test, probs)
 
        return self.X_train, self.y_train, self.X_test, self.y_test

    
def run_feature_ablation(X: pd.DataFrame, y: pd.Series, feature_cols: list, drop_features: list,
                          cv_splitter, pipeline_cls, model_type: str = 'xgboost', target_ws: list | None = None, **kwargs):
    """
    Retrain without the features listed in drop_features and compare to the full model, fold by fold
    Both runs use the same cv_splitter (same split on both sides)
    """
    reduced_cols = [f for f in feature_cols if f not in drop_features]
    print(f"\n\n########## ABLATION : {model_type.upper()} — sans {drop_features} ##########")

    print("\n--- FULL MODEL ---")
    full_pipeline = pipeline_cls(feature_cols=feature_cols, model_type=model_type)
    full_pipeline.execute_time_series_validation(X, y, cv_splitter, target_ws=target_ws, **kwargs)

    print("\n--- MODEL WITHOUT DOMINANT FEATURES ---")
    reduced_pipeline = pipeline_cls(feature_cols=reduced_cols, model_type=model_type)
    reduced_pipeline.execute_time_series_validation(X, y, cv_splitter, target_ws=target_ws, **kwargs)

    return full_pipeline, reduced_pipeline
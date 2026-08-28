import numpy as np
import pandas as pd
 
from data_loader import PowerDataLoader
from cv_splitters import build_cv_splitter
from pipeline_utils import WeeklyNativePipeline, run_feature_ablation



class WeeklyFeatureEngineer:
    """
    Builds weekly features from hourly data. The process is as follows:
    1 row = 1 (Bidding Zone, WS, TY, ISO week).
    Hourly features (Net_Load, Net_Load_Core, ramps, RES) are first calculated at the hourly level 
    (essential to capture extremes/slopes), THEN summarized by week never naively averaged to avoid
    diluting rare events in a 168h average.
    """
    def __init__(self, res_columns: list, low_res_quantile: float = 0.10, ramp_quantile: float = 0.90,
                 min_hours_per_week: int = 150):
        self.res_columns = res_columns
        self.low_res_quantile = low_res_quantile
        self.ramp_quantile = ramp_quantile
        self.min_hours_per_week = min_hours_per_week

    def _build_hourly_base(self, df_system: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
        X_df = df_system.reset_index().merge(
            df_gen.reset_index(),
            on=['Datetime', 'Bidding Zone', 'WS', 'TY'],
            how='inner'
        )
        # Tri explicite zone/scenario/temps : indispensable pour que diff() et le calcul
        # de "plus longue série" (longest run) soient corrects par groupe.
        X_df = X_df.sort_values(['Bidding Zone', 'WS', 'TY', 'Datetime']).set_index('Datetime')
        mask_peakhours = X_df.index.hour.isin([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) & X_df.index.dayofweek.isin([1, 2, 3, 4, 5])
        X_df['Total_RES'] = X_df[self.res_columns].sum(axis=1)
        X_df['Net_Load'] = X_df['Load_MW'] - X_df['Total_RES']
        X_df['Net_Load_Peak'] = X_df['Net_Load'].where(mask_peakhours)
        X_df['Net_Load_OffPeak'] = X_df['Net_Load'].where(~mask_peakhours)

        X_df['Net_Load_Core'] = X_df.groupby([X_df.index, 'WS', 'TY'])['Net_Load'].transform('sum')
        X_df['Net_Load_Core_Peak'] = X_df.groupby([X_df.index, 'WS', 'TY'])['Net_Load_Peak'].transform('sum')
        X_df['Net_Load_Core_OffPeak'] = X_df.groupby([X_df.index, 'WS', 'TY'])['Net_Load_OffPeak'].transform('sum')
        # Isole la pression EXTERNE (retire l'auto-corrélation avec le Net_Load propre de la zone,
        # cf. discussion précédente sur la dominance de Net_Load_Core dans les SHAP)    
        X_df['Net_Load_Core_ExSelf'] = X_df['Net_Load_Core'] - X_df['Net_Load']

        grouped = X_df.groupby(['Bidding Zone', 'WS', 'TY'])
        X_df['Net_Load_Ramp'] = grouped['Net_Load'].diff()
        X_df['Positive_Ramp'] = X_df['Net_Load_Ramp'].clip(lower=0).fillna(0)
        X_df['Negative_Ramp'] = X_df['Net_Load_Ramp'].clip(upper=0).fillna(0)
        X_df['Total_RES_Ramp'] = grouped['Total_RES'].diff()
        X_df['Up_Ramp_Max'] = grouped['Net_Load_Ramp'].cummax()
        iso = X_df.index.isocalendar()
        X_df['ISO_Year'] = iso['year'].values
        X_df['ISO_Week'] = iso['week'].values
        X_df['Month'] = X_df.index.month
        X_df['Scarcity_Hour'] = (X_df['Price_Euro_MWh'] > 500).astype(int)

        X_df['PV_Peak'] = X_df['PV'].where(mask_peakhours)

        return X_df
    def _thresholds_per_zone(self, X_df: pd.DataFrame) -> pd.DataFrame:
            """
            Calculate thresholds for low RES and high ramp hours per Bidding Zone.
            """
            return X_df.groupby('Bidding Zone').agg(
                Low_RES_Threshold=('Total_RES', lambda s: s.quantile(self.low_res_quantile)),
                High_Ramp_Threshold=('Net_Load_Ramp', lambda s: s.abs().quantile(self.ramp_quantile)),
            )
    
    @staticmethod
    def _longest_run(binary_values: np.ndarray) -> int:
        """ Return length of the longest consecutive run of 1s in a binary array."""
        max_run = current = 0
        for v in binary_values:
            if v == 1:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run

    def generate_weekly_features(self, df_system: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
        X_df = self._build_hourly_base(df_system, df_gen)
        thresholds = self._thresholds_per_zone(X_df)
        X_df = X_df.join(thresholds, on='Bidding Zone')

        X_df['Is_Low_RES_Hour'] = (X_df['Total_RES'] < X_df['Low_RES_Threshold']).astype(int)
        X_df['Is_High_Ramp_Hour'] = (X_df['Net_Load_Ramp'].abs() > X_df['High_Ramp_Threshold']).astype(int)

        group_cols = ['Bidding Zone', 'WS', 'TY', 'ISO_Year', 'ISO_Week']

        weekly = X_df.groupby(group_cols).agg(
            Hours_In_Week=('Net_Load', 'size'),
            Net_Load_Peak_mean=('Net_Load_Peak', 'mean'),
            Net_Load_OffPeak_mean=('Net_Load_OffPeak', 'mean'),
            Net_Load_Peak_p95=('Net_Load_Peak', lambda s: s.quantile(0.95)),
            # Net_Load_OffPeak_p95=('Net_Load_OffPeak', lambda s: s.quantile(0.95)),
            Net_Load_max=('Net_Load', 'max'),
            Net_Load_Core_mean=('Net_Load_Core', 'mean'),
            Net_Load_Core_Peak_p95=('Net_Load_Core_Peak', lambda s: s.quantile(0.95)),
            # Net_Load_Core_OffPeak_p95=('Net_Load_Core_OffPeak', lambda s: s.quantile(0.95)),
            Net_Load_Core_ExSelf_mean=('Net_Load_Core_ExSelf', 'mean'),
            # Rampes = "changements rapides du profil de charge" -> lien direct avec les
            # déclenchements soudains d'ENS évoqués dans la demande.
            Net_Load_Ramp_max=('Net_Load_Ramp', lambda s: s.abs().max()),
            Pos_Ramp_Max=('Positive_Ramp', 'max'),
            Neg_Ramp_Max=('Negative_Ramp', 'min'),
            Net_Load_Ramp_std=('Net_Load_Ramp', 'std'),
            Up_Ramp_Max=('Up_Ramp_Max', 'max'),
            High_Ramp_Hours=('Is_High_Ramp_Hour', 'sum'),
            # Positive_Ramp_Hours=('Positive_Ramp_Flag', 'sum'),
            # Negative_Ramp_Hours=('Negative_Ramp_Flag', 'sum'),
            Load_MW_mean=('Load_MW', 'mean'),
            Total_RES_mean=('Total_RES', 'mean'),
            Total_RES_min=('Total_RES', 'min'),
            Onshore_min=('Onshore', 'min'),
            Offshore_min=('Offshore', 'min'),
            # PV_min serait ~0 presque chaque semaine (nuit) -> non discriminant, volontairement omis.
            PV_Peak_mean=('PV_Peak', 'mean'),
            RES_Drop_max=('Total_RES_Ramp', 'min'),  # plus négatif = plus grosse chute de RES
            Low_RES_Hours=('Is_Low_RES_Hour', 'sum'),
            Longest_Low_RES_Run=('Is_Low_RES_Hour', lambda s: WeeklyFeatureEngineer._longest_run(s.values)),
            Month=('Month', 'first'),
            Scarcity_Week_Flag=('Scarcity_Hour', 'max'),
            Scarcity_Hours_Count=('Scarcity_Hour', 'sum'),  # diagnostic only — JAMAIS en feature (= dérivé de la cible)
        ).reset_index()

        # Semaines partielles aux bords de l'historique (ISO week 1 / 52-53) -> stats
        # dégénérées (percentiles calculés sur 2-3 heures) -> on les enlève.
        weekly = weekly[weekly['Hours_In_Week'] >= self.min_hours_per_week].copy()

        weekly = weekly.sort_values(['Bidding Zone', 'WS', 'TY', 'ISO_Year', 'ISO_Week'])
        grouped_weekly = weekly.groupby(['Bidding Zone', 'WS', 'TY'])
        # Lags à l'échelle semaine (pas heure) : persistance des régimes météo (Dunkelflaute
        # dure souvent plusieurs semaines). Prev_Week_Scarcity_Flag utilise le label de la
        # semaine PRÉCÉDENTE, connu avant le début de la semaine courante -> pas de fuite.
        weekly['Net_Load_Core_mean_Lag_1w'] = grouped_weekly['Net_Load_Core_mean'].shift(1)
        weekly['Net_Load_Peak_mean_Lag_1w'] = grouped_weekly['Net_Load_Peak_mean'].shift(1)
        weekly['Prev_Week_Scarcity_Flag'] = grouped_weekly['Scarcity_Week_Flag'].shift(1)

        weekly['Bidding Zone'] = weekly['Bidding Zone'].astype('category')
        weekly['TY'] = weekly['TY'].astype('category')
        print(len(weekly), "Weeks generated after weekly feature engineering.")
        # print(weekly[weekly.isna().any(axis=1)][['Bidding Zone', 'WS', 'TY', 'ISO_Year', 'ISO_Week']])
        return weekly.dropna(subset=[
            'Net_Load_Core_mean_Lag_1w', 'Net_Load_Peak_mean_Lag_1w', 'Prev_Week_Scarcity_Flag'
        ]).reset_index(drop=True)



def run(config: dict) -> dict:
    data_cfg = config['data']
    loader = PowerDataLoader(
        base_path=data_cfg['path'],
        bidding_zone=data_cfg['bidding_zones'],
        weather_scenario=data_cfg['weather_scenarios'],
        target_year=data_cfg['target_years'],
    )
    df_sys = loader.load_system_data()
    df_gen = loader.load_gen_data()
 
    wfe_cfg = config.get('weekly_feature_engineering', {})
    res_techs = ['Offshore', 'PV', 'Onshore']
    wfe = WeeklyFeatureEngineer(
        res_columns=res_techs,
        low_res_quantile=wfe_cfg.get('low_res_quantile', 0.10),
        ramp_quantile=wfe_cfg.get('ramp_quantile', 0.90),
        min_hours_per_week=wfe_cfg.get('min_hours_per_week', 150),
    )
    weekly_data = wfe.generate_weekly_features(df_sys, df_gen)
    pos_rate = 100 * weekly_data['Scarcity_Week_Flag'].mean()
    print(f"Weekly rows generated: {len(weekly_data)} | positive rate: {pos_rate:.2f}%")
 
    y = weekly_data['Scarcity_Week_Flag']
    feature_cols = config['features']['weekly']
    cv_splitter = build_cv_splitter(config.get('cv', {}))
    target_ws = data_cfg.get('target_ws')

    thresholds_cfg = config.get('thresholds', {})
    eval_thresholds = thresholds_cfg.get('weekly_evaluation', [0.3, 0.5, 0.75])
 
    results = {}
    for model_type in config['model']['types']:
        print(f"\n\n########## [WEEKLY-NATIVE] MODEL: {model_type.upper()} ##########")
        pipeline = WeeklyNativePipeline(feature_cols=feature_cols, model_type=model_type)
        pipeline.execute_time_series_validation(weekly_data, y, cv_splitter, target_ws=target_ws, eval_thresholds=eval_thresholds)
        results[model_type] = pipeline
 
    ablation_cfg = config.get('ablation', {})
    if ablation_cfg.get('enabled', False):
        run_feature_ablation(
            weekly_data, y, feature_cols=feature_cols, drop_features=ablation_cfg['drop_features'],
            cv_splitter=cv_splitter, pipeline_cls=WeeklyNativePipeline,
            model_type=ablation_cfg.get('model_type', 'xgboost'), target_ws=target_ws,
            eval_thresholds=eval_thresholds
        )
 
    return results
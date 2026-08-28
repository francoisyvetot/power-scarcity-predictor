import numpy as np
import pandas as pd
import holidays
 
from data_loader import PowerDataLoader
from cv_splitters import build_cv_splitter
from pipeline_utils import HourlyToWeeklyPipeline, run_feature_ablation
from metrics import PowerSystemDiagnostics


class TargetBuilder:
    """Class responsible for defining the target variable to predict"""
    @staticmethod
    def build_hourly_scarcity_flag(df_system: pd.DataFrame) -> pd.Series:
        """Transforms the unserved energy into a binary or continuous target."""
        flag = (df_system['Price_Euro_MWh'] > 500).astype(int)  # Or Unserved_Energy_MWh if we are working on post CS solution files
        flag.name = 'Scarcity_Target'
        return flag
    

class FeatureEngineer:
    """Class responsible for creating features"""
    def __init__(self,res_columns: list ):
        self.res_columns = res_columns
    
    def generate_features(self, df_system: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
        # Joint on Datetime index + important columns
        X_df = df_system.reset_index().merge(
            df_gen.reset_index(), 
            on=['Datetime', 'Bidding Zone', 'WS', 'TY'], 
            how='inner'
        ).set_index('Datetime')

        df_countries = X_df["Bidding Zone"].str[:2].unique()
        df_years = X_df.index.year.unique().tolist()
        country_holiday_dict = {}
        for country in df_countries:
            try:
                h = holidays.country_holidays(country, years=df_years)
                country_holiday_dict[country] = set(h.keys())
            except NotImplementedError:
                country_holiday_dict[country] = set()
        dates = X_df.index.date
        countries = X_df["Bidding Zone"].str[:2].values
        is_holiday = [d in country_holiday_dict.get(c, set()) for d, c in zip(dates, countries)]
        is_weekend = X_df.index.dayofweek >= 5
        is_yesterday_holiday = np.array([d in country_holiday_dict.get(c, set()) for d, c in zip((X_df.index - pd.Timedelta(days=1)).date, countries)])
        is_tomorrow_holiday = np.array([d in country_holiday_dict.get(c, set()) for d, c in zip((X_df.index + pd.Timedelta(days=1)).date, countries)])
        bridge_friday = (X_df.index.dayofweek.values == 4) & is_yesterday_holiday
        bridge_monday = (X_df.index.dayofweek.values == 0) & is_tomorrow_holiday

        X_df['Total_RES'] = X_df[self.res_columns].sum(axis=1)
        X_df['Net_Load'] = X_df['Load_MW'] - X_df['Total_RES']
        X_df['Net_Load_Core'] = X_df.groupby([X_df.index, 'WS','TY'])['Net_Load'].transform('sum')
        # Features Temporelles
        X_df['Hour'] = X_df.index.hour
        is_peak = X_df['Hour'].between(7, 22)
        pv_peak_only = X_df['PV'].where(is_peak, np.nan)
        X_df['Hour_sin'] = np.sin(2 * np.pi * X_df['Hour'] / 24)
        X_df['Hour_cos'] = np.cos(2 * np.pi * X_df['Hour'] / 24)
        X_df['DayOfWeek'] = X_df.index.dayofweek
        X_df["is_off_day"] = (is_weekend | np.array(is_holiday)).astype(int)
        X_df["bridge_day"] = (bridge_friday | bridge_monday).astype(int)        
        X_df['Ramp_Net_Load_1h'] = X_df.groupby(['Bidding Zone', 'WS', 'TY'])['Net_Load'].diff(1)
        X_df['Ramp_Net_Load_3h'] = X_df.groupby(['Bidding Zone', 'WS', 'TY'])['Net_Load'].diff(3)
        X_df['Wind'] = X_df['Onshore'] + X_df['Offshore']
        grouped_wind = X_df.groupby(['Bidding Zone', 'WS', 'TY'])['Wind']
        X_df['Wind_mean_3h'] = grouped_wind.transform(lambda x: x.rolling(3, min_periods=1).mean())
        X_df['Wind_mean_12h'] = grouped_wind.transform(lambda x: x.rolling(12, min_periods=1).mean())
        X_df['Wind_mean_24h'] = grouped_wind.transform(lambda x: x.rolling(24, min_periods=1).mean())
        X_df['PV_Peak_mean_24h'] = pv_peak_only.groupby([X_df["Bidding Zone"], X_df["WS"], X_df["TY"]]).transform(lambda s: s.rolling(24, min_periods=1).mean())
        X_df['Month'] = X_df.index.month
        X_df['ISO_Week'] = X_df.index.isocalendar().week.astype(int)
        
        grouped = X_df.groupby(['Bidding Zone', 'WS', 'TY'])
        for lag in [1, 2, 24]:
            X_df[f'Load_Lag_{lag}'] = grouped['Load_MW'].shift(lag)
            X_df[f'Net_Load_Lag_{lag}'] = grouped['Net_Load'].shift(lag)
        
        X_df['Bidding Zone'] = X_df['Bidding Zone'].astype('category')

        print(f"Features generated: {list(X_df.columns)}")
        
        return X_df.dropna()

    
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
    PowerSystemDiagnostics.log_spatial_correlations(df_sys)
 
    y = TargetBuilder.build_hourly_scarcity_flag(df_sys)
 
    res_techs = ['Offshore', 'PV', 'Onshore']
    X = FeatureEngineer(res_columns=res_techs).generate_features(df_sys, df_gen)
 
    feature_cols = config['features']['hourly']
    cv_splitter = build_cv_splitter(config.get('cv', {}))
    target_ws = data_cfg.get('target_ws')

    thresholds_cfg = config.get('thresholds', {})
    agg_threshold = thresholds_cfg.get('hourly_to_weekly_agg', 0.75)
    eval_thresholds = thresholds_cfg.get('weekly_evaluation', [0.3, 0.5, 0.75])

    results = {}
    for model_type in config['model']['types']:
        print(f"\n\n########## [HOURLY] MODEL: {model_type.upper()} ##########")
        pipeline = HourlyToWeeklyPipeline(feature_cols=feature_cols, model_type=model_type)
        pipeline.execute_time_series_validation(X, y, cv_splitter, target_ws=target_ws, agg_threshold=agg_threshold, eval_thresholds=eval_thresholds)
        results[model_type] = pipeline
 
    ablation_cfg = config.get('ablation', {})
    if ablation_cfg.get('enabled', False):
        run_feature_ablation(
            X, y, feature_cols=feature_cols, drop_features=ablation_cfg['drop_features'],
            cv_splitter=cv_splitter, pipeline_cls=HourlyToWeeklyPipeline,
            model_type=ablation_cfg.get('model_type', 'xgboost'), target_ws=target_ws,
            agg_threshold=agg_threshold, eval_thresholds=eval_thresholds
        )
 
    return results
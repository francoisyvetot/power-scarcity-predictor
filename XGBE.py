import pandas as pd
import holidays
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from metrics import PowerSystemDiagnostics
from pipeline_utils import ScarcityPredictorPipeline, run_feature_ablation
from cv_splitters import LeaveOneTargetYearOut

class PowerDataLoader:
    def __init__(self, base_path: str, bidding_zone: list, weather_scenario: list, target_year: list):
        self.base_path = base_path
        self.bidding_zone = bidding_zone
        self.weather_scenario = weather_scenario
        self.target_year = target_year

    def load_system_data(self) -> pd.DataFrame:
        """Load demand and ENS"""
        df = pd.read_csv(f"{self.base_path}/SystemData_Hourly.csv")
        df = df[(df['Bidding Zone'].isin(self.bidding_zone)) & 
                (df['WS'].isin(self.weather_scenario)) & 
                (df['TY'].isin(self.target_year))].copy()
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.set_index('Datetime').sort_index()

    def load_gen_data(self) -> pd.DataFrame:
        """Load the gen and pivot it to get 1 column by tech type"""
        df = pd.read_csv(f"{self.base_path}/Generation_By_Tech_Hourly.csv")

        df = df[(df['Bidding Zone'].isin(self.bidding_zone)) & 
                (df['WS'].isin(self.weather_scenario)) & 
                (df['TY'].isin(self.target_year))].copy()
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        df_pivot = df.pivot_table(
            index=['Datetime', 'Bidding Zone', 'WS', 'TY'], 
            columns='Technology_group', 
            values='Generation_MW', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        return df_pivot.set_index('Datetime').sort_index()

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

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\fyvetot\Documents\Datasets\EVA_IDP37"
    # BZONE = ["DE00","FR00","BE00","CZ00","HU00","IE00","NL00","PL00","SK00","UKNI"]

    BZONE = ["DE00","FR00","NL00","BE00","AT00","ES00","PT00","ITN1",
                  "PL00","CZ00","HU00","SE03","DKW1","UK00","IE00"]

    # Weather scenarios targeted for EVA so tested on it
    TARGET_WS = ['WS03','WS06','WS07','WS10','WS12','WS14','WS15','WS16',
                 'WS17','WS18','WS19','WS20','WS25','WS27','WS29','WS31']

    WEATHER_SCENARIO = ['WS01','WS02','WS03','WS05','WS06','WS07','WS08','WS09','WS10','WS11',
                        'WS12','WS13','WS14','WS15','WS16','WS17','WS18','WS19','WS20','WS21',
                        'WS22','WS23','WS24','WS25','WS26','WS27','WS28','WS29','WS30','WS31',
                        'WS32','WS35','WS36']

    # The LOTO function needs at least 2 target years
    TARGET_YEAR = [2030, 2033, 2035]

    loader = PowerDataLoader(base_path=DATA_PATH, bidding_zone=BZONE, weather_scenario=WEATHER_SCENARIO, target_year=TARGET_YEAR)
    df_sys = loader.load_system_data()
    df_gen = loader.load_gen_data()
    # PowerSystemDiagnostics.log_spatial_correlations(df_sys)
    # Call TargetBuilder to build "y" (Scarcity Target)x
    y = TargetBuilder.build_hourly_scarcity_flag(df_sys)

    # Call FeatureEngineer to generate features
    res_techs = ['Offshore', 'PV', 'Onshore']
    fe_engineer = FeatureEngineer(res_columns=res_techs)
    X = fe_engineer.generate_features(df_sys,df_gen)
    
    features_to_use = [
        'Bidding Zone', 'Net_Load', 'Net_Load_Core',
        'DayOfWeek', 'ISO_Week', 'Month',
        'Hour_sin', 'Hour_cos',
        'is_off_day','bridge_day',
        'Ramp_Net_Load_1h', 'Ramp_Net_Load_3h', 'Wind_mean_3h', 'Wind_mean_12h', 'Wind_mean_24h',
        'PV_Peak_mean_24h',
        'Net_Load_Lag_1', 'Net_Load_Lag_24'
    ] + res_techs

    cv_splitter = LeaveOneTargetYearOut() # Or GroupKFold(n_splits=4) / StratifiedGroupKFold(n_splits=4) to split on the WS
    # cv_splitter = GroupKFold(n_splits=4)
    # cv_splitter = StratifiedGroupKFold(n_splits=4)
    results = {}
    for model_type in ['lightgbm']:
        print(f"\n\n########## MODEL: {model_type.upper()} ##########")
        pipeline = ScarcityPredictorPipeline(feature_cols=features_to_use, model_type=model_type)
        pipeline.execute_time_series_validation(X, y, cv_splitter,target_ws = TARGET_WS)
        results[model_type] = pipeline # Last fold of each model for viz

    # Uncomment to check the impact of dropping a dominant feature (Net_Load_Core) on the model's performance
    # run_feature_ablation(X, y, feature_cols=features_to_use, drop_features=['Net_Load_Core'], model_type='xgboost')

    

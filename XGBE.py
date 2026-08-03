import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve, auc
import xgboost as xgb
import shap 
from Scarcity_visualizer import InteractiveScarcityVisualizer
from metrics import PowerSystemDiagnostics
class PowerDataLoader:
    def __init__(self, base_path: str, bidding_zone: list, weather_scenario: list, target_year: list):
        self.base_path = base_path
        self.bidding_zone = bidding_zone
        self.weather_scenario = weather_scenario
        self.target_year = target_year

    def load_system_data(self) -> pd.DataFrame:
        """Load demand and ENS"""
        df = pd.read_csv(f"{self.base_path}/System_data.csv")
        # Filter
        df = df[(df['Bidding Zone'].isin(self.bidding_zone)) & 
                (df['WS'].isin(self.weather_scenario)) & 
                (df['TY'].isin(self.target_year))].copy()
        
        # Time index 
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        return df.set_index('Datetime').sort_index()

    def load_gen_data(self) -> pd.DataFrame:
        """Load the gen and pivot it to get 1 column by tech type"""
        df = pd.read_csv(f"{self.base_path}/Gen_by_tech.csv")

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

    def build_target(self, df_system: pd.DataFrame) -> pd.Series:
        """Transforms the unserved energy into a binary or continuous target."""
        target = (df_system['Unserved_Energy_MWh'] > 0.1).astype(int)
        target.name = 'Scarcity_Target'
        return target

class FeatureEngineer:
    """Class responsible for creating features for ML models"""
    def __init__(self,res_columns: list ):
        self.res_columns = res_columns

    def generate_features(self, df_system: pd.DataFrame, df_gen: pd.DataFrame) -> pd.DataFrame:
        # Joint on Datetime index + important columns
        # Pour éviter les doublons, on utilise les colonnes de clés dans le join
        X_df = df_system.reset_index().merge(
            df_gen.reset_index(), 
            on=['Datetime', 'Bidding Zone', 'WS', 'TY'], 
            how='inner'
        ).set_index('Datetime')
        
        X_df['Total_RES'] = X_df[self.res_columns].sum(axis=1)
        X_df['Net_Load'] = X_df['Load_MW'] - X_df['Total_RES']
                
        # Features Temporelles
        X_df['Hour'] = X_df.index.hour
        X_df['DayOfWeek'] = X_df.index.dayofweek
        X_df['Month'] = X_df.index.month
        X_df['IsWeekend'] = X_df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # CORRECTION CRITIQUE : Calcul des Lags PAR Bidding Zone et PAR Scenario Météo
        # On utilise groupby pour éviter que les données d'une zone ne bavent sur une autre
        grouped = X_df.groupby(['Bidding Zone', 'WS', 'TY'])
        
        for lag in [1, 2, 24]:
            X_df[f'Load_Lag_{lag}'] = grouped['Load_MW'].shift(lag)
            X_df[f'Net_Load_Lag_{lag}'] = grouped['Net_Load'].shift(lag)
            
        # Encodage de la Bidding Zone (XGBoost sait gérer les catégories s'ils sont convertis en type 'category')
        X_df['Bidding Zone'] = X_df['Bidding Zone'].astype('category')
            
        return X_df.dropna()

class ScarcityPredictorPipeline:
    """Classe orchestrant l'entraînement du modèle XGBoost et l'évaluation."""
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.model = None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None

    def execute_time_series_validation(self, X: pd.DataFrame, y: pd.Series):
        """Découpe les horizons (ex: 2028-2035) proprement pour valider le modèle."""
        # Alignement final des indexes
        common_idx = X.index.intersection(y.index)
        X, y = X.loc[X.index.isin(common_idx)], y.loc[y.index.isin(common_idx)]        
        groups = X['WS'].values
        
        X_features = X[self.feature_cols]
        
        # GroupKFold : garantit qu'un WS présent dans le train ne sera JAMAIS dans le test
        gkf = GroupKFold(n_splits=2)

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_features, y, groups)):
            self.X_train, self.X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            self.y_train, self.y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            test_scenarios = np.unique(groups[test_idx])
            # Gestion du déséquilibre de classe (Scarcity = événement rare)
            # scale_pos_weight = (nb d'heures normales) / (nb d'heures de scarcity)
            scale_weight = (sum(self.y_train == 0) / max(1, sum(self.y_train == 1)))
            print(f"\n--- Training for Fold {fold + 1} ---")
            print(f"Weather Scenarios used for TEST in this Fold : {test_scenarios}")
            print(f"Size of Training set: {len(self.X_train)} hours | Scarcity Events in trainset: {sum(self.y_train == 1)}")
            
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_weight,
                eval_metric='logloss',
                enable_categorical=True, # because column Bidding Zone is categorical
                random_state=42
            )
            
            self.model.fit(self.X_train, self.y_train)
            PowerSystemDiagnostics.log_feature_importance_gain(self.model, self.feature_cols)
            # PowerSystemDiagnostics.log_permutation_importance(self.model, self.X_test, self.y_test)
            PowerSystemDiagnostics.log_shap_scarcity_impact(self.model, self.X_test, self.y_test)
            # Évaluation
            preds = self.model.predict(self.X_test)
            probs = self.model.predict_proba(self.X_test)[:, 1]
            
            print(classification_report(self.y_test, preds, zero_division=0))
            precision, recall, _ = precision_recall_curve(self.y_test, probs)
            print(f"PR-AUC (Indicateur clé Crises Rares): {auc(recall, precision):.4f}")
            PowerSystemDiagnostics.analyze_prediction_thresholds(self.y_test, probs)

        return self.X_train, self.y_train, self.X_test, self.y_test  # Retourne le dernier fold pour visualisation



if __name__ == "__main__":
    DATA_PATH = r"C:\Users\fyvetot\Documents\XGBE\Datasets"
    BZONE = ["DE00","FR00","BE00","CZ00","HU00","IE00","NL00","PL00","SK00","UKNI"]
    WEATHER_SCENARIO = ["WS03","WS06","WS07","WS10","WS12","WS14","WS15","WS16","WS17","W18","WS19","WS20","WS25","WS27","WS31","WS32","WS35"]
    TARGET_YEAR = [2035]

    loader = PowerDataLoader(base_path=DATA_PATH, bidding_zone=BZONE, weather_scenario=WEATHER_SCENARIO, target_year=TARGET_YEAR)
    df_sys = loader.load_system_data()
    df_gen = loader.load_gen_data()
    PowerSystemDiagnostics.log_spatial_correlations(df_sys)
    # Call TargetBuilder to build "y" (Scarcity Target)x
    target_builder = TargetBuilder()
    y = target_builder.build_target(df_sys)
    
    # Call FeatureEngineer to generate features
    res_techs = ['Offshore', 'PV', 'Onshore']
    fe_engineer = FeatureEngineer(res_columns=res_techs)
    X = fe_engineer.generate_features(df_sys,df_gen)
    
    # Définition des colonnes à envoyer au XGBoost et exécution du pipeline
    features_to_use = [
        'Bidding Zone','Load_MW', 'Net_Load', 
        'Hour', 'DayOfWeek', 'IsWeekend', # 'Month',
        'Load_Lag_1', 'Net_Load_Lag_1', 'Load_Lag_24', 'Net_Load_Lag_24'
    ] + res_techs

    pipeline = ScarcityPredictorPipeline(feature_cols=features_to_use)
    X_train, y_train, X_test, y_test = pipeline.execute_time_series_validation(X, y)

    probs_test = pipeline.model.predict_proba(X_test[features_to_use])[:, 1]

    # Génération et sauvegarde automatique de l'HTML global sans blocage d'affichage
    InteractiveScarcityVisualizer.save_interactive_profile(
        df_test_features=X_test,
        y_true=y_test,
        y_probs=probs_test,
        bidding_zone="BE00",      # Zone à visualiser
        threshold=0.05,          # Seuil configuré
        output_dir="output",     # Dossier cible
        file_name="scarcity_global_profile_BE00.html"
    )
    

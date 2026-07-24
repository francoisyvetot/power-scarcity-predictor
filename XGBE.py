import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve, auc
import xgboost as xgb
import shap 

class PowerDataLoader:
    """Gère l'ingestion, le filtrage et le pivotage des fichiers de l'ERAA."""
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
        target = (df_system['Price_Euro_MWh'] > 200).astype(int)
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

    def execute_time_series_validation(self, X: pd.DataFrame, y: pd.Series):
        """Découpe les horizons (ex: 2028-2035) proprement pour valider le modèle."""
        # Alignement final des indexes
        common_idx = X.index.intersection(y.index)
        X, y = X.iloc[X.index.isin(common_idx)], y.iloc[y.index.isin(common_idx)]        
        groups = X['WS'].values
        
        # Sélection des features finales pour le modèle
        X_features = X[self.feature_cols]
        
        # GroupKFold : garantit qu'un WS présent dans le train ne sera JAMAIS dans le test
        gkf = GroupKFold(n_splits=5)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_features, y, groups)):
            X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            test_scenarios = np.unique(groups[test_idx])
            # Gestion du déséquilibre de classe (Scarcity = événement rare)
            # scale_pos_weight = (nb d'heures normales) / (nb d'heures de scarcity)
            scale_weight = (sum(y_train == 0) / max(1, sum(y_train == 1)))
            print(f"\n--- Training for Fold {fold + 1} ---")
            print(f"Weather Scenarios used for TEST in this Fold : {test_scenarios}")
            print(f"Size of Training set: {len(X_train)} hours | Scarcity Events in trainset: {sum(y_train == 1)}")
            
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                scale_pos_weight=scale_weight,
                eval_metric='logloss',
                enable_categorical=True, # because column Bidding Zone is categorical
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            PowerSystemDiagnostics.log_feature_importance_gain(self.model, self.feature_cols)
            PowerSystemDiagnostics.log_permutation_importance(self.model, X_test, y_test)
            PowerSystemDiagnostics.log_shap_scarcity_impact(self.model, X_test, y_test)
            # Évaluation
            preds = self.model.predict(X_test)
            probs = self.model.predict_proba(X_test)[:, 1]
            
            print(classification_report(y_test, preds, zero_division=0))
            precision, recall, _ = precision_recall_curve(y_test, probs)
            print(f"PR-AUC (Indicateur clé Crises Rares): {auc(recall, precision):.4f}")
            PowerSystemDiagnostics.analyze_prediction_thresholds(y_test, probs)

class PowerSystemDiagnostics:
    """Outils d'analyse physique et spatiale pour l'adéquation du réseau."""
    
    def log_spatial_correlations(df_system: pd.DataFrame):
        """Calcule et affiche la co-occurrence des événements de scarcity entre les pays."""
        print("\n==================================================")
        print("DIAGNOSTIC : CO-OCCURRENCE DE LA SCARCITY (CORRÉLATION SPATIALE)")
        print("==================================================")
        
        # On crée une table pivot : Index = Datetime/WS, Colonnes = Pays, Valeurs = 1 si scarcity
        pivot_scarcity = df_system.reset_index().pivot_table(
            index=['Datetime', 'WS', 'TY'],
            columns='Bidding Zone',
            values='Price_Euro_MWh',
            aggfunc='sum'
        ).fillna(0)
        
        binary_scarcity = (pivot_scarcity > 0).astype(int)
        
        # Calcul de la matrice de corrélation de Pearson sur les événements binaires
        corr_matrix = binary_scarcity.corr()
        print("Matrice de corrélation spatiale des événements de Scarcity :")
        print(corr_matrix.round(3))
        print("\nInterprétation : Un score proche de 1 indique que les deux zones tombent en panne en même temps.")
        print("==================================================\n")

    @staticmethod
    def analyze_prediction_thresholds(y_true, y_probs):
        """Analyse l'impact du seuil de probabilité sur le Recall et la Précision."""
        print("\n==================================================")
        print("DIAGNOSTIC : OPTIMISATION DU SEUIL DE DÉCISION (THRESHOLD)")
        print("==================================================")
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        # On cherche différents compromis
        df_thresholds = pd.DataFrame({
            'Threshold': np.append(thresholds, 1.0),
            'Precision': precisions,
            'Recall': recalls
        })
        
        # On filtre pour afficher des seuils clés
        print(f"{'Seuil Proba':<15} | {'Précision':<12} | {'Recall (Taux de détection)':<25}")
        print("-" * 60)
        for target_recall in [0.90, 0.75, 0.50, 0.25]:
            # Trouver le seuil qui s'approche le plus du recall visé
            idx = (df_thresholds['Recall'] - target_recall).abs().idxmin()
            row = df_thresholds.loc[idx]
            print(f"{row['Threshold']*100:.1f}%{'':<11} | {row['Precision']*100:.1f}%{'':<7} | {row['Recall']*100:.1f}%")
        print("==================================================\n")
    
    # Logs of importance for each features
    @staticmethod
    def log_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series):
        """2. Calcule la baisse de PR-AUC lorsqu'une variable est permutée (bruitée)."""
        print("\n==================================================")
        print("DIAGNOSTIC : PERMUTATION IMPORTANCE (IMPACT SUR PR-AUC)")
        print("==================================================")
        
        # Définition d'un scorer personnalisé basé sur le PR-AUC
        def pr_auc_scorer(estimator, X, y):
            probs = estimator.predict_proba(X)[:, 1]
            p, r, _ = precision_recall_curve(y, probs)
            return auc(r, p)

        # Permutation sur le Test set
        result = permutation_importance(
            model, X_test, y_test, 
            scoring=pr_auc_scorer, 
            n_repeats=5, 
            random_state=42,
            n_jobs=-1
        )
        
        df_perm = pd.DataFrame({
            'Feature': X_test.columns,
            'Mean_PR_AUC_Drop': result.importances_mean,
            'Std_Dev': result.importances_std
        }).sort_values(by='Mean_PR_AUC_Drop', ascending=False)

        print(df_perm.to_string(index=False))
        print("\nInterprétation : Une valeur positive élevée indique que dégrader la variable fait chuter le PR-AUC.")
        print("==================================================\n")

    @staticmethod
    def log_feature_importance_gain(model, feature_names: list):
        """1. Affiche l'importance native basée sur le Gain."""
        booster = model.get_booster()
        score_gain = booster.get_score(importance_type='gain')
        
        # Formatage sous forme de DataFrame propre
        df_imp = pd.DataFrame({
            'Feature': list(score_gain.keys()),
            'Gain': list(score_gain.values())
        }).sort_values(by='Gain', ascending=False)
        
        print("\n==================================================")
        print("DIAGNOSTIC : FEATURE IMPORTANCE (NATIVE GAIN)")
        print("==================================================")
        print(df_imp.to_string(index=False))
        print("==================================================\n")
    
    @staticmethod
    def log_shap_scarcity_impact(model, X_test: pd.DataFrame, y_test: pd.Series):
        """3. Affiche l'impact SHAP moyen lors des vraies heures de Scarcity (y=1)."""
        print("\n==================================================")
        print("DIAGNOSTIC : SHAP VALUES (IMPACT LORS DES CRISES)")
        print("==================================================")
        
        # Isoler uniquement les heures où une réelle scarcity a eu lieu
        scarcity_idx = (y_test == 1)
        if scarcity_idx.sum() == 0:
            print("Aucune heure de scarcity dans ce test set pour calculer SHAP.")
            return

        X_scarcity = X_test.loc[scarcity_idx]
        
        # Explainer XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scarcity)
        
        # Calcul de l'impact moyen absolu sur la prédiction lors des crises
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        df_shap = pd.DataFrame({
            'Feature': X_test.columns,
            'Avg_SHAP_Impact': mean_abs_shap
        }).sort_values(by='Avg_SHAP_Impact', ascending=False)

        print(df_shap.to_string(index=False))
        print("==================================================\n")

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
        'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
        'Load_Lag_1', 'Net_Load_Lag_1', 'Load_Lag_24', 'Net_Load_Lag_24'
    ] + res_techs   
    
    pipeline = ScarcityPredictorPipeline(feature_cols=features_to_use)
    pipeline.execute_time_series_validation(X, y)

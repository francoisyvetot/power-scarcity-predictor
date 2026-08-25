import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.inspection import permutation_importance
import shap


class PowerSystemDiagnostics:  
    @staticmethod  
    def log_spatial_correlations(df_system: pd.DataFrame):
        """Co-occurrence of scarcity events between countries"""
        print("\n==================================================")
        print("DIAGNOSTIC : CO-OCCURRENCE OF SCARCITY EVENTS (SPATIAL CORRELATION)")
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
        print("Correlation Matrix of Spatial Correlation of Scarcity Events :")
        print(corr_matrix.round(3))
        print("\nInterpretation : A score close to 1 indicates that the two zones fail at the same time.")
        print("==================================================\n")

    @staticmethod
    def analyze_prediction_thresholds(y_true, y_probs):
        """Analyse the impact of the probability threshold on Recall and Precision."""
        print("\n==================================================")
        print("DIAGNOSTIC : OPTIMIZATION OF THE DECISION THRESHOLD (THRESHOLD)")
        print("==================================================")
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        # On cherche différents compromis
        df_thresholds = pd.DataFrame({
            'Threshold': np.append(thresholds, 1.0),
            'Precision': precisions,
            'Recall': recalls
        })
        
        # On filtre pour afficher des seuils clés
        print(f"{'Threshold':<15} | {'Precision':<12} | {'Recall (Detection Rate)':<25}")
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
        """Calculates the drop in PR-AUC when a feature is permuted (noised)."""
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
        print("\nInterpretation : A high positive value indicates that degrading the feature reduces the PR-AUC.")
        print("==================================================\n")

    @staticmethod
    def log_feature_importance_gain(model, feature_names: list):
        """Displays the native importance based on the Gain."""
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
        """3. Displays the average SHAP impact during actual scarcity hours (y=1)."""
        print("\n==================================================")
        print("DIAGNOSTIC : SHAP VALUES (IMPACT DURING CRISES)")
        print("==================================================")
        
        # Isolate only the hours where a real scarcity occurred
        scarcity_idx = (y_test == 1)
        if scarcity_idx.sum() == 0:
            print("No scarcity hours in this test set for calculating SHAP.")
            return

        X_scarcity = X_test.loc[scarcity_idx]
        
        # Explainer XGBoost, LightGBM or RandomForest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scarcity)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] 
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:,:,1]  
        
        # Calcul of the average absolute impact on the prediction during crises
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        df_shap = pd.DataFrame({
            'Feature': X_test.columns,
            'Avg_SHAP_Impact': mean_abs_shap
        }).sort_values(by='Avg_SHAP_Impact', ascending=False)

        print(df_shap.to_string(index=False))
        print("==================================================\n")
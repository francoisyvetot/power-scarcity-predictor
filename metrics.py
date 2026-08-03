



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
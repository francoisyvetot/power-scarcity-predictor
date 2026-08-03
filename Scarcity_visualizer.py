import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InteractiveScarcityVisualizer:
    """Génère un profil interactif complet sauvegardé sous forme de fichier HTML."""

    @staticmethod
    def save_interactive_profile(
        df_test_features: pd.DataFrame,
        y_true: pd.Series,
        y_probs: np.ndarray,
        bidding_zone: str = "BE00",
        threshold: float = 0.05,
        output_dir: str = "output",
        file_name: str = "scarcity_profile_interactive.html"
    ):
        """
        Crée un fichier HTML interactif avec un Range Slider pour explorer tout l'historique.
        """
        # 1. Création du dossier d'export s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)

        # 2. Masque booléen direct pour la zone (évite la démultiplication d'index)
        mask_zone = (df_test_features['Bidding Zone'] == bidding_zone)

        # Filtrage aligné par masque booléen
        df_sub = df_test_features.loc[mask_zone].copy()
        y_true_sub = y_true.loc[mask_zone]
        
        # Gestion propre de y_probs selon qu'il s'agisse d'un numpy array ou d'une Series Pandas
        if isinstance(y_probs, pd.Series):
            y_probs_sub = y_probs.loc[mask_zone].values
        else:
            y_probs_sub = y_probs[mask_zone.values]

        # 3. Initialisation de la figure Plotly à double axe Y
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Trace 1 : Probabilité prédite par XGBoost (Axe Y1 - Gauche)
        fig.add_trace(
            go.Scatter(
                x=df_sub.index,
                y=y_probs_sub,
                name="Proba XGBoost P(Scarcity)",
                line=dict(color="#1f77b4", width=1.5),
                hovertemplate="<b>Date</b>: %{x}<br><b>Proba</b>: %{y:.2%}<extra></extra>"
            ),
            secondary_y=False
        )

        # Trace 2 : Net Load physique (Axe Y2 - Droite)
        fig.add_trace(
            go.Scatter(
                x=df_sub.index,
                y=df_sub['Net_Load'],
                name="Net Load (MW)",
                line=dict(color="#7f7f7f", width=1, dash="dot"),
                opacity=0.6,
                hovertemplate="<b>Net Load</b>: %{y:,.0f} MW<extra></extra>"
            ),
            secondary_y=True
        )

        # Trace 3 : Points d'alertes réelles de Scarcity (y_true == 1)
        scarcity_mask = (y_true_sub == 1)
        scarcity_dates = df_sub.index[scarcity_mask]
        scarcity_probs = y_probs_sub[scarcity_mask]

        if len(scarcity_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=scarcity_dates,
                    y=scarcity_probs,
                    mode="markers",
                    name="Vraie Scarcity (ENS > 0)",
                    marker=dict(color="red", size=8, symbol="triangle-up"),
                    hovertemplate="<b>CRISE RÉELLE</b><br>Date: %{x}<extra></extra>"
                ),
                secondary_y=False
            )

        # Ligne horizontale du seuil d'alerte
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Seuil d'Alerte ({threshold:.1%})",
            annotation_position="bottom left",
            secondary_y=False
        )

        # 4. Configuration des axes, titres et Range Slider pour le zoom
        fig.update_layout(
            title=dict(
                text=f"Profil Global de Prédiction de Scarcity — Zone {bidding_zone}",
                x=0.5,
                font=dict(size=16)
            ),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Axe X avec Slider interactif au bas de la page
        fig.update_xaxes(
            title_text="Datetime",
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )

        # Config des axes Y
        fig.update_yaxes(title_text="Probabilité P(Scarcity)", range=[-0.02, 1.05], secondary_y=False)
        fig.update_yaxes(title_text="Net Load (MW)", secondary_y=True)

        # 5. Sauvegarde directe du fichier HTML (sans affichage)
        fig.write_html(file_path, include_plotlyjs="cdn", full_html=True)
        print(f"Fichier interactif sauvegardé dans : {os.path.abspath(file_path)}")
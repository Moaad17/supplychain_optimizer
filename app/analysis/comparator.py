import pandas as pd
import plotly.graph_objects as go


def compare_forecasts(
    df: pd.DataFrame,
    product_name: str,
    prophet_result: dict,
    xgboost_result: dict,
    save_path: str | None = None
) -> go.Figure:
    """
    Affiche sur un même graphique l'historique et les prédictions de
    Prophet et XGBoost pour un produit donné, afin de les comparer
    visuellement.

    Parameters
    ----------
    df : pd.DataFrame
        Données historiques contenant 'date', 'product', 'quantity'.
    product_name : str
        Le produit affiché.
    prophet_result, xgboost_result : dict
        Résultats retournés par forecast_prophet / forecast_xgboost
        (doivent contenir 'dates', 'predictions', 'mae').
    save_path : str, optionnel
        Si fourni, sauvegarde le graphique en HTML à ce chemin
        au lieu de l'afficher.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    history = (
        df[df["product"] == product_name]
        .sort_values("date")
    )

    def _label(name, result):
        if result["mae"] is not None:
            return f"{name} (MAE={result['mae']:.1f})"
        return f"{name} (MAE indisponible)"

    fig = go.Figure()

    # Historique (ligne bleue)
    fig.add_trace(go.Scatter(
        x=history["date"],
        y=history["quantity"],
        mode="lines+markers",
        name="Historique",
        line=dict(color="royalblue")
    ))

    # Prédictions Prophet
    fig.add_trace(go.Scatter(
        x=prophet_result["dates"],
        y=prophet_result["predictions"],
        mode="lines+markers",
        name=_label("Prophet", prophet_result),
        line=dict(color="orange")
    ))

    # Prédictions XGBoost
    fig.add_trace(go.Scatter(
        x=xgboost_result["dates"],
        y=xgboost_result["predictions"],
        mode="lines+markers",
        name=_label("XGBoost", xgboost_result),
        line=dict(color="seagreen")
    ))

    fig.update_layout(
        title=f"Prophet vs XGBoost — {product_name}",
        xaxis_title="Date",
        yaxis_title="Quantité",
        template="plotly_white",
        hovermode="x unified"
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

    return fig


def compare_mae(prophet_result: dict, xgboost_result: dict) -> dict:
    """
    Compare les deux modèles sur la base de leur MAE de backtest.

    Returns
    -------
    dict
        {"prophet_mae": ..., "xgboost_mae": ..., "winner": ...}
    """

    prophet_mae = prophet_result["mae"]
    xgboost_mae = xgboost_result["mae"]

    if prophet_mae is None or xgboost_mae is None:
        winner = "indéterminé (backtest manquant pour au moins un modèle)"
    elif prophet_mae < xgboost_mae:
        winner = "Prophet"
    else:
        winner = "XGBoost"

    return {
        "prophet_mae": prophet_mae,
        "xgboost_mae": xgboost_mae,
        "winner": winner
    }

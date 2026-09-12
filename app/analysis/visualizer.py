import plotly.graph_objects as go


def plot_forecast(df, product_name, forecast_result, save_path=None):
    """
    Trace l'historique, les prédictions et l'intervalle de confiance
    pour un produit donné.

    Parameters
    ----------
    df : pd.DataFrame
        Données historiques contenant 'date', 'product', 'quantity'.
    product_name : str
        Le produit affiché.
    forecast_result : dict
        Le résultat retourné par forecast_prophet
        (doit contenir 'dates', 'predictions', 'lower', 'upper').
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

    dates = forecast_result["dates"]
    lower = forecast_result["lower"]
    upper = forecast_result["upper"]
    predictions = forecast_result["predictions"]

    fig = go.Figure()

    # Historique (ligne bleue)
    fig.add_trace(go.Scatter(
        x=history["date"],
        y=history["quantity"],
        mode="lines+markers",
        name="Historique",
        line=dict(color="royalblue")
    ))

    # Intervalle de confiance (zone grisée/transparente)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor="rgba(255,165,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Intervalle de confiance"
    ))

    # Prédictions (ligne orange)
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode="lines+markers",
        name="Prédictions",
        line=dict(color="orange")
    ))

    fig.update_layout(
        title=f"Prévisions Prophet — {product_name}",
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

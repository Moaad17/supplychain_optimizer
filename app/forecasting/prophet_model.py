import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# En dessous de ce nombre de mois d'historique, Prophet est peu fiable.
MIN_MONTHS_RECOMMENDED = 24


def prepare_for_prophet(df, product_name):
    """
    Filtre le DataFrame pour un produit donné et le convertit
    au format attendu par Prophet : colonnes 'ds' (date) et 'y' (valeur).
    """

    product_df = df[df["product"] == product_name].copy()

    product_df = product_df.rename(
        columns={
            "date": "ds",
            "quantity": "y"
        }
    )

    return (
        product_df[["ds", "y"]]
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _backtest(prophet_df, horizon):
    """
    Cache les `horizon` derniers mois, entraîne Prophet sur le reste,
    puis compare les prédictions aux valeurs réelles cachées.

    Returns
    -------
    float ou None
        Le MAE du backtest, ou None si pas assez de données pour le faire.
    """

    if len(prophet_df) <= horizon:
        return None

    train = prophet_df.iloc[:-horizon]
    test = prophet_df.iloc[-horizon:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)

    # Les ventes ne peuvent pas être négatives
    predictions_test = forecast["yhat"].tail(horizon).clip(lower=0).values

    return mean_absolute_error(test["y"].values, predictions_test)


def forecast_prophet(df, product_name, horizon=3):
    """
    Prédit les `horizon` prochains mois pour un produit avec Prophet.

    Parameters
    ----------
    df : pd.DataFrame
        Données contenant au minimum 'date', 'product', 'quantity'.
    product_name : str
        Le produit à prévoir.
    horizon : int
        Nombre de mois à prédire (et taille du backtest).

    Returns
    -------
    dict
        {
            "predictions": [...],       # yhat pour les mois futurs
            "lower": [...],             # borne basse de l'intervalle
            "upper": [...],             # borne haute de l'intervalle
            "dates": [...],             # dates prédites
            "mae": float ou None,       # erreur mesurée par backtest
            "n_months_history": int,
            "warnings": [...]
        }
    """

    warnings_list = []

    # 1. Préparer les données
    prophet_df = prepare_for_prophet(df, product_name)

    if prophet_df.empty:
        raise ValueError(
            f"Produit '{product_name}' introuvable dans les données."
        )

    n_months = len(prophet_df)

    if n_months < MIN_MONTHS_RECOMMENDED:
        warnings_list.append(
            f"Seulement {n_months} mois d'historique pour '{product_name}' "
            f"(recommandé : {MIN_MONTHS_RECOMMENDED}+ mois). "
            "Les prévisions sont moins fiables."
        )

    # 2. Backtest : mesurer la qualité du modèle sur des mois déjà connus
    mae = _backtest(prophet_df, horizon)

    if mae is None:
        warnings_list.append(
            "Pas assez de données pour effectuer un backtest "
            f"(il faut au moins {horizon + 1} mois d'historique)."
        )

    # 3. Entraîner sur tout l'historique pour prédire le vrai futur
    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)

    future_forecast = forecast.tail(horizon).copy()

    # 4. Cas limite : pas de ventes négatives
    for column in ["yhat", "yhat_lower", "yhat_upper"]:
        future_forecast[column] = future_forecast[column].clip(lower=0)

    return {
        "predictions": future_forecast["yhat"].tolist(),
        "lower": future_forecast["yhat_lower"].tolist(),
        "upper": future_forecast["yhat_upper"].tolist(),
        "dates": future_forecast["ds"].tolist(),
        "mae": mae,
        "n_months_history": n_months,
        "warnings": warnings_list
    }

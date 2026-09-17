import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# En dessous de ce nombre de mois d'historique, les features (surtout
# lag_12) sont peu fiables.
MIN_MONTHS_RECOMMENDED = 24

BASE_FEATURE_COLUMNS = [
    "month", "year", "quarter",
    "lag_1", "lag_3", "lag_12",
    "rolling_mean_3", "rolling_mean_12",
    "trend"
]


def _feature_columns_for(features_df: pd.DataFrame) -> list[str]:
    """
    Liste des colonnes de features à utiliser pour ce DataFrame :
    les colonnes temporelles de base, + "unit_price" si elle est
    présente (toutes les données n'ont pas forcément un prix unitaire).
    """

    columns = list(BASE_FEATURE_COLUMNS)
    if "unit_price" in features_df.columns:
        columns.append("unit_price")

    return columns


def create_temporal_features(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """
    Crée les features temporelles (+ prix, si disponible) pour un
    produit donné.

    XGBoost ne comprend pas le temps : il faut lui fournir des colonnes
    numériques qui décrivent la saisonnalité et la tendance.

    Colonnes créées : month, year, quarter, lag_1, lag_3, lag_12,
    rolling_mean_3, rolling_mean_12, trend, + unit_price (recopiée
    telle quelle si la colonne existe dans `df` -- le prix peut
    influencer la demande, cf. élasticité-prix).

    Attention : lag_12 a besoin de 12 mois d'historique avant la première
    ligne exploitable -> les premières lignes contiendront des NaN
    (à supprimer avant l'entraînement, cf. _train_test_split_temporal).
    """

    product_df = (
        df[df["product"] == product_name]
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    product_df["month"] = product_df["date"].dt.month
    product_df["year"] = product_df["date"].dt.year
    product_df["quarter"] = product_df["date"].dt.quarter

    # Lags : uniquement des valeurs passées, jamais la valeur du mois
    # courant -> pas de fuite de données (data leakage).
    product_df["lag_1"] = product_df["quantity"].shift(1)
    product_df["lag_3"] = product_df["quantity"].shift(3)
    product_df["lag_12"] = product_df["quantity"].shift(12)

    # Moyennes glissantes calculées AVANT le mois courant (shift(1))
    product_df["rolling_mean_3"] = (
        product_df["quantity"].shift(1).rolling(window=3).mean()
    )
    product_df["rolling_mean_12"] = (
        product_df["quantity"].shift(1).rolling(window=12).mean()
    )

    # Tendance linéaire : 1, 2, 3, ...
    product_df["trend"] = range(1, len(product_df) + 1)

    # unit_price est déjà dans product_df si elle existait dans df
    # (filtrage par produit ne perd aucune colonne) -- rien à faire de
    # plus, elle est prête à être utilisée comme feature.

    return product_df


def _train_test_split_temporal(
    features_df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Split chronologique STRICT : les `horizon` dernières lignes vont au
    test, tout le reste va au train. Jamais de split aléatoire : un split
    aléatoire mélangerait des lignes dont les lags viennent du "futur"
    par rapport au train -> data leakage, métriques irréalistes.

    Returns
    -------
    (train, test) ou (None, None) si pas assez de données.
    """

    clean_df = features_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if len(clean_df) <= horizon:
        return None, None

    train = clean_df.iloc[:-horizon]
    test = clean_df.iloc[-horizon:]

    return train, test


def _backtest(features_df: pd.DataFrame, horizon: int) -> float | None:
    """
    Entraîne XGBoost sur tout sauf les `horizon` derniers mois connus,
    prédit ces mois cachés, et calcule le MAE.

    Returns
    -------
    float ou None si pas assez de données pour backtester.
    """

    train, test = _train_test_split_temporal(features_df, horizon)

    if train is None:
        return None

    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(train[FEATURE_COLUMNS], train["quantity"])

    predictions = model.predict(test[FEATURE_COLUMNS])
    predictions = np.clip(predictions, 0, None)  # pas de ventes négatives

    return mean_absolute_error(test["quantity"].values, predictions)


def _next_feature_row(
    history: pd.DataFrame, next_date: pd.Timestamp, trend: int
) -> pd.DataFrame:
    """
    Construit la ligne de features pour le mois `next_date`, à partir de
    l'historique connu (auquel s'ajoutent les prédictions déjà générées
    lors d'une prévision récursive multi-mois).
    """

    quantities = history["quantity"]

    lag_1 = quantities.iloc[-1]
    lag_3 = quantities.iloc[-3] if len(quantities) >= 3 else np.nan
    lag_12 = quantities.iloc[-12] if len(quantities) >= 12 else np.nan
    rolling_mean_3 = (
        quantities.iloc[-3:].mean() if len(quantities) >= 3 else np.nan
    )
    rolling_mean_12 = (
        quantities.iloc[-12:].mean() if len(quantities) >= 12 else np.nan
    )

    return pd.DataFrame([{
        "month": next_date.month,
        "year": next_date.year,
        "quarter": (next_date.month - 1) // 3 + 1,
        "lag_1": lag_1,
        "lag_3": lag_3,
        "lag_12": lag_12,
        "rolling_mean_3": rolling_mean_3,
        "rolling_mean_12": rolling_mean_12,
        "trend": trend
    }])


def _forecast_future(
    features_df: pd.DataFrame, horizon: int, quantile_alpha: float | None = None
) -> tuple[list[pd.Timestamp], list[float]]:
    """
    Entraîne un modèle sur tout l'historique disponible puis prédit
    récursivement les `horizon` prochains mois : chaque prédiction sert
    de lag pour prédire le mois suivant.

    Si `quantile_alpha` est fourni (ex: 0.1 ou 0.9), entraîne un modèle
    de régression quantile (reg:quantileerror) au lieu d'une régression
    classique -> permet d'obtenir une borne basse/haute sans supposer
    une distribution d'erreur symétrique comme la méthode ±1.5xMAE.
    """

    clean_df = features_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if quantile_alpha is None:
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    else:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="reg:quantileerror",
            quantile_alpha=quantile_alpha
        )

    model.fit(clean_df[FEATURE_COLUMNS], clean_df["quantity"])

    # Historique glissant : sert à calculer les lags des mois futurs
    history = features_df[["date", "quantity"]].copy()
    last_trend = features_df["trend"].iloc[-1]

    dates = []
    predictions = []

    for step in range(1, horizon + 1):
        next_date = history["date"].iloc[-1] + pd.DateOffset(months=1)
        trend = last_trend + step

        feature_row = _next_feature_row(history, next_date, trend)
        prediction = max(float(model.predict(feature_row[FEATURE_COLUMNS])[0]), 0)

        dates.append(next_date)
        predictions.append(prediction)

        # La prédiction devient un "fait connu" pour calculer les lags
        # du mois suivant.
        history = pd.concat(
            [history, pd.DataFrame([{"date": next_date, "quantity": prediction}])],
            ignore_index=True
        )

    return dates, predictions


def forecast_xgboost(df: pd.DataFrame, product_name: str, horizon: int = 3) -> dict:
    """
    Prédit les `horizon` prochains mois pour un produit avec XGBoost
    et des features temporelles manuelles (lags, moyennes glissantes,
    tendance, saisonnalité).

    Même interface de sortie que forecast_prophet : {predictions, lower,
    upper, dates, mae, n_months_history, warnings}. Le sélecteur et
    l'optimisation utilisent l'un ou l'autre modèle sans distinction
    (polymorphisme) — c'est tout l'intérêt de standardiser le format.
    """

    warnings_list = []

    features_df = create_temporal_features(df, product_name)

    if features_df.empty:
        raise ValueError(
            f"Produit '{product_name}' introuvable dans les données."
        )

    n_months = len(features_df)

    if n_months < MIN_MONTHS_RECOMMENDED:
        warnings_list.append(
            f"Seulement {n_months} mois d'historique pour '{product_name}' "
            f"(recommandé : {MIN_MONTHS_RECOMMENDED}+ mois). "
            "Les prévisions sont moins fiables."
        )

    # 1. Backtest : mesurer la qualité du modèle sur des mois déjà connus
    mae = _backtest(features_df, horizon)

    if mae is None:
        warnings_list.append(
            "Pas assez de données pour effectuer un backtest "
            f"(il faut au moins 12 mois d'historique + {horizon} mois à cacher, "
            "à cause de lag_12)."
        )

    # 2. Prédiction sur le vrai futur (récursive)
    dates, predictions = _forecast_future(features_df, horizon)

    # 3. Intervalle de confiance
    clean_df = features_df.dropna(subset=FEATURE_COLUMNS)

    if len(clean_df) >= 12:
        # Régression quantile : bornes 10e / 90e percentile
        _, lower = _forecast_future(features_df, horizon, quantile_alpha=0.1)
        _, upper = _forecast_future(features_df, horizon, quantile_alpha=0.9)

        # Les 3 modèles (0.1 / 0.5 / 0.9) sont entraînés indépendamment :
        # avec peu de données, il arrive qu'ils se "croisent" (quantile
        # crossing) et que la borne basse dépasse la prédiction médiane.
        # On force l'ordre logique lower <= prediction <= upper.
        lower = [min(l, p) for l, p in zip(lower, predictions)]
        upper = [max(u, p) for u, p in zip(upper, predictions)]
    else:
        # Pas assez de données pour un modèle quantile fiable : on
        # retombe sur la méthode simple prédiction ± 1.5 x MAE.
        warnings_list.append(
            "Pas assez de données pour une régression quantile fiable : "
            "intervalle approximé avec ±1.5 x MAE."
        )
        margin = 1.5 * (mae if mae is not None else 0)
        lower = [max(p - margin, 0) for p in predictions]
        upper = [p + margin for p in predictions]

    return {
        "predictions": predictions,
        "lower": lower,
        "upper": upper,
        "dates": dates,
        "mae": mae,
        "n_months_history": n_months,
        "warnings": warnings_list
    }

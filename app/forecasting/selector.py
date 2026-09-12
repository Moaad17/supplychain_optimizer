import logging

import pandas as pd

from forecasting.prophet_model import forecast_prophet
from forecasting.ml_model import forecast_xgboost


logger = logging.getLogger(__name__)

MODEL_FUNCTIONS = {
    "prophet": forecast_prophet,
    "xgboost": forecast_xgboost
}


def forecast_all_products(
    df: pd.DataFrame, model_type: str = "prophet", horizon: int = 3
) -> dict:
    """
    Applique UN modèle (Prophet ou XGBoost) à tous les produits du
    DataFrame.

    Un produit qui échoue (pas assez de données, données trop
    irrégulières, etc.) n'interrompt pas le traitement des autres :
    l'erreur est loggée et le produit est listé dans "failed".

    Parameters
    ----------
    df : pd.DataFrame
        Données contenant 'date', 'product', 'quantity'.
    model_type : str
        "prophet" ou "xgboost".
    horizon : int
        Nombre de mois à prédire.

    Returns
    -------
    dict
        {
            "results": {produit: résultat_forecast, ...},
            "failed": {produit: raison_de_l_échec, ...}
        }
    """

    if model_type not in MODEL_FUNCTIONS:
        raise ValueError(
            f"model_type invalide : '{model_type}'. "
            f"Choix possibles : {list(MODEL_FUNCTIONS)}"
        )

    forecast_function = MODEL_FUNCTIONS[model_type]

    results = {}
    failed = {}

    for product in df["product"].unique():

        try:
            results[product] = forecast_function(df, product, horizon)

        except Exception as error:
            logger.warning(
                "Échec du forecast %s pour '%s' : %s",
                model_type, product, error
            )
            failed[product] = str(error)

    return {"results": results, "failed": failed}


def select_best_model(df: pd.DataFrame, product_name: str, horizon: int = 3) -> dict:
    """
    Entraîne Prophet ET XGBoost pour un produit, compare leur MAE de
    backtest, et retourne le résultat du modèle gagnant.

    Si un des deux modèles n'a pas pu être backtesté (mae=None, pas
    assez de données), on garde automatiquement l'autre. Si aucun des
    deux n'a de MAE, on garde Prophet par défaut (avec ses warnings).

    Returns
    -------
    dict
        Le résultat du modèle gagnant, avec en plus la clé "model"
        ("prophet" ou "xgboost").
    """

    prophet_result = forecast_prophet(df, product_name, horizon)
    xgboost_result = forecast_xgboost(df, product_name, horizon)

    prophet_mae = prophet_result["mae"]
    xgboost_mae = xgboost_result["mae"]

    if prophet_mae is None and xgboost_mae is None:
        model_name, chosen = "prophet", prophet_result
    elif prophet_mae is None:
        model_name, chosen = "xgboost", xgboost_result
    elif xgboost_mae is None:
        model_name, chosen = "prophet", prophet_result
    elif prophet_mae <= xgboost_mae:
        model_name, chosen = "prophet", prophet_result
    else:
        model_name, chosen = "xgboost", xgboost_result

    return {
        "model": model_name,
        "mae": chosen["mae"],
        "predictions": chosen["predictions"],
        "lower": chosen["lower"],
        "upper": chosen["upper"],
        "dates": chosen["dates"],
        "n_months_history": chosen["n_months_history"],
        "warnings": chosen["warnings"]
    }


def forecast_all_products_auto(df: pd.DataFrame, horizon: int = 3) -> dict:
    """
    Boucle sur chaque produit, sélectionne automatiquement le meilleur
    modèle (Prophet vs XGBoost) via select_best_model, et retourne un
    dictionnaire complet prêt à être consommé par le module
    d'optimisation.

    Un produit qui échoue n'interrompt pas les autres (cf.
    forecast_all_products).

    Returns
    -------
    dict
        {
            "results": {
                produit: {
                    "model": "prophet" ou "xgboost",
                    "mae": float,
                    "predictions": [...],
                    "lower": [...],
                    "upper": [...],
                    "dates": [...],
                    "n_months_history": int,
                    "warnings": [...]
                },
                ...
            },
            "failed": {produit: raison_de_l_échec, ...}
        }
    """

    results = {}
    failed = {}

    for product in df["product"].unique():

        try:
            results[product] = select_best_model(df, product, horizon)

        except Exception as error:
            logger.warning(
                "Échec du forecast pour '%s' : %s", product, error
            )
            failed[product] = str(error)

    return {"results": results, "failed": failed}

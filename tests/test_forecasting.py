"""
Validation end-to-end de la Phase 2 (forecasting) sur les 3 CSV
d'exemple : chargement, validation, preprocessing, puis sélection
automatique du meilleur modèle (Prophet vs XGBoost) pour chaque produit.

Lancer avec `pytest tests/test_forecasting.py` pour les assertions,
ou `python tests/test_forecasting.py` pour le résumé lisible.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Permet `python tests/test_forecasting.py` en direct (pytest le fait
# déjà via conftest.py, mais un run direct n'exécute pas les fixtures).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from data.loader import load_data
from data.validator import validate_data
from data.preprocessor import preprocess_data
from forecasting.prophet_model import forecast_prophet
from forecasting.ml_model import forecast_xgboost, create_temporal_features, _feature_columns_for
from forecasting.selector import forecast_all_products_auto


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
HORIZON = 3


def _load_and_prepare(filename):
    """
    Charge et valide un CSV. Retourne (df_preprocessed, validation)
    -- df_preprocessed est None si les données sont invalides.
    """

    df = load_data(EXAMPLES_DIR / filename)
    validation = validate_data(df)

    if not validation["valid"]:
        return None, validation

    return preprocess_data(df), validation


def _summarize(filename, forecast_result, validation):
    """Construit un résumé texte lisible pour un fichier testé."""

    lines = [f"\n--- {filename} ---"]

    if forecast_result is None:
        lines.append("❌ Données invalides, forecasting ignoré :")
        lines.extend(f"  - {error}" for error in validation["errors"])
        return "\n".join(lines)

    for product, result in forecast_result["results"].items():
        mae = result["mae"]
        mae_str = f"{mae:.1f}" if mae is not None else "N/A"
        lines.append(
            f"  {product} → {result['model']} (MAE={mae_str})"
        )

    for product, reason in forecast_result["failed"].items():
        lines.append(f"  {product} → ÉCHEC ({reason})")

    return "\n".join(lines)


# ==========================================================
# Tests pytest
# ==========================================================

@pytest.mark.parametrize("filename", [
    "sample_beverages.csv",
    "sample_retail.csv",
])
def test_forecast_all_products_auto_valid_datasets(filename):
    """
    Sur les CSV valides (colonnes date/product/quantity présentes),
    chaque produit doit obtenir un résultat, avec un MAE mesuré et
    des prédictions de la bonne longueur.
    """

    df, validation = _load_and_prepare(filename)
    assert validation["valid"], validation["errors"]

    forecast_result = forecast_all_products_auto(df, horizon=HORIZON)

    all_products = set(df["product"].unique())
    succeeded = set(forecast_result["results"].keys())
    failed = set(forecast_result["failed"].keys())

    # Tous les produits sont pris en compte, aucun n'est perdu
    assert succeeded | failed == all_products

    # Sur ces jeux de données propres (36 mois, pas de trous), tout
    # doit réussir
    assert not failed, forecast_result["failed"]

    for product, result in forecast_result["results"].items():
        assert result["model"] in ("prophet", "xgboost")
        assert result["mae"] is not None
        assert result["mae"] >= 0
        assert len(result["predictions"]) == HORIZON
        assert len(result["dates"]) == HORIZON
        assert all(p >= 0 for p in result["predictions"])


def test_forecast_all_products_auto_invalid_dataset():
    """
    sample_port_traffic.csv n'a pas les colonnes 'product'/'quantity'
    -> validate_data doit le détecter AVANT qu'on lance le forecasting.
    """

    df, validation = _load_and_prepare("sample_port_traffic.csv")

    assert not validation["valid"]
    assert df is None


def test_forecast_all_products_auto_partial_failure():
    """
    Un produit avec trop peu de mois d'historique doit échouer sans
    faire planter les autres produits.
    """

    df = load_data(EXAMPLES_DIR / "sample_beverages.csv")
    df = preprocess_data(df)

    # On tronque un seul produit à 4 mois d'historique (insuffisant),
    # les autres gardent leurs 36 mois.
    short_product = df["product"].unique()[0]
    other_rows = df[df["product"] != short_product]
    short_rows = df[df["product"] == short_product].head(4)
    df_partial = pd.concat([other_rows, short_rows], ignore_index=True)

    forecast_result = forecast_all_products_auto(df_partial, horizon=HORIZON)

    assert short_product in forecast_result["failed"]
    assert len(forecast_result["results"]) == df_partial["product"].nunique() - 1


# ==========================================================
# Invariants Prophet / XGBoost (indépendants du sélecteur)
# ==========================================================

@pytest.fixture(scope="module")
def beverages_df():
    """DataFrame prétraité, réutilisé par les tests d'invariants
    ci-dessous pour éviter de recharger le CSV à chaque test."""

    df = load_data(EXAMPLES_DIR / "sample_beverages.csv")
    return preprocess_data(df)


@pytest.mark.parametrize("forecast_function", [forecast_prophet, forecast_xgboost])
def test_predictions_are_non_negative(beverages_df, forecast_function):
    """Les ventes ne peuvent pas être négatives : predictions >= 0."""

    product = beverages_df["product"].unique()[0]
    result = forecast_function(beverages_df, product, horizon=HORIZON)

    assert all(prediction >= 0 for prediction in result["predictions"])


@pytest.mark.parametrize("forecast_function", [forecast_prophet, forecast_xgboost])
def test_confidence_interval_is_coherent(beverages_df, forecast_function):
    """
    Pour chaque mois prédit : lower <= prediction <= upper, et
    lower >= 0. Un intervalle inversé ou négatif indiquerait un bug.
    """

    product = beverages_df["product"].unique()[0]
    result = forecast_function(beverages_df, product, horizon=HORIZON)

    for lower, prediction, upper in zip(
        result["lower"], result["predictions"], result["upper"]
    ):
        assert lower <= prediction <= upper
        assert lower >= 0


def test_selector_returns_result_for_every_product(beverages_df):
    """forecast_all_products_auto doit retourner un résultat pour
    chaque produit du dataset (aucun perdu, aucun dupliqué)."""

    forecast_result = forecast_all_products_auto(beverages_df, horizon=HORIZON)

    assert len(forecast_result["results"]) == beverages_df["product"].nunique()


@pytest.mark.parametrize("forecast_function", [forecast_prophet, forecast_xgboost])
def test_mae_beats_naive_mean_baseline(beverages_df, forecast_function):
    """
    Un modèle utile doit faire mieux qu'une prédiction naïve = la
    moyenne des ventes. Si la MAE dépasse 50% de la moyenne, le modèle
    est pire qu'une moyenne constante -> quelque chose ne va pas.
    """

    for product in beverages_df["product"].unique():
        result = forecast_function(beverages_df, product, horizon=HORIZON)
        mean_quantity = beverages_df.loc[
            beverages_df["product"] == product, "quantity"
        ].mean()

        assert result["mae"] is not None
        assert result["mae"] < 0.5 * mean_quantity


# ==========================================================
# unit_price comme feature XGBoost
# ==========================================================

def test_create_temporal_features_includes_unit_price_when_present():
    df, validation = _load_and_prepare("sample_beverages.csv")
    assert validation["valid"]
    assert "unit_price" in df.columns  # préservé par preprocess_data

    product = df["product"].unique()[0]
    features = create_temporal_features(df, product)

    assert "unit_price" in features.columns
    assert "unit_price" in _feature_columns_for(features)
    assert not features["unit_price"].isna().any()


def test_create_temporal_features_omits_unit_price_when_absent():
    df, _ = _load_and_prepare("sample_beverages.csv")
    df_no_price = df.drop(columns=["unit_price"])

    product = df_no_price["product"].unique()[0]
    features = create_temporal_features(df_no_price, product)

    assert "unit_price" not in features.columns
    assert "unit_price" not in _feature_columns_for(features)


def test_forecast_xgboost_still_works_without_unit_price():
    """La feature est optionnelle : le forecasting ne doit pas planter
    sur un dataset qui n'a pas cette colonne."""

    df, _ = _load_and_prepare("sample_beverages.csv")
    df_no_price = df.drop(columns=["unit_price"])
    product = df_no_price["product"].unique()[0]

    result = forecast_xgboost(df_no_price, product, horizon=HORIZON)

    assert len(result["predictions"]) == HORIZON


# ==========================================================
# Exécution manuelle : affiche un résumé lisible sur les 3 CSV
# ==========================================================

if __name__ == "__main__":

    for filename in [
        "sample_beverages.csv",
        "sample_retail.csv",
        "sample_port_traffic.csv",
    ]:
        df, validation = _load_and_prepare(filename)

        forecast_result = (
            forecast_all_products_auto(df, horizon=HORIZON)
            if df is not None else None
        )

        print(_summarize(filename, forecast_result, validation))

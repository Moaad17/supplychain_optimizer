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

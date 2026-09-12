"""
Tests pour data/loader.py (chargement) et data/validator.py (validation).
"""

from pathlib import Path

import pandas as pd
import pytest

from data.loader import load_data
from data.validator import validate_data


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# ==========================================================
# load_data
# ==========================================================

def test_load_data_valid_csv():
    """Un CSV valide doit être chargé en DataFrame non vide avec une
    colonne 'date' au format datetime."""

    df = load_data(EXAMPLES_DIR / "sample_beverages.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_data_missing_file():
    """Un fichier inexistant doit lever une FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_data("fichier_qui_nexiste_pas.csv")


def test_load_data_unsupported_format(tmp_path):
    """Un fichier dans un format non supporté (.txt) doit lever une
    erreur explicite plutôt que d'être chargé silencieusement."""

    bad_file = tmp_path / "donnees.txt"
    bad_file.write_text("ceci n'est pas un CSV ni un Excel")

    with pytest.raises(ValueError):
        load_data(bad_file)


# ==========================================================
# validate_data
# ==========================================================

def _make_valid_df(n_months=24):
    """DataFrame minimal valide : n_months mensuels pour un produit."""

    dates = pd.date_range("2023-01-01", periods=n_months, freq="MS")

    return pd.DataFrame({
        "date": dates,
        "product": ["Produit A"] * n_months,
        "quantity": range(100, 100 + n_months)
    })


def test_validate_data_correct_dataframe():
    """Un DataFrame correct doit être valide, sans erreurs."""

    df = _make_valid_df()

    result = validate_data(df)

    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_data_missing_date_column():
    """Un DataFrame sans colonne 'date' doit être invalide, avec une
    erreur qui le mentionne explicitement."""

    df = _make_valid_df().drop(columns=["date"])

    result = validate_data(df)

    assert result["valid"] is False
    assert any("date" in error for error in result["errors"])


def test_validate_data_too_many_missing_values():
    """50% de NaN dans 'quantity' doit déclencher un warning ou une
    erreur sur les valeurs manquantes (le seuil d'erreur du validateur
    est fixé à 30%)."""

    df = _make_valid_df()
    df.loc[::2, "quantity"] = None  # une ligne sur deux -> 50%

    result = validate_data(df)

    all_messages = result["errors"] + result["warnings"]
    assert any("manquante" in message for message in all_messages)

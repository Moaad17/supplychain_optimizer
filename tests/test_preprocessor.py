"""
Tests pour data/preprocessor.py, en particulier la conservation de
unit_price (utilisé comme feature XGBoost et comme coût réel par
produit dans l'optimisation).
"""

import numpy as np
import pandas as pd

from data.preprocessor import preprocess_data


def _make_df_with_price():
    """2 produits, quelques mois, avec un unit_price constant par
    produit et UN mois manquant (mars) pour 'A' -- pour vérifier que
    le prix est bien reporté (ffill) sur le mois comblé."""

    return pd.DataFrame({
        "date": [
            "2023-01-01", "2023-02-01", "2023-04-01",  # 'A' : mars manquant
            "2023-01-01", "2023-02-01", "2023-03-01",
        ],
        "product": ["A", "A", "A", "B", "B", "B"],
        "quantity": [100, 110, 130, 50, 55, 60],
        "unit_price": [9.9, 9.9, 9.9, 20.0, 20.0, 20.0],
    })


def test_preprocess_data_keeps_unit_price_column():
    df = preprocess_data(_make_df_with_price())

    assert "unit_price" in df.columns
    assert not df["unit_price"].isna().any()


def test_preprocess_data_fills_missing_month_price_by_carrying_forward():
    df = preprocess_data(_make_df_with_price())

    march_row = df[(df["product"] == "A") & (df["date"] == "2023-03-01")]

    assert len(march_row) == 1
    assert march_row["quantity"].iloc[0] == 0  # quantité comblée à 0
    assert march_row["unit_price"].iloc[0] == 9.9  # prix reporté, pas 0/NaN


def test_preprocess_data_price_varies_correctly_per_product():
    df = preprocess_data(_make_df_with_price())

    assert (df.loc[df["product"] == "A", "unit_price"] == 9.9).all()
    assert (df.loc[df["product"] == "B", "unit_price"] == 20.0).all()


def test_preprocess_data_without_unit_price_column_still_works():
    """Un DataFrame sans unit_price doit continuer à fonctionner
    normalement (colonne optionnelle)."""

    df_no_price = _make_df_with_price().drop(columns=["unit_price"])

    df = preprocess_data(df_no_price)

    assert "unit_price" not in df.columns
    assert len(df) > 0

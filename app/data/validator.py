import pandas as pd


def validate_data(df: pd.DataFrame) -> dict:
    """
    Valide les données avant leur utilisation dans le pipeline
    de forecasting.

    Vérifie : colonnes obligatoires, types (date/quantity), valeurs
    négatives, valeurs manquantes, volume de données (nombre de mois),
    présence de produits et doublons date+produit.

    Parameters
    ----------
    df : pd.DataFrame
        Données à valider. Colonnes attendues : 'date', 'product',
        'quantity'.

    Returns
    -------
    dict
        {
            "valid": True/False,
            "errors": [...],
            "warnings": [...],
            "stats": {...}
        }
    """

    errors = []
    warnings = []

    # ==========================================================
    # 1. Colonnes obligatoires
    # ==========================================================

    required_columns = ["date", "product", "quantity"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Colonnes obligatoires manquantes : {missing_columns}"
        )

        # On ne peut pas continuer certaines vérifications
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "stats": {}
        }

    # ==========================================================
    # 2. Vérification des types
    # ==========================================================

    # Date
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):

        converted_date = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        if converted_date.isna().any():
            errors.append(
                "La colonne 'date' contient des valeurs "
                "qui ne peuvent pas être converties en datetime."
            )
        else:
            df["date"] = converted_date

    # Quantity
    if not pd.api.types.is_numeric_dtype(df["quantity"]):

        converted_quantity = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        )

        if converted_quantity.isna().any():
            errors.append(
                "La colonne 'quantity' contient des valeurs "
                "qui ne peuvent pas être converties en numérique."
            )
        else:
            df["quantity"] = converted_quantity

    # ==========================================================
    # 3. Valeurs négatives
    # ==========================================================

    negative_count = (df["quantity"] < 0).sum()

    if negative_count > 0:
        warnings.append(
            f"{negative_count} valeur(s) négative(s) "
            "détectée(s) dans 'quantity'."
        )

    # ==========================================================
    # 4. Valeurs manquantes
    # ==========================================================

    missing_pct = {}

    for column in df.columns:

        percentage = df[column].isna().mean() * 100
        missing_pct[column] = round(percentage, 2)

        if percentage > 30:
            errors.append(
                f"La colonne '{column}' contient "
                f"{percentage:.1f}% de valeurs manquantes "
                "(> 30%)."
            )

        elif percentage >= 5:
            warnings.append(
                f"La colonne '{column}' contient "
                f"{percentage:.1f}% de valeurs manquantes."
            )

    # ==========================================================
    # 5. Volume de données
    # ==========================================================

    # Nombre de mois
    if df["date"].notna().any():

        date_min = df["date"].min()
        date_max = df["date"].max()

        n_months = (
            (date_max.year - date_min.year) * 12
            + date_max.month - date_min.month
            + 1
        )

        if n_months < 12:
            errors.append(
                f"Seulement {n_months} mois de données disponibles. "
                "Pas assez de données pour faire du forecasting fiable."
            )

        date_range = (
            f"{date_min.strftime('%Y-%m')} "
            f"to {date_max.strftime('%Y-%m')}"
        )

    else:
        n_months = 0
        date_range = None

    # Nombre de produits
    n_products = df["product"].nunique()

    if n_products == 0:
        errors.append(
            "Aucun produit trouvé dans les données."
        )

    # ==========================================================
    # 6. Doublons date + produit
    # ==========================================================

    duplicate_count = df.duplicated(
        subset=["date", "product"]
    ).sum()

    if duplicate_count > 0:
        warnings.append(
            f"{duplicate_count} ligne(s) dupliquée(s) "
            "pour la combinaison date + produit."
        )

    # ==========================================================
    # Résultat final
    # ==========================================================

    stats = {
        "n_products": n_products,
        "n_months": n_months,
        "date_range": date_range,
        "missing_pct": missing_pct,
        "n_negative": int(negative_count),
        "n_duplicates": int(duplicate_count),
        "n_rows": len(df)
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats
    }
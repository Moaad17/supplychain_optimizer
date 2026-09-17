import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare les données pour le forecasting.

    Étapes :
    - Supprime les doublons produit + date en faisant la somme
      (moyenne pour unit_price, si présent).
    - Crée les mois manquants pour chaque produit.
    - Remplit les mois absents avec 0 (unit_price : dernier prix connu).
    - Trie les données chronologiquement.
    - Ajoute year, month et quarter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant date, product et quantity. La colonne
        unit_price, si présente, est conservée (utilisée comme feature
        par le forecasting XGBoost et comme coût réel par produit par
        l'optimisation -- cf. optimization.baselines.compute_unit_costs).

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé.
    """

    # Faire une copie pour ne pas modifier le DataFrame original
    df = df.copy()

    has_unit_price = "unit_price" in df.columns

    # ==========================================================
    # 1. S'assurer que la date est au bon format
    # ==========================================================

    df["date"] = pd.to_datetime(df["date"])

    # ==========================================================
    # 2. Supprimer les doublons
    # ==========================================================

    # Même produit + même date → somme des quantités (et moyenne du
    # prix unitaire, s'il est présent)
    agg = {"quantity": "sum"}
    if has_unit_price:
        agg["unit_price"] = "mean"

    df = df.groupby(["product", "date"], as_index=False).agg(agg)

    # ==========================================================
    # 3. Créer les mois manquants pour chaque produit
    # ==========================================================

    processed = []

    for product, group in df.groupby("product"):

        # Trier les dates
        group = group.sort_values("date")

        # Premier et dernier mois
        start_date = group["date"].min()
        end_date = group["date"].max()

        # Créer toutes les dates mensuelles
        full_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="MS"
        )

        # Mettre la date comme index
        group = group.set_index("date")

        # Reindexer avec tous les mois
        group = group.reindex(full_dates)

        # Remettre product
        group["product"] = product

        # Les mois absents deviennent 0
        group["quantity"] = group["quantity"].fillna(0)

        # Le prix unitaire n'est pas "0" pour un mois absent : on
        # reprend le dernier prix connu (et, s'il manque au tout début,
        # le premier prix connu qui suit)
        if has_unit_price:
            group["unit_price"] = group["unit_price"].ffill().bfill()

        # Remettre la date comme colonne
        group.index.name = "date"
        group = group.reset_index()

        processed.append(group)

    # Combiner tous les produits
    df = pd.concat(processed, ignore_index=True)

    # ==========================================================
    # 4. Trier par produit puis par date
    # ==========================================================

    df = df.sort_values(
        ["product", "date"]
    ).reset_index(drop=True)

    # ==========================================================
    # 5. Ajouter les variables temporelles
    # ==========================================================

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    return df
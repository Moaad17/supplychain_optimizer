import pandas as pd


def compute_naive_orders(df: pd.DataFrame, horizon: int) -> dict:
    """
    Commande "naïve" : la moyenne mensuelle historique par produit,
    multipliée par l'horizon -- sans prévision ni optimisation. Sert de
    référence "zéro effort" pour mesurer ce qu'apportent le forecasting
    et l'optimisation stochastique (cf. optimization.solver.compute_naive_cost).

    Parameters
    ----------
    df : pd.DataFrame
        Données prétraitées (cf. data.preprocessor.preprocess_data),
        contenant 'product' et 'quantity'.
    horizon : int
        Nombre de mois sur lesquels projeter la moyenne mensuelle.

    Returns
    -------
    dict
        {produit: quantité totale naïve sur l'horizon}.
    """

    monthly_mean = df.groupby("product")["quantity"].mean()

    return (monthly_mean * horizon).to_dict()


def compute_unit_costs(df: pd.DataFrame, default_unit_cost: float) -> dict:
    """
    Coût d'achat unitaire cᵢ réel par produit, à partir de la colonne
    'unit_price' des données (moyenne, au cas où elle varierait dans le
    temps) -- au lieu d'un unit_cost unique partagé par tous les
    produits, qui n'a pas de sens quand les produits ont des prix très
    différents (ex : Coca-Cola vs Smartphone).

    NB : 'unit_price' est le prix du produit tel qu'il apparaît dans
    les données (souvent un prix de vente) ; en l'absence d'une colonne
    de coût d'achat séparée, c'est la meilleure approximation
    disponible de cᵢ.

    Parameters
    ----------
    df : pd.DataFrame
        Données contenant 'product', et idéalement 'unit_price'.
    default_unit_cost : float
        Valeur de repli (cf. config.yaml -> unit_cost) utilisée pour
        un produit sans unit_price connu, ou si la colonne est absente.

    Returns
    -------
    dict
        {produit: coût unitaire}.
    """

    products = df["product"].unique()

    if "unit_price" not in df.columns:
        return {product: default_unit_cost for product in products}

    mean_price = df.groupby("product")["unit_price"].mean()

    return {
        product: float(mean_price[product]) if product in mean_price.index and pd.notna(mean_price[product])
        else default_unit_cost
        for product in products
    }

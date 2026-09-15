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

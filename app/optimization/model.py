import numpy as np
import pulp

from optimization.constraints import (
    add_budget_constraint,
    add_capacity_constraint,
    add_recourse_constraints
)


def _as_dict(value, keys) -> dict:
    """Normalise un paramètre scalaire OU déjà-dict en dict {clé: valeur}
    (ex: unit_cost=10 partagé par tous les produits, ou unit_cost={"A": 8, "B": 12})."""

    if isinstance(value, dict):
        return value

    return {key: value for key in keys}


def build_stochastic_model(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True
) -> tuple[pulp.LpProblem, dict, dict, dict]:
    """
    Construit le programme déterministe équivalent (PDE) d'un problème
    de gestion de stock multi-produits sous incertitude de la demande
    (recours simple), formulation "Here and Now" (HN) -- cf.
    méthodologie, Étape 2 et 4.

    Variables
    ---------
    order[p]        : quantité commandée du produit p (1ère étape,
                       décidée AVANT de connaître la demande réelle).
                       Entière (qᵢ ∈ ℤ⁺) si `integer=True`.
    surplus[p, s]    : surstock du produit p si le scénario s se réalise
                       (2ème étape / recours).
    shortage[p, s]   : rupture du produit p si le scénario s se réalise
                       (2ème étape / recours).

    Objectif
    --------
    Min Σᵢ cᵢqᵢ + Σₛ πˢ Σᵢ (h·surplus[i,s] + p·shortage[i,s])

    Parameters
    ----------
    scenarios : dict
        {produit: np.ndarray de demandes simulées}, cf.
        constraints.generate_demand_scenarios. Tous les produits
        doivent avoir le même nombre de scénarios que `probabilities`.
    probabilities : np.ndarray
        Probabilité πˢ de chaque scénario (doit sommer à 1).
    unit_cost, min_order : float ou dict {produit: valeur}
        Coût d'achat unitaire cᵢ et commande minimale qᵢᵐⁱⁿ. Un float
        est appliqué à tous les produits.
    holding_cost, shortage_cost : float
        Coûts unitaires de stockage (h) et de rupture (p), partagés
        par tous les produits.
    max_budget, max_capacity : float
        Contraintes globales (B, C).
    integer : bool
        Si True (par défaut), les commandes qᵢ sont contraintes à des
        valeurs entières (conforme à qᵢ ∈ ℤ⁺, méthodologie 2.3).

    Returns
    -------
    tuple
        (problem, order_vars, surplus_vars, shortage_vars)
    """

    products = list(scenarios.keys())
    n_scenarios = len(probabilities)

    unit_cost_by_product = _as_dict(unit_cost, products)
    min_order_by_product = _as_dict(min_order, products)

    order_category = pulp.LpInteger if integer else pulp.LpContinuous

    problem = pulp.LpProblem("gestion_stock_stochastique", pulp.LpMinimize)

    order_vars = {
        product: pulp.LpVariable(
            f"order_{product}",
            lowBound=min_order_by_product[product],
            cat=order_category
        )
        for product in products
    }

    surplus_vars = {
        (product, s): pulp.LpVariable(f"surplus_{product}_{s}", lowBound=0)
        for product in products
        for s in range(n_scenarios)
    }

    shortage_vars = {
        (product, s): pulp.LpVariable(f"shortage_{product}_{s}", lowBound=0)
        for product in products
        for s in range(n_scenarios)
    }

    # Objectif : coût d'achat certain + espérance des coûts de recours
    purchase_cost = pulp.lpSum(
        unit_cost_by_product[product] * order_vars[product] for product in products
    )

    expected_recourse_cost = pulp.lpSum(
        probabilities[s] * (
            holding_cost * surplus_vars[(product, s)]
            + shortage_cost * shortage_vars[(product, s)]
        )
        for product in products
        for s in range(n_scenarios)
    )

    problem += purchase_cost + expected_recourse_cost

    # Contraintes
    add_recourse_constraints(problem, order_vars, surplus_vars, shortage_vars, scenarios)
    add_budget_constraint(problem, order_vars, unit_cost_by_product, max_budget)
    add_capacity_constraint(problem, order_vars, max_capacity)

    return problem, order_vars, surplus_vars, shortage_vars

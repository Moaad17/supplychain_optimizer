import logging

import numpy as np
import pulp

from optimization.constraints import generate_demand_scenarios
from optimization.model import _as_dict, build_stochastic_model
from optimization.lshaped import solve_lshaped


logger = logging.getLogger(__name__)

# Au-delà de ce nombre de produits, le PDE direct devient trop gros
# (N + 2*N*S variables) -- on bascule sur la décomposition L-shaped.
# Cf. méthodologie, Étape 4.
LSHAPED_PRODUCT_THRESHOLD = 100


def solve_recourse_problem(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True
) -> dict:
    """
    Résout le PDE (build_stochastic_model) pour un jeu de scénarios
    donné et retourne un résultat exploitable. Fonction de bas niveau
    réutilisée par solve_optimal (HN, plusieurs scénarios) et
    compute_wait_and_see / evaluate_fixed_order (un seul scénario à la
    fois, avec probabilité 1.0).

    Returns
    -------
    dict
        {
            "status": "Optimal" / "Infeasible" / ...,
            "total_cost": float ou None,
            "orders": {produit: quantité commandée},
            "expected_shortage": {produit: rupture moyenne pondérée},
            "expected_surplus": {produit: surstock moyen pondéré},
            "service_level": {produit: proba pondérée d'absence de rupture}
        }
    """

    problem, order_vars, surplus_vars, shortage_vars = build_stochastic_model(
        scenarios, probabilities, unit_cost, holding_cost, shortage_cost,
        max_budget, max_capacity, min_order=min_order, integer=integer
    )

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[problem.status]

    if status != "Optimal":
        logger.warning(
            "Le solveur n'a pas trouvé de solution optimale : %s", status
        )
        return {
            "status": status,
            "total_cost": None,
            "orders": {},
            "expected_shortage": {},
            "expected_surplus": {},
            "service_level": {}
        }

    products = list(scenarios.keys())
    n_scenarios = len(probabilities)

    orders = {product: order_vars[product].value() for product in products}

    expected_shortage = {
        product: sum(
            probabilities[s] * shortage_vars[(product, s)].value()
            for s in range(n_scenarios)
        )
        for product in products
    }

    expected_surplus = {
        product: sum(
            probabilities[s] * surplus_vars[(product, s)].value()
            for s in range(n_scenarios)
        )
        for product in products
    }

    service_level = {
        product: sum(
            probabilities[s] for s in range(n_scenarios)
            if shortage_vars[(product, s)].value() <= 1e-6
        )
        for product in products
    }

    return {
        "status": status,
        "total_cost": pulp.value(problem.objective),
        "orders": orders,
        "expected_shortage": expected_shortage,
        "expected_surplus": expected_surplus,
        "service_level": service_level
    }


def evaluate_fixed_order(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    orders: dict
) -> dict:
    """
    Évalue une commande FIXE (déjà décidée, pas ré-optimisée par
    scénario) sur un jeu de scénarios : coût total attendu + stats de
    recours (rupture/surstock/niveau de service) par produit.

    À commande fixée, le recours optimal est trivial (pas besoin de
    résoudre un LP) : surplus = max(order-demande, 0),
    shortage = max(demande-order, 0). Utilisé pour la phase 2 de EV,
    pour la stratégie naïve, et pour compléter le résultat de
    solve_lshaped (qui ne renvoie que les commandes).
    """

    products = list(scenarios.keys())
    unit_cost_by_product = _as_dict(unit_cost, products)

    total_cost = 0.0
    expected_shortage = {}
    expected_surplus = {}
    service_level = {}

    for product in products:
        demand = np.asarray(scenarios[product], dtype=float)
        order = orders[product]

        shortage = np.maximum(demand - order, 0)
        surplus = np.maximum(order - demand, 0)

        expected_shortage[product] = float(np.dot(probabilities, shortage))
        expected_surplus[product] = float(np.dot(probabilities, surplus))
        service_level[product] = float(probabilities[shortage <= 1e-6].sum())

        total_cost += (
            unit_cost_by_product[product] * order
            + holding_cost * expected_surplus[product]
            + shortage_cost * expected_shortage[product]
        )

    return {
        "status": "Optimal",
        "total_cost": total_cost,
        "orders": dict(orders),
        "expected_shortage": expected_shortage,
        "expected_surplus": expected_surplus,
        "service_level": service_level
    }


def _scale_to_constraints(
    orders: dict,
    unit_cost_by_product: dict,
    max_budget: float,
    max_capacity: float
) -> tuple[dict, float]:
    """
    Réduit proportionnellement une commande FIXE pour qu'elle respecte
    le budget et la capacité, si elle les dépasse. Un vrai gestionnaire
    ne peut pas stocker plus que l'entrepôt ne le permet, même avec une
    règle "naïve" (ex : moyenne historique) qui ne connaît pas ces
    contraintes.

    Returns
    -------
    (commande ajustée, facteur d'échelle appliqué -- 1.0 si aucun
    ajustement n'était nécessaire).
    """

    total_units = sum(orders.values())
    total_purchase_cost = sum(
        unit_cost_by_product[product] * quantity for product, quantity in orders.items()
    )

    scale = 1.0
    if max_capacity > 0 and total_units > max_capacity:
        scale = min(scale, max_capacity / total_units)
    if max_budget > 0 and total_purchase_cost > max_budget:
        scale = min(scale, max_budget / total_purchase_cost)

    if scale >= 1.0:
        return dict(orders), 1.0

    return {product: quantity * scale for product, quantity in orders.items()}, scale


def solve_optimal(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True,
    lshaped_threshold: int = LSHAPED_PRODUCT_THRESHOLD,
    lshaped_tol: float = 0.01
) -> dict:
    """
    Résout le problème HN (Here and Now) en choisissant automatiquement
    la méthode de résolution selon la taille du problème (méthodologie,
    Étape 4) :

        - N produits < lshaped_threshold : PDE direct (solve_recourse_problem)
        - N produits >= lshaped_threshold : décomposition L-shaped
          (solve_lshaped), car le PDE direct devient trop gros
          (N + 2*N*S variables).

    Retourne le même format que solve_recourse_problem, avec en plus
    "method" ("PDE direct" ou "L-shaped").
    """

    n_products = len(scenarios)

    if n_products < lshaped_threshold:
        result = solve_recourse_problem(
            scenarios, probabilities, unit_cost, holding_cost, shortage_cost,
            max_budget, max_capacity, min_order=min_order, integer=integer
        )
        result["method"] = "PDE direct"
        return result

    lshaped_result = solve_lshaped(
        scenarios, probabilities, unit_cost, holding_cost, shortage_cost,
        max_budget, max_capacity, min_order=min_order, integer=integer,
        tol=lshaped_tol
    )

    if lshaped_result["status"] not in ("Optimal", "MaxIterations"):
        return {
            "status": lshaped_result["status"],
            "total_cost": None,
            "orders": {},
            "expected_shortage": {},
            "expected_surplus": {},
            "service_level": {},
            "method": "L-shaped"
        }

    # solve_lshaped ne renvoie que les commandes -- on complète avec les
    # stats de recours (rupture/surstock/service), calculables
    # directement à partir de la commande finale (cf. evaluate_fixed_order).
    result = evaluate_fixed_order(
        scenarios, probabilities, unit_cost, holding_cost, shortage_cost,
        lshaped_result["orders"]
    )
    result["status"] = lshaped_result["status"]
    result["method"] = "L-shaped"
    result["n_iterations"] = lshaped_result["n_iterations"]
    result["gap"] = lshaped_result["gap"]

    return result


def compute_wait_and_see(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True
) -> float:
    """
    WS (Wait and See) : suppose une information parfaite -- pour
    chaque scénario, on résout le problème comme si la demande de ce
    scénario était connue à l'avance, puis on moyenne les coûts
    optimaux (pondérés par leur probabilité). C'est une borne
    théorique : le coût qu'on aurait si on pouvait attendre de
    connaître la demande avant de commander.

    WS <= HN toujours (l'information parfaite ne peut qu'aider).
    """

    n_scenarios = len(probabilities)

    total_cost = 0.0

    for s in range(n_scenarios):
        single_scenario = {
            product: np.array([scenarios[product][s]]) for product in scenarios
        }

        result = solve_recourse_problem(
            single_scenario, np.array([1.0]), unit_cost, holding_cost,
            shortage_cost, max_budget, max_capacity,
            min_order=min_order, integer=integer
        )

        if result["status"] != "Optimal":
            raise RuntimeError(
                f"WS : le scénario {s} est infaisable ({result['status']})."
            )

        total_cost += probabilities[s] * result["total_cost"]

    return total_cost


def compute_expected_value(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True
) -> tuple[float, dict]:
    """
    EV (Expected Value) : résout le problème déterministe avec la
    demande MOYENNE (espérance pondérée des scénarios), puis évalue le
    coût RÉEL de cette commande fixe sur chaque scénario (le fait
    d'ignorer l'incertitude a un coût, qu'on mesure ici).

    Contrairement à HN, la commande n'est PAS ré-optimisée par
    scénario : c'est une seule commande figée, évaluée après coup.

    Returns
    -------
    (ev_cost, ev_orders)
    """

    mean_demand = {
        product: float(np.average(scenarios[product], weights=probabilities))
        for product in scenarios
    }

    # Phase 1 : résoudre le déterministe sur la demande moyenne
    mean_scenario = {
        product: np.array([mean_demand[product]]) for product in scenarios
    }

    phase1_result = solve_recourse_problem(
        mean_scenario, np.array([1.0]), unit_cost, holding_cost, shortage_cost,
        max_budget, max_capacity, min_order=min_order, integer=integer
    )

    if phase1_result["status"] != "Optimal":
        raise RuntimeError(
            f"EV : le problème déterministe est infaisable "
            f"({phase1_result['status']})."
        )

    ev_orders = phase1_result["orders"]

    # Phase 2 : évaluer ev_orders (commande FIGÉE) sur tous les scénarios
    ev_cost = evaluate_fixed_order(
        scenarios, probabilities, unit_cost, holding_cost, shortage_cost, ev_orders
    )["total_cost"]

    return ev_cost, ev_orders


def compute_naive_cost(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    naive_orders: dict,
    max_budget: float,
    max_capacity: float
) -> dict:
    """
    Stratégie "naïve" : commande = moyenne historique par produit (cf.
    optimization.baselines.compute_naive_orders), sans prévision ni
    optimisation. Sert de repère "zéro effort" pour mesurer ce
    qu'apporte tout le reste du pipeline (forecasting + stochastique).

    La moyenne historique brute peut dépasser le budget ou la capacité
    (elle ne les connaît pas) -- dans ce cas, elle est réduite
    proportionnellement pour rester réalisable (cf.
    _scale_to_constraints), sinon la comparaison avec HN/EV/WS serait
    faussée (un plan qui ne rentre pas dans l'entrepôt n'est pas
    vraiment "moins cher", il est juste infaisable).

    Returns
    -------
    dict : résultat de evaluate_fixed_order, + "scale_applied" (1.0 si
    aucun ajustement) et "orders_before_scaling".
    """

    products = list(scenarios.keys())
    unit_cost_by_product = _as_dict(unit_cost, products)

    adjusted_orders, scale = _scale_to_constraints(
        naive_orders, unit_cost_by_product, max_budget, max_capacity
    )

    result = evaluate_fixed_order(
        scenarios, probabilities, unit_cost, holding_cost, shortage_cost, adjusted_orders
    )
    result["scale_applied"] = scale
    result["orders_before_scaling"] = dict(naive_orders)

    return result


def evaluate_strategies(
    forecast_results: dict,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    historical_orders: dict | None = None,
    min_order=0,
    integer: bool = True,
    n_sampled: int = 20,
    extreme_prob: float = 0.15,
    normal_prob: float = 0.50,
    confidence_level: float = 0.95,
    seed: int = 42,
    lshaped_threshold: int = LSHAPED_PRODUCT_THRESHOLD
) -> dict:
    """
    Calcule et compare les stratégies de la méthodologie (Étapes 4-6) :
    HN (stochastique, PDE direct ou L-shaped selon N -- "here and now"),
    WS (information parfaite), EV (déterministe sur la moyenne, évalué
    sur les scénarios réels), et Naïf (moyenne historique, si
    `historical_orders` est fourni), et en déduit :

        EVPI = HN - WS   coût de l'incertitude
        VSS  = EV - HN   valeur apportée par l'optimisation stochastique

    Toutes les stratégies utilisent le MÊME jeu de scénarios (généré
    une seule fois ici), pour que la comparaison soit valide.

    Parameters
    ----------
    historical_orders : dict, optionnel
        {produit: quantité}, cf. optimization.baselines.compute_naive_orders.
        Si omis, la stratégie "Naïf" n'est pas calculée. Si elle dépasse
        le budget/la capacité (la moyenne historique ne les connaît
        pas), elle est réduite proportionnellement avant d'être
        évaluée -- cf. compute_naive_cost.

    Returns
    -------
    dict
        {
            "hn": résultat complet (avec "method": "PDE direct"/"L-shaped"),
            "ws_cost": float,
            "ev_cost": float, "ev_orders": {...},
            "naive_cost": float ou None, "naive_orders": dict ou None,
            "evpi": float,
            "vss": float,
            "checks_ok": bool,  # WS <= HN <= EV (à la tolérance numérique près)
            "comparison": [
                {"strategy": "WS", "cost": ..., "vs_hn_abs": ..., "vs_hn_pct": ...},
                {"strategy": "HN", ...}, {"strategy": "EV", ...},
                {"strategy": "Naïf", ...} si historical_orders fourni
            ]  # trié par coût croissant
        }
    """

    scenarios, probabilities = generate_demand_scenarios(
        forecast_results,
        n_sampled=n_sampled,
        extreme_prob=extreme_prob,
        normal_prob=normal_prob,
        confidence_level=confidence_level,
        seed=seed
    )

    common_args = dict(
        unit_cost=unit_cost, holding_cost=holding_cost, shortage_cost=shortage_cost,
        max_budget=max_budget, max_capacity=max_capacity,
        min_order=min_order, integer=integer
    )

    hn_result = solve_optimal(
        scenarios, probabilities, lshaped_threshold=lshaped_threshold, **common_args
    )

    if hn_result["status"] not in ("Optimal", "MaxIterations"):
        raise RuntimeError(f"HN : problème infaisable ({hn_result['status']}).")

    ws_cost = compute_wait_and_see(scenarios, probabilities, **common_args)
    ev_cost, ev_orders = compute_expected_value(scenarios, probabilities, **common_args)

    naive_result = None
    naive_cost = None
    if historical_orders is not None:
        naive_result = compute_naive_cost(
            scenarios, probabilities, unit_cost, holding_cost, shortage_cost,
            historical_orders, max_budget=max_budget, max_capacity=max_capacity
        )
        naive_cost = naive_result["total_cost"]

        if naive_result["scale_applied"] < 1.0:
            logger.warning(
                "La commande naïve (moyenne historique) dépassait le budget "
                "et/ou la capacité -- réduite à %.0f%% pour rester réalisable.",
                naive_result["scale_applied"] * 100
            )

    hn_cost = hn_result["total_cost"]
    evpi = hn_cost - ws_cost
    vss = ev_cost - hn_cost

    tolerance = 1e-4 * max(abs(hn_cost), 1.0)
    checks_ok = (ws_cost <= hn_cost + tolerance) and (hn_cost <= ev_cost + tolerance)

    if not checks_ok:
        logger.warning(
            "L'inégalité WS <= HN <= EV n'est pas respectée "
            "(WS=%.2f, HN=%.2f, EV=%.2f) : vérifier le modèle.",
            ws_cost, hn_cost, ev_cost
        )

    comparison = [
        {"strategy": "WS", "cost": ws_cost},
        {"strategy": "HN", "cost": hn_cost},
        {"strategy": "EV", "cost": ev_cost},
    ]
    if naive_cost is not None:
        comparison.append({"strategy": "Naïf", "cost": naive_cost})

    for row in comparison:
        row["vs_hn_abs"] = row["cost"] - hn_cost
        row["vs_hn_pct"] = (row["cost"] - hn_cost) / hn_cost * 100 if hn_cost else 0.0

    comparison.sort(key=lambda row: row["cost"])

    return {
        "hn": hn_result,
        "ws_cost": ws_cost,
        "ev_cost": ev_cost,
        "ev_orders": ev_orders,
        "naive_cost": naive_cost,
        "naive_orders": naive_result["orders"] if naive_result else None,
        "naive_scale_applied": naive_result["scale_applied"] if naive_result else None,
        "evpi": evpi,
        "vss": vss,
        "checks_ok": checks_ok,
        "comparison": comparison
    }

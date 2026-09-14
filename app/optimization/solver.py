import logging

import numpy as np
import pulp

from optimization.constraints import generate_demand_scenarios
from optimization.model import _as_dict, build_stochastic_model


logger = logging.getLogger(__name__)


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
    réutilisée par optimize_inventory (HN, plusieurs scénarios) et
    compute_wait_and_see / compute_expected_value (un seul scénario à
    la fois, avec probabilité 1.0).

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


def optimize_inventory(
    forecast_results: dict,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True,
    n_sampled: int = 20,
    extreme_prob: float = 0.15,
    normal_prob: float = 0.50,
    confidence_level: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Détermine la quantité optimale à commander pour chaque produit,
    sous incertitude de la demande ("Here and Now" -- décision prise
    avant de connaître la demande), à partir des résultats du
    forecasting (Phase 2). Génère les scénarios (méthode C, cf.
    constraints.generate_demand_scenarios) puis résout le PDE direct.

    Voir solve_recourse_problem pour le format du résultat -- avec en
    plus "scenarios" et "probabilities", pour être réutilisés par
    compute_wait_and_see / compute_expected_value sans regénérer un
    tirage différent (cf. evaluate_strategies).
    """

    scenarios, probabilities = generate_demand_scenarios(
        forecast_results,
        n_sampled=n_sampled,
        extreme_prob=extreme_prob,
        normal_prob=normal_prob,
        confidence_level=confidence_level,
        seed=seed
    )

    result = solve_recourse_problem(
        scenarios, probabilities, unit_cost, holding_cost, shortage_cost,
        max_budget, max_capacity, min_order=min_order, integer=integer
    )

    result["scenarios"] = scenarios
    result["probabilities"] = probabilities

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

    products = list(scenarios.keys())
    n_scenarios = len(probabilities)

    total_cost = 0.0

    for s in range(n_scenarios):
        single_scenario = {
            product: np.array([scenarios[product][s]]) for product in products
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

    products = list(scenarios.keys())
    n_scenarios = len(probabilities)

    unit_cost_by_product = _as_dict(unit_cost, products)

    mean_demand = {
        product: float(np.average(scenarios[product], weights=probabilities))
        for product in products
    }

    # Phase 1 : résoudre le déterministe sur la demande moyenne
    mean_scenario = {
        product: np.array([mean_demand[product]]) for product in products
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
    total_cost = 0.0

    for s in range(n_scenarios):
        scenario_cost = 0.0

        for product in products:
            demand = scenarios[product][s]
            order = ev_orders[product]
            surplus = max(order - demand, 0)
            shortage = max(demand - order, 0)

            scenario_cost += (
                unit_cost_by_product[product] * order
                + holding_cost * surplus
                + shortage_cost * shortage
            )

        total_cost += probabilities[s] * scenario_cost

    return total_cost, ev_orders


def evaluate_strategies(
    forecast_results: dict,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True,
    n_sampled: int = 20,
    extreme_prob: float = 0.15,
    normal_prob: float = 0.50,
    confidence_level: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Calcule et compare les 3 stratégies de la méthodologie (Étape 5) :
    HN (stochastique, "here and now"), WS (information parfaite) et EV
    (déterministe sur la moyenne, évalué sur les scénarios réels), et
    en déduit :

        EVPI = HN - WS   coût de l'incertitude
        VSS  = EV - HN   valeur apportée par l'optimisation stochastique

    Les 3 stratégies utilisent le MÊME jeu de scénarios (généré une
    seule fois ici), pour que la comparaison soit valide -- WS et EV
    ne doivent pas être évalués sur un tirage Monte Carlo différent de
    celui utilisé pour HN.

    Returns
    -------
    dict
        {
            "hn": résultat complet de solve_recourse_problem,
            "ws_cost": float,
            "ev_cost": float,
            "ev_orders": {produit: quantité},
            "evpi": float,
            "vss": float,
            "checks_ok": bool  # WS <= HN <= EV (à la tolérance numérique près)
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

    hn_result = solve_recourse_problem(scenarios, probabilities, **common_args)

    if hn_result["status"] != "Optimal":
        raise RuntimeError(f"HN : problème infaisable ({hn_result['status']}).")

    ws_cost = compute_wait_and_see(scenarios, probabilities, **common_args)
    ev_cost, ev_orders = compute_expected_value(scenarios, probabilities, **common_args)

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

    return {
        "hn": hn_result,
        "ws_cost": ws_cost,
        "ev_cost": ev_cost,
        "ev_orders": ev_orders,
        "evpi": evpi,
        "vss": vss,
        "checks_ok": checks_ok
    }

import logging

import numpy as np
import pulp

from optimization.constraints import add_budget_constraint, add_capacity_constraint
from optimization.model import _as_dict


logger = logging.getLogger(__name__)


def _solve_recourse_subproblem(
    fixed_order: dict,
    demand: dict,
    holding_cost: float,
    shortage_cost: float
) -> tuple[float, dict]:
    """
    Résout le sous-problème de recours (2ème étape) pour UN scénario,
    la commande q étant FIXÉE à des valeurs numériques (pas des
    variables PuLP) -- cf. méthodologie, Étape 6.3, Étape B.

    Returns
    -------
    (objective_value, duals)
        objective_value : coût optimal du recours pour ce scénario.
        duals : {produit: variable duale de la contrainte de lien},
            utilisée pour construire la coupe d'optimalité (cf.
            méthodologie, Étape 6.4 : λ = h si surstock actif,
            λ = -p si rupture active).
    """

    products = list(fixed_order.keys())

    problem = pulp.LpProblem("sous_probleme_recours", pulp.LpMinimize)

    surplus = {p: pulp.LpVariable(f"surplus_{p}", lowBound=0) for p in products}
    shortage = {p: pulp.LpVariable(f"shortage_{p}", lowBound=0) for p in products}

    problem += pulp.lpSum(
        holding_cost * surplus[p] + shortage_cost * shortage[p] for p in products
    )

    link_constraints = {}
    for p in products:
        constraint = surplus[p] - shortage[p] == fixed_order[p] - demand[p]
        link_constraints[p] = constraint
        problem += constraint, f"lien_{p}"

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    duals = {p: link_constraints[p].pi for p in products}
    objective_value = pulp.value(problem.objective)

    return objective_value, duals


def solve_lshaped(
    scenarios: dict,
    probabilities: np.ndarray,
    unit_cost,
    holding_cost: float,
    shortage_cost: float,
    max_budget: float,
    max_capacity: float,
    min_order=0,
    integer: bool = True,
    tol: float = 0.01,
    max_iterations: int = 50
) -> dict:
    """
    Résout le problème stochastique par décomposition de Benders /
    L-shaped (single-cut, recours simple) -- cf. méthodologie, Étape 6.

    Sépare le problème en :
    - un MASTER : décide q (1ère étape) + une variable θ qui
      approxime, par le bas, le coût de recours espéré ;
    - S SOUS-PROBLÈMES indépendants (un par scénario, 2ème étape),
      résolus pour un q fixé, dont les variables duales servent à
      construire une coupe d'optimalité ajoutée au master à
      l'itération suivante.

    Utile quand N produits x S scénarios rend le PDE direct
    (solver.solve_recourse_problem) trop gros pour être résolu d'un
    bloc. Pour la taille de ce projet (quelques produits), le PDE
    direct suffit largement ; cette fonction sert de démonstration /
    validation de la méthode (elle doit converger vers le même coût
    optimal que le PDE direct, cf. tests).

    Parameters
    ----------
    scenarios, probabilities : cf. constraints.generate_demand_scenarios.
    unit_cost, holding_cost, shortage_cost, max_budget, max_capacity,
    min_order, integer : cf. model.build_stochastic_model.
    tol : float
        Tolérance de convergence sur le gap relatif (UB-LB)/UB.
    max_iterations : int
        Nombre maximal d'itérations avant d'abandonner.

    Returns
    -------
    dict
        {
            "status": "Optimal" / "MaxIterations" / statut du solveur,
            "total_cost": float ou None,
            "orders": {produit: quantité},
            "n_iterations": int,
            "gap": float ou None,
            "history": [{"iteration", "lower_bound", "upper_bound", "gap"}, ...]
        }
    """

    products = list(scenarios.keys())
    n_scenarios = len(probabilities)

    unit_cost_by_product = _as_dict(unit_cost, products)
    min_order_by_product = _as_dict(min_order, products)

    order_category = pulp.LpInteger if integer else pulp.LpContinuous

    cuts = []  # liste de (alpha, beta_dict)
    upper_bound = float("inf")
    history = []

    order_solution = {}

    for iteration in range(1, max_iterations + 1):

        # ---- ÉTAPE A : résoudre le Master ----
        master = pulp.LpProblem("master", pulp.LpMinimize)

        order_vars = {
            p: pulp.LpVariable(
                f"order_{p}", lowBound=min_order_by_product[p], cat=order_category
            )
            for p in products
        }
        theta = pulp.LpVariable("theta", lowBound=0)

        master += pulp.lpSum(
            unit_cost_by_product[p] * order_vars[p] for p in products
        ) + theta

        add_budget_constraint(master, order_vars, unit_cost_by_product, max_budget)
        add_capacity_constraint(master, order_vars, max_capacity)

        for i, (alpha, beta) in enumerate(cuts):
            master += (
                theta >= alpha + pulp.lpSum(
                    beta[p] * order_vars[p] for p in products
                ),
                f"coupe_{i}"
            )

        master.solve(pulp.PULP_CBC_CMD(msg=False))
        master_status = pulp.LpStatus[master.status]

        if master_status != "Optimal":
            logger.warning("Master L-shaped infaisable : %s", master_status)
            return {
                "status": master_status,
                "total_cost": None,
                "orders": {},
                "n_iterations": iteration,
                "gap": None,
                "history": history
            }

        order_solution = {p: order_vars[p].value() for p in products}
        purchase_cost_value = sum(
            unit_cost_by_product[p] * order_solution[p] for p in products
        )
        lower_bound = purchase_cost_value + theta.value()

        # ---- ÉTAPE B : résoudre les S sous-problèmes de recours ----
        scenario_costs = []
        duals_per_scenario = []

        for s in range(n_scenarios):
            demand_s = {p: scenarios[p][s] for p in products}
            cost_s, duals_s = _solve_recourse_subproblem(
                order_solution, demand_s, holding_cost, shortage_cost
            )
            scenario_costs.append(cost_s)
            duals_per_scenario.append(duals_s)

        expected_recourse_cost = sum(
            probabilities[s] * scenario_costs[s] for s in range(n_scenarios)
        )

        # ---- ÉTAPE C : borne supérieure ----
        upper_bound = min(upper_bound, purchase_cost_value + expected_recourse_cost)

        gap = (upper_bound - lower_bound) / upper_bound if upper_bound > 0 else 0.0
        history.append({
            "iteration": iteration,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "gap": gap
        })

        # ---- ÉTAPE D : convergence ----
        if gap <= tol:
            return {
                "status": "Optimal",
                "total_cost": upper_bound,
                "orders": order_solution,
                "n_iterations": iteration,
                "gap": gap,
                "history": history
            }

        # ---- ÉTAPE E : coupe d'optimalité ----
        e_vector = {
            p: sum(
                probabilities[s] * duals_per_scenario[s][p]
                for s in range(n_scenarios)
            )
            for p in products
        }
        alpha = expected_recourse_cost - sum(
            e_vector[p] * order_solution[p] for p in products
        )
        cuts.append((alpha, e_vector))

    logger.warning(
        "L-shaped n'a pas convergé en %d itérations (gap=%.4f).",
        max_iterations, gap
    )

    return {
        "status": "MaxIterations",
        "total_cost": upper_bound,
        "orders": order_solution,
        "n_iterations": max_iterations,
        "gap": gap,
        "history": history
    }

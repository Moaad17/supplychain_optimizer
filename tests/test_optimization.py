"""
Tests pour la Phase 3 (optimisation des stocks sous incertitude) :
génération de scénarios (constraints.py), PDE direct (model.py /
solver.py), métriques WS/HN/EV/EVPI/VSS (solver.py), et décomposition
L-shaped (lshaped.py).
"""

import numpy as np
import pytest
from scipy.stats import norm

from optimization.constraints import generate_demand_scenarios
from optimization.solver import (
    solve_recourse_problem,
    optimize_inventory,
    evaluate_strategies
)
from optimization.lshaped import solve_lshaped


UNIT_COST = 10
HOLDING_COST = 5
SHORTAGE_COST = 20

Z_95 = 1.959963984540054


def _fake_forecast_result(mean: float, std_95: float, horizon: int = 1) -> dict:
    """
    Construit un résultat de forecast minimal (predictions/lower/upper
    constants sur l'horizon), pour tester l'optimisation indépendamment
    du module de forecasting (Phase 2). `std_95` est l'écart-type visé
    (la demi-largeur de l'intervalle à 95% vaut Z_95 * std_95).
    """

    half_width = Z_95 * std_95

    return {
        "predictions": [mean] * horizon,
        "lower": [mean - half_width] * horizon,
        "upper": [mean + half_width] * horizon,
        "dates": None,
        "mae": None
    }


# ==========================================================
# generate_demand_scenarios (méthode C)
# ==========================================================

def test_generate_demand_scenarios_probabilities_sum_to_one():
    forecast_results = {"A": _fake_forecast_result(mean=100, std_95=20)}

    _, probabilities = generate_demand_scenarios(forecast_results, n_sampled=20)

    assert probabilities.sum() == pytest.approx(1.0)
    assert len(probabilities) == 3 + 20


def test_generate_demand_scenarios_structural_scenarios_ordered():
    """Les 3 scénarios structurels doivent être pessimiste < normal <
    optimiste (indices 0, 1, 2)."""

    forecast_results = {"A": _fake_forecast_result(mean=100, std_95=20)}

    scenarios, _ = generate_demand_scenarios(forecast_results, n_sampled=5, seed=1)

    pessimiste, normal, optimiste = scenarios["A"][:3]
    assert pessimiste < normal < optimiste
    assert normal == pytest.approx(100)


def test_generate_demand_scenarios_non_negative():
    forecast_results = {"A": _fake_forecast_result(mean=10, std_95=20)}  # std > mean

    scenarios, _ = generate_demand_scenarios(forecast_results, n_sampled=200, seed=1)

    assert (scenarios["A"] >= 0).all()


# ==========================================================
# solve_recourse_problem / optimize_inventory (PDE direct, HN)
# ==========================================================

def test_optimize_inventory_respects_budget_and_capacity():
    forecast_results = {"Produit": _fake_forecast_result(mean=1000, std_95=100)}

    result = optimize_inventory(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=2000, max_capacity=10**9, n_sampled=50
    )

    assert result["status"] == "Optimal"
    assert result["orders"]["Produit"] * UNIT_COST <= 2000 + 1e-6


def test_optimize_inventory_orders_are_integers_by_default():
    forecast_results = {
        "A": _fake_forecast_result(mean=137, std_95=15),
        "B": _fake_forecast_result(mean=283, std_95=30),
    }

    result = optimize_inventory(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, n_sampled=30
    )

    for order in result["orders"].values():
        assert order == pytest.approx(round(order))


def test_optimize_inventory_infeasible_reports_status_without_crashing():
    forecast_results = {"Produit": _fake_forecast_result(mean=100, std_95=10)}

    result = optimize_inventory(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=-1, max_capacity=10**9, n_sampled=10
    )

    assert result["status"] != "Optimal"
    assert result["orders"] == {}


def test_optimize_inventory_matches_newsvendor_with_fine_grained_scenarios():
    """
    Avec un échantillonnage quasi-SAA pur (peu de poids sur les 3
    scénarios structurels, beaucoup de scénarios échantillonnés), la
    commande optimale doit converger vers le quantile critique du
    modèle newsvendor : CR = (shortage_cost - unit_cost) / (shortage_cost + holding_cost).

    NB : avec les poids par défaut de la méthode C (normal_prob=0.50
    concentré en un seul point = la moyenne), ce n'est PAS vrai -- voir
    test_default_method_c_snaps_to_an_anchor_point ci-dessous.
    """

    mean, std = 1000.0, 150.0
    critical_ratio = (SHORTAGE_COST - UNIT_COST) / (SHORTAGE_COST + HOLDING_COST)

    forecast_results = {"Produit": _fake_forecast_result(mean=mean, std_95=std)}

    result = optimize_inventory(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9,
        n_sampled=500, extreme_prob=0.001, normal_prob=0.001,
        integer=False, seed=1
    )

    expected_order = mean + norm.ppf(critical_ratio) * std

    assert result["orders"]["Produit"] == pytest.approx(expected_order, rel=0.05)


def test_default_method_c_snaps_to_an_anchor_point():
    """
    Caractéristique DOCUMENTÉE (pas un bug) de la méthode C par
    défaut : avec 50% de la probabilité concentrée en un seul point (le
    scénario "normal" = la moyenne), la commande optimale "accroche"
    souvent cette moyenne au lieu du quantile continu théorique, dès
    que le ratio critique tombe dans la plage couverte par cet atome de
    probabilité. Ici CR=0.4 tombe dans cette plage -> la commande vaut
    exactement la moyenne (1000), pas ~962 comme le donnerait la
    théorie newsvendor continue (cf. test ci-dessus avec plus de
    scénarios échantillonnés et moins de poids structurel).
    """

    forecast_results = {"Produit": _fake_forecast_result(mean=1000, std_95=150)}

    result = optimize_inventory(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9,
        n_sampled=100, integer=False, seed=1
    )

    assert result["orders"]["Produit"] == pytest.approx(1000, abs=1.0)


# ==========================================================
# evaluate_strategies (WS / HN / EV / EVPI / VSS)
# ==========================================================

def test_evaluate_strategies_respects_ws_hn_ev_ordering():
    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=150),
        "B": _fake_forecast_result(mean=500, std_95=100),
    }

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, n_sampled=30, seed=1
    )

    assert result["hn"]["status"] == "Optimal"
    assert result["checks_ok"]
    assert result["ws_cost"] <= result["hn"]["total_cost"] + 1e-6
    assert result["hn"]["total_cost"] <= result["ev_cost"] + 1e-6
    assert result["evpi"] >= -1e-6
    assert result["vss"] >= -1e-6


def test_evaluate_strategies_with_binding_capacity():
    """Les inégalités WS <= HN <= EV doivent tenir aussi quand la
    capacité est la contrainte active (pas seulement en marché libre)."""

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=150),
        "B": _fake_forecast_result(mean=1000, std_95=100),
    }

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=800, n_sampled=30, seed=1
    )

    assert result["checks_ok"]


# ==========================================================
# solve_lshaped (décomposition de Benders)
# ==========================================================

@pytest.mark.parametrize("max_capacity", [10**9, 1500])
def test_lshaped_matches_direct_pde(max_capacity):
    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=150),
        "B": _fake_forecast_result(mean=500, std_95=100),
        "C": _fake_forecast_result(mean=700, std_95=120),
    }

    scenarios, probabilities = generate_demand_scenarios(
        forecast_results, n_sampled=30, seed=1
    )

    direct = solve_recourse_problem(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=max_capacity
    )

    lshaped = solve_lshaped(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=max_capacity, tol=0.01
    )

    assert direct["status"] == "Optimal"
    assert lshaped["status"] == "Optimal"
    assert sum(lshaped["orders"].values()) <= max_capacity + 1e-6
    assert lshaped["total_cost"] == pytest.approx(direct["total_cost"], rel=0.02)

"""
Tests pour la Phase 3 (optimisation des stocks sous incertitude) :
génération de scénarios (constraints.py), PDE direct (model.py /
solver.py), métriques WS/HN/EV/EVPI/VSS (solver.py), et décomposition
L-shaped (lshaped.py).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from optimization.constraints import generate_demand_scenarios
from optimization.solver import (
    solve_recourse_problem,
    solve_optimal,
    evaluate_strategies
)
from optimization.lshaped import solve_lshaped
from optimization.baselines import compute_unit_costs


def _optimize(forecast_results, n_sampled=20, extreme_prob=0.15, normal_prob=0.50,
              seed=42, integer=True, **kwargs):
    """Raccourci de test : génère les scénarios (méthode C) puis résout
    avec solve_optimal (équivalent de l'ancien optimize_inventory)."""

    scenarios, probabilities = generate_demand_scenarios(
        forecast_results, n_sampled=n_sampled, extreme_prob=extreme_prob,
        normal_prob=normal_prob, seed=seed
    )
    return solve_optimal(scenarios, probabilities, integer=integer, **kwargs)


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

    result = _optimize(
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

    result = _optimize(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, n_sampled=30
    )

    for order in result["orders"].values():
        assert order == pytest.approx(round(order))


def test_optimize_inventory_infeasible_reports_status_without_crashing():
    forecast_results = {"Produit": _fake_forecast_result(mean=100, std_95=10)}

    result = _optimize(
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

    result = _optimize(
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

    result = _optimize(
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

def test_solve_optimal_dispatches_to_pde_direct_below_threshold():
    forecast_results = {"A": _fake_forecast_result(mean=1000, std_95=150)}
    scenarios, probabilities = generate_demand_scenarios(forecast_results, n_sampled=20, seed=1)

    result = solve_optimal(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, lshaped_threshold=100
    )

    assert result["method"] == "PDE direct"
    assert result["status"] == "Optimal"


def test_solve_optimal_dispatches_to_lshaped_above_threshold():
    """Force le seuil très bas (1) pour vérifier le branchement
    L-shaped sans avoir besoin de 100 produits réels dans le test."""

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=150),
        "B": _fake_forecast_result(mean=500, std_95=100),
    }
    scenarios, probabilities = generate_demand_scenarios(forecast_results, n_sampled=20, seed=1)

    direct = solve_recourse_problem(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9
    )
    dispatched = solve_optimal(
        scenarios, probabilities,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, lshaped_threshold=1
    )

    assert dispatched["method"] == "L-shaped"
    assert dispatched["status"] in ("Optimal", "MaxIterations")
    # Complété par evaluate_fixed_order -- doit avoir les mêmes clés que le PDE direct
    assert set(dispatched) >= {"expected_shortage", "expected_surplus", "service_level"}
    assert dispatched["total_cost"] == pytest.approx(direct["total_cost"], rel=0.02)


# ==========================================================
# Stratégie Naïve (moyenne historique) + tableau de comparaison
# ==========================================================

def test_evaluate_strategies_with_naive_baseline():
    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=150),
        "B": _fake_forecast_result(mean=500, std_95=100),
    }
    # Une commande naïve délibérément mauvaise (bien en dessous de la
    # moyenne) pour vérifier qu'elle ressort bien comme la plus chère.
    historical_orders = {"A": 400, "B": 200}

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9,
        historical_orders=historical_orders, n_sampled=30, seed=1
    )

    assert result["naive_cost"] is not None
    assert result["naive_orders"] == historical_orders
    assert result["naive_cost"] >= result["hn"]["total_cost"]

    strategies_present = {row["strategy"] for row in result["comparison"]}
    assert strategies_present == {"WS", "HN", "EV", "Naïf"}

    # Trié par coût croissant
    costs = [row["cost"] for row in result["comparison"]]
    assert costs == sorted(costs)

    # HN est la référence (vs_hn_abs=0, vs_hn_pct=0)
    hn_row = next(row for row in result["comparison"] if row["strategy"] == "HN")
    assert hn_row["vs_hn_abs"] == pytest.approx(0.0, abs=1e-6)
    assert hn_row["vs_hn_pct"] == pytest.approx(0.0, abs=1e-6)

    naive_row = next(row for row in result["comparison"] if row["strategy"] == "Naïf")
    assert naive_row["vs_hn_abs"] == pytest.approx(
        result["naive_cost"] - result["hn"]["total_cost"]
    )


def test_naive_orders_exceeding_capacity_are_scaled_down():
    """
    Non-régression : la commande naïve (moyenne historique) ne connaît
    pas le budget/la capacité et peut les dépasser. Sans réduction, son
    coût "évalué" ignore la contrainte de capacité et ressort
    artificiellement moins cher que HN (qui la respecte) -- un plan
    physiquement impossible ne doit jamais sembler "moins cher".
    """

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=100),
        "B": _fake_forecast_result(mean=1000, std_95=100),
    }
    # Somme = 2000, largement au-dessus de la capacité choisie (800)
    historical_orders = {"A": 1000, "B": 1000}

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=800,
        historical_orders=historical_orders, n_sampled=30, seed=1
    )

    assert result["naive_scale_applied"] == pytest.approx(800 / 2000)
    assert sum(result["naive_orders"].values()) == pytest.approx(800)
    # La commande naïve réduite doit rester cohérente avec les
    # proportions d'origine (1:1 ici)
    assert result["naive_orders"]["A"] == pytest.approx(result["naive_orders"]["B"])


def test_evaluate_strategies_without_naive_baseline_omits_it():
    forecast_results = {"A": _fake_forecast_result(mean=1000, std_95=150)}

    result = evaluate_strategies(
        forecast_results,
        unit_cost=UNIT_COST, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=10**9, n_sampled=20, seed=1
    )

    assert result["naive_cost"] is None
    assert {row["strategy"] for row in result["comparison"]} == {"WS", "HN", "EV"}


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


# ==========================================================
# baselines.compute_unit_costs (prix réel par produit)
# ==========================================================

def _make_df_with_price():
    return pd.DataFrame({
        "product": ["A", "A", "B", "B"],
        "quantity": [10, 12, 5, 6],
        "unit_price": [9.5, 9.5, 25.0, 25.0],
    })


def test_compute_unit_costs_uses_real_price_per_product():
    costs = compute_unit_costs(_make_df_with_price(), default_unit_cost=1.0)

    assert costs == {"A": pytest.approx(9.5), "B": pytest.approx(25.0)}


def test_compute_unit_costs_falls_back_without_price_column():
    df_no_price = _make_df_with_price().drop(columns=["unit_price"])

    costs = compute_unit_costs(df_no_price, default_unit_cost=7.0)

    assert costs == {"A": 7.0, "B": 7.0}


def test_compute_unit_costs_can_be_used_directly_by_the_optimizer():
    """Vérifie que le dict retourné s'intègre tel quel dans solve_optimal
    (cᵢ par produit, cf. model.build_stochastic_model)."""

    forecast_results = {
        "A": _fake_forecast_result(mean=1000, std_95=100),
        "B": _fake_forecast_result(mean=1000, std_95=100),
    }
    unit_cost = compute_unit_costs(_make_df_with_price(), default_unit_cost=1.0)

    scenarios, probabilities = generate_demand_scenarios(forecast_results, n_sampled=20, seed=1)
    result = solve_optimal(
        scenarios, probabilities,
        unit_cost=unit_cost, holding_cost=HOLDING_COST, shortage_cost=SHORTAGE_COST,
        max_budget=10**9, max_capacity=800
    )

    assert result["status"] == "Optimal"
    # Le produit le moins cher (A, 9.5) doit être favorisé par rapport
    # au plus cher (B, 25.0) sous une capacité contrainte
    assert result["orders"]["A"] > result["orders"]["B"]

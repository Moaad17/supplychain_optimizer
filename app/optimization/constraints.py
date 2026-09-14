import numpy as np
import pulp
from scipy.stats import norm


def _aggregate_forecast_to_mu_sigma(result: dict, confidence_level: float) -> tuple[float, float]:
    """
    Agrège un résultat de forecast (predictions/lower/upper par mois,
    cf. forecasting/prophet_model.py et ml_model.py) en une seule
    distribution de la demande TOTALE sur l'horizon :

        μ = somme des prédictions mensuelles
        σ = somme quadratique des écarts-types mensuels déduits des
            intervalles de confiance (hypothèse : les mois sont
            indépendants -> Var(somme) = somme des Var)

    L'écart-type mensuel est déduit de upper = mean + z*std, avec z le
    quantile normal correspondant à `confidence_level`.
    """

    predictions = np.array(result["predictions"], dtype=float)
    lower = np.array(result["lower"], dtype=float)
    upper = np.array(result["upper"], dtype=float)

    z_score = norm.ppf(0.5 + confidence_level / 2)
    monthly_std = np.clip((upper - lower) / (2 * z_score), a_min=1e-6, a_max=None)

    mu = float(predictions.sum())
    sigma = float(np.sqrt((monthly_std ** 2).sum()))

    return mu, sigma


def generate_demand_scenarios(
    forecast_results: dict,
    n_sampled: int = 20,
    extreme_prob: float = 0.15,
    normal_prob: float = 0.50,
    confidence_level: float = 0.95,
    seed: int = 42
) -> tuple[dict, np.ndarray]:
    """
    Discrétise la distribution de demande de chaque produit en un
    nombre fini de scénarios (méthode C : 3 scénarios structurels +
    n_sampled scénarios échantillonnés -- cf. méthodologie, Étape 3).

    Les scénarios sont CORRÉLÉS entre produits : un même indice de
    scénario s représente une réalisation conjointe de la demande pour
    tous les produits (ex : le scénario "pessimiste" place TOUS les
    produits à leur borne basse simultanément, modélisant un choc de
    marché commun ; les scénarios échantillonnés tirent chaque produit
    indépendamment, mais sous le même indice s pour tous).

    3 scénarios structurels (mêmes pour tous les produits) :
        - pessimiste : μᵢ - 1.5σᵢ   pour tout i    (proba extreme_prob)
        - normal     : μᵢ            pour tout i    (proba normal_prob)
        - optimiste  : μᵢ + 1.5σᵢ   pour tout i    (proba extreme_prob)

    + n_sampled scénarios mixtes, un tirage indépendant par produit
    dans N(μᵢ, σᵢ²) par scénario (proba totale restante répartie
    également entre eux). Ça couvre les combinaisons intermédiaires
    sans l'explosion combinatoire de la discrétisation 3^N (méthode B
    du cours, écartée pour cette raison).

    μᵢ, σᵢ sont estimés à partir de forecast_results[produit] (cf.
    _aggregate_forecast_to_mu_sigma) : demande totale sur l'horizon de
    prévision, pas mois par mois.

    Parameters
    ----------
    forecast_results : dict
        {produit: résultat de forecast_prophet / forecast_xgboost /
        select_best_model}, doit contenir 'predictions', 'lower', 'upper'.
    n_sampled : int
        Nombre de scénarios mixtes échantillonnés (en plus des 3
        scénarios structurels).
    extreme_prob : float
        Probabilité de CHACUN des scénarios "pessimiste" et
        "optimiste".
    normal_prob : float
        Probabilité du scénario "normal".
    confidence_level : float
        Niveau de confiance des intervalles [lower, upper] fournis.
    seed : int
        Graine aléatoire, pour des résultats reproductibles.

    Returns
    -------
    (scenarios, probabilities)
        scenarios : dict {produit: np.ndarray de shape (S,)},
            S = 3 + n_sampled.
        probabilities : np.ndarray de shape (S,), qui somme à 1.
    """

    if n_sampled < 1:
        raise ValueError("n_sampled doit être >= 1.")

    remaining_prob = 1 - normal_prob - 2 * extreme_prob
    if not (0 <= extreme_prob and 0 <= normal_prob and remaining_prob >= 0):
        raise ValueError(
            "extreme_prob et normal_prob doivent être >= 0 et vérifier "
            "normal_prob + 2*extreme_prob <= 1."
        )

    rng = np.random.default_rng(seed)

    scenarios = {}

    for product, result in forecast_results.items():

        mu, sigma = _aggregate_forecast_to_mu_sigma(result, confidence_level)

        # 3 scénarios structurels : pessimiste, normal, optimiste
        structural = np.array([
            max(mu - 1.5 * sigma, 0),
            mu,
            mu + 1.5 * sigma
        ])

        # n_sampled scénarios mixtes échantillonnés
        sampled = np.clip(
            rng.normal(loc=mu, scale=sigma, size=n_sampled),
            a_min=0, a_max=None
        )

        scenarios[product] = np.concatenate([structural, sampled])

    probabilities = np.array(
        [extreme_prob, normal_prob, extreme_prob]
        + [remaining_prob / n_sampled] * n_sampled
    )

    return scenarios, probabilities


def add_recourse_constraints(
    problem: pulp.LpProblem,
    order_vars: dict,
    surplus_vars: dict,
    shortage_vars: dict,
    scenarios: dict
) -> None:
    """
    Ajoute, pour chaque produit p et chaque scénario s, la contrainte
    de recours qui relie la commande à la demande réalisée :

        order[p] - demande[p, s] = surplus[p, s] - shortage[p, s]

    surplus[p, s] et shortage[p, s] sont des variables >= 0 : si la
    demande dépasse la commande, shortage absorbe l'écart (rupture) ;
    sinon, surplus l'absorbe (stock excédentaire).
    """

    for product, demand_scenarios in scenarios.items():
        for s, demand in enumerate(demand_scenarios):
            problem += (
                order_vars[product] - demand
                == surplus_vars[(product, s)] - shortage_vars[(product, s)],
                f"recours_{product}_{s}"
            )


def add_budget_constraint(
    problem: pulp.LpProblem,
    order_vars: dict,
    unit_cost_by_product: dict,
    max_budget: float
) -> None:
    """Le coût d'achat total (Σᵢ cᵢqᵢ) ne doit pas dépasser le budget
    disponible."""

    problem += (
        pulp.lpSum(
            order_vars[product] * unit_cost_by_product[product]
            for product in order_vars
        ) <= max_budget,
        "budget_max"
    )


def add_capacity_constraint(
    problem: pulp.LpProblem,
    order_vars: dict,
    max_capacity: float
) -> None:
    """La quantité totale commandée (tous produits confondus) ne doit
    pas dépasser la capacité de stockage disponible."""

    problem += (
        pulp.lpSum(order_vars.values()) <= max_capacity,
        "capacite_max"
    )

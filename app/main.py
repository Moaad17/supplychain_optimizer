from config import config
from data.loader import load_data
from data.validator import validate_data
from data.preprocessor import preprocess_data
from forecasting.prophet_model import forecast_prophet
from forecasting.ml_model import forecast_xgboost
from forecasting.selector import forecast_all_products_auto
from analysis.visualizer import plot_forecast
from analysis.comparator import compare_forecasts, compare_mae
from optimization.solver import evaluate_strategies
from optimization.baselines import compute_naive_orders


filepath = "examples/sample_beverages.csv"

# 1. Charger
df = load_data(filepath)

print("=== DONNÉES CHARGÉES ===")
print(df.head())


# 2. Valider
validation = validate_data(df)

print("\n=== VALIDATION ===")
print(validation)


# Si les données sont invalides, on arrête
if not validation["valid"]:
    print("\n❌ Données invalides :")
    for error in validation["errors"]:
        print("-", error)

else:
    # 3. Preprocess
    df_processed = preprocess_data(df)

    print("\n=== DONNÉES APRÈS PREPROCESSING ===")
    print(df_processed.head(20))

    # 4. Forecast Prophet sur un produit
    product_name = df_processed["product"].unique()[1]

    result = forecast_prophet(df_processed, product_name, horizon=3)

    print(f"\n=== FORECAST PROPHET — {product_name} ===")
    print("Dates :", result["dates"])
    print("Prédictions :", result["predictions"])
    print("Intervalle bas :", result["lower"])
    print("Intervalle haut :", result["upper"])
    print("MAE (backtest) :", result["mae"])

    if result["warnings"]:
        print("\n⚠️ Avertissements :")
        for warning in result["warnings"]:
            print("-", warning)

    # 5. Visualisation
    plot_forecast(df_processed, product_name, result)

    # 6. Forecast XGBoost sur le même produit, pour comparaison
    result_xgb = forecast_xgboost(df_processed, product_name, horizon=3)

    print(f"\n=== FORECAST XGBOOST — {product_name} ===")
    print("Dates :", result_xgb["dates"])
    print("Prédictions :", result_xgb["predictions"])
    print("Intervalle bas :", result_xgb["lower"])
    print("Intervalle haut :", result_xgb["upper"])
    print("MAE (backtest) :", result_xgb["mae"])

    if result_xgb["warnings"]:
        print("\n⚠️ Avertissements :")
        for warning in result_xgb["warnings"]:
            print("-", warning)

    # 7. Comparaison Prophet vs XGBoost
    print("\n=== COMPARAISON PROPHET vs XGBOOST ===")
    print(compare_mae(result, result_xgb))

    compare_forecasts(df_processed, product_name, result, result_xgb)

    # 8. Forecast + sélection automatique du modèle sur TOUS les produits
    forecast_auto = forecast_all_products_auto(df_processed, horizon=3)

    print("\n=== FORECAST AUTO (TOUS PRODUITS) ===")
    for product, forecast in forecast_auto["results"].items():
        mae = forecast["mae"]
        mae_str = f"{mae:.1f}" if mae is not None else "N/A"
        print(f"  {product} → {forecast['model']} (MAE={mae_str})")

    for product, reason in forecast_auto["failed"].items():
        print(f"  {product} → ÉCHEC ({reason})")

    # 9. Optimisation des stocks sous incertitude (Phase 3)
    #    HN (stochastique, PDE direct ou L-shaped selon N) vs WS (info
    #    parfaite) vs EV (déterministe) vs Naïf (moyenne historique)
    naive_orders = compute_naive_orders(df_processed, horizon=3)

    strategies = evaluate_strategies(
        forecast_auto["results"],
        unit_cost=config["unit_cost"],
        holding_cost=config["holding_cost"],
        shortage_cost=config["shortage_cost"],
        max_budget=config["max_budget"],
        max_capacity=config["max_capacity"],
        historical_orders=naive_orders,
        confidence_level=config["confidence_level"]
    )

    hn = strategies["hn"]

    print(f"\n=== OPTIMISATION DES STOCKS ({hn['status']} — {hn.get('method', '?')}) ===")

    if hn["status"] == "Optimal":
        for product, order in hn["orders"].items():
            print(
                f"  {product} → commander {order:.0f} unités "
                f"(niveau de service ≈ {hn['service_level'][product]:.0%}, "
                f"rupture attendue ≈ {hn['expected_shortage'][product]:.1f}, "
                f"surstock attendu ≈ {hn['expected_surplus'][product]:.1f})"
            )

        print("\n--- Comparaison des stratégies (triées par coût) ---")
        for row in strategies["comparison"]:
            marker = " ← référence" if row["strategy"] == "HN" else ""
            print(
                f"  {row['strategy']:<5s} {row['cost']:>12,.2f} MAD  "
                f"({row['vs_hn_pct']:+.1f}% vs HN){marker}"
            )

        print(f"\nEVPI (coût de l'incertitude)           : {strategies['evpi']:,.2f} MAD")
        print(f"VSS  (valeur de l'optim. stochastique)  : {strategies['vss']:,.2f} MAD")
        if strategies["naive_cost"] is not None:
            savings_vs_naive = strategies["naive_cost"] - hn["total_cost"]
            pct_vs_naive = savings_vs_naive / strategies["naive_cost"] * 100
            print(
                f"Économie vs stratégie naïve : {savings_vs_naive:,.2f} MAD "
                f"({pct_vs_naive:.1f}%)"
            )
        if not strategies["checks_ok"]:
            print("⚠️ WS <= HN <= EV n'est pas respecté — vérifier le modèle.")
    else:
        print("Pas de solution optimale trouvée — vérifier le budget/la capacité.")
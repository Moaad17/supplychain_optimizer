from data.loader import load_data
from data.validator import validate_data
from data.preprocessor import preprocess_data
from forecasting.prophet_model import forecast_prophet
from forecasting.ml_model import forecast_xgboost
from analysis.visualizer import plot_forecast
from analysis.comparator import compare_forecasts, compare_mae


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
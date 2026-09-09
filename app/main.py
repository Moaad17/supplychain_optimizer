from data.loader import load_data
from data.validator import validate_data
from data.preprocessor import preprocess_data


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
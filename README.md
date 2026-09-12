# Supply Chain Optimizer

Outil de prévision de la demande et d'optimisation des stocks pour la
supply chain. Le projet se construit en 3 phases : **chargement/validation
des données**, **forecasting** (Prophet + XGBoost avec sélection
automatique du meilleur modèle), puis **optimisation** des stocks sous
contraintes.

Ce README suit l'avancement jour par jour (Jour 1 à Jour 6 pour l'instant).

## Fonctionnalités implémentées

### Phase 1 — Données ([app/data/](app/data/))

- **`loader.py`** : `load_data(filepath)` charge un fichier CSV ou Excel
  et convertit la colonne `date` en datetime.
- **`validator.py`** : `validate_data(df)` vérifie les colonnes
  obligatoires (`date`, `product`, `quantity`), les types, les valeurs
  négatives, les valeurs manquantes, le volume de données (mois
  d'historique) et les doublons date+produit. Retourne
  `{valid, errors, warnings, stats}`.
- **`preprocessor.py`** : `preprocess_data(df)` fusionne les doublons,
  comble les mois manquants avec 0, trie chronologiquement et ajoute
  `year`, `month`, `quarter`.

### Phase 2 — Forecasting ([app/forecasting/](app/forecasting/), [app/analysis/](app/analysis/))

- **`prophet_model.py`** : `forecast_prophet(df, product_name, horizon)`
  — prévision par produit avec [Prophet](https://facebook.github.io/prophet/),
  backtest sur les derniers mois connus pour mesurer la MAE, intervalle
  de confiance natif, valeurs négatives clippées à 0.
- **`ml_model.py`** : `forecast_xgboost(df, product_name, horizon)` —
  même interface que Prophet, mais avec XGBoost et des features
  temporelles manuelles (lags, moyennes glissantes, tendance,
  saisonnalité). Split train/test strictement chronologique (pas de
  data leakage). Intervalle de confiance par régression quantile
  (`reg:quantileerror`).
- **`selector.py`** :
  - `forecast_all_products(df, model_type, horizon)` — applique un
    modèle à tous les produits, isole les échecs par produit sans
    interrompre les autres.
  - `select_best_model(df, product_name, horizon)` — compare Prophet et
    XGBoost sur la MAE de backtest et retourne le gagnant.
  - `forecast_all_products_auto(df, horizon)` — sélection automatique du
    meilleur modèle pour chaque produit du dataset.
- **`visualizer.py`** : `plot_forecast(...)` — graphique Plotly
  (historique, prédictions, intervalle de confiance).
- **`comparator.py`** : `compare_forecasts(...)` et `compare_mae(...)` —
  comparaison visuelle et chiffrée Prophet vs XGBoost.

### Tests ([tests/](tests/))

- `test_loader.py` : chargement (CSV valide, fichier absent, format non
  supporté) et validation (DataFrame correct, colonne manquante, trop de
  valeurs manquantes).
- `test_forecasting.py` : bout en bout sur les 3 CSV d'exemple
  (multi-produits, gestion d'échec partiel) + invariants (prédictions
  positives, intervalle cohérent, sélecteur exhaustif, MAE raisonnable).

Lancer tous les tests :

```bash
pytest -v
```

### À venir

- Phase 3 — Optimisation des stocks ([app/optimization/](app/optimization/) :
  contraintes, modèle, solveur — pas encore implémenté).
- API ([app/API/](app/API/)) et frontend ([frontend/](frontend/)) — squelettes
  pas encore implémentés.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
cd app
python main.py
```

Charge `examples/sample_beverages.csv`, valide et prétraite les
données, puis lance et compare Prophet et XGBoost sur un produit avec
visualisation.

## Données d'exemple ([examples/](examples/))

| Fichier | Colonnes | Usage |
|---|---|---|
| `sample_beverages.csv` | date, product, quantity, unit_price | Démo principale (36 mois, 5 produits) |
| `sample_retail.csv` | date, product, quantity, unit_price | Test multi-produits (8 produits) |
| `sample_port_traffic.csv` | date, port, tonnage_milliers, nb_navires | Test du rejet par le validateur (colonnes incompatibles) |

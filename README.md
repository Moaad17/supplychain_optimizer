# Supply Chain Optimizer

Un outil qui répond à une question simple mais coûteuse à mal gérer :
**combien commander de chaque produit, pour le mois prochain, quand on
ne connaît pas la demande exacte à l'avance ?**

Commander trop → argent immobilisé en stock, coût de stockage.
Commander trop peu → ruptures, ventes perdues, clients mécontents.
Ce projet automatise cette décision en combinant **prévision de la
demande** (machine learning) et **optimisation sous incertitude**
(recherche opérationnelle), plutôt que de se fier à une intuition ou à
une simple moyenne historique.

## Le principe en une image

```
Historique de ventes (CSV)
         │
         ▼
┌─────────────────────┐
│   1. FORECASTING     │   Pour chaque produit : Prophet ET XGBoost
│  (Prophet / XGBoost) │   sont entraînés, le meilleur des deux est
│                      │   choisi automatiquement (backtest MAE).
└─────────┬────────────┘   Résultat : une prévision + un intervalle
          │                 d'incertitude (pas juste un chiffre).
          ▼
┌─────────────────────┐
│  2. OPTIMISATION     │   L'incertitude de la prévision devient des
│   (recherche         │   scénarios de demande (optimiste/pessimiste/
│   opérationnelle)    │   intermédiaires). Le programme calcule la
│                      │   commande qui minimise le coût ATTENDU
└─────────┬────────────┘   (achat + rupture + surstock), sous
          │                 contrainte de budget et de capacité.
          ▼
   Quantité à commander
   par produit, + le coût
   que ça permet d'économiser
   par rapport à des méthodes
   plus simples.
```

## Pourquoi c'est plus qu'une prévision + une règle de calcul

La partie qui distingue ce projet d'un simple "je prévois puis
j'arrondis" : l'optimisation ne travaille jamais avec un seul chiffre
de demande. Elle raisonne sur une **distribution** de scénarios
possibles (tirés à partir de l'intervalle de confiance de la
prévision), et choisit la commande qui minimise le coût *en moyenne*
sur tous ces scénarios — pas seulement pour le scénario le plus
probable. C'est ce qu'on appelle la **programmation stochastique avec
recours**.

Le projet va plus loin en répondant à une question que beaucoup
d'outils n'abordent pas : **est-ce que tout cet effort en vaut la
peine ?** Il calcule et compare 4 stratégies :

| Stratégie | Ce qu'elle représente |
|---|---|
| **Naïf** | Commander la moyenne historique — le réflexe "zéro effort" |
| **EV** | Commander en fonction de la demande moyenne prévue, sans tenir compte de l'incertitude |
| **HN** *(la stratégie retenue)* | Optimiser en tenant compte explicitement de l'incertitude |
| **WS** | Le coût qu'on aurait si on connaissait la demande à l'avance — une borne théorique, inatteignable |

De ça découlent deux indicateurs concrets :
- **EVPI** (Expected Value of Perfect Information) — combien coûte le
  fait de ne pas connaître la demande à l'avance. Investir dans une
  meilleure prévision n'a de sens que si l'EVPI est élevé.
- **VSS** (Value of the Stochastic Solution) — combien on gagne en
  utilisant l'optimisation stochastique plutôt qu'une règle basée sur
  la simple moyenne.

## Le pipeline en détail

### 1. Données ([app/data/](app/data/))

Charge un CSV (`date`, `product`, `quantity`, `unit_price` optionnel),
le valide (colonnes obligatoires, types, valeurs manquantes, volume
d'historique suffisant) et le nettoie (doublons fusionnés, mois
manquants comblés, variables temporelles ajoutées).

### 2. Prévision ([app/forecasting/](app/forecasting/))

Pour chaque produit, deux modèles sont entraînés et comparés par
backtest (les derniers mois sont cachés puis prédits, pour mesurer
l'erreur sur des données jamais vues) :

- **Prophet** : modèle spécialisé séries temporelles, capte bien la
  saisonnalité et la tendance.
- **XGBoost** : modèle généraliste avec des features construites à la
  main (mois précédent, moyennes glissantes, tendance, prix du produit
  s'il est disponible). Meilleur sur des patterns plus irréguliers.

Le modèle avec la MAE (erreur moyenne) la plus basse est retenu
automatiquement, produit par produit — pas besoin de choisir à la
main.

### 3. Optimisation ([app/optimization/](app/optimization/))

L'intervalle de confiance de chaque prévision est converti en
scénarios de demande (un mélange de cas extrêmes et de tirages
intermédiaires). Un programme linéaire (résolu avec
[PuLP](https://github.com/coin-or/pulp)) détermine ensuite la commande
qui minimise :

```
coût d'achat (certain) + coût de rupture/surstock attendu (moyenne sur tous les scénarios)
```

sous contrainte de budget et de capacité de stockage. Pour un grand
nombre de produits, une méthode de décomposition (L-shaped / Benders)
prend le relais automatiquement pour rester rapide.

### 4. Interface ([frontend/app.py](frontend/app.py))

Une application [Streamlit](https://streamlit.io/) qui exécute tout le
pipeline avec des paramètres ajustables (coûts, budget, capacité,
horizon), et affiche le plan de commande, la comparaison des 4
stratégies et l'interprétation de l'EVPI/VSS.

## Démarrage rapide

```bash
pip install -r requirements.txt

# Interface interactive
streamlit run frontend/app.py

# Ou le script de démonstration en ligne de commande
cd app && python main.py
```

## Données d'exemple ([examples/](examples/))

| Fichier | Contenu | Usage |
|---|---|---|
| `sample_beverages.csv` | 5 boissons, 36 mois | Démo principale |
| `sample_retail.csv` | 8 produits retail, 24 mois | Test multi-produits, prix très hétérogènes |
| `sample_port_traffic.csv` | Trafic portuaire (colonnes différentes) | Vérifie que le validateur rejette proprement un format incompatible |

## Tests

```bash
pytest -v
```

44 tests couvrent le chargement/validation, le forecasting (précision,
cohérence des intervalles, gestion des échecs), et l'optimisation
(contraintes respectées, cohérence WS ≤ HN ≤ EV, non-régression du
solveur L-shaped contre le calcul direct).

## Ce qui n'est pas encore fait

- `app/API/` : pas encore d'API HTTP, seulement le script/l'interface Streamlit.
- `app/analysis/metrics.py` : fichier réservé, pas encore implémenté.
- Containerisation (`Dockerfile`, `docker-compose.yaml`) : pas encore écrite.

"""
Interface Streamlit : charge des données de ventes, lance le
forecasting (Prophet/XGBoost, sélection automatique) puis l'optimisation
des stocks sous incertitude (WS/HN/EV/EVPI/VSS), et affiche les
résultats avec leur interprétation.

Lancer avec : streamlit run frontend/app.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# app/ utilise des imports "à plat" (from data.loader import ...) en
# supposant qu'il est à la racine du sys.path -- on reproduit ça ici.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from config import config as default_config  # noqa: E402
from data.loader import load_data  # noqa: E402
from data.validator import validate_data  # noqa: E402
from data.preprocessor import preprocess_data  # noqa: E402
from forecasting.selector import forecast_all_products_auto  # noqa: E402
from optimization.solver import evaluate_strategies  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


st.set_page_config(page_title="Supply Chain Optimizer", layout="wide")
st.title("📦 Supply Chain Optimizer")
st.caption(
    "Prévision de la demande (Prophet / XGBoost) puis optimisation des "
    "stocks sous incertitude (programmation stochastique avec recours)."
)


# ==========================================================
# Sidebar : données et paramètres
# ==========================================================

st.sidebar.header("Données")

csv_files = sorted(p.name for p in EXAMPLES_DIR.glob("*.csv"))
selected_file = st.sidebar.selectbox("Fichier d'exemple", csv_files)
uploaded_file = st.sidebar.file_uploader("...ou importer ton propre CSV", type="csv")

st.sidebar.header("Paramètres économiques")
unit_cost = st.sidebar.number_input(
    "Coût d'achat unitaire", value=float(default_config.get("unit_cost", 10)), min_value=0.0
)
holding_cost = st.sidebar.number_input(
    "Coût de stockage (surplus/unité)", value=float(default_config.get("holding_cost", 5)), min_value=0.0
)
shortage_cost = st.sidebar.number_input(
    "Coût de rupture (unité manquante)", value=float(default_config.get("shortage_cost", 20)), min_value=0.0
)
max_budget = st.sidebar.number_input(
    "Budget maximum", value=float(default_config.get("max_budget", 500000)), min_value=0.0
)
max_capacity = st.sidebar.number_input(
    "Capacité de stockage maximum", value=float(default_config.get("max_capacity", 14000)), min_value=0.0
)

st.sidebar.header("Prévision")
horizon = st.sidebar.slider("Horizon (mois)", 1, 6, int(default_config.get("forecast_horizon", 3)))
confidence_level = st.sidebar.slider(
    "Niveau de confiance", 0.80, 0.99, float(default_config.get("confidence_level", 0.95))
)

run = st.sidebar.button("🚀 Lancer l'analyse", type="primary", width="stretch")


# ==========================================================
# Étapes coûteuses (chargement + forecasting), mises en cache
# ==========================================================

@st.cache_data(show_spinner="Chargement et validation des données...")
def _load_and_prepare(file_bytes: bytes | None, example_filename: str):
    if file_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        df = load_data(tmp_path)
    else:
        df = load_data(EXAMPLES_DIR / example_filename)

    validation = validate_data(df)

    if not validation["valid"]:
        return None, validation

    return preprocess_data(df), validation


@st.cache_data(show_spinner="Prévision de la demande (Prophet / XGBoost)...")
def _forecast(df: pd.DataFrame, horizon: int):
    return forecast_all_products_auto(df, horizon=horizon)


@st.cache_data(show_spinner="Optimisation des stocks sous incertitude...")
def _optimize(
    forecast_results: dict,
    unit_cost: float, holding_cost: float, shortage_cost: float,
    max_budget: float, max_capacity: float, confidence_level: float
):
    return evaluate_strategies(
        forecast_results,
        unit_cost=unit_cost, holding_cost=holding_cost, shortage_cost=shortage_cost,
        max_budget=max_budget, max_capacity=max_capacity,
        confidence_level=confidence_level
    )


# ==========================================================
# Corps de la page
# ==========================================================

if not run:
    st.info("Configure tes paramètres dans la barre latérale, puis clique sur **Lancer l'analyse**.")
    st.stop()

file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
df_processed, validation = _load_and_prepare(file_bytes, selected_file)

with st.expander("Validation des données", expanded=not validation["valid"]):
    if validation["valid"]:
        st.success("Données valides.")
    else:
        st.error("Données invalides :")
        for error in validation["errors"]:
            st.write(f"- {error}")
    for warning in validation["warnings"]:
        st.warning(warning)

if df_processed is None:
    st.stop()

st.dataframe(df_processed.head(10), width="stretch")

# ---- Forecasting ----

forecast_auto = _forecast(df_processed, horizon)

st.subheader("📈 Prévisions par produit")

if forecast_auto["failed"]:
    st.warning(
        f"{len(forecast_auto['failed'])} produit(s) n'ont pas pu être prévus : "
        + ", ".join(f"{p} ({reason})" for p, reason in forecast_auto["failed"].items())
    )

forecast_table = pd.DataFrame([
    {
        "Produit": product,
        "Modèle retenu": result["model"],
        "MAE (backtest)": round(result["mae"], 1) if result["mae"] is not None else None,
        "Prévision horizon (total)": round(sum(result["predictions"]), 0),
    }
    for product, result in forecast_auto["results"].items()
])
st.dataframe(forecast_table, width="stretch", hide_index=True)

if not forecast_auto["results"]:
    st.error("Aucun produit n'a pu être prévu, impossible d'optimiser.")
    st.stop()

# ---- Optimisation ----

strategies = _optimize(
    forecast_auto["results"],
    unit_cost=unit_cost, holding_cost=holding_cost, shortage_cost=shortage_cost,
    max_budget=max_budget, max_capacity=max_capacity, confidence_level=confidence_level
)

hn = strategies["hn"]

st.subheader("📦 Solution optimale (commande à passer)")

if hn["status"] != "Optimal":
    st.error(f"Pas de solution optimale trouvée ({hn['status']}) — vérifie le budget/la capacité.")
    st.stop()

orders_table = pd.DataFrame([
    {
        "Produit": product,
        "Commande optimale (qᵢ*)": round(order),
        "Niveau de service attendu": f"{hn['service_level'][product]:.0%}",
        "Rupture attendue": round(hn["expected_shortage"][product], 1),
        "Surstock attendu": round(hn["expected_surplus"][product], 1),
    }
    for product, order in hn["orders"].items()
])
st.dataframe(orders_table, width="stretch", hide_index=True)

total_units = sum(hn["orders"].values())
total_units_cost = total_units * unit_cost
c1, c2, c3 = st.columns(3)
c1.metric("Unités totales commandées", f"{total_units:,.0f}")
c2.metric("Coût d'achat", f"{total_units_cost:,.0f}", help="Σ cᵢ × qᵢ* -- ne compte pas le recours (rupture/surstock).")
c3.metric("Capacité utilisée", f"{total_units / max_capacity:.0%}" if max_capacity else "—")

st.markdown(
    f"**Coût total attendu de cette stratégie (HN) : {hn['total_cost']:,.2f}** "
    "(achat certain + espérance des coûts de rupture/surstock sur tous les scénarios)."
)

# ---- WS / HN / EV / EVPI / VSS ----

st.subheader("🔍 Comparaison des stratégies")

ws_cost = strategies["ws_cost"]
ev_cost = strategies["ev_cost"]
hn_cost = hn["total_cost"]
evpi = strategies["evpi"]
vss = strategies["vss"]

col_ws, col_hn, col_ev = st.columns(3)
col_ws.metric("WS — information parfaite", f"{ws_cost:,.2f}")
col_hn.metric("HN — stochastique (recommandé)", f"{hn_cost:,.2f}")
col_ev.metric("EV — déterministe (moyenne)", f"{ev_cost:,.2f}")

st.caption(
    "**WS** (Wait and See) : coût si on connaissait la demande à l'avance -- une borne "
    "théorique inatteignable, elle sert juste de référence. "
    "**HN** (Here and Now) : coût de la stratégie recommandée par ce modèle, qui tient "
    "compte de l'incertitude. "
    "**EV** (Expected Value) : coût si on avait commandé une quantité fixe basée sur la "
    "demande moyenne (stratégie naïve), évalué sur les mêmes scénarios réels."
)

col_evpi, col_vss = st.columns(2)

with col_evpi:
    st.metric("EVPI = HN − WS", f"{evpi:,.2f}")
    if evpi < 1e-6 * max(hn_cost, 1):
        st.info(
            "EVPI ≈ 0 : connaître la demande à l'avance n'apporterait presque rien. "
            "Ça veut dire que la décision est dominée par une contrainte dure "
            "(budget ou capacité), pas par l'incertitude de la demande -- investir "
            "dans une meilleure prévision n'aiderait pas ici."
        )
    else:
        pct = evpi / hn_cost * 100 if hn_cost else 0
        st.info(
            f"EVPI = **{evpi:,.0f}** (≈{pct:.1f}% du coût HN) : c'est ce que coûte "
            "l'incertitude sur la demande. Un EVPI élevé indique qu'investir dans de "
            "meilleures prévisions (plus de données, un meilleur modèle) aurait de la valeur."
        )

with col_vss:
    st.metric("VSS = EV − HN", f"{vss:,.2f}")
    if vss < 1e-6 * max(hn_cost, 1):
        st.info(
            "VSS ≈ 0 : la solution stochastique n'apporte quasiment rien de plus "
            "qu'une simple règle \"commander la demande moyenne\". Le modèle "
            "stochastique n'est pas nécessaire ici."
        )
    else:
        pct = vss / hn_cost * 100 if hn_cost else 0
        st.info(
            f"VSS = **{vss:,.0f}** (≈{pct:.1f}% du coût HN) : c'est ce que fait gagner "
            "l'optimisation stochastique par rapport à une politique naïve basée sur la "
            "moyenne. Plus VSS est grand, plus ce modèle apporte de valeur par rapport à "
            "une règle simple."
        )

if not strategies["checks_ok"]:
    st.error(
        "⚠️ L'inégalité attendue WS ≤ HN ≤ EV n'est pas respectée "
        f"(WS={ws_cost:.2f}, HN={hn_cost:.2f}, EV={ev_cost:.2f}) -- "
        "signe possible d'un problème numérique, à vérifier."
    )

st.bar_chart(
    pd.DataFrame({"Coût attendu": [ws_cost, hn_cost, ev_cost]}, index=["WS", "HN", "EV"])
)

import sys
sys.setrecursionlimit(20000)

import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from Black_scholes_class import BlackScholesOptionPricing
from Market_class import Market
from Option_class import Option
from Model_class import Model
from Derivatives_class import Derivatives

# -----------------------------------------------------------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🌳 Trinomial Option Pricer", layout="wide")
st.title("🌳 Pricer d'Option avec Arbre Trinomial")

# -----------------------------------------------------------------------------
# PARAMÈTRES DU MODÈLE (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres du modèle")
S0 = st.sidebar.number_input("Prix spot (S₀)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike (K)", value=100.0, step=1.0)
r = st.sidebar.number_input("Taux sans risque (r)", value=0.03, step=0.01)
sigma = st.sidebar.number_input("Volatilité (σ)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
dividend_price = st.sidebar.number_input("Dividende", min_value=0.0, value=0.0, step=0.1)
gap_str = st.sidebar.text_input("Écart de prix souhaité avec B&S", value="")
gap = float(gap_str) if gap_str else None
steps_default = st.session_state.get("actual_steps", 50)
steps = int(st.sidebar.number_input(
    "Nombre de steps",
    min_value=2,
    value=steps_default,
    step=1
))

option_type = st.sidebar.selectbox("Type de payoff", ["call", "put"])
exercise_type = st.sidebar.selectbox("Type d'option", ["EU", "US"])
maturity_input = st.sidebar.date_input("Date de maturité", datetime(2026, 9, 1))
pricing_date_input = st.sidebar.date_input("Date de valorisation", datetime(2025, 9, 1))
date_div_input = st.sidebar.date_input("Date de dividende", datetime(2026, 4, 10))

# -----------------------------------------------------------------------------
# CONVERSION DES DATES
# -----------------------------------------------------------------------------
pricing_date = datetime.combine(pricing_date_input, datetime.min.time())
maturity = datetime.combine(maturity_input, datetime.min.time())
date_div = datetime.combine(date_div_input, datetime.min.time())

# -----------------------------------------------------------------------------
# CALCUL DES PRIX AVEC CACHE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def calculate_prices(S0, K, r, sigma, dividend, option_type, exercise_type, pricing_date, maturity, steps, gap, date_div):
    """Calcule les prix via Black-Scholes et l'arbre trinomial."""
    try:
        # Création du marché et de l’option
        market = Market(rate=r, vol=sigma, spot_price=S0, dividend_price=dividend, dividend_date=date_div)
        option = Option(payoff_type=option_type, option_type=exercise_type, K=K, maturity=maturity)

        # Prix Black-Scholes
        bs_price = BlackScholesOptionPricing(option, pricing_date, market).calculate_option_price()

        # Modèle trinomial
        model = Model(market, option, pricing_date, steps, gap)

        # Calcul des prix récursif et backward
        results = model.pricing(recursive_pricing=True, backward_pricing=True)

        result_recursif = results["recursive_price"]
        recursive_pricing_time = results["recursive_pricing_exec_time"]
        result_backward = results["backward_price"]
        backward_pricing_time = results["backward_pricing_exec_time"]
        tree_creation_time = results["tree_creation_time"]

        return bs_price, result_recursif, recursive_pricing_time, result_backward, backward_pricing_time, tree_creation_time


    except Exception as e:
        st.error(f"Erreur de calcul : {e}")
        return None, None, None, None, None, None

# -----------------------------------------------------------------------------
# CALCUL PRINCIPAL
# -----------------------------------------------------------------------------
with st.spinner("🧮 Calcul en cours..."):
    bs_price, result_recursif, recursive_pricing_time, result_backward, backward_pricing_time, tree_creation_time = calculate_prices(
        S0, K, r, sigma, dividend_price, option_type, exercise_type, pricing_date, maturity, steps, gap, date_div
    )

# -----------------------------------------------------------------------------
# ONGLET D’AFFICHAGE
# -----------------------------------------------------------------------------
tabs = st.tabs(["🌳 Arbre", "💰 Prix", "📊 Greeks", "📈 Convergence", "⏱ Temps d'exécution"])

# -----------------------------------------------------------------------------
# ONGLET 1 : ARBRE TRINOMIAL
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🌳 Visualisation de l'Arbre Trinomial")

    if bs_price is not None:
        # 1️⃣ Clé unique pour le cache
        key = f"model_{steps}_{S0}_{K}_{r}_{sigma}_{gap}_{date_div}_{dividend_price}"

        # 2️⃣ Construction ou récupération depuis le cache
        if key not in st.session_state:
            market = Market(
                rate=r,
                vol=sigma,
                spot_price=S0,
                dividend_price=dividend_price,
                dividend_date=date_div
            )
            option = Option(
                payoff_type=option_type,
                option_type=exercise_type,
                K=K,
                maturity=maturity
            )
            model = Model(market, option, pricing_date, steps, gap)

            # 🔒 Calcul des prix de l’arbre (option_price rempli)
            model.pricing(backward_pricing=True)

            # 🔁 Mettre à jour steps dans le session_state si le modèle l'a ajusté
            if hasattr(model, "tree") and hasattr(model.tree, "steps"):
                actual_steps = model.tree.steps
                if actual_steps != st.session_state.get("actual_steps", steps):
                    st.session_state["actual_steps"] = actual_steps
                    st.rerun()


            # 🔒 On sauvegarde tout dans le cache Streamlit
            st.session_state[key] = {
                "model": model,
                "market": market,
                "option": option
            }
        else:
            model = st.session_state[key]["model"]
            market = st.session_state[key]["market"]
            option = st.session_state[key]["option"]

        # 3️⃣ Interface des options d’affichage
        if steps > 500:
            st.warning("Arbre trop grand (> 500 steps). Aucune donnée n'est affichée.")
        else:
            # Sélection d'une seule information à afficher
            show_choice = st.radio(
                "Afficher dans l'arbre",
                ("Prix du sous-jacent", "Prix de l’option", "Probabilité du nœud"),
                index=0
            )

            show_underlying = show_option = show_prob = False
            if show_choice == "Prix du sous-jacent":
                show_underlying = True
            elif show_choice == "Prix de l’option":
                show_option = True
            elif show_choice == "Probabilité du nœud":
                show_prob = True

            # 4️⃣ Affichage du graphe
            if hasattr(model.tree, "plot_tree"):
                fig = model.tree.plot_tree(
                    print_underlying_asset_price=show_underlying,
                    print_option_price=show_option,
                    print_prob_node=show_prob
                )
                st.plotly_chart(fig, width=True)
            else:
                st.info("🧩 Implémentez la méthode `plot_tree()` dans TrinomialTree pour afficher l’arbre interactif.")

# -----------------------------------------------------------------------------
# ONGLET 2 : PRIX DES OPTIONS
# -----------------------------------------------------------------------------

with tabs[1]:
    st.subheader("💰 Comparaison des prix")

    if bs_price is not None:
        # Disposition en colonnes côte à côte
        col1, col2, col3 = st.columns(3)

        col1.metric("Prix Black-Scholes", f"{bs_price:.4f}")
        col2.metric("Prix Récursif", f"{result_recursif:.4f}")
        col3.metric("Prix Backward", f"{result_backward:.4f}")

        # Création du DataFrame uniquement pour les deux autres méthodes
        df_results = pd.DataFrame({
            "Méthode": ["Récursif", "Backward"],
            "Écart avec BS": [
                abs(result_recursif - bs_price),
                abs(result_backward - bs_price)
            ],
            "Temps de calcul (s)": [
                recursive_pricing_time,
                backward_pricing_time
            ]
        }).reset_index(drop=True)

        st.dataframe(df_results, use_container_width=True)
        
    else:
        st.warning("Les prix n'ont pas pu être calculés.")


# -----------------------------------------------------------------------------
# ONGLET 3 : GREEKS
# -----------------------------------------------------------------------------
with tabs[2]:
    steps = 1000

    if bs_price is None:
        st.warning("⚠️ Les prix doivent être calculés avant de pouvoir évaluer les Greeks.")
    else:
        # --- Marché et option ---
        market = Market(
            rate=r,
            vol=sigma,
            spot_price=S0,
            dividend_price=dividend_price,
            dividend_date=date_div
        )
        option = Option(
            payoff_type=option_type,
            option_type=exercise_type,
            K=K,
            maturity=maturity
        )

        # --- Fonction de pricing avec trace ---
        def pricing_func(market_in, pricing_date_in, nb_steps, option_in):
            model = Model(market_in, option_in, pricing_date_in, steps=nb_steps)
            result = model.pricing(recursive_pricing=False, backward_pricing=True)
            try:
                price = float(result["backward_price"])
            except Exception as e:
                st.error(f"Erreur conversion en float: {e}")
                st.write("backward_price contenu:", result["backward_price"])
                price = None
            return price

        # --- Calcul du Delta avec trace ---
        derivative = Derivatives(
            derivative_function=pricing_func,
            epsilon_price=0.0001,
            epsilon_vol=0.0001
        )

        try:
            with st.spinner("🧮 Calcul des Greeks via modèle trinomial..."):
                tri_delta = derivative.delta(market, steps, pricing_date, option)
                tri_gamma = derivative.gamma(market, steps, pricing_date, option)
                tri_vega = derivative.vega(market, steps, pricing_date, option)
                tri_voma = derivative.voma(market, steps, pricing_date, option)
                tri_vanna = derivative.vanna(market, steps, pricing_date, option)

            # --- Analyse holistique des Greeks ---
            result_analysis = derivative.holistic_greek_analysis(market, steps, pricing_date, option)
            st.markdown("### 🧠 Analyse holistique des Greeks")
            for line in result_analysis['holistic_analysis']:
                st.write(f"- {line}")

            # --- Tableau comparatif ---
            bs_model = BlackScholesOptionPricing(option, pricing_date, market)
            bs_delta = bs_model.calculate_delta()
            bs_gamma = bs_model.calculate_gamma()
            bs_vega  = bs_model.calculate_vega()
            bs_voma  = bs_model.calculate_vomma()
            bs_vanna = bs_model.calculate_vanna()

            data = {
                "Greek": ["Delta", "Gamma", "Vega", "Voma", "Vanna"],
                "Black-Scholes": [bs_delta, bs_gamma, bs_vega, bs_voma, bs_vanna],
                "Trinomial": [tri_delta, tri_gamma, tri_vega, tri_voma, tri_vanna],
                "Écart (%)": [
                    100 * abs((tri_delta - bs_delta) / bs_delta) if bs_delta != 0 else 0,
                    100 * abs((tri_gamma - bs_gamma) / bs_gamma) if bs_gamma != 0 else 0,
                    100 * abs((tri_vega - bs_vega) / bs_vega) if bs_vega != 0 else 0,
                    100 * abs((tri_voma - bs_voma) / bs_voma) if bs_voma != 0 else 0,
                    100 * abs((tri_vanna - bs_vanna) / bs_vanna) if bs_vanna != 0 else 0
                ]
            }

            df_greeks = pd.DataFrame(data)
            df_greeks["Black-Scholes"] = df_greeks["Black-Scholes"].map(lambda x: f"{x:.6f}")
            df_greeks["Trinomial"] = df_greeks["Trinomial"].map(lambda x: f"{x:.6f}")
            df_greeks["Écart (%)"] = df_greeks["Écart (%)"].map(lambda x: f"{x:.3f}%")

            st.markdown("### 🔍 Tableau comparatif des Greeks")
            st.dataframe(df_greeks, width=800)

            # --- Graphique optionnel ---
            st.markdown("### 📊 Visualisation graphique")
            fig, ax = plt.subplots()
            labels = df_greeks["Greek"]
            x = np.arange(len(labels))
            width = 0.35
            bs_vals  = [float(v) for v in df_greeks["Black-Scholes"]]
            tri_vals = [float(v) for v in df_greeks["Trinomial"]]

            ax.bar(x - width/2, bs_vals, width, label="Black-Scholes")
            ax.bar(x + width/2, tri_vals, width, label="Trinomial")
            ax.set_ylabel("Valeur")
            ax.set_title("Comparaison des Greeks")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur pendant le calcul des Greeks: {e}")



# -----------------------------------------------------------------------------
# ONGLET 4 : CONVERGENCE
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📈 Convergence selon le nombre d'étapes")

    try:
        from convergence_data import dict_convergence

        if dict_convergence:
            steps = list(dict_convergence.keys())
            diffs = list(dict_convergence.values())

            fig, ax = plt.subplots()
            ax.plot(steps, diffs, linewidth=1.5, label="Écart réel")

            # Ajustement de l'échelle pour que 1/steps soit comparable
            scale = diffs[0] * steps[0]  # met les deux sur un ordre de grandeur similaire
            ax.plot(steps, [scale / s for s in steps], 'k--', label="∝ f(x) = 1/x")

            ax.set_xlabel("Nombre d'étapes")
            ax.set_ylabel("Écart absolu |Trinomial - Black-Scholes|")
            ax.set_title("Convergence du modèle trinomial")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            st.success(f"Données de convergence chargées ({len(steps)} points).")

        else:
            st.warning("Le dictionnaire `dict_convergence` est vide.")

    except ModuleNotFoundError:
        st.error("❌ Le fichier `convergence_data.py` est introuvable.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

# -----------------------------------------------------------------------------
# ONGLET 5 : TEMPS D'EXÉCUTION
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("⏱ Temps de calcul selon le nombre d'étapes")

    try:
        from convergence_data import dict_recursive_pricing,dict_backward_pricing, dict_tree

        if dict_recursive_pricing and dict_backward_pricing and dict_tree:
            steps = list(dict_recursive_pricing.keys())
            recursive_pricing_times = list(dict_recursive_pricing.values())
            backward_pricing_times = list(dict_backward_pricing.values())
            tree_times = list(dict_tree.values())

            fig, ax = plt.subplots()
            ax.plot(steps, recursive_pricing_times, label="Pricing récursif", linewidth=1.5)
            ax.plot(steps, backward_pricing_times, label="Pricing backward", linewidth=1.5)
            ax.plot(steps, tree_times, label="Construction arbre", linewidth=1.5)
            ax.set_xlabel("Nombre d'étapes")
            ax.set_ylabel("Temps (secondes)")
            ax.set_title("Temps d'exécution selon le nombre d'étapes (en Python)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            st.success(f"Données de temps chargées ({len(steps)} points).")

            # Tableau résumé des dernières valeurs
            st.write("### Dernières valeurs")
            st.dataframe({
                "Étapes": steps[-10:],
                "Temps Pricing Récursif (s)": [f"{t:.6f}" for t in recursive_pricing_times[-10:]],
                "Temps Pricing Backward (s)": [f"{t:.6f}" for t in backward_pricing_times[-10:]],
                "Temps Arbre (s)": [f"{t:.6f}" for t in tree_times[-10:]],
            })

        else:
            st.warning("Les dictionnaires `dict_pricing` et `dict_tree` sont vides.")

    except ModuleNotFoundError:
        st.error("❌ Le fichier `convergence_data.py` est introuvable.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des temps : {e}")


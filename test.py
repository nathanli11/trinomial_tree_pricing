from time import time
from Black_scholes_class import BlackScholesOptionPricing
from Market_class import Market
from Option_class import Option
from Model_class import Model
from Derivatives_class import Derivatives

from math import exp, sqrt

from datetime import datetime
import sys

sys.setrecursionlimit(20000)

if __name__ == "__main__":
    from datetime import datetime, timedelta

    # Paramètres de marché
    market = Market(
        rate=0.04,
        vol=0.28, 
        spot_price=102.45, 
        dividend_price=0.0, 
        dividend_date=datetime(2026, 6, 9)
    )

    # Option
    maturity = datetime(2026, 8, 27)
    option = Option(
        payoff_type="call", 
        option_type="EU", 
        K=100, 
        maturity=maturity
    )

    # Construction de l'arbre
    pricing_date = datetime(2025, 10, 9)
    steps = 100
    gap = None
    model = Model(market, option, pricing_date, steps, gap)

    # epsilon pour les dérivées
    epsilon_price = 0.0001
    epsilon_vol = 0.0001

    b_s_model = BlackScholesOptionPricing(option, pricing_date, market)
    b_s_price = b_s_model.calculate_option_price()

    # Type de pricing
    recursive_pricing = True
    backward_pricing = True
    greek_letter = False

    # Affichage de l'arbre
    plot_tree = False
    print_underlying_asset_price = False
    print_option_price = False
    print_prob_node = False

    download_convergence = False 

    if recursive_pricing or backward_pricing:
        results = model.pricing(recursive_pricing, backward_pricing)

        result_recursif = results["recursive_price"]
        recursive_pricing_time = results["recursive_pricing_exec_time"]
        result_backward = results["backward_price"]
        backward_pricing_time = results["backward_pricing_exec_time"]
        tree_creation_time = results["tree_creation_time"]
    
    if recursive_pricing:
        # Affichage
        ecart = abs(result_recursif - b_s_price)
        print(f"L'écart de convergence avec récursif pour {model.tree.steps} steps est de {ecart:.6f}")
        print(f"Prix récursif : {result_recursif:.6f}, Prix Black-Scholes : {b_s_price:.6f}")

    if backward_pricing:
        # Affichage
        ecart = abs(result_backward - b_s_price)
        print(f"L'écart de convergence avec récursif pour {model.tree.steps} steps est de {ecart:.6f}")
        print(f"Prix backward : {result_backward:.6f}, Prix Black-Scholes : {b_s_price:.6f}")
    
    if plot_tree : 
        fig = model.tree.plot_tree(print_option_price=print_option_price, print_underlying_asset_price=print_underlying_asset_price, print_prob_node=print_prob_node)
        fig.show()    
   
    if greek_letter:
        # 1️⃣ Définir le callable de pricing pour l'arbre trinomial
        def pricing_func(market_in, pricing_date_in, nb_steps, option_in):
            # Appel du modèle
            result = Model(market_in, option_in, pricing_date_in, steps=nb_steps).pricing(
                recursive_pricing=False,
                backward_pricing=True
            )

            # Récupération du prix backward
            result_backward = result["backward_price"] 

            # Retourner un float
            return float(result_backward)

        # 2️⃣ Créer l'objet Derivatives
        greeks = Derivatives(derivative_function=pricing_func,
                            epsilon_price=epsilon_price,
                            epsilon_vol=epsilon_vol)

        # 3️⃣ Calculer les Greeks via l'arbre trinomial
        delta = greeks.delta_streamlit(market, steps, pricing_date, option)
        gamma = greeks.gamma_streamlit(market, steps, pricing_date, option)
        vega = greeks.vega_streamlit(market, steps, pricing_date, option)
        vanna = greeks.vanna_streamlit(market, steps, pricing_date, option)
        voma = greeks.voma_streamlit(market, steps, pricing_date, option)

        delta_tt = greeks.delta(market, steps, pricing_date, option)
        gamma_tt = greeks.gamma(market, steps, pricing_date, option)
        vega_tt  = greeks.vega(market, steps, pricing_date, option)
        vanna_tt = greeks.vanna(market, steps, pricing_date, option)
        voma_tt  = greeks.voma(market, steps, pricing_date, option)

        # 4️⃣ Calculer les Greeks via Black-Scholes
        delta_bs  = b_s_model.calculate_delta()
        gamma_bs  = b_s_model.calculate_gamma()
        vega_bs   = b_s_model.calculate_vega()
        vanna_bs  = b_s_model.calculate_vanna()
        voma_bs   = b_s_model.calculate_vomma()

        # 5️⃣ Affichage des résultats et des écarts
        print(f"{'Greek':<10} {'Trinomial':>12} {'Black-Scholes':>15} {'Écart':>12}")
        print("-" * 50)
        print(f"{'Delta':<10} {delta_tt:12.6f} {delta_bs:15.6f} {abs(delta_tt - delta_bs):12.6f}")
        print(f"{'Gamma':<10} {gamma_tt:12.6f} {gamma_bs:15.6f} {abs(gamma_tt - gamma_bs):12.6f}")
        print(f"{'Vega':<10} {vega_tt:12.6f} {vega_bs:15.6f} {abs(vega_tt - vega_bs):12.6f}")
        print(f"{'Vanna':<10} {vanna_tt:12.6f} {vanna_bs:15.6f} {abs(vanna_tt - vanna_bs):12.6f}")
        print(f"{'Voma':<10} {voma_tt:12.6f} {voma_bs:15.6f} {abs(voma_tt - voma_bs):12.6f}")

    if download_convergence : 
            
        import matplotlib.pyplot as plt 

        # Dictionnaire pour stocker les résultats
        dict_convergence = {}
        dict_recursive_pricing = {}
        dict_backward_pricing = {}
        dict_tree = {}

        # Boucle sur le nombre d'étapes de l'arbre trinomial
        for step in range(100, 10001, 100):  # commence à 5, va jusqu'à 1000, par pas de 5
            print(step)
            model = Model(market, option, pricing_date, step, gap)
            b_s_price = BlackScholesOptionPricing(option, pricing_date, market).calculate_option_price()
            option_price, recursive_pricing_time, _, backward_pricing_time, tree_creation_time = model.pricing(recursive_pricing=True, backward_pricing=True)
            dict_convergence[step] = abs(option_price - b_s_price)
            dict_recursive_pricing[step] = recursive_pricing_time
            dict_backward_pricing[step] = backward_pricing_time
            dict_tree[step] = tree_creation_time

        # ---------------------------------------------------------------------
        # ÉCRITURE DANS UN FICHIER PYTHON
        # ---------------------------------------------------------------------
        output_file = "convergence_data.py"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# ================================================================\n")
            f.write("# Dictionnaires générés automatiquement pour l'analyse de convergence\n")
            f.write("# ================================================================\n\n")

            f.write("dict_convergence = {\n")
            for steps, diff in dict_convergence.items():
                f.write(f"    {steps}: {diff:.10f},\n")
            f.write("}\n\n")

            f.write("dict_recursive_pricing = {\n")
            for steps, t in dict_recursive_pricing.items():
                f.write(f"    {steps}: {t:.10f},\n")
            f.write("}\n\n")

            f.write("dict_backward_pricing = {\n")
            for steps, t in dict_backward_pricing.items():
                f.write(f"    {steps}: {t:.10f},\n")
            f.write("}\n\n")

            f.write("dict_tree = {\n")
            for steps, t in dict_tree.items():
                f.write(f"    {steps}: {t:.10f},\n")
            f.write("}\n")

        print(f"\n✅ Résultats enregistrés dans le fichier : {output_file}")
        

        
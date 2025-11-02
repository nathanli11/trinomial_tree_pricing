from time import time
from Market_class import Market
from Option_class import Option
from Trinomial_Tree_class import TrinomialTree
from math import exp, sqrt
from datetime import datetime
import sys
from typing import Dict

sys.setrecursionlimit(20000)


class Model:
    """
    Classe principale permettant de construire un modèle de marché
    et de pricer une option à l’aide d’un arbre trinomial.

    Cette classe sert de "chef d’orchestre" :
      - Crée un arbre trinomial (TrinomialTree)
      - Exécute un calcul de prix (récursif ou backward)
      - Mesure les temps d’exécution
      - Renvoie le résultat du pricing

    Args:
        market (Market): Informations de marché (spot, taux, volatilité...).
        option (Option): Caractéristiques de l’option.
        pricing_date (datetime): Date d’évaluation.
        steps (int, optional): Nombre d’étapes de l’arbre.
        gap (float, optional): Pas pour l’arbre.
    """

    def __init__(self, market: Market, option: Option, pricing_date: datetime, steps: int = None,
                 gap: float = None):
        # Création de l’arbre trinomial
        self.tree = TrinomialTree(market, pricing_date, steps, gap)

        # Enregistrement du marché et de l’option
        self.market = market
        self.option = option


    # -------------------------------------------------------------------------
    # Méthode principale de pricing
    # -------------------------------------------------------------------------
    def pricing(self, recursive_pricing: bool = False, backward_pricing: bool = False) -> Dict:
        """
        Calcule le prix de l’option en utilisant soit :
          - la méthode récursive
          - la méthode backward

        L’une des deux méthodes doit être activée.

        Returns:
            dict: Prix et temps d’exécution pour chaque méthode
        """
        # Étape 1 : Construction de l’arbre
        _, tree_creation_time = Model.mesurer_temps_execution(self.tree._build_tree, self.option)

        # Étape 2 : Pricing via la méthode choisie
        if recursive_pricing:
            result_recursif, recursive_pricing_time = Model.mesurer_temps_execution(
                self.tree.root.recursive_pricing, self.option
            )
        else:
            result_recursif = "You did not choose to use recursive pricing"
            recursive_pricing_time = result_recursif

        if backward_pricing:
            result_backward, backward_pricing_time = Model.mesurer_temps_execution(
                self.tree.root.backward_pricing, self.option
            )
        else:
            result_backward = "You did not choose to use backward pricing"
            backward_pricing_time = result_backward

        if not recursive_pricing and not backward_pricing:
            raise ValueError(
                "❌ Choisir entre recursive_pricing=True ou backward_pricing=True"
            )

        return {
            "recursive_price": result_recursif,
            "recursive_pricing_exec_time": recursive_pricing_time,
            "backward_price": result_backward,
            "backward_pricing_exec_time": backward_pricing_time,
            "tree_creation_time": tree_creation_time
        }


    # -------------------------------------------------------------------------
    # Fonction utilitaire : mesure du temps d’exécution
    # -------------------------------------------------------------------------
    @staticmethod
    def mesurer_temps_execution(fonction, *args, **kwargs):
        """
        Mesure le temps d’exécution d’une fonction donnée.

        Args:
            fonction (callable): Fonction à chronométrer
            *args, **kwargs: Arguments de la fonction

        Returns:
            tuple: (résultat, durée d’exécution en secondes)
        """
        debut = time()
        resultat = fonction(*args, **kwargs)
        fin = time()
        duree = fin - debut
        return resultat, duree
from datetime import datetime, timedelta
from math import exp, sqrt, pi, log
import plotly.io as pio
import formula as f
from TruncNode_class import TruncNode
from Market_class import Market
from Option_class import Option
from Proba_class import Proba

import sys
sys.setrecursionlimit(20000)  # Augmente la limite pour le pricing récursif

pio.renderers.default = "browser"

import matplotlib.pyplot as plt
from collections import deque


class TrinomialTree:
    """
    Classe représentant un arbre trinomial pour le pricing d'options.

    Cet arbre modélise l'évolution du prix d'un sous-jacent sur une période
    définie, avec des probabilités calculées à chaque étape. 

    Attributes:
        market (Market): Informations de marché (spot, taux, volatilité, dividende).
        steps (int): Nombre d'étapes de l'arbre.
        pricing_date (datetime): Date de valorisation.
        p_critic (float): Seuil de pruning
        nb_days (int): Nombre de jours dans une année (utilisé pour delta_t).
    """

    def __init__(self, market: Market, pricing_date: datetime, steps: int, gap : int = None):
        self.market = market
        self.p_critic = 10**(-7)
        self.steps = steps
        self.pricing_date = pricing_date
        self.nb_days: int = 365
        self.gap = gap

    # -------------------------------------------------------------------------
    # Construction de l'arbre
    # -------------------------------------------------------------------------
    def _build_tree(self, option: Option) -> None:
        """
        Méthode principale pour construire l'arbre trinomial.

        Args:
            option (Option): L'option à valoriser (utilisée pour maturité et K).
        """

        # Si steps = 0, utiliser une approximation GAP pour déterminer le nombre de pas
        if self.steps is None or self.gap is not None:
            if self.gap is not None:
                self.steps = int(
                    (self.market.vol ** 2 * (option.maturity - self.pricing_date).days) /
                    log(1+((8 * sqrt(2 * pi) * self.gap * sqrt(exp(self.market.vol ** 2 * (option.maturity - self.pricing_date).days) - 1))/(3*self.market.spot_price)))
                )
            else : 
                raise ValueError("Please input a step or a gap")

        # Calcul de l'incrément temporel Δt
        self.delta_t = abs(((option.maturity - self.pricing_date).days / self.steps) / self.nb_days)
        self.delta_day = self.delta_t * self.nb_days

        # Calcul d'alpha (facteur d'asymétrie) à partir de la volatilité et Δt
        self.alpha = f.calculate_alpha(self.market.vol, self.delta_t)

        # Création de la racine (tronc de l'arbre)
        self.root = TruncNode(self, self.pricing_date, self.market.spot_price)
        trunc = self.root
        trunc.prob_node = 1  # Probabilité cumulée initiale

        # Pré-calcul des probabilités sans dividendes (utilisées la plupart du temps)
        proba_without_div = Proba(
            next_mid_price=f.calculate_fwd(
                underlying_asset_price=self.market.spot_price,
                rate=self.market.rate,
                delta_t=self.delta_t,
                with_dividend=False,
                dividend_price=self.market.dividend_price
            ),
            variance=f.calculate_variance(
                self.market.spot_price,
                self.market.rate,
                self.delta_t,
                self.market.vol
            ),
            alpha=self.alpha,
            esperance=f.calculate_esperance(
                underlying_asset_price=self.market.spot_price,
                rate=self.market.rate,
                delta_t=self.delta_t,
                with_dividend=False,
                dividend_price=self.market.dividend_price
            ),
            with_dividend=False
        )

        next_generation_node_date = self.pricing_date

        # Boucle sur les étapes de l'arbre
        for _ in range(self.steps):
            next_generation_node_date += timedelta(days=self.delta_day)
            trunc = self.__create_new_generation(trunc, next_generation_node_date, proba_without_div)

        # on retourne None pour éviter le problème de mesurer temps d'excution dans modèle
        return None
    
    # -------------------------------------------------------------------------
    # Création d'une génération suivante
    # -------------------------------------------------------------------------
    def __create_new_generation(self, base_node: TruncNode, next_generation_node_date: datetime, proba_without_div: Proba) -> TruncNode:
        """
        Crée la génération suivante à partir d'un noeud donné (base_node).

        Args:
            base_node (TruncNode): Noeud parent servant de base pour la nouvelle génération.
            next_generation_node_date (datetime): Date du prochain niveau.
            proba_without_div (Proba): Probabilités pré-calculées sans dividende.

        Returns:
            TruncNode: Le noeud central de la nouvelle génération.
        """

        # Création du bloc trinomial pour le noeud de base
        base_node.create_trinomial_block(node_date=next_generation_node_date)

        # Cas où aucun dividende n'affecte ce noeud
        if not base_node.with_dividend:
            base_node.compute_esperance()
            base_node.compute_variance()
            base_node.compute_probabilities(proba=proba_without_div)

        else:
            # Cas où un dividende modifie le calcul des probabilités
            base_node.compute_esperance()
            base_node.compute_variance()

            # Recalcul des probabilités avec dividende
            proba_with_div = Proba(
                next_mid_price=base_node.next_mid_node.underlying_asset_price,
                variance=base_node.variance,
                alpha=self.alpha,
                esperance=base_node.esperance,
                with_dividend=base_node.with_dividend
            )

            base_node.compute_probabilities(proba=proba_with_div)

        # Mise à jour des probabilités cumulées du bloc
        base_node.compute_block_prob_node()

        # Construction de la partie supérieure et inférieure de l'arbre
        base_node.move_up_down(node_date=next_generation_node_date, proba=proba_without_div)

        return base_node.next_mid_node

    def get_nodes_data(self, option):
        """
        Récupère les coordonnées des nœuds de l'arbre pour affichage graphique.

        Retourne un dictionnaire contenant :
            - les coordonnées (étape, valeur du sous-jacent)
            - les coordonnées (étape, prix de l’option) si disponibles

        Returns:
            dict: { 'S_x': [...], 'S_y': [...], 'price_x': [...], 'price_y': [...] }
        """

        # --- Vérifie que l’arbre a été construit ---
        if not hasattr(self, "root") or self.root is None:
            self._build_tree(option)

        nodes_spot_x, nodes_spot_y = [], []
        nodes_price_x, nodes_price_y = [], []

        # --- Parcours en largeur (BFS) de tous les nœuds ---
        queue = [(self.root, 0)]  # (noeud, niveau)
        while queue:
            node, level = queue.pop(0)

            # Stocke la valeur du sous-jacent (spot)
            nodes_spot_x.append(level)
            nodes_spot_y.append(node.underlying_asset_price)

            # Si le prix de l’option est déjà calculé
            if hasattr(node, "option_price") and node.option_price is not None:
                nodes_price_x.append(level)
                nodes_price_y.append(node.option_price)

            # Ajout des fils (u, m, d)
            for child in [node.next_upper_node, node.next_mid_node, node.next_lower_node]:
                if child is not None:
                    queue.append((child, level + 1))

        # --- Retour des résultats ---
        return {
            "S_x": nodes_spot_x,
            "S_y": nodes_spot_y,
            "price_x": nodes_price_x,
            "price_y": nodes_price_y,
        }

    def plot_tree(
        self,
        option: Option = None,
        print_option_price: bool = False,
        print_underlying_asset_price: bool = False,
        print_prob_node: bool = False
    ):
        """
        Trace l’arbre trinomial sous forme interactive avec Plotly.
        Les informations des nœuds n'apparaissent qu'au survol.
        Supporte un grand nombre de steps avec adaptation de la taille des markers.
        """

        import plotly.graph_objects as go
        from collections import deque

        # --- 1️⃣ Vérification de l’existence de l’arbre ---
        if not hasattr(self, "root") or self.root is None:
            if option is None:
                raise ValueError("⚠️ L'arbre n'est pas encore construit et aucun 'option' n'a été fourni à plot_tree().")
            self._build_tree(option)

        # --- 2️⃣ Parcours de l’arbre en largeur (BFS) ---
        queue = deque([(0, self.root)])
        visited = set()
        coords = {}

        while queue:
            step, node = queue.popleft()
            if node is None or node in visited:
                continue
            visited.add(node)
            coords[node] = (step, getattr(node, "underlying_asset_price", 0))

            for child in [
                getattr(node, "next_lower_node", None),
                getattr(node, "next_mid_node", None),
                getattr(node, "next_upper_node", None)
            ]:
                if child is not None:
                    queue.append((step + 1, child))

        # --- 3️⃣ Construction des arêtes ---
        edge_x, edge_y = [], []
        for node, (x, y) in coords.items():
            for child in [
                getattr(node, "next_lower_node", None),
                getattr(node, "next_mid_node", None),
                getattr(node, "next_upper_node", None)
            ]:
                if child in coords:
                    x_child, y_child = coords[child]
                    edge_x += [x, x_child, None]
                    edge_y += [y, y_child, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="black"),
            hoverinfo="none"
        )

        # --- 4️⃣ Construction des nœuds ---
        node_x, node_y, node_color, node_size, hover_texts = [], [], [], [], []

        for node, (x, y) in coords.items():
            prob_node = getattr(node, "prob_node", 0)
            s_price = getattr(node, "underlying_asset_price", 0)
            o_price = getattr(node, "option_price", None)

            node_x.append(x)
            node_y.append(y)
            node_color.append(prob_node)
            node_size.append(2 + prob_node*4)

            texts = []
            if print_underlying_asset_price:
                texts.append(f"S={s_price:.2f}")
            if print_option_price and o_price is not None:
                texts.append(f"O={o_price:.2f}")
            if print_prob_node:
                texts.append(f"p={prob_node:.4f}")
            hover_texts.append("<br>".join(texts))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="RdYlBu",
                cmin=0,
                cmax=1,
                showscale=True,
                line=dict(width=1, color="black"),
                colorbar=dict(
                    title="Probabilité cumulée",
                    orientation="h",
                    x=0, 
                    y=-0.2,       # décalage sous l'axe
                    xanchor="left",
                    yanchor="top",
                    len=1.0,
                    tickvals=[0, 0.5, 1],
                    ticktext=["0", "0.5", "1"]
                )
            ),
            hoverinfo="text",
            hovertext=hover_texts
        )

        # --- 5️⃣ Mise en page finale ---
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="🌳 Arbre Trinomial",
            xaxis=dict(title="Étape", showgrid=True, zeroline=False),
            yaxis=dict(title="Prix du sous-jacent", showgrid=True, zeroline=False),
            showlegend=False,
            height=700,
            margin=dict(t=50, b=120)  # espace plus large pour la colorbar
        )

        return fig

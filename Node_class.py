from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
from math import log10, exp
import formula as f
from Option_class import Option
from Proba_class import Proba
import sys

sys.setrecursionlimit(20000)


class Node:
    """
    Classe représentant un nœud dans un arbre trinomial.

    Chaque nœud contient :
        - le prix du sous-jacent à ce nœud,
        - les probabilités de montée, de maintien ou de descente,
        - les liens vers les nœuds voisins (up, down, next_mid, etc.),
        - le calcul du prix de l'option via recursive ou backward pricing.

    Attributes:
        nb_nodes (int): compteur statique de nœuds créés.
        option_price (Optional[float]): prix de l'option calculé à ce nœud.
    """

    nb_nodes = 0
    option_price: Optional[float] = None

    def __init__(self, tree, node_date: datetime, underlying_asset_price: float):
        """Initialisation d'un nœud."""
        type(self).nb_nodes += 1

        # Attributs principaux
        self.tree = tree
        self.node_date: datetime = node_date
        self.underlying_asset_price: float = underlying_asset_price
        self.prob_node: float = 0.0
        self.is_trinomial_block: bool = True

        # Test dividende
        if (self.node_date is not None and
            self.node_date + timedelta(days=self.tree.delta_t * self.tree.nb_days) >=
            self.tree.market.dividend_date >= self.node_date):
            self.with_dividend = True
        else:
            self.with_dividend = False

        # Connexions aux nœuds voisins
        self.node_up: Node = None
        self.node_down: Node = None
        self.next_mid_node: Node = None
        self.next_upper_node: Node = None
        self.next_lower_node: Node = None
        self.previous_node: Node = None


    # -------------------------------------------------------------------------
    # Calculs financiers pour chaque nœud
    # -------------------------------------------------------------------------
    def compute_fwd(self) -> None:
        """Calcule le prix forward au nœud."""
        self.forward_price = f.calculate_fwd(
            self.underlying_asset_price,
            self.tree.market.rate,
            self.tree.delta_t,
            self.with_dividend,
            self.tree.market.dividend_price
        )

    def compute_esperance(self) -> None:
        """Calcule l'espérance du prix au nœud."""
        self.esperance = f.calculate_esperance(
            self.underlying_asset_price,
            self.tree.market.rate,
            self.tree.delta_t,
            self.with_dividend,
            self.tree.market.dividend_price
        )

    def compute_variance(self) -> None:
        """Calcule la variance du prix au nœud."""
        self.variance = f.calculate_variance(
            self.underlying_asset_price,
            self.tree.market.rate,
            self.tree.delta_t,
            self.tree.market.vol
        )

    def compute_probabilities(self, proba: Optional[Proba] = None) -> None:
        """Assigne les probabilités p_up, p_mid, p_down au nœud."""
        if self.is_trinomial_block and proba is not None:
            self.p_up = proba.p_up
            self.p_mid = proba.p_mid
            self.p_down = proba.p_down
        else:
            self.p_up = 0
            self.p_mid = 1
            self.p_down = 0

    def compute_block_prob_node(self) -> None:
        """Ajoute les probabilités cumulées pour chaque nœud suivant."""
        self.next_mid_node.prob_node += self.prob_node * self.p_mid
        if self.is_trinomial_block:
            self.next_upper_node.prob_node += self.prob_node * self.p_up
            self.next_lower_node.prob_node += self.prob_node * self.p_down


    # -------------------------------------------------------------------------
    # Connexions entre nœuds
    # -------------------------------------------------------------------------
    def connect_up_down(self, up_neighbor: Node) -> None:
        """Connecte deux nœuds verticalement (self en bas, up_neighbor en haut)."""
        self.node_up = up_neighbor
        up_neighbor.node_down = self


    # -------------------------------------------------------------------------
    # Construction des blocs supérieurs et inférieurs
    # -------------------------------------------------------------------------
    def move_up_down(self, node_date: datetime, proba: Proba) -> None:
        """
        Construit les blocs vers le haut et le bas à partir de ce nœud.
        """
        p_critic = self.tree.p_critic
        alpha = self.tree.alpha

        for direction in ("up", "down"):
            neighbor = getattr(self, f"node_{direction}")

            while neighbor is not None:
                neighbor.compute_esperance()
                getattr(neighbor, "compute_next_mid")(**{f"move_{direction}": True})

                if neighbor.prob_node >= p_critic and neighbor.next_mid_node is not None:
                    if neighbor.with_dividend:
                        neighbor.compute_variance()
                        proba = Proba(
                            next_mid_price=neighbor.next_mid_node.underlying_asset_price,
                            variance=neighbor.variance,
                            alpha=alpha,
                            esperance=neighbor.esperance,
                            with_dividend=neighbor.with_dividend
                        )

                    if direction == "up":
                        neighbor.next_lower_node = neighbor.next_mid_node.node_down
                    else:
                        neighbor.next_upper_node = neighbor.next_mid_node.node_up

                    neighbor.add_node(node_date, **{f"move_{direction}": True})
                    neighbor.compute_probabilities(proba)
                    neighbor.compute_block_prob_node()
                    neighbor.compute_variance()
                else:
                    neighbor.is_trinomial_block = False
                    neighbor.compute_probabilities()
                    neighbor.compute_block_prob_node()
                    neighbor.compute_variance()

                neighbor = getattr(neighbor, f"node_{direction}")

    def add_node(self, node_date: datetime, move_up: bool = False, move_down: bool = False) -> None:
        """Ajoute un nœud au-dessus ou en dessous de la génération."""
        if move_up and self.next_upper_node is None:
            self.next_upper_node = Node(
                self.tree,
                node_date,
                self.next_mid_node.underlying_asset_price * self.tree.alpha
            )
            self.next_mid_node.connect_up_down(self.next_upper_node)

        if move_down and self.next_lower_node is None:
            self.next_lower_node = Node(
                self.tree,
                node_date,
                self.next_mid_node.underlying_asset_price / self.tree.alpha
            )
            self.next_lower_node.connect_up_down(self.next_mid_node)

    def compute_next_mid(self, move_up: bool = False, move_down: bool = False) -> None:
        """
        Recherche et instancie le prochain nœud central (next_mid) pour la génération suivante.
        """
        if move_up or move_down:
            ref_attr = "node_down" if move_up else "node_up"
            ref_node = getattr(self, ref_attr)
            next_attr = "next_upper_node" if move_up else "next_lower_node"

            if not ref_node.is_trinomial_block:
                coef = self.tree.alpha if move_up else 1 / self.tree.alpha
                next_mid = ref_node.next_mid_node
                expected_next_mid_node = Node(
                    self.tree,
                    next_mid.node_date,
                    next_mid.underlying_asset_price * coef
                )
                connect_src, connect_dst = ((next_mid, expected_next_mid_node)
                                           if move_up else (expected_next_mid_node, next_mid))
                connect_src.connect_up_down(connect_dst)
            else:
                expected_next_mid_node = getattr(ref_node, next_attr)

        upper_bound = expected_next_mid_node.underlying_asset_price * (1 + self.tree.alpha) / 2
        lower_bound = expected_next_mid_node.underlying_asset_price * (1 + 1 / self.tree.alpha) / 2

        if self.esperance < 0:
            self.node_up = None
            self.next_mid_node = None
            return

        if lower_bound <= self.esperance <= upper_bound:
            self.next_mid_node = expected_next_mid_node
        elif self.esperance < lower_bound:
            self.next_mid_node = self.search_down(expected_next_mid_node)
        else:
            self.next_mid_node = self.search_up(expected_next_mid_node)


    # -------------------------------------------------------------------------
    # Recherche de nœuds
    # -------------------------------------------------------------------------
    def search_down(self, expected_next_mid_node: Node) -> Node:
        """Cherche le nœud mid en dessous du nœud attendu."""
        while self.esperance < (expected_next_mid_node.underlying_asset_price * (1 + 1 / self.tree.alpha) / 2):
            if expected_next_mid_node.node_down is not None:
                expected_next_mid_node = expected_next_mid_node.node_down
            else:
                new_node_down = Node(
                    self.tree,
                    expected_next_mid_node.node_date,
                    expected_next_mid_node.underlying_asset_price / self.tree.alpha
                )
                new_node_down.connect_up_down(expected_next_mid_node)
                expected_next_mid_node = new_node_down
        return expected_next_mid_node

    def search_up(self, expected_next_mid_node: Node) -> Node:
        """Cherche le nœud mid au-dessus du nœud attendu."""
        while self.esperance > (expected_next_mid_node.underlying_asset_price * (1 + self.tree.alpha) / 2):
            if expected_next_mid_node.node_up is not None:
                expected_next_mid_node = expected_next_mid_node.node_up
            else:
                new_node_up = Node(
                    self.tree,
                    expected_next_mid_node.node_date,
                    expected_next_mid_node.underlying_asset_price * self.tree.alpha
                )
                expected_next_mid_node.connect_up_down(new_node_up)
                expected_next_mid_node = new_node_up
        return expected_next_mid_node


    # -------------------------------------------------------------------------
    # Pricing
    # -------------------------------------------------------------------------
    def recursive_pricing(self, opt: Option) -> float:
        """Calcule le prix de l'option via un parcours récursif dans l'arbre."""
        if self.next_mid_node is None:
            self.option_price = opt.payoff(self.underlying_asset_price)
        elif self.option_price is None:
            self.option_price = f.discount(self.tree.market.rate, self.tree.delta_t) * (
                self.p_mid * self.next_mid_node.recursive_pricing(opt) +
                (self.p_up * self.next_upper_node.recursive_pricing(opt) if self.next_upper_node else 0) +
                (self.p_down * self.next_lower_node.recursive_pricing(opt) if self.next_lower_node else 0)
            )
        if opt.option_type == "US":
            self.option_price = max(self.option_price, opt.payoff(self.underlying_asset_price))
        return self.option_price

    def backward_pricing(self, opt: Option) -> float:
        """Calcule le prix de l'option via backward pricing à partir de ce nœud."""
        def discounted_value(node) -> float:
            r, dt = self.tree.market.rate, self.tree.delta_t
            return f.discount(r, dt) * (
                (node.p_mid * node.next_mid_node.option_price if node.next_mid_node else 0) +
                (node.p_up * node.next_upper_node.option_price if node.next_upper_node else 0) +
                (node.p_down * node.next_lower_node.option_price if node.next_lower_node else 0)
            )

        def compute_option_price(node) -> float:
            if node.next_mid_node is None:
                return opt.payoff(node.underlying_asset_price)
            val = discounted_value(node)
            if opt.option_type == "US":
                return max(val, opt.payoff(node.underlying_asset_price))
            return val

        trunc = self
        while trunc.next_mid_node is not None:
            trunc = trunc.next_mid_node

        while trunc.previous_node is not None:
            trunc.option_price = compute_option_price(trunc)
            for neighbor_attr in ('node_up', 'node_down'):
                neighbor = getattr(trunc, neighbor_attr)
                while neighbor is not None:
                    neighbor.option_price = compute_option_price(neighbor)
                    neighbor = getattr(neighbor, neighbor_attr)
            trunc = trunc.previous_node

        self.option_price = compute_option_price(self)
        return self.option_price

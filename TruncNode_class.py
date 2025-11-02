##### changer pour une classe truncnode
from Node_class import Node
import datetime

class TruncNode(Node):
    """
    Classe représentant un nœud du tronc principal (axe central) de l’arbre trinomial.
    Hérite de Node mais avec un comportement spécifique pour la création des blocs suivants.
    """

    def __init__(self, tree, node_date: datetime, underlying_asset_price: float):
        super().__init__(tree, node_date, underlying_asset_price)
        # Les noeuds du tronc sont toujours des blocs trinômiaux
        self.is_trinomial_block = True
        self.previous_node = None

    def create_trinomial_block(self, node_date: datetime) -> None:
        """
        Crée le bloc trinomial depuis un nœud du tronc principal.
        Ce comportement est spécifique : la suite du tronc reste composée de TrunkNode.
        """
        self.compute_fwd()

        # Le next_mid_node reste un TrunkNode

        self.next_mid_node = TruncNode(self.tree, node_date, self.forward_price)
        self.next_mid_node.previous_node = self

        self.add_node(node_date, move_up=True, move_down=True)

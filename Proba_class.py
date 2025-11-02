import formula as f


class Proba:
    """
    Classe représentant le calcul des probabilités d’évolution du prix.

    Cette classe permet de calculer les probabilités :
    - p_up : probabilité de hausse du sous-jacent
    - p_mid : probabilité de stabilité (état intermédiaire)
    - p_down : probabilité de baisse du sous-jacent

    Ces probabilités sont dérivées de la variance, de l’espérance
    et du paramètre d’asymétrie alpha.
    """

    def __init__(self, next_mid_price: float, variance: float, alpha: float, esperance: float, with_dividend: bool):
        """
        Initialise les paramètres nécessaires au calcul des probabilités.

        Args:
            next_mid_price (float): Prix du sous jacent au next_mid_node.
            variance (float): Variance du sous-jacent sur la période Δt.
            alpha (float): Facteur d’asymétrie.
            esperance (float): Espérance mathématique du prix futur de l’actif.
            with_dividend (bool): Indique si le dividende tombe entre la date du noeud et la prochaine.
        """

        # Attributs principaux utilisés pour le calcul
        self.next_mid_price = next_mid_price
        self.variance = variance
        self.alpha = alpha
        self.esperance = esperance
        self.with_dividend = with_dividend

        # Déclenche immédiatement le calcul des probabilités
        self.compute_proba()

    def compute_proba(self):
        """
        Calcule les probabilités de montée, descente et stabilité
        du sous-jacent à partir des formules fournies dans le module `formula`.

        Raises:
            ValueError: Si les probabilités calculées ne sont pas cohérentes (hors [0,1]).
        """

        # Calcul de la probabilité de baisse à partir de la variance et de l’espérance
        self.p_down = f.calculate_p_down(
            self.next_mid_price, self.variance, self.esperance, self.alpha
        )

        # Calcul de la probabilité de hausse, dépendante de p_down
        self.p_up = f.calculate_p_up(
            self.next_mid_price, self.esperance, self.alpha, self.with_dividend, self.p_down
        )

        # Vérification : les probabilités doivent être comprises entre 0 et 1,
        # et leur somme ne doit pas dépasser 1.
        if 0 <=self.p_down <= 1 and 0 <= self.p_up <= 1 and self.p_down + self.p_up <= 1:
            # Si tout est cohérent, on déduit la probabilité intermédiaire
            self.p_mid = 1 - self.p_down - self.p_up
        else:
            # En cas d’erreur, on renvoie un message explicite
            raise ValueError(
                f"Erreur : probabilités incohérentes.\n"
                f"p_up = {self.p_up:.5f}, p_down = {self.p_down:.5f}, somme = {self.p_up + self.p_down:.5f}"
            )

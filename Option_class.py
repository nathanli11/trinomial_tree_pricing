from datetime import datetime
from typing import Literal


class Option:
    """
    Classe représentant une option financière.

    Cette classe contient :
      - le type de payoff ("call" ou "put"),
      - le type d'option ("US" pour américaine, "EU" pour européenne),
      - le prix d'exercice (K),
      - la date de maturité.
    """

    def __init__(self, payoff_type: Literal["call", "put"],
                 option_type: Literal["US", "EU"],
                 K: float,
                 maturity: datetime):
        """
        Initialise une option avec ses caractéristiques.

        Args:
            payoff_type (Literal["call","put"]): Type de payoff de l'option.
            option_type (Literal["US","EU"]): Type de l'option : américaine ou européenne.
            K (float): Prix d'exercice.
            maturity (datetime): Date de maturité de l'option.
        """
        self.payoff_type = payoff_type
        self.option_type = option_type
        self.K = K
        self.maturity = maturity

    # -------------------------------------------------------------------------
    # Calcul du payoff
    # -------------------------------------------------------------------------
    def payoff(self, S_t: float) -> float:
        """
        Calcule le payoff de l'option à un prix du sous-jacent donné.

        Args:
            S_t (float): Prix du sous-jacent au temps t.

        Returns:
            float: Valeur du payoff (≥ 0).

        Notes:
            - Pour un call : max(S_t - K, 0)
            - Pour un put  : max(K - S_t, 0)
        """

        if self.payoff_type == "call":
            return max(S_t - self.K, 0)
        else:
            return max(self.K - S_t, 0)

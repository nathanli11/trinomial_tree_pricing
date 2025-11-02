from datetime import datetime, date


class Market:
    """
    Classe représentant les informations de marché nécessaires pour la valorisation
    d'options ou pour d'autres modèles financiers.

    Cette classe contient les principaux paramètres du marché :
        - taux sans risque (rate)
        - volatilité du sous-jacent (vol)
        - prix spot du sous-jacent (spot_price)
        - dividendes (dividend_price et dividend_date)
    """

    def __init__(self, rate: float, vol: float, spot_price: float, dividend_price: float, dividend_date: datetime):
        """
        Initialise un objet Market avec les paramètres fournis.

        Args:
            rate (float): Taux d'intérêt sans risque annuel.
            vol (float): Volatilité annuelle du sous-jacent.
            spot_price (float): Prix actuel (spot) du sous-jacent.
            dividend_price (float): Montant du dividende.
            dividend_date (datetime): Date de versement du dividende.

        Notes:
            - Si `dividend_date` est fourni en tant que `date` (sans heure),
              il est converti en `datetime` avec heure = 00:00.
        """

        # Conversion automatique de date en datetime si nécessaire
        if isinstance(dividend_date, date) and not isinstance(dividend_date, datetime):
            dividend_date = datetime.combine(dividend_date, datetime.min.time())

        # Attributs principaux
        self.rate = rate                    # Taux sans risque annuel
        self.vol = vol                      # Volatilité annuelle du sous-jacent
        self.spot_price = spot_price        # Prix spot actuel du sous-jacent
        self.dividend_price = dividend_price  # Montant du dividende
        self.dividend_date = dividend_date    # Date du dividende (datetime)

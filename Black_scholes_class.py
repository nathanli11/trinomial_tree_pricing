from datetime import datetime
import numpy as np
from math import exp
from scipy.stats import norm
from Market_class import Market
from Option_class import Option
import formula as f


class BlackScholesOptionPricing:
    """
    Classe permettant de calculer le prix et toutes les sensibilités (Greeks)
    d'une option européenne selon le modèle de Black-Scholes.
    """

    def __init__(self, opt: Option, pricing_date: datetime, market: Market) -> None:
        """
        Initialise les paramètres nécessaires au modèle de Black-Scholes.
        """
        self.K = opt.K
        # Temps restant jusqu’à maturité exprimé en années
        self.T = ((opt.maturity - pricing_date).days) / 365
        self.r = market.rate
        self.sigma = market.vol
        self.payoff_type = opt.payoff_type
        self.S = market.spot_price

    # ---------------------------------------------------------------------
    # Paramètres intermédiaires : d1 et d2
    # ---------------------------------------------------------------------

    def __d1(self) -> float:
        """Calcule le paramètre d1 de la formule de Black-Scholes."""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        return d1

    def __d2(self) -> float:
        """Calcule le paramètre d2 de la formule de Black-Scholes."""
        d2 = self.__d1() - self.sigma * np.sqrt(self.T)
        return d2

    # ---------------------------------------------------------------------
    # Prix de l'option
    # ---------------------------------------------------------------------

    def calculate_option_price(self) -> float:
        """Calcule le prix théorique de l’option européenne (call ou put)."""
        d1 = self.__d1()
        d2 = self.__d2()

        if self.payoff_type == "call":
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.payoff_type == "put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError("Type de payoff non supporté. Utilisez 'call' ou 'put'.")

    # ---------------------------------------------------------------------
    # Sensibilités primaires (les 'Greeks' classiques)
    # ---------------------------------------------------------------------

    def calculate_delta(self) -> float:
        """Calcule le Delta : sensibilité du prix de l’option au prix du sous-jacent."""
        d1 = self.__d1()
        if self.payoff_type == "call":
            return norm.cdf(d1)
        elif self.payoff_type == "put":
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Type de payoff non supporté. Utilisez 'call' ou 'put'.")

    def calculate_gamma(self) -> float:
        """Calcule le Gamma : sensibilité du Delta au prix du sous-jacent."""
        d1 = self.__d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def calculate_vega(self) -> float:
        """Calcule le Vega : sensibilité du prix de l’option à la volatilité."""
        d1 = self.__d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

    def calculate_theta(self) -> float:
        """Calcule le Theta : sensibilité du prix de l’option à l’écoulement du temps."""
        d1 = self.__d1()
        d2 = self.__d2()
        first_term = - (self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.payoff_type == "call":
            theta = first_term - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.payoff_type == "put":
            theta = first_term + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("Type de payoff non supporté. Utilisez 'call' ou 'put'.")
        return theta

    def calculate_rho(self) -> float:
        """Calcule le Rho : sensibilité du prix de l’option au taux d’intérêt."""
        d2 = self.__d2()
        if self.payoff_type == "call":
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.payoff_type == "put":
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        else:
            raise ValueError("Type de payoff non supporté. Utilisez 'call' ou 'put'.")
        return rho

    # ---------------------------------------------------------------------
    # Sensibilités de second ordre : Vanna et Vomma
    # ---------------------------------------------------------------------

    def calculate_vanna(self) -> float:
        """
        Calcule la Vanna : sensibilité croisée du prix de l’option
        au prix du sous-jacent et à la volatilité.

        Vanna = ∂²V / (∂S ∂σ)
        
        Formule correcte :
            Vanna = n(d1) * sqrt(T) * (1 - d1 / (sigma * sqrt(T)))
        où n(d1) = N'(d1).

        Remarque :
            La Vanna mesure comment le Vega évolue lorsque le sous-jacent change.
        """
        d1 = self.__d1()
        sigma = self.sigma
        T = self.T
        n_d1 = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)  
        vanna = n_d1 * np.sqrt(T) * (1.0 - d1 / (sigma * np.sqrt(T)))
        return vanna

    def calculate_vomma(self) -> float:
        """
        Calcule la Vomma (aussi appelée Volga) :
        sensibilité du Vega à une variation de la volatilité.

        Définition :
            Vomma = ∂²V / ∂σ²

        Formule :
            Vomma = Vega * (d1 * d2 / σ)

        Interprétation :
            - Si Vomma > 0 : le Vega augmente quand la volatilité augmente.
            - Si Vomma < 0 : le Vega diminue quand la volatilité augmente.
        """
        d1 = self.__d1()
        d2 = self.__d2()
        vega = self.calculate_vega()
        vomma = vega * (d1 * d2) / self.sigma
        return vomma

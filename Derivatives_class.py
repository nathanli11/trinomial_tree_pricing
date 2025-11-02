from datetime import datetime
from Market_class import Market
from Option_class import Option
from Model_class import Model
from typing import Callable, Optional
import streamlit as st


class Derivatives:
    def __init__(
        self,
        derivative_function: Optional[Callable[[Market, datetime, int, Option], float]] = None,
        epsilon_price: float = 0.01,
        epsilon_vol: float = 0.01
    ):
        """
        Classe pour calculer les Greeks d’une option via un callable de pricing
        ou directement via backward pricing si derivative_function=None.

        Args:
            derivative_function (Callable, optional): Fonction de pricing acceptant 
                (Market, datetime, nb_steps, Option) et retournant un float.
                Si None, utilise le pricing backward interne.
            epsilon_price (float): Pas pour dérivée par rapport au prix.
            epsilon_vol (float): Pas pour dérivée par rapport à la volatilité.
        """
        self.f: Optional[Callable[[Market, datetime, int, Option], float]] = derivative_function
        self.epsilon_price = epsilon_price
        self.epsilon_vol = epsilon_vol


    # -------------------------------
    # Helpers pour modifier le marché
    # -------------------------------
    def _shift_price(self, market: Market, shift: float) -> Market:
        return Market(
            market.rate,
            market.vol,
            market.spot_price + shift,
            market.dividend_price,
            market.dividend_date
        )


    def _shift_vol(self, market: Market, shift: float) -> Market:
        return Market(
            market.rate,
            market.vol + shift,
            market.spot_price,
            market.dividend_price,
            market.dividend_date
        )


    # -------------------------------
    # Greeks
    # -------------------------------
    def delta(self, market: Market, steps: int, pricing_date: datetime, option: Option) -> float:
        price_up = self.f(
            self._shift_price(market, self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        price_down = self.f(
            self._shift_price(market, -self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        return (price_up - price_down) / (2 * self.epsilon_price)


    def gamma(self, market: Market, steps: int, pricing_date: datetime, option: Option) -> float:
        price_up = self.f(
            self._shift_price(market, self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        price_down = self.f(
            self._shift_price(market, -self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        price = self.f(market, pricing_date, steps, option)
        return (price_up + price_down - 2 * price) / (self.epsilon_price ** 2)


    def vega(self, market: Market, steps: int, pricing_date: datetime, option: Option) -> float:
        price_up = self.f(
            self._shift_vol(market, self.epsilon_vol),
            pricing_date,
            steps,
            option
        )
        price_down = self.f(
            self._shift_vol(market, -self.epsilon_vol),
            pricing_date,
            steps,
            option
        )
        return (price_up - price_down) / (2 * self.epsilon_vol)


    def voma(self, market: Market, steps: int, pricing_date: datetime, option: Option) -> float:
        price_up = self.f(
            self._shift_vol(market, self.epsilon_vol),
            pricing_date,
            steps,
            option
        )
        price_down = self.f(
            self._shift_vol(market, -self.epsilon_vol),
            pricing_date,
            steps,
            option
        )
        price = self.f(market, pricing_date, steps, option)
        return (price_up + price_down - 2 * price) / (self.epsilon_vol ** 2)


    def vanna(self, market: Market, steps: int, pricing_date: datetime, option: Option) -> float:
        price_uu = self.f(
            self._shift_price(self._shift_vol(market, self.epsilon_vol), self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        price_ud = self.f(
            self._shift_price(self._shift_vol(market, -self.epsilon_vol), self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        price_du = self.f(
            self._shift_price(self._shift_vol(market, self.epsilon_vol), -self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        price_dd = self.f(
            self._shift_price(self._shift_vol(market, -self.epsilon_vol), -self.epsilon_price),
            pricing_date,
            steps,
            option
        )
        return (price_uu - price_ud - price_du + price_dd) / (4 * self.epsilon_price * self.epsilon_vol)


    def holistic_greek_analysis(self, market: Market, steps: int, pricing_date: datetime, option: Option) -> dict:
        """
        Analyse combinée des Greeks pour produire une interprétation globale et pratique
        selon le type d’option et les interactions entre Delta, Gamma, Vega, Vanna, Vomma.
        """
        # Calcul des Greeks
        delta = self.delta(market, steps, pricing_date, option)
        gamma = self.gamma(market, steps, pricing_date, option)
        vega = self.vega(market, steps, pricing_date, option)
        vanna = self.vanna(market, steps, pricing_date, option)
        vomma = self.voma(market, steps, pricing_date, option)

        analysis = []

        # --- Analyse Delta/Gamma ---
        if abs(delta) > 0.7:
            moneyness = "profondément dans la monnaie"
        elif abs(delta) < 0.3:
            moneyness = "hors de la monnaie"
        else:
            moneyness = "proche du strike"

        if gamma > 0.05:
            gamma_comment = "Gamma élevé : ajustements Delta fréquents nécessaires"
        else:
            gamma_comment = "Gamma faible : Delta stable, peu d'ajustements nécessaires"

        analysis.append(
            f"L’option est {moneyness} avec Delta = {delta:.2f} et Gamma = {gamma:.2f}. {gamma_comment}."
        )

        # --- Analyse Vega/Vomma ---
        if vega > 0.1 and vomma > 0.05:
            vol_comment = "Option très sensible à la volatilité et à son évolution (Vega et Vomma élevés)."
        elif vega > 0.1:
            vol_comment = "Option sensible à la volatilité (Vega élevé)."
        else:
            vol_comment = "Option peu sensible à la volatilité."

        analysis.append(f"Vega = {vega:.2f}, Vomma = {vomma:.2f}. {vol_comment}")

        # --- Analyse Delta + Vanna ---
        if abs(vanna) > 0.05:
            vanna_comment = "Delta varie avec la volatilité (Vanna significatif) : risque directionnel modulé par la volatilité."
            analysis.append(f"Vanna = {vanna:.2f}. {vanna_comment}")

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vanna': vanna,
            'vomma': vomma,
            'holistic_analysis': analysis
        }

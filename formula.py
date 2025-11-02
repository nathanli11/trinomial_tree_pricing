from math import exp, sqrt

def calculate_fwd(underlying_asset_price: float, rate: float, delta_t: float,
                  with_dividend: bool, dividend_price: float) -> float:
    """
    Calcule le prix forward théorique d’un actif sous-jacent.

    Paramètres
    ----------
    underlying_asset_price : float
        Prix spot (actuel) de l’actif sous-jacent.
    rate : float
        Taux d’intérêt sans risque (en continu).
    delta_t : float
        Durée en années jusqu’à l’échéance.
    with_dividend : bool
        True si l’actif verse un dividende connu avant l’échéance.
    dividend_price : float
        Valeur actuelle du dividende à retrancher du prix forward.

    Retourne
    --------
    float
        Le prix forward théorique :
        - Sans dividende : F = S0 * e^(r * Δt)
        - Avec dividende : F = S0 * e^(r * Δt) - D
    """
    if with_dividend:
        return underlying_asset_price * exp(rate * delta_t) - dividend_price
    else:
        return underlying_asset_price * exp(rate * delta_t)


def calculate_esperance(underlying_asset_price: float, rate: float, delta_t: float,
                        with_dividend: bool, dividend_price: float) -> float:
    """
    Calcule l'espérance (valeur moyenne attendue) du prix futur de l’actif.

    Paramètres
    ----------
    underlying_asset_price : float
        Prix actuel de l’actif.
    rate : float
        Taux d’intérêt sans risque.
    delta_t : float
        Horizon temporel (en années).
    with_dividend : bool
        Indique si un dividende doit être pris en compte.
    dividend_price : float
        Montant du dividende connu à soustraire.

    Retourne
    --------
    float
        L'espérance du prix futur selon :
        E[S] = S0 * e^(r * Δt) - D (si dividende)
    """
    if with_dividend:
        return underlying_asset_price * exp(rate * delta_t) - dividend_price
    else:
        return underlying_asset_price * exp(rate * delta_t)


def calculate_variance(underlying_asset_price: float, rate: float,
                       delta_t: float, vol: float) -> float:
    """
    Calcule la variance du prix futur d’un actif selon sa volatilité.

    Paramètres
    ----------
    underlying_asset_price : float
        Prix actuel de l’actif.
    rate : float
        Taux sans risque.
    delta_t : float
        Durée jusqu’à l’échéance.
    vol : float
        Volatilité annualisée (écart-type des rendements logarithmiques).

    Retourne
    --------
    float
        Variance du prix futur :
        Var(S) = S0² * e^(2rΔt) * (e^(σ²Δt) - 1)
    """
    return (underlying_asset_price ** 2) * exp(2 * rate * delta_t) * (exp((vol ** 2) * delta_t) - 1)


def calculate_p_down(next_mid_price: float, variance: float,
                     esperance: float, alpha: float) -> float:
    """
    Calcule la probabilité de baisse du prix dans un modèle discret (trinomial).

    Paramètres
    ----------
    next_mid_price : float
        Prix médian (ou central) du nœud suivant dans l’arbre.
    variance : float
        Variance du prix futur (issue de calculate_variance).
    esperance : float
        Espérance du prix futur (issue de calculate_esperance).
    alpha : float
        Facteur d’amplitude des mouvements de prix (hausse/baisse).

    Retourne
    --------
    float
        Probabilité de baisse p_down calculée par calibration du modèle :
        On ajuste les moments (espérance, variance) pour que le modèle discret
        reproduise les propriétés du processus continu.
    """
    return ((1 / (next_mid_price ** 2)) * (variance + esperance ** 2)
            - 1 - (alpha + 1) * ((1 / next_mid_price) * esperance - 1)) \
           / ((1 - alpha) * ((1 / (alpha ** 2)) - 1))


def calculate_p_up(next_mid_price: float, esperance: float, alpha: float,
                   with_dividend: bool, p_down: float) -> float:
    """
    Calcule la probabilité de hausse du prix dans un modèle discret.

    Paramètres
    ----------
    next_mid_price : float
        Prix médian (ou central) du nœud suivant.
    esperance : float
        Espérance du prix futur.
    alpha : float
        Facteur d’échelle des variations (lié à la volatilité).
    with_dividend : bool
        True si un dividende est pris en compte.
    p_down : float
        Probabilité de baisse (calculée précédemment).

    Retourne
    --------
    float
        Probabilité de hausse p_up :
        - Si dividende : formule ajustée selon les flux.
        - Sinon : p_up = p_down / alpha
    """
    if with_dividend:
        return (((1 / next_mid_price) * esperance - 1)
                - (((1 / alpha) - 1) * p_down)) / (alpha - 1)
    else:
        return p_down / alpha


def calculate_alpha(vol: float, delta_t: float) -> float:
    """
    Calcule le facteur d’amplitude alpha, qui détermine la taille
    relative des mouvements de prix dans un modèle discret.

    Paramètres
    ----------
    vol : float
        Volatilité annualisée de l’actif.
    delta_t : float
        Durée de chaque pas de temps.

    Retourne
    --------
    float
        Facteur alpha :
        α = e^(σ * √(3Δt))
        Le coefficient '3' est arbitraire et ajuste la dispersion du modèle.
    """
    return exp(vol * sqrt(3 * delta_t))


def discount(rate: float, delta_t: float) -> float:
    """
    Calcule le facteur d’actualisation pour ramener une valeur future
    à la date présente selon un taux d’intérêt continu.

    Paramètres
    ----------
    rate : float
        Taux d’intérêt sans risque (en continu).
    delta_t : float
        Horizon temporel (en années).

    Retourne
    --------
    float
        Facteur d’actualisation :
        DF = e^(-r * Δt)
    """
    return exp(-rate * delta_t)
import numpy as np
import scipy
import math
import scipy.integrate
import pandas as pd


def C_CF(u, t, r, nu, kappa, sigma, rho, bj, uj):
    d = np.sqrt(
        (rho * sigma * u * 1j - bj) ** 2 - sigma**2 * (2 * uj * u * 1j - u**2)
    )
    g = (bj - rho * sigma * u * 1j + d) / (bj - rho * sigma * u * 1j - d)
    out = r * u * t * 1j + (nu * kappa) / sigma**2 * (
        (bj - rho * sigma * u * 1j + d) * t
        - 2 * np.log((1 - g * np.exp(d * t)) / (1 - g))
    )
    return out


def D_CF(u, t, sigma, rho, bj, uj):
    d = np.sqrt(
        (rho * sigma * u * 1j - bj) ** 2 - sigma**2 * (2 * uj * u * 1j - u**2)
    )
    g = (bj - rho * sigma * u * 1j + d) / (bj - rho * sigma * u * 1j - d)
    out = (
        (bj - rho * sigma * u * 1j + d)
        / (sigma**2)
        * ((1 - np.exp(d * t)) / (1 - g * np.exp(d * t)))
    )
    return out


def Call_Heston(K, T, r, nu, kappa, sigma, rho, S, V):

    """Call_Heston:  Compute the value of call option using the formula in
    Heston[1993], see also formula (6) in Albrecher et Al.[2006].

    USAGE: P = Call_Heston(K, T, r, nu, kappa, sigma, rho, S, V, modif);

    PARAMETERS:
       Input:
            K : float
              strike price of the call option

            T : float
              maturity of the call option

            r : float
              risk free rate

            nu : float
               long term variance

            kappa : float
                  speed of mean reversion

            sigma : float
                  parameters of the Heston model

            rho : float
                correlation parameter between the stock and vol processes

            S, V : float , float
                 initial stock price and volatility
       Output:
           P : float
             price of the call option.
    """
    b1 = kappa - rho * sigma
    b2 = kappa
    u1 = 0.5
    u2 = -0.5

    x = np.log(S)
    alpha = np.log(K)  # log-strike

    ##for simplicity
    real = np.real
    pi = np.pi

    integrand = lambda u: (
        S
        * np.real(
            np.exp(-1j * u * alpha)
            * np.exp(
                C_CF(u, T, r, nu, kappa, sigma, rho, b1, u1)
                + V * D_CF(u, T, sigma, rho, b1, u1)
                + 1j * u * x
            )
            / (1j * u)
        )
        - K
        * np.exp(-r * T)
        * real(
            np.exp(-1j * u * alpha)
            * np.exp(
                C_CF(u, T, r, nu, kappa, sigma, rho, b2, u2)
                + V * D_CF(u, T, sigma, rho, b2, u2)
                + 1j * u * x
            )
            / (1j * u)
        )
    )

    integral = scipy.integrate.quad(integrand, 0, 100)[0]
    P = 0.5 * (S - K * np.exp(-r * T)) + 1 / pi * integral
    return P

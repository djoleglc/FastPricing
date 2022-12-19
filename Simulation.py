from HestonFFT import *
import numpy as np
import pandas as pd
import math
import scipy 
import math
import scipy.integrate
import time 
from Simulation import TimeHestonFFT 


def GenerateGridsfromUniform(N, bound, seed=10):
    np.random.seed(seed)

    n_col = 7
    data = np.random.uniform(size=(N, 3))
    gridnu = data[:, 0] * (bound["nu"][1] - bound["nu"][0]) + bound["nu"][0]
    gridkappa = data[:, 1] * (bound["kappa"][1] - bound["kappa"][0]) + bound["kappa"][0]
    gridsigma = data[:, 2] * (bound["sigma"][1] - bound["sigma"][0]) + bound["sigma"][0]
    to_remove = gridsigma**2 >= 2 * gridnu * gridkappa
    to_rem_N = to_remove.sum()

    # Feller Condition
    while to_rem_N != 0:
        data = np.random.uniform(size=(to_rem_N, 3))
        gridnu[to_remove] = (
            data[:, 0] * (bound["nu"][1] - bound["nu"][0]) + bound["nu"][0]
        )
        gridkappa[to_remove] = (
            data[:, 1] * (bound["kappa"][1] - bound["kappa"][0]) + bound["kappa"][0]
        )
        gridsigma[to_remove] = (
            data[:, 2] * (bound["sigma"][1] - bound["sigma"][0]) + bound["sigma"][0]
        )

        # index of values to remove
        to_remove = gridsigma**2 <= 2 * gridnu * gridkappa
        to_rem_N = to_remove.sum()

    data_ = np.random.uniform(size=(N, 4))
    gridK = data_[:, 0] * (bound["K"][1] - bound["K"][0]) + bound["K"][0]
    gridT = data_[:, 1] * (bound["T"][1] - bound["T"][0]) + bound["T"][0]
    gridr = data_[:, 2] * (bound["r"][1] - bound["r"][0]) + bound["r"][0]
    gridrho = data_[:, 3] * (bound["rho"][1] - bound["rho"][0]) + bound["rho"][0]

    # generate the grid of V0
    x = np.random.exponential(5, N)
    x = (x - x.min()) / (x.max() - x.min())
    gridV = x * (bound["V"][1] - bound["V"][0]) + bound["V"][0]

    return gridK, gridT, gridr, gridnu, gridkappa, gridV, gridrho, gridsigma


def SimulateGridFFTUniform(
    N, bound, seed=1000, save=True, name="HestonSimulationUnif.csv"
) -> pd.DataFrame:
    """
    Function to Simulate data that were drawn from Uniform distribution
    using the function GenerateGridsfromUniform. Note that in this case
    we are not doing all the possible combinations
    """
    (
        gridK,
        gridT,
        gridr,
        gridnu,
        gridkappa,
        gridV,
        gridrho,
        gridsigma,
    ) = GenerateGridsfromUniform(N, bound, seed)
    # creation of a Dataframe to store the data
    # do we store the data in a pandas dataframe ?
    values = []
    for i, V in enumerate(gridV):
        if i % 1000 == 0:
            print("To Do:   ", N - i)
        rho = gridrho[i]
        T = gridT[i]
        kappa = gridkappa[i]
        nu = gridnu[i]
        K = gridK[i]
        r = gridr[i]
        sigma = gridsigma[i]
        # they use moneyness log(S/K) but it is not stated for which values ?
        # they also have dividends should be fine to set them to zero and have a parameter less
        # storing the values in df can be slow to it is better to save the data
        P = Call_Heston(K, T, r, nu, kappa, sigma, rho, 1, V)
        values.append((P, rho, sigma, kappa, nu, r, T, K, V))

    df = pd.DataFrame(
        values, columns=["Price", "rho", "sigma", "kappa", "nu", "r", "T", "K", "V"]
    )
    if save:
        df.to_csv(name)
    return df


def SimulateGridFFTCombination(
    gridV,
    gridK,
    gridT,
    gridr,
    gridnu,
    gridkappa,
    gridsigma,
    gridrho,
    save=True,
    name="HestonSimulationComb.csv",
) -> pd.DataFrame:
    """
    Function to simulate price from linspaces calculating the price
    for all the combinations of the parameters
    """

    # creation of a Dataframe to store the data
    # do we store the data in a pandas dataframe ?
    values = []
    for nu in gridnu:
        for kappa in gridkappa:
            for sigma in gridsigma:
                # Feller Condition
                if sigma**2 <= 2 * nu * kappa:
                    for V in gridV:
                        for K in gridK:
                            for T in gridT:
                                for r in gridr:
                                    for rho in gridrho:
                                        # they use moneyness log(S/K) but it is not stated for which values ?
                                        # they also have dividends should be fine to set them to zero and have a parameter less
                                        # storing the values in df can be slow to it is better to save the data
                                        P = Call_Heston(
                                            K, T, r, nu, kappa, sigma, rho, 1, V
                                        )
                                        # P=0
                                        values.append(
                                            (P, rho, sigma, kappa, nu, r, T, K, V)
                                        )

    df = pd.DataFrame(
        values, columns=["Price", "rho", "sigma", "kappa", "nu", "r", "T", "K", "V"]
    )
    if save:
        df.to_csv(name)
    return df





def TimeHestonFFT(df):
    K = df["K"].to_list()
    T = df["T"].to_list()
    r = df["r"].to_list()
    nu = df["nu"].to_list()
    kappa = df["kappa"].to_list()
    sigma = df["sigma"].to_list()
    rho = df["rho"].to_list()
    V = df["V"].to_list()
    start = time.time()
    price = []
    for j in range(len(V)):
        price.append(
            Call_Heston(K[j], T[j], r[j], nu[j], kappa[j], sigma[j], rho[j], 1, V[j])
        )
    end = time.time()
    return end - start

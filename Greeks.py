from HestonFFT import Call_Heston
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
import scipy.integrate
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
import pandas as pd
import time
import torch.nn as nn


def derivativeNN(model, x, name, names, diff, h=1e-10):
    """

    Input:

        model : nn.Module
               pytorch neural network model

        x : torch tensor
           tensor of point in which we need to calculate the derivative

        name : str
             name of the greek to calculate

        names : dict
              dictionary containing for each greek the index corresponding to the column of x containing the feature connected

        diff : str
             string that specifies which finite difference method to use

        h : float
           parameter of the finite difference method

    Output:

         torch tensor containing the derivative


    """

    j = names[name]
    if diff == "forward":
        # forward difference
        x_dt = torch.clone(x)
        x_dt[:, j] += h
        p, p_dt = model(x), model(x_dt)
        sens = (1 / h) * (p_dt - p)

    elif diff == "backward":
        x_dt = torch.clone(x)
        x_dt[:, j] -= h
        p, p_dt = model(x), model(x_dt)
        sens = (1 / h) * (-p_dt + p)

    elif diff == "doublecentral":
        x_dplus = torch.clone(x)
        x_dminus = torch.clone(x)
        x_plus = torch.clone(x)
        x_minus = torch.clone(x)
        x_dplus[:, j] += 2 * h
        x_dminus[:, j] -= 2 * h
        x_plus[:, j] += h
        x_minus[:, j] -= h
        sens = (
            -model(x_dplus) + 8 * model(x_plus) - 8 * model(x_minus) + model(x_dminus)
        ) / (12 * h)

    elif diff == "central":
        x_plus = torch.clone(x)
        x_minus = torch.clone(x)
        x_plus[:, j] += h
        x_minus[:, j] -= h
        sens = (model(x_plus) - model(x_minus)) / (2 * h)

    if name == "delta":
        factor = (-x[:, j]).reshape((-1, 1))
        return factor * sens + model(x)

    if name == "theta":
        return -sens

    elif name == "rho":
        return sens

    elif name in ["vega", "vegalt"]:
        factor = (2 * x[:, j] ** 0.5).reshape((-1, 1))
        return sens * factor


def derivativeGPR(model, X, Xtrain, name, names):
    """

    Input:

        model : sklearn model
               Gaussian process regression model

        X : numpy array
           array of point in which we need to calculate the derivative

        Xtrain : numpy array
               array containing the datasets used to train the corresponding model

        name : str
             name of the greek to calculate

        names : dict
              dictionary containing for each greek the index corresponding to the column of x containing the feature connected


    Output:

         numpy array containing the derivative


    """
    j = names[name]
    l = model.kernel_.length_scale
    RBF = k.RBF(length_scale=l)
    alpha = model.alpha_
    KS = RBF(X, Xtrain)
    # take the vector at which we are interested in
    x = X[:, j].reshape(-1, 1)
    xtrain = Xtrain[:, j].reshape(-1, 1)

    if name in ["vega", "vegalt"]:
        GradKS = (2 * x**0.5) * (xtrain.T - x) * (KS / l**2)
        return (GradKS @ alpha.reshape(-1, 1)).reshape(-1)

    elif name == "delta":
        GradKS = -x * (xtrain.T - x) * (KS / l**2)
        return (
            GradKS @ alpha.reshape(-1, 1) + model.predict(X).reshape(-1, 1)
        ).reshape(-1)

    elif name == "rho":
        GradKS = (xtrain.T - x) * (KS / l**2)
        return (GradKS @ alpha.reshape(-1, 1)).reshape(-1)

    elif name == "theta":
        GradKS = -(xtrain.T - x) * (KS / l**2)
        return (GradKS @ alpha.reshape(-1, 1)).reshape(-1)


def derivativeRF(model, x, name, names, diff, h=1e-10):
    """

    Input:

        model : sklearn model
                random forest model

        x : numpy array
           array of point in which we need to calculate the derivative

        name : str
             name of the greek to calculate

        names : dict
              dictionary containing for each greek the index corresponding to the column of x containing the feature connected

        diff : str
             string that specifies which finite difference method to use

        h : float
           parameter of the finite difference method

    Output:

         numpy array containing the derivative


    """
    j = names[name]
    if diff == "forward":
        # forward difference
        x_dt = x.copy(x)
        x_dt[:, j] += h
        p, p_dt = model.predict(x), model.predict(x_dt)
        sens = (1 / h) * (p_dt - p)

    elif diff == "backward":
        x_dt = np.copy(x)
        x_dt[:, j] -= h
        p, p_dt = model.predict(x), model.predict(x_dt)
        sens = (1 / h) * (-p_dt + p)

    elif diff == "doublecentral":
        x_dplus = np.copy(x)
        x_dminus = np.copy(x)
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_dplus[:, j] += 2 * h
        x_dminus[:, j] -= 2 * h
        x_plus[:, j] += h
        x_minus[:, j] -= h
        sens = (
            -model.predict(x_dplus)
            + 8 * model.predict(x_plus)
            - 8 * model.predict(x_minus)
            + model.predict(x_dminus)
        ) / (12 * h)

    elif diff == "central":
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[:, j] += h
        x_minus[:, j] -= h
        sens = (model.predict(x_plus) - model.predict(x_minus)) / (2 * h)
        print(sens.shape)

    if name == "delta":
        factor = (-x[:, j]).reshape((-1, 1))
        return factor.reshape(-1) * sens + model.predict(x).reshape(-1)

    if name == "theta":
        return -sens

    elif name == "rho":
        return sens

    elif name in ["vega", "vegalt"]:
        factor = (2 * x[:, j] ** 0.5).reshape((-1, 1))
        return sens * factor.reshape(-1)

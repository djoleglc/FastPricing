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

import pandas as pd
import time
from joblib import dump
import numpy as np
import joblib
import multiprocessing
from TestPerformance import test_performanceGPR


def fitGreeksGPR(
    greek_df, price_df, model, name_model=None, save=False, time_=True, restart=2
):
    """
    Function to fit a Parallel Gaussian Process Regressor given a Pandas dataframe containing in the first column
    the price to learn, and in the remaining columns the feature used to learn the pricing function
    Note that both greeks ans prices dataframe need to be pandas dataframe

    Inputs:

        greek_df : string variable describing the name of the csv containing the greeks to use to help
                 the training phase ( in the following order "theta", "delta", "vega", "vegalt", "rho" )

        price_df : str
                 variable describing the name of the csv containing prices (first columns) and features

        model : GreeksGPR model

        save : bool
              variable that describes if the model need to be fitted

        name_model : str
                   variable describing the how the model need to be saved, to use without .joblib extension.
                   needed only when save = True

       time_ : bool
             variable describing if it needed to return the fitting time

    Output:

        self : GreeksGPR model

    """
    greeks = pd.read_csv(greek_df, header=None)
    greeks.columns = ["theta", "delta", "vega", "vegalt", "rho"]

    if model.__class__.__name__ == "DeltaGPR":
        greek_variable = greeks.delta
    elif model.__class__.__name__ == "RhoGPR":
        greek_variable = greeks.rho
    else:
        raise Exception("Model not supported")

    df = pd.read_csv(price_df).to_numpy()[:, 1:]
    X = df[:, 1:]
    price, greek_variable = df[:, 0], greek_variable.to_numpy()

    if model.__class__.__name__ == "DeltaGPR":
        s = time.time()
        model.fit(X, price, greek_variable - price, times=restart)
        e = time.time()
    if model.__class__.__name__ == "RhoGPR":
        s = time.time()
        model.fit(X, price, greek_variable, times=restart)
        e = time.time()

    if time_:
        print(f"Fitting Time:  {e-s}")
    if save:
        dump(model, name_model)
    return model


class cv_alpha_GreeksGPR:
    """

    Class that define an validation procedure

    """

    def __init__(self):
        self.mae = []
        self.aae = []
        self.bestMAE = None
        self.bestAAE = None

    def fit(self, alphas, class_, greeks_dataset, price_dataset, validation_dataset):
        """
        Function to fit the model for each alpha

        Inputs:

            alphas : list or numpy array
                    contains the values of alpha to use in the model

            class_ : class
                    class of a GreeksGPR model

            greeks_dataset : pd.DataFrame

            price_dataset : pd.DataFrame

            validation_dataset : pd.DataFrame

        Output:

            cv_alpha_GreeksGPR object

        """

        for i, alpha_ in enumerate(alphas):
            mod = class_(alpha=alpha_)
            mod = fitGreeksGPR(
                greek_df=greeks_dataset,
                price_df=price_dataset,
                model=mod,
                save=False,
                time_=False,
                restart=2,
            )

            # performance of the model
            print(f"alpha:   {alpha_}")
            print(validation_dataset)
            df = pd.read_csv(validation_dataset).to_numpy()[:, 1:]
            X = df[:, 1:]
            y = df[:, 0]
            aae_, mae_ = test_performanceGPR(X, y, mod, type_="both", to_return=True)
            self.aae.append(aae_)
            self.mae.append(mae_)

            if i > 0:
                if mae_ <= np.min(self.mae):
                    self.bestMAE = mod
                if aae_ <= np.min(self.aae):
                    self.bestAAE = mod
            else:
                self.bestMAE = mod
                self.bestAAE = mod
            print("\n")

        return self

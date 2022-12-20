import pandas as pd
import time
from joblib import dump
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
from optim import optim
from TestPerformance import test_performanceGPR


def fitGPR(name, model, name_model=None, save=True, time_=False):
    """
    Function to fit a Parallel Gaussian Process Regressor given a Pandas dataframe containing in the first column
    the price to learn, and in the remaining columns the feature used to learn the pricing function

    Inputs:

        name : str
            variable describing the name of the dataset to use as training sample.

        model : sklearn model

        number_models : int
                       variable that specifies how many models need to be fitted.
        save : bool
             variable that describes if the model need to be fitted

        name_model: str
                  variable describing the how the model need to be saved, to use without .joblib extension.
                  needed only when save = True

       time_ : bool
              variable describing if it needed to return the fitting time

    Output:

        self : sklearn model


    """
    df = pd.read_csv(name).to_numpy()[:, 1:]
    X = df[:, 1:]
    y = df[:, 0]
    s = time.time()
    model.fit(X, y)
    e = time.time()
    if time_:
        print(f"Fitting Time:  {e-s}")
    if save:
        dump(model, name_model)
    return model


class cv_alpha_GPR:

    """

    Class that define an validation procedure

    """

    def __init__(self):
        self.mae = []
        self.aae = []
        self.bestMAE = None
        self.bestAAE = None

    def fit(self, alphas, price_dataset, validation_dataset):
        """
        Function to fit the model for each alpha

        Inputs:

            alphas : list or numpy array
                    contains the values of alpha to use in the model

            price_dataset : pd.DataFrame

            validation_dataset : pd.DataFrame

        Output:

            cv_alpha_GPR object

        """

        for i, alpha_ in enumerate(alphas):
            kernel = k.RBF()
            mod = GaussianProcessRegressor(
                kernel=kernel, optimizer=optim, alpha=alpha_, random_state=10
            )
            mod = fitGPR(
                name=price_dataset,
                model=mod,
                save=False,
                time_=True,
            )

            # performance of the model
            print(f"alpha:   {alpha_}")
            print(validation_dataset)
            df = pd.read_csv(validation_dataset).to_numpy()[:, 1:]
            X = df[:, 1:]
            y = df[:, 0]
            aae_, mae_ = test_performanceGPR(X, y, mod, type_="both", to_return=True)
            self.mae.append(mae_)
            self.aae.append(aae_)

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

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
import numpy as np
from joblib import dump, load
import joblib
from joblib import Parallel, delayed
import scipy
import math
import pandas as pd


class ParallelGPR:
    """

    class that defines a ParallelGPR object

    """

    def __init__(self, N_Model):
        """

        Inputs:

            N_model : int
                    number of models to fit

        """
        self.N_Model = N_Model

    def optim(self, obj_func, initial_theta, bounds):
        """
        Inputs:

            obj_func : function
                     function to minimize

            initial_theta : float or list of float

            bounds : list of tuples

        Outputs:

            theta_opt : float

            func_min : float


        """

        optimResult = scipy.optimize.minimize(
            obj_func, initial_theta, method="COBYLA", jac=False
        )
        theta_opt = optimResult.x
        func_min = optimResult.fun
        return theta_opt, func_min

    def functionGPR(self, xy):
        """

        Inputs:

            xy : list or tuple of numpy array
                tuple or list containing the y array and the X matrix

        Output:

            gprM : sklearn model


        """
        x, y = xy
        kernel = k.RBF()
        gprM = GaussianProcessRegressor(
            kernel=kernel, optimizer=self.optim, alpha=1e-12, random_state=10
        ).fit(x, y)
        return gprM

    def predictGPR(self, xm):
        """
        Inputs:

            xy : list or tuple of numpy array
                tuple or list containing the x matrix and a sklearn model

        Output:

            m.predict(x) : numpy array
                         array of predicted values computed usinf the model

        """
        x, m = xm
        return m.predict(x)

    def GetWeight(self, y, X):
        """
        Inputs:

            y : numpy array
               contains the true values

            X : numpy array
               contains the predicted values obtained through a model


        Outputs:

            w : numpy array
               weight calculated on MAE loss

        """
        MAE = X - y.reshape(-1, 1)
        mae_mean = np.mean(MAE, 0) ** (-1)
        w = (mae_mean / mae_mean.sum()).reshape(-1, 1)
        return w

    def fit(self, X, y):
        """
        Inputs:

            X : numpy array
              matrix containing the features

            y : numpy array
              array containing the true prices

        """
        Xs = np.split(X, self.N_Model)
        ys = np.split(y, self.N_Model)
        # fitting the models
        self.mod = Parallel(n_jobs=-2, verbose=1)(
            delayed(self.functionGPR)(xy) for xy in zip(Xs, ys)
        )

        # now we want to fit also some weights
        predicted = np.empty((X.shape[0], self.N_Model))
        for i, m in enumerate(self.mod):
            pred_m = Parallel(n_jobs=-2)(delayed(self.predictGPR)([x, m]) for x in Xs)
            pred_m = np.concatenate(pred_m, axis=0)
            predicted[:, i] = pred_m
        self.w = self.GetWeight(y, predicted)

    def predict(self, X):
        """
        Inputs:

            X : numpy array
              matrix containing feautures corresponding to the price to predict

        Outputs:

            pred_w : numpy array
                    predicted prices

        """
        pred = Parallel(n_jobs=-2)(delayed(self.predictGPR)([X, m]) for m in self.mod)
        pred = np.array(pred).T
        pred_w = pred @ self.w
        return pred_w

    def save(self, name):
        """
        Inputs:

            name : str
                  name of the file to save

        """
        tosave = [self.mod, self.w]
        dump(tosave, f"{name}.joblib")

    def load(self, name):
        """
        Inputs:

            name : str
                 name of the file to load

        """
        self.mod, self.w = load(f"{name}.joblib")


def fitParallelGPR(
    name, number_models=40, name_model="Parallel", save=True, time_=True
):
    """
    Function to fit a Parallel Gaussian Process Regressor given a Pandas dataframe containing in the first column
    the price to learn, and in the remaining columns the feature used to learn the pricing function

    Inputs:

        name : str
             variable describing the name of the dataset to use as training sample.

        number_models : int
                      variable that specifies how many models need to be fitted.

        save : bool
             variable that describes if the model need to be fitted


        name_model : str
                    variable describing the how the model need to be saved, to use without .joblib extension.
                    needed only when save = True

    Outputs:

        m : ParalleGPR

    """
    df = pd.read_csv(name).to_numpy()[:, 1:]
    X = df[:, 1:]
    y = df[:, 0]
    m = ParallelGPR(number_models)
    s = time.time()
    m.fit(X, y)
    e = time.time()
    if time_:
        print(f"Time:  {e-s}")
    if save:
        m.save(name_model)
    return m

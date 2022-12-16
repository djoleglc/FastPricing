from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
import numpy as np
from joblib import dump, load
import joblib
from joblib import Parallel, delayed
import scipy
import math 
import pandas as pd
import time 


class ParallelGPR:
    def __init__(self, N_Model):
        self.N_Model = N_Model
        

    def optim(self, obj_func, initial_theta, bounds):
        optimResult = scipy.optimize.minimize(
            obj_func, initial_theta, method="COBYLA", jac=False
        )
        theta_opt = optimResult.x
        func_min = optimResult.fun
        return theta_opt, func_min

    def functionGPR(self, xy):
        x, y = xy
        kernel = k.RBF()
        gprM = GaussianProcessRegressor(
            kernel=kernel, optimizer=self.optim, alpha=1e-12, random_state=10
        ).fit(x, y)
        return gprM

    def predictGPR(self, xm):
        x, m = xm
        return m.predict(x)

    def GetWeight(self, y, X):
        MAE = X - y.reshape(-1, 1)
        mae_mean = np.mean(MAE, 0) ** (-1)
        return (mae_mean / mae_mean.sum()).reshape(-1, 1)

    def fit(self, X, y):
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
        pred = Parallel(n_jobs=-2)(delayed(self.predictGPR)([X, m]) for m in self.mod)
        pred = np.array(pred).T
        return pred @ self.w

    def save(self, name):
        tosave = [self.mod, self.w]
        dump(tosave, f"{name}.joblib")

    def load(self, name):
        self.mod, self.w = load(f"{name}.joblib")
        

def fitParallelGPR(name, number_models = 40, name_model = "Parallel", save = True, time_ = True):
    """
    Function to fit a Parallel Gaussian Process Regressor given a Pandas dataframe containing in the first column 
    the price to learn, and in the remaining columns the feature used to learn the pricing function 
    - name: string variable describing the name of the dataset to use as training sample. 
    - number_models: int variable that specifies how many models need to be fitted. 
    - save: boolean variable that describes if the model need to be fitted
    - name_model: string variable describing the how the model need to be saved, to use without .joblib extension.
                  needed only when save = True
    -time_ : boolean variable describing if it needed to return the fitting time 
    """
    df = pd.read_csv(name).to_numpy()[:,1:]
    X = df[:,1:]
    y = df[:,0]
    m = ParallelGPR(number_models)
    s = time.time()
    m.fit(X,y)
    e = time.time()
    if time_:
        print(f"Fitting Time:  {e-s}\n")
    if save:
        m.save(name_model)
    return m 

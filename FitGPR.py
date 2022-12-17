import pandas as pd
import time
from joblib import dump
import numpy as np


def fitGPR(name, model, name_model=None, save=True, time_=False):
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

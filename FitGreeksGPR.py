import pandas as pd
import time
from joblib import dump
import numpy as np
import joblib
import multiprocessing

def fitGreeksGPR(
    greek_df, price_df, model, name_model=None, save=False, time_=True, restart=2
):
    """
    Function to fit a Parallel Gaussian Process Regressor given a Pandas dataframe containing in the first column
    the price to learn, and in the remaining columns the feature used to learn the pricing function
    Note that both greeks ans prices dataframe need to be pandas dataframe 
    
    - greek_df: string variable describing the name of the csv containing the greeks to use to help 
                the training phase ( in the following order "theta", "delta", "vega", "vegalt", "rho" )
    - price_df: string variable describing the name of the csv containing prices (first columns) and features
    - number_models: int variable that specifies how many models need to be fitted.
    - save: boolean variable that describes if the model need to be fitted
    - name_model: string variable describing the how the model need to be saved, to use without .joblib extension.
                  needed only when save = True
    -time_ : boolean variable describing if it needed to return the fitting time
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
    price, greek_variable = df[:, 0],  greek_variable.to_numpy()

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

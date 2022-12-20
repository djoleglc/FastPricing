import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_errorNN(df, model, variable, title = None):
    """
    Function used to visualize error of a Neural Network

        df : pd.DataFrame

        model : nn.Module
               pytorch neural network model

        variable : list of str
                 list of the name of variables for which we need to visualize the error


    """
    df_ = df.to_numpy()[:, 1:]
    X = df_[:, 1:]
    y = df_[:, 0]

    X = torch.from_numpy(X).double()
    prediction = model(X).detach().numpy()
    prediction = prediction.flatten()
    y = y.flatten()
    error = np.absolute(prediction - y)

    for var in variable:
        var_array = df.loc[:, var].to_numpy()
        plt.scatter(var_array, error, s=3)
        plt.xlabel(var)
        plt.ylabel("Absolute Error")
        if title == None:
            plt.title(var)
        else: 
            plt.title(title)
        plt.savefig(f"NeuralNet - {var}")
        plt.show()


def visualize_errorGPR(df, model, variable, title = None):
    """
    Function used to visualize error of a GPR model or of sklearn model 

        df : pd.DataFrame

        model : sklearn model or GreeksGPR model

        variable : list of str
                 list of the name of variables for which we need to visualize the error


    """

    df_ = df.to_numpy()[:, 1:]
    X = df_[:, 1:]
    y = df_[:, 0]

    prediction = model.predict(X)
    prediction = prediction.flatten()
    y = y.flatten()
    error = np.absolute(prediction - y)

    for var in variable:
        var_array = df.loc[:, var].to_numpy()
        plt.scatter(var_array, error, s=3)
        plt.xlabel(var)
        plt.ylabel("Absolute Error")
        if title == None:
            plt.title(var)
        else: 
            plt.title(title)
        plt.savefig(f"GPR - {var}")
        plt.show()

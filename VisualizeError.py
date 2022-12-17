import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_errorNN(df, model, variable):
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
        plt.title(var)
        plt.savefig(f"NeuralNet - {var}")
        plt.show()


def visualize_errorGPR(df, model, variable):

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
        plt.title(var)
        plt.savefig(f"GPR - {var}")
        plt.show()

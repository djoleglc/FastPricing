import scipy.optimize
import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cholesky
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k


class DeltaGPR:
    def __init__(self, alpha):
        """
        Inputs:

            alpha : float
                   value to set as default alpha (regularization parameter)

        """
        self.alpha = alpha

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
        opt = {}
        opt["maxiter"] = 5000
        optimResult = scipy.optimize.minimize(
            obj_func,
            initial_theta,
            tol=1e-20,
            method="COBYLA",
            bounds=bounds,
            jac=False,
            options=opt,
        )
        theta_opt = optimResult.x
        func_min = optimResult.fun
        return theta_opt, func_min

    def kernel(self, l, X, index_k):
        """
        Function used to calculate the kernel of the GreeksGPR

        Inputs:

            l : float
               lenght scale parameter

            X  : numpy array
                matrix of features

            index_k : int
                     index of the feature corresponding to the greek in the X matrix

        Output:

            result : numpy array

        """
        n = X.shape[0]
        RBF = k.RBF(length_scale=l)
        K_price = RBF(X)
        x = X[:, index_k].reshape(-1, 1)
        M = x.T - x
        cov_delta_price = -x * M * (K_price / l**2)
        var_delta = x * x.T * (K_price / l**2) * (1 - (M / l) ** 2)

        # build the resulting matrix
        result = np.empty((2 * n, 2 * n))
        result[0:n, 0:n] = K_price
        result[n:, 0:n] = cov_delta_price
        result[0:n, n:] = cov_delta_price.T
        result[n:, n:] = var_delta
        return result

    def loglikeGPR(self, X, y, sigma, K):
        """
        function to calculate the loglike to maximize

        Inputs:

            X  : numpy array
                matrix of features

            y : numpy array
               true values of prices

            sigma : float
                   regularization parameter

            K : numpy array
               kernel matrix

        Output:

            result : float

        """
        M = K + np.eye(K.shape[0]) * sigma
        L = cholesky(M, lower=True)
        alpha = cho_solve((L, True), y)
        LogDet = np.log(np.diag(L)).sum()
        loglike = -0.5 * y.T @ alpha - LogDet - K.shape[0] / 2 * np.log(2 * np.pi)
        result = loglike.sum(axis=-1)
        return result

    def Loglike(self, X, price, delta, l, sigma):
        """
        Inputs:

            X  : numpy array
                matrix of features

            price : numpy array
               true values of prices

            delta : numpy array
                   values of the greeks to use to train the model

            l : float
               length scale parameter

            sigma : float
                   regularization parameter

        Output:

            loglikelihood : float


        """

        K = self.kernel(l, X, 6)
        loglikelihood = self.loglikeGPR(np.r_[X, X], np.r_[price, delta], sigma, K)
        return loglikelihood

    def optimization(self, initial):
        """
        Inputs:

            initial : float
                    initial value to use for the optimization

        Output :

            opt : scipy minimization object

        """
        sigma = self.alpha
        X = self.Xtrain
        price = self.price
        delta = self.delta
        tominimize = lambda l: -self.Loglike(X, price, delta, l, sigma)
        opt = scipy.optimize.minimize(
            tominimize, initial, bounds=[(1e-5, 1e5)], method="L-BFGS-B"
        )
        return opt

    def fit(self, Xtrain, price, delta, times=2):
        """

        Xtrain : numpy array
                matrix of feature for the training

        price : numpy array
               prices to use for the training

        delta : numpy array
               greeks to use for the training

        times : int
               number of times used to repeat the optimization

        """
        self.Xtrain = Xtrain
        self.price = price.reshape(-1, 1)
        self.delta = delta.reshape(-1, 1)
        self.y = np.r_[self.price, self.delta]
        initials = np.random.uniform(1e-6, 3, size=(times, 1))
        optimals = Parallel(n_jobs=4)(
            delayed(self.optimization)(init) for init in initials
        )
        # selecting the best parameter
        minimum = np.inf
        index = 0
        for j, opt in enumerate(optimals):
            fun = opt.fun.item()
            if fun <= minimum:
                minimum = fun
                index = j
        self.length_scale = np.absolute(optimals[index].x.item())
        self.kernel_ = self.kernel(self.length_scale, self.Xtrain, 6)
        self.L = cholesky(
            self.kernel_ + self.alpha * np.eye(self.kernel_.shape[0]),
            lower=True,
            check_finite=False,
        )
        self.alpha_ = cho_solve((self.L, True), self.y)

    def predict(self, Xtest):
        """
        Inputs:

            Xtest : numpy array
                   matrix of feature for which estimate the price

        Output:

            predicted : numpy array
                       prices predicted by the model

        """
        kern = k.RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
        gprM = GaussianProcessRegressor(
            kernel=kern, alpha=self.alpha, optimizer=self.optim
        ).fit(self.Xtrain, self.price)
        predicted = gprM.predict(Xtest)
        return predicted


class RhoGPR:
    def __init__(self, alpha):
        """
        Inputs:

            alpha : float
                   value to set as default alpha (regularization parameter)

        """
        self.alpha = alpha

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

        opt = {}
        opt["maxiter"] = 5000
        optimResult = scipy.optimize.minimize(
            obj_func,
            initial_theta,
            tol=1e-20,
            method="COBYLA",
            bounds=bounds,
            jac=False,
            options=opt,
        )
        theta_opt = optimResult.x
        func_min = optimResult.fun
        return theta_opt, func_min

    def kernel(self, l, X, index_k):
        """
        Function used to calculate the kernel of the GreeksGPR

        Inputs:

            l : float
               lenght scale parameter

            X  : numpy array
                matrix of features

            index_k : int
                     index of the feature corresponding to the greek in the X matrix

        Output:

            result : numpy array

        """

        n = X.shape[0]
        RBF = k.RBF(length_scale=l)
        K_price = RBF(X)
        x = X[:, index_k].reshape(-1, 1)
        M = x.T - x
        cov_rho_price = M * (K_price / l**2)
        var_rho = K_price / l**2 * (1 - (M / l) ** 2)

        # build the resulting matrix
        result = np.empty((2 * n, 2 * n))
        result[0:n, 0:n] = K_price
        result[n:, 0:n] = cov_rho_price
        result[0:n, n:] = cov_rho_price.T
        result[n:, n:] = var_rho
        return result

    def loglikeGPR(self, X, y, sigma, K):
        """
        function to calculate the loglike to maximize

        Inputs:

            X  : numpy array
                matrix of features

            y : numpy array
               true values of prices

            sigma : float
                   regularization parameter

            K : numpy array
               kernel matrix

        Output:

            result : float

        """
        M = K + np.eye(K.shape[0]) * sigma
        L = cholesky(M, lower=True)
        alpha = cho_solve((L, True), y)
        LogDet = np.log(np.diag(L)).sum()
        loglike = -0.5 * y.T @ alpha - LogDet - K.shape[0] / 2 * np.log(2 * np.pi)
        return loglike.sum(axis=-1)

    def Loglike(self, X, price, rho, l, sigma):
        """
        Inputs:

            X  : numpy array
                matrix of features

            price : numpy array
               true values of prices

            rho : numpy array
                   values of the greeks to use to train the model

            l : float
               length scale parameter

            sigma : float
                   regularization parameter

        Output:

            loglikelihood : float


        """

        K = self.kernel(l, X, 4)
        loglikelihood = self.loglikeGPR(np.r_[X, X], np.r_[price, rho], sigma, K)
        return loglikelihood

    def optimization(self, initial):
        """
        Inputs:

            initial : float
                    initial value to use for the optimization

        Output :

            opt : scipy minimization object

        """
        sigma = self.alpha
        X = self.Xtrain
        price = self.price
        rho = self.rho
        tominimize = lambda l: -self.Loglike(X, price, rho, l, sigma)
        opt = scipy.optimize.minimize(
            tominimize, initial, bounds=[(1e-5, 1e5)], method="COBYLA"
        )
        return opt

    def fit(self, Xtrain, price, rho, index_rho=4, times=2):
        """

        Xtrain : numpy array
                matrix of feature for the training

        price : numpy array
               prices to use for the training

        rho : numpy array
               greeks to use for the training

        times : int
               number of times used to repeat the optimization

        """
        self.Xtrain = Xtrain
        self.price = price.reshape(-1, 1)
        self.rho = rho.reshape(-1, 1)
        self.y = np.r_[self.price, self.rho]
        initials = np.random.uniform(1e-6, 3, size=(times, 1))
        optimals = Parallel(n_jobs=4)(
            delayed(self.optimization)(init) for init in initials
        )

        # selecting the best parameter
        minimum = np.inf
        index = 0
        for j, opt in enumerate(optimals):
            fun = opt.fun.item()
            if fun <= minimum:
                minimum = fun
                index = j
        self.length_scale = np.absolute(optimals[index].x.item())
        self.kernel_ = self.kernel(self.length_scale, self.Xtrain, 4)
        self.L = cholesky(
            self.kernel_ + self.alpha * np.eye(self.kernel_.shape[0]),
            lower=True,
            check_finite=False,
        )
        self.alpha_ = cho_solve((self.L, True), self.y)

    def predict(self, Xtest, method="Greeks"):
        """
        Inputs:

            Xtest : numpy array
                   matrix of feature for which estimate the price

        Output:

            predicted : numpy array
                       prices predicted by the model

        """
        kern = k.RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
        gprM = GaussianProcessRegressor(
            kernel=kern, alpha=self.alpha, optimizer=self.optim
        ).fit(self.Xtrain, self.price)
        predicted = gprM.predict(Xtest)
        return predicted

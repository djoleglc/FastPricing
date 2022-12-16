import scipy.optimize
import joblib
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k



class deltaGPR:
    def __init__(self, alpha):
        self.alpha = alpha

    def optim(self, obj_func, initial_theta, bounds):
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
        """
        M = K + np.eye(K.shape[0]) * sigma
        L = cholesky(M, lower=True)
        alpha = cho_solve((L, True), y)
        LogDet = np.log(np.diag(L)).sum()
        loglike = -0.5 * y.T @ alpha - LogDet - K.shape[0] / 2 * np.log(2 * np.pi)
        return loglike.sum(axis=-1)

    def Loglike(self, X, price, delta, l, sigma):
        K = self.kernel(l, X, 6)
        loglikelihood = self.loglikeGPR(np.r_[X, X], np.r_[price, delta], sigma, K)
        return loglikelihood

    def optimization(self, initial):
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

    def predict(self, Xtest, method="Greeks"):
        if method == "Greeks":
            RBF = k.RBF(length_scale=self.length_scale)
            deltatrain = self.Xtrain[:, 6].reshape(-1, 1)
            deltatest = Xtest[:, 6].reshape(-1, 1)
            A = self.kernel_
            C = RBF(self.Xtrain, Xtest)
            D = (
                (1 / self.length_scale**2)
                * (-deltatrain)
                * (deltatest.T - deltatrain)
                * C
            )
            CD = np.r_[C, D]
            return CD.T @ self.alpha_
        if method == "Prices":
            kern = k.RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
            gprM = GaussianProcessRegressor(
                kernel=kern, alpha=self.alpha, optimizer=self.optim
            ).fit(self.Xtrain, self.price)
            return gprM.predict(Xtest)

        
        
        
class RhoGPR:
    def optim(self, obj_func, initial_theta, bounds):
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
        """
        M = K + np.eye(K.shape[0]) * sigma
        L = cholesky(M, lower=True)
        alpha = cho_solve((L, True), y)
        LogDet = np.log(np.diag(L)).sum()
        loglike = -0.5 * y.T @ alpha - LogDet - K.shape[0] / 2 * np.log(2 * np.pi)
        return loglike.sum(axis=-1)

    def Loglike(self, X, price, rho, l, sigma):
        K = self.kernel(l, X, 4)
        loglikelihood = self.loglikeGPR(np.r_[X, X], np.r_[price, rho], sigma, K)
        return loglikelihood

    def optimization(self, initial):
        sigma = self.alpha
        X = self.Xtrain
        price = self.price
        rho = self.rho
        tominimize = lambda l: -self.Loglike(X, price, rho, l, sigma)
        opt = scipy.optimize.minimize(
            tominimize, initial, bounds=[(1e-5, 1e5)], method="COBYLA"
        )
        return opt

    def fit(self, Xtrain, price, rho, index_rho=4, alpha=1e-12, times=2):
        self.alpha = alpha
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
        if method == "Greeks":
            RBF = k.RBF(length_scale=self.length_scale)
            rhotrain = self.Xtrain[:, 4].reshape(-1, 1)
            rhotest = Xtest[:, 4].reshape(-1, 1)
            A = self.kernel_
            C = RBF(self.Xtrain, Xtest)
            D = (1 / self.length_scale**2) * (rhotest.T - rhotrain) * C
            CD = np.r_[C, D]
            return CD.T @ self.alpha_
        if method == "Prices":
            kern = k.RBF(length_scale=self.length_scale, length_scale_bounds="fixed")
            gprM = GaussianProcessRegressor(
                kernel=kern, alpha=self.alpha, optimizer=self.optim
            ).fit(self.Xtrain, self.price)
            return gprM.predict(Xtest)
# linear regression with different training methods
# author: Hyung Jip Lee

# how to write a good class:
# https://towardsdatascience.com/how-to-write-awesome-python-classes-f2e1f05e51a9

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.linalg import solve_triangular
from dataclasses import dataclass, field
from typing import Optional, Union
from scipy.optimize import minimize

@dataclass
class LinearRegression:

    def _ols(self, X: np.ndarray, y: np.ndarray):

        # fitting the LR model to data
        # with ordinary least squares

        # add column of ones to compute y-intersect = Beta_0
        X = np.c_[X, np.ones(X.shape[0])]

        # equation minimizing residual sum of squares:
        # solving del RSS / del beta = 0
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat

    def _qr(self, X: np.array, y: np.array):
        q, r = np.linalg.qr(X)
        return solve_triangular(r, q.T @ y)

    def _cholesky(self, X: np.ndarray, y: np.ndarray):
        A = np.linalg.cholesky(X.T @ X)
        B = solve_triangular(A, X.T @ y, lower=True)
        return solve_triangular(A.T, B)

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "ols"):
        
        if method == "ols":
            self.weights = self._ols(X, y)
        if method == "qr":
            self.weights = self._qr(X, y)
        if method == "cholesky":
            self.weights = self._cholesky(X, y)
        return self
    
    def predict(self, X: np.array, method: str, params: Optional[np.array] = None):
        if params is None:
            return np.dot(X, self.weights)
        return np.dot(X, params)

@dataclass
class LinearRegression_MLE:

    
    theta: Optional[np.ndarray] = field(init = False, default = np.array([]))
    # specify type of  variables in class (self.something)

    def loglikelihood(self, y, yhat):
        err = y - yhat
        return 0.5 * (err ** 2).sum()

    def obj_func(self, yhat, X, y):
        yguess = self.predict(X, thetas = yhat)
        return self.loglikelihood(y=y, yhat=yguess)

    def jacobian(self, yhat, X, y):
        return X.T @ (yhat @ X.T - y)

    def fit(self, X: np.array, y: np.array, method: str):
        # random guess
        rg = np.random.RandomState(1)
        param_guess = rg.uniform(low=0, high=10, size=X.shape[1])
        
        # below code will depend on method
        # mle bfgs or mle newton cg
        self.theta = minimize(
            self.obj_func,
            param_guess,
            jac=self.jacobian,
            method="BFGS",
            options={"disp": True},
            args=(X,y)
        )
        return self

    def predict(self, X, thetas):
        return X @ thetas

@dataclass
class RegressionMetrics:
    model: Union[LinearRegression, LinearRegression_MLE]
    X: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    
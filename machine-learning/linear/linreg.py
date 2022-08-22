# linear regression with different training methods
# author: Hyung Jip Lee

# how to write a good class:
# https://towardsdatascience.com/how-to-write-awesome-python-classes-f2e1f05e51a9

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.linalg import solve_triangular
from dataclasses import dataclass, field
from typing import Optional, Union
from scipy.optimize import minimize

# dataclass just allows you to write less code
# self.bla -> bla: type hint

# field is to give additional information

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
    predictions: Optional[np.ndarray] = field(init=False, default=np.array([]))
    residuals: np.ndarray = field(init=False, default=np.array([]))
    rss: float = field(init=False, default=0.0) # residual sum of sq
    ess: float = field(init=False, default=0.0) # explained sum of sq
    tss: float = field(init=False, default=0.0) # total sum of sq
    r2: float = field(init=False, default=0.0)
    mse: float = field(init=False, default=0.0)
    mae: float = field(init=False, default=0.0)
    # parameter covariance?
    # param_covar = s2 * inv(X.T @ params @ X)
    # https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwwwbrr.cr.usgs.gov%2Fhill_tiedeman_book%2Fpresentation%2520files%2F07a-ParamStats.ppt&wdOrigin=BROWSELINK

    def __post_init__(self) -> None:
        # n: number of samples
        # p: number of features
        n, p = self.X.shape[0], self.X.shape[1]
        # baseline y:
        ybar = self.y.mean()

        # getattr 오브젝트의 attribute 값을 리턴
        # getattr(c, 'x') <=> c.x
        self.predictions = getattr(self.obj, "predict")(self.X)
        self.residuals = self.y - self.predictions
        self.rss = self.residuals @ self.residuals
        self.tss = ((self.y - ybar) @ (self.y - ybar))
        self.ess = self.tss - self.rss
        self.r2 = self.ess / self.tss
        self.mse = self.rss / n
        self.mae = abs(self.residuals) / n
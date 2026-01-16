# linear regression with different training methods
# author: Hyung Jip Lee

# how to write a good class:
# https://towardsdatascience.com/how-to-write-awesome-python-classes-f2e1f05e51a9

#from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.linalg import solve_triangular
from dataclasses import dataclass, field
from typing import Optional, Union
from scipy.optimize import minimize
from abc import ABCMeta, abstractmethod

# dataclass just allows you to write less code
# self.bla -> bla: type hint

# field is to give additional information

class LinearABC(metaclass = ABCMeta):
    """
    @abstractmethod
    def fit(self):
        pass
    """
    @abstractmethod
    def predict(self):
        pass

"""
@dataclass
class RegressionMetrics:
    #model: Union[LinearRegression, LinearRegression_MLE]
    X: np.ndarray #= field(init=False, default_factory=np.array([]))
    y: np.ndarray #= field(init=False, default_factory=np.array([]))
    #theta: np.ndarray
    predictions: Optional[np.ndarray] #= field(init=False, default_factory=np.array([]))
    residuals: np.ndarray #= field(init=False, default_factory=np.array([]))
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
        self.predictions = getattr(self.obj, "predict")(self.X) # to do this abstract class is needed
        self.residuals = self.y - self.predictions
        self.rss = self.residuals @ self.residuals
        self.tss = ((self.y - ybar) @ (self.y - ybar))
        self.ess = self.tss - self.rss
        self.r2 = self.ess / self.tss
        self.mse = self.rss / n
        self.mae = abs(self.residuals) / n
"""

@dataclass
class LinearRegression(object):
    theta = np.array([])
    #theta: np.array = field(init=False, default_factory=np.array([]))
    #metrics: Optional[RegressionMetrics] = field(init=False)
    metrics: Optional[dict] = field(init=False)

    def _ols(self, X: np.ndarray, y: np.ndarray) -> np.array:

        # fitting the LR model to data
        # with ordinary least squares

        # add column of ones to compute y-intersect = Beta_0
        X = np.c_[X, np.ones(X.shape[0])]

        # equation minimizing residual sum of squares:
        # solving del RSS / del beta = 0
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat

    def _qr(self, X: np.array, y: np.array) -> np.array:
        q, r = np.linalg.qr(X)
        return solve_triangular(r, q.T @ y)

    def _cholesky(self, X: np.ndarray, y: np.ndarray) -> np.array:
        A = np.linalg.cholesky(X.T @ X)
        B = solve_triangular(A, X.T @ y, lower=True)
        return solve_triangular(A.T, B)

    def get_metrics(self, X: np.array, y: np.array) -> dict:
        X = X.reshape(-1,1)
        n, p = X.shape[0], X.shape[1]
        ybar = y.mean()
        self.predictions = self.predict(X)
        self.residuals = y - self.predictions
        self.rss = self.residuals.T @ self.residuals
        self.tss = (y-ybar).T@(y-ybar)
        self.ess = self.tss - self.rss
        self.r2 = self.ess/self.tss
        self.mse = self.rss/n
        self.mae = sum(abs(self.residuals))/n
        
        # prints anova table:
        # anova table style from: Regression Modeling with Actuarial and Financial Applications by Edward W. Frees
        print("\n")
        print("-------------------------------------------------------")
        print("                      ANOVA table                     ")
        print("-------------------------------------------------------")
        print("|  source  |   sum of squares  |  df  |  mean square  |")
        print("-------------------------------------------------------")
        print("| residual |  ", self.rss[0][0], "   |  ", len(self.theta), "  |  ", self.mae[0], "  |")
        print("|  error   |  ", self.ess[0][0], "   |  ", n - len(self.theta) - 1, "  |  ", self.mse[0][0], "  |")
        print("|  total   |  ", self.tss[0][0], "  |  ", n - 1, " |                      |")
        

        return {"r2": self.r2, "mse": self.mse, "mae": self.mae}

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "ols", get_metric: bool = False):
        
        if method == "ols":
            self.theta = self._ols(X, y)
        elif method == "qr":
            self.theta = self._qr(X, y)
        elif method == "cholesky":
            self.theta = self._cholesky(X, y)
        
        if get_metric:
            self.metrics = self.get_metrics(X, y)

        return self
    
    def predict(self, X: np.array, params: Optional[np.array] = None):
        if params is None:
            return X * self.theta[0] + self.theta[1]
        return X * params[0] + params[1]

@dataclass
class LinearRegression_MLE(object):

    theta: Optional[np.ndarray] = field(init = False, default_factory = np.array([]))
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
"""
def compute_metrics(
    #model: Union[LinearRegression, LinearRegression_MLE],
    X: np.ndarray, y: np.ndarray
    ) -> RegressionMetrics:
    return RegressionMetrics(X, y)
    #return RegressionMetrics(model, X, y, model.theta)
"""
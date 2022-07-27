# linear regression with different training methods
# author: Hyung Jip Lee

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.linalg import solve_triangular

class LinearRegression:

    def _ols(self, X: np.array, y: np.array):

        # fitting the LR model to data

        # add column of ones to compute y-intersect = Beta_0
        X = np.c_[X, np.ones(X.shape[0])]

        # equation minimizing residual sum of squares:
        # solving del RSS / del beta = 0
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat

    def _qr(self, X: np.array, y: np.array):
        q, r = np.linalg.qr(X)
        return solve_triangular(r, q.T @ y)

    def _mle(X: np.array, y: np.array):
        
        # other way to fit model

        return

    def fit(self, X: np.array, y: np.array, method: str = "ols"):
        
        if method == "ols":
            self.weights = self._ols(X, y)
        if method == "qr":
            self.weights = self._qr(X, y)
        
        return self
    
    def predict(self, X: np.array, params: Optional[np.array] = None):
        if params is None:
            return np.dot(X, self.weights)
        return np.dot(X, params)

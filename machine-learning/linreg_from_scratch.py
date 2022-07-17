import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def _ols(self, X: np.array, y: np.array):

        # fitting the LR model to data

        # add column of ones to compute y-intersect = Beta_0
        X = np.c_[X, np.ones(X.shape[0])]

        # equation minimizing residual sum of squares:
        # solving del RSS / del beta = 0
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat

    def _mle(X: np.array, y: np.array):
        
        # other way to fit model

        return

    def fit(self, X: np.array, y: np.array, method: str = "ols") -> LinearRegression:
        
        
        
        return self
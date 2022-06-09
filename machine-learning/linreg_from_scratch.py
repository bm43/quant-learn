import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, X: np.array, y: np.array) -> None:
        # fit intercept by default
        self.optimal_parameters = self.OLS(X, y)

    def OLS(self, X: np.array, y: np.array):

        # fitting the LR model to data

        # add column of ones to compute y-intersect = Beta_0
        X = np.c_[X, np.ones(X.shape[0])]

        # equation minimizing residual sum of squares:
        # solving del RSS / del beta = 0
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta_hat

    def MLE(X: np.array, y: np.array):
        
        # other way to fit model

        return
# implementing logistic regression from scratch
# author: Hyung Jip Lee, 2022

from dataclasses import dataclass, field
from typing import Union
import numpy as np

@dataclass
class LogisticRegression:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _hessian(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        return X.T @ w @ X

    def _irls(self, X: np.ndarray, y: np.ndarray, iter: int = 30) -> np.ndarray:
        
        # training implementation source:
        # https://cedar.buffalo.edu/~srihari/CSE574/Chap4/4.3.3-IRLS.pdf
        
        n = X.shape[0]
        guess = np.zeros(X.shape[1])
        w = np.zeros((n,n))
        w_inv = np.zeros((n,n))

        for _ in range(iter):
            p = self.predict(X, guess)
            np.fill_diagonal(w, p*(1-p))
            H = self._hessian(X, w)
            np.fill_diagonal(w_inv, p*(1-p))
            zbar = X @ guess + w_inv @ (y - p)
            guess = np.linalg.solve(H, X.T @ w @ zbar)

        return guess

    def fit(self,
        X: np.ndarray, y: np.ndarray, method: str = "irls", iter: int = 30):
        if method == "irls":
            self.theta = self._irls(X, y, iter=iter)
            return self

    def predict(self, X: np.ndarray, thetas: np.ndarray = None):
        if thetas is None:
            if isinstance(self.theta, np.ndarray):
                return self.sigmoid(X @ self.theta)
        return self.sigmoid(X @ thetas)
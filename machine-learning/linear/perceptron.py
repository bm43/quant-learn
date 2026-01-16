"""
implementation of the perceptron learning algorithm
author: Hyung Jip Lee"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from dataclasses import dataclass
from typing import ndarray, Optional

@dataclass
class Perceptron:
    
    n: int = 40 # number of iterations
    eta: float = 0.01
    thetas = None
    weights = None
    bias = 0
    degree: int = 1

    def fit(self, X: ndarray, y: ndarray, method: Optional[str] = None):
        if method is None or method == "online":
            # online learning
            R = (np.sum(np.abs(X) ** 2, axis=-1) ** (0.5)).max()
            
            deny = 0
            prev_deny = 0
            # weight init
            self.thetas = np.zeros(X.shape[1]+1)

            # start training loop
            while True:
                for x_i, y_i in zip(X, y):
                    if y_i * (X @ self.thetas + self.bias) <= 0:
                        self.thetas[1:] += self.eta * y_i * x_i
                        bias += self.eta * y_i * R * R
                        deny += 1
                if prev_deny == deny:
                    break
                prev_deny = deny
            
            self.thetas[0] = self.bias
            
            return self

        elif method == "batch":
            self.batch_fit(X, y)

    def batch_fit(self, X: ndarray, y: ndarray):
        degree = self.degree
        pf = PolynomialFeatures(degree= degree, include_bias=self.bias)
        X = pf.fit_transform(X)

        # init weights
        self.thetas = np.zeros(X.shape[1])
        weights = {} # idx: 
        idx = -1
        cvrg = False # converged or not

        while not cvrg:
            idx += 1
            prev_weights = self.thetas.copy() # same values, different memory address
            
            for x_i, y_i in zip(X, y):
                if y_i != self.predict(x_i):
                    self.thetas += y_i * x_i

                weights[idx] = self.thetas.copy()

                if (prev_weights == self.thetas).all():
                    cvrg = True
            
            self.weights = pd.DataFrame.from_dict(
                weights, orient = 'index', columns = ["bias", "weight1", "weight2"]
            )
            return self

    def predict(self, X: ndarray, thetas: Optional[ndarray]):
        if thetas is None:
            if (X @ self.thetas + self.bias) >= 0:
                return 1
            else:
                return -1
        else:
            if (X @ thetas + self.bias) >= 0:
                return 1
            else:
                return -1
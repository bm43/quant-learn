"""
implementation of the perceptron learning algorithm
author: Hyung Jip Lee"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import ndarray, Optional

@dataclass
class Perceptron:
    
    n: int = 40 # number of iterations
    eta: float = 0.01
    thetas = None
    weights = None
    degree: int = 1

    def fit(self, X: ndarray, y: ndarray):
        # online learning
        R = (np.sum(np.abs(X) ** 2, axis=-1) ** (0.5)).max()
        bias = 0
        deny = 0
        prev_deny = 0
        # weight init
        self.thetas = np.zeros(X.shape[1]+1)

        # start training loop
        while True:
            for x_i, y_i in zip(X, y):
                if y_i * (X @ self.thetas + bias) <= 0:
                    self.thetas[1:] += self.eta * y_i * x_i
                    bias += self.eta * y_i * R * R
                    deny += 1
            if prev_deny == deny:
                break
            prev_deny = deny
        
        self.thetas[0] = bias

        return self
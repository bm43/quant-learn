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
        
        R = (np.sum(np.abs(X) ** 2, axis=-1) ** (0.5)).max()

        # weight init
        self.thetas = np.zeros(X.shape[1]+1)

        # start training loop

        return self
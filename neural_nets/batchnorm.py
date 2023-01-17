import numpy as np
from typing import Tuple, ndarray

class BatchNormalization:
    def __init__(self, shape: Tuple, epsilon: float = 1e-8):
        self.shape = shape
        self.epsilon = epsilon
        self.gamma = np.ones(shape)
        self.beta = np.zeros(shape)
        self.running_mean = np.zeros(shape)
        self.running_var = np.zeros(shape)
    
    def forward(self, x: ndarray, is_training: bool = True) -> np.array:
        if is_training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * x_norm + self.beta
        return out
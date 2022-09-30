from dataclasses import dataclass
import numpy as np

@dataclass
class HiddenMarkovModel():
    transmission_prob: np.ndarray # (n+2) x (n+2)
    emission_prob: np.ndarray
    
    
    def fit(self):
        return self

    def predict(self):
        return self
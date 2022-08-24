"""
hamilton's regime switching model

https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf

  https://personal.eur.nl/kole/rsexample.pdf

  https://www.stata.com/features/overview/markov-switching-models/

  https://homepage.ntu.edu.tw/~ckuan/pdf/Lec-Markov_note.pdf
"""

import import random
import numpy as np

import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, field

@dataclass
class MarkovSwitch:
  
  n_regime: int = 2

  def _sigmoid(self, x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

  def _em(self, obs: np.ndarray, iter: int = 10):
    n = obs.shape[0]
    n_regime = self.n_regime


    idx = np.random.randint(low=0, high=n, size=n_regime)
    mu_k = obs[idx] # means for regimes
    p_k = np.zeros(n_regime)
    theta = [0] * iter
    sig = np.ones(1)

    for i in range(iter):
      theta[i]
  def fit(self, obs: np.ndarray, iter: int = 10):

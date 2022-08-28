"""
hamilton's regime switching model

https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf

  https://personal.eur.nl/kole/rsexample.pdf

  https://www.stata.com/features/overview/markov-switching-models/

  https://homepage.ntu.edu.tw/~ckuan/pdf/Lec-Markov_note.pdf
"""

import random
import numpy as np

import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, field
from scipy.stats import norm

@dataclass
class MarkovSwitch:
  
  trans_matrix: np.ndarray = field(init=False)
  n_regime: int = 2

  def __post_init__(self) -> None:
    self.trans_matrix = np.zeros((self.n_regime, self.n_regime))
    

  def _sigmoid(self, x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

  def _normpdf(self, obs: np.ndarray[np.float64], mean: np.ndarray[np.float64], std: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    # = f(y_t | s_t, F_{t-1})
    likelihood = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, std)
    return np.column_stack(tuple(likelihood))

  def _transition_matrix(self, p_00: float, p_11: float) -> None:
    """
    this function constructs tr matrix for binary process
    if s_t is binary process, then tr matrix is 2x2
    | p_00     1-p_11 |
    | 1-p_00    p_11  |
    (2nd paper p.2)
    """
    self.trans_matrix[0,0] = p_00
    self.trans_matrix[0,1] = 1-p_11
    self.trans_matrix[1,0] = 1-p_00
    self.trans_matrix[1,1] = p_11

  def _hamilton_filter(self, obs: np.ndarray, theta: np.ndarray):
    p = theta[:2] # first two
    means = [2:4] # third and fourth
    std = np.array([theta[-1], theta[-1]])

    # I. init
    H_filter = np.zeros((n, self.n_regime))
    n = obs.shape[0]
    pred_p = np.zeros((n,self.n_regime))

    # Transition
    self._transition_matrix(*self._sigmoid(p))

    # init guess
    H_filter[0] = np.array([0.5, 0.5])
    pred_p[1] = self.trans_matrix.T @ H_filter[0]

    # p density through time
    eta = self._normpdf(obs, means, std)

    # II. input to filter t=1,...,T-1
    for t in range(1, n-1): # for every observation
      exp = eta[t] * pred_p[t]
      H_filter[t] = exp/exp.sum()

      # predict t= 2, 3, ... T-1
      pred_p[t+1] = self.trans_matrix.T @ H_filter[t]
    
    # H_filter[T]
    exp = eta[-1] * pred_p[-1]
    H_filter[-1] = exp/exp.sum()

    return [H_filter, pred_p]
    
  def _qp(self, filter_p: np.ndarray, pred_p: np.ndarray, P: np.ndarray):
    return

  def _e_step(self, obs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # obs for observations
    # theta = initial guess, input to em algo
    H_filter, pred_p = self._hamilton_filter(obs, theta)

    return self._qp(filter_p = H_filter, pred_p = )
    return

  def _em(self, obs: np.ndarray, iter: int = 10):
    # http://www.columbia.edu/~mh2078/MachineLearningORFE/EM_Algorithm.pdf
    n = obs.shape[0]
    n_regime = self.n_regime


    idx = np.random.randint(low=0, high=n, size=n_regime)
    mu_k = obs[idx] # means for regimes
    p_k = np.zeros(n_regime)
    theta = [0] * iter
    sig = np.ones(1)

    for i in range(iter):
      theta[i] = np.concatenate((p_k, mu_k, sig))
      max_low_bound = self._estep
    
  def fit(self, obs: np.ndarray, iter: int = 10):

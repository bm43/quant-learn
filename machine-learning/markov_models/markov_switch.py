"""
hamilton's regime switching model

https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf

  https://personal.eur.nl/kole/rsexample.pdf

  https://www.stata.com/features/overview/markov-switching-models/

  https://homepage.ntu.edu.tw/~ckuan/pdf/Lec-Markov_note.pdf
"""

# EM algo soures:
# sources:
# http://www.columbia.edu/~mh2078/MachineLearningORFE/EM_Algorithm.pdf
# https://personal.eur.nl/kole/rsexample.pdf

import random
from typing import Tuple, List
import numpy as np
from itertools import chain
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, field
from scipy.stats import norm

@dataclass
class MarkovSwitch:
  
  trans_matrix: np.ndarray = field(init=False)
  n_regime: int = 2
  filtered_p: np.ndarray = field(init=False)
  pred_p: np.ndarray = field(init=False)
  smoothed_p: np.ndarray = field(init=False)
  em_params: pd.Series = field(init=False)

  def __post_init__(self) -> None:
    self.trans_matrix = np.zeros((self.n_regime, self.n_regime))
    self.filtered_p = np.array([])
    self.pred_p = np.array([])
    self.smoothed_p = np.array([])
    self.theta = np.array([])
    self.em_params = pd.Series([])

  def _sigmoid(self, x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))
  
  def _inv_sigmoid(self, x: np.ndarray) -> np.ndarray[np.float64]:
    return -np.log((1-x)/x)

  def _normpdf(self, obs: np.ndarray[np.float64], mean: np.ndarray[np.float64], std: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    # = f(y_t | s_t, F_{t-1})
    likelihood = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, std)
    return np.column_stack(tuple(likelihood))

  def _loglikelihood(self, obs: np.ndarray[np.float64],\
    theta: np.ndarray[np.float64], store: bool=False) -> float:
    p: np.ndarray[np.float64] = theta[:2]
    means = theta[2:4]
    std = np.array([theta[-1], theta[-1]])

    # 1. initialize values
    n = obs.shape[0] # number of observations
    hfilter = np.zeros((n, self.n_regime))
    eta = np.zeros((n, self.n_regime))
    pred_p = np.zeros((n, self.n_regime))

    # joint p density Pr(yt|St, Ft-1) * Pr(st|Ft-1)
    jointdist = np.zeros(n)

    p_00, p_11 = self._sigmoid(p)
    self._transition_matrix(p_00, p_11)

    # initial input to filter
    hfilter[0] = np.array([0.5, 0.5])
    pred_p[1] = self.trans_matrix.T @ hfilter[0]

    eta = self._normpdf(obs, means, std)

    # 2. filter for t = 1, ... , T-1
    for t in range(1, n-1):
      exp = eta[t] * pred_p[t]
      loglike = exp.sum()

      jointdist[t] = loglike
      hfilter[t] = exp /loglike

      # predict
      pred_p[t + 1] = self.trans_matrix.T @ hfilter[t]
    
    # hfilter and jointdist at time T
    exp = eta[-1] * pred_p[-1]
    loglike = exp.sum()
    jointdist[-1] = loglike
    hfilter[-1] = exp/loglike
    
    if store:
      self.filtered_p = hfilter
      self.pred_p = pred_p
    
    return np.log(jointdist[1:]).mean()

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
    means = theta[2:4] # third and fourth
    std = np.array([theta[-1], theta[-1]])

    # I. init
    n = obs.shape[0]
    H_filter = np.zeros((n, self.n_regime))
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
  
  def _kims_algo(self, filtered_p: np.ndarray[np.float64], pred_p: np.ndarray[np.float64], P: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    # get Pr(St=i|FT)
    
    n = filtered_p.shape[0] # how many data points
    smoothed_p = np.zeros_like(filtered_p)

    # set smoothed_p[T] = filtered_p[T]
    smoothed_p[-1] = filtered_p[-1]

    # get smooth probas
    for t in range(n-1, 0, -1):
      a = P @ (smoothed_p[t] / pred_p[t])
      smoothed_p[t-1] = filtered_p[t-1] * a

    return smoothed_p

  def _qp(self, filtered_p: np.ndarray, pred_p: np.ndarray, P: np.ndarray) -> np.ndarray:
    # posterior joint proba computed w/ kim's smoothing algorithm
    # https://homepage.ntu.edu.tw/~ckuan/pdf/Lec-Markov_note.pdf
    # page 8
    smoothed_p = self._kims_algo(filtered_p, pred_p, P)
    n = smoothed_p.shape[0]
    qp = np.zeros((n, 2** self.n_regime))
    # compute joint proba at t, t-1, ... 1
    for t in range(1, n):
      # (st-1 = 0, st = 0) and (st-1 = 0, st = 1)
      qp[t, :2] = (P[0] * smoothed_p[t] * filtered_p[t-1, 0] / pred_p[t])
      # (st-1 = 1, st = 0) and (st-1 = 1, st = 1)
      qp[t, 2:] = (P[1] * smoothed_p[t] * filtered_p[t-1, 1] / pred_p[t])
    return np.concatenate((smoothed_p, qp), axis=1)

  def _e_step(self, obs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # obs for observations
    # theta = initial guess, input to em algo
    H_filter, pred_p = self._hamilton_filter(obs, theta)
    return self._qp(filtered_p = H_filter, pred_p = pred_p, P=self.trans_matrix)    

  def _m_step(self, obs: np.ndarray, qp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    # qp is posterior probas
    p00 = qp[2:, 2].sum()/qp[1:, 0].sum()
    p11 = qp[2:, 4].sum()/qp[1:, 1].sum()
    pkk = np.array([p00, 1-p11])
    
    mu0 = (qp[1:, 0] * obs[1:]).sum()/qp[1:, 0].sum()
    mu1 = (qp[1:, 1] * obs[1:]).sum()/qp[1:, 1].sum()
    muk = np.array([mu0,mu1])
    spr1 = obs[1:]-mu0
    spr2 = obs[1:]-mu1
    var = qp[1:, 0] * spr1 ** 2 + qp[1:, 1] * spr2 ** 2
    return pkk, muk, [np.sqrt(var.mean())]

  def _em(self, obs: np.ndarray, iter: int = 10):
    # sources:
    # http://www.columbia.edu/~mh2078/MachineLearningORFE/EM_Algorithm.pdf
    # https://personal.eur.nl/kole/rsexample.pdf
    n = obs.shape[0]
    n_regime = self.n_regime

    idx = np.random.randint(low=0, high=n, size=n_regime)
    mu_k = obs[idx] # means for regimes
    p_k = np.zeros(n_regime)
    theta = [0] * iter
    sig = np.ones(1)

    for i in range(iter):
      theta[i] = np.concatenate((p_k, mu_k, sig))

      # E step:
      qp = self._e_step(obs, theta[i])

      # M step:
      p_kk, mu_k, sig = self._m_step(obs, qp)
      p_k = self._inv_sigmoid(p_kk)
    
    cols = self._make_titles()
    self.em_params = pd.DataFrame(theta, columns=cols)
    self.em_params.index.name = "em_iters"
    self.em_params[["p11", "p22"]] = self.em_params[["p11", "p22"]].apply(self._sigmoid)
    return self
  
  def _objective_function(self, guess: np.ndarray[np.float64], \
    obs: np.ndarray[np.float64], store: bool = False) -> float:
    # obs = observed response variable
    f = self._loglikelihood(obs, theta=guess, store=store)
    return -f

  def fit(self, obs: np.ndarray, iter: int = 10):
    self._em(obs, iter = iter)
    param_guess = self.em_params.tail(1).values.ravel()

    # first two param (=transition probs) change
    param_guess[:2] = self._inv_sigmoid(param_guess[:2])

    # fit
    self.theta = minimize(
      self._objective_function,
      param_guess,
      method="SLSQP",
      options={"disp": True},
      args=(obs, True)
    )["x"]

    # get smooth p
    self.smoothed_p = self._kims_algo(
      self.filtered_p, self.pred_p, self.trans_matrix
    )

    self.theta[:2] = self._sigmoid(self.theta[:2])

    return self

  def _make_titles(self) -> list:
    n = self.n_regime
    col1 = list("p{i}{i}".format(i=i) for i in range(1, n+1))
    col2 = list("regime{i}_mean".format(i=i) for i in range(1, n+1))
    col3 = ["regime_vol"]
    return list(chain(col1, col2, col3))
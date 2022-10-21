from dataclasses import dataclass, field
from typing import List
import numpy as np
# implemented using:
# https://web.stanford.edu/~jurafsky/slp3/A.pdf

# 1.
# 2.
# 3. forward-backward algo

@dataclass
class HiddenMarkovModel():
    # n = no. of hidden states
    # m = number of possible observation outcomes
    transmission_prob: np.ndarray # (n+2) x (n+2) input
    emission_prob: np.ndarray # m x n
    n: int
    m: int

    # observation labels:
    obs: List[str]

    psi: np.ndarray
    gamma: np.ndarray

    forward: np.ndarray
    backward: np.ndarray

    def __post_init__(self) -> None:
        self.psi = np.zeros((self.n, self.n, len(self.obs)-1))
        self.gamma = np.zeros((len(self.obs), self.n))
        return
    
    def fit(self, obs: List[str], iter: int):
        for i in range(iter):
            old_transmission = self.transmission_prob.copy()
            old_emission = self.emission_prob.copy()
            print("Iteration: {}".format(i + 1))
            # perform em
            self._e_step()
            self._m_step()
        return
    
    def _e_step(self) -> None:
        self._forward = self._forward_recursion(len(self.obs))
        self._backward = self.backward_recursion(0)
        self._get_gamma()
        self._get_psi()
        return

    def _forward_recursion(self, idx):
        return

    

    def _get_gamma(self) -> None:
        self.gamma = np.zeros((2,len(self.obs)))
        
        # first column:
        self.gamma[:, 0] = self._forward[0, :] * self._backward[0, :] \
            / self._forward[0, :] * self._backward[0, :] \
                + self._forward[1, :] * self._backward[1, :]
        
        # second column:
        self.gamma[:, 1] = self._forward[1, :] * self._backward[1, :] \
            / self._forward[0, :] * self._backward[0, :] \
                + self._forward[1, :] * self._backward[1, :]

    def _get_psi(self) -> None:
        return

    def _compute_psi(self, t, i, j):
        return

    def _m_step(self) -> None:
        return

    def _get_state_probas(self):
        return

    def _estimate_transmission(self, i, j):
        return
    
    def _estimate_emission(self, j ,obs):
        return

    def _backward_recursion(self, idx) -> np.ndarray:
        return self.backward

    def _backward_init(self, state):
        return self.transmission_prob[self.n + 1][state + 1]

    def _backward_proba(self, idx, backward, state, last) -> np.float:
        return

    
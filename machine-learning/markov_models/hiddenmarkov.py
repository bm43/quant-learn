from dataclasses import dataclass, field
from turtle import hideturtle
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

    # declare inputs to class
    transmission_prob: np.ndarray # (n+2) x (n+2) input
    emission_prob: np.ndarray # m x n

    def __post_init__(self) -> None:
        self.n = self.emission_prob.shape[1]
        self.m = self.emission_prob.shape[0]

        # observation labels:
        self.obs = np.array([[]])

        self.forward = np.array([])
        self.backward = np.array([])
        self.psi = np.zeros((self.n, self.n, len(self.obs)-1))
        self.gamma = np.zeros((len(self.obs), self.n))
        return
    
    def assume_obs(self):
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
        if idx == 0:
            #fwd = [[0.0] * (len(self.obs)) for i in range(self.n)]
            fwd = np.zeros((self.n, len(self.obs)))
        return fwd

    def _forward_init(self, obs, state):
        return

    def _forward_proba(self, idx, forward, state):
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

    def _backward_proba(self, idx, backward, state, last) -> float:
        return 0.0

    def likelihood(self, new_obs) -> float:
        return 0.0

if __name__ == "__main__":
    emission = np.array([[0.7, 0], [0.2, 0.3], [0.1, 0.7]])
    transmission = np.array([ [0, 0, 0, 0], [0.5, 0.8, 0.2, 0], [0.5, 0.1, 0.7, 0], [0, 0.1, 0.1, 0]])
    model = HiddenMarkovModel(transmission, emission)
    print(model._forward_recursion(0))
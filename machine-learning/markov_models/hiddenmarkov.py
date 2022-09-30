from dataclasses import dataclass, field
from typing import List
import numpy as np

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

    def __post_init__(self) -> None:
        self.psi = np.zeros((self.n, self.n, len(self.obs)-1))
        self.gamma = np.zeros((len(self.obs), self.n))
        return
    
    def fit(self, obs: List[str], iter: int):
        for i in range(iter):
            old_transmission = self.transmission_prob.copy()
            old_emission = self.emission_prob.copy()
            print("Iteration: {}".format(i + 1))
            self._expectation()
            self._maximize()
        return

    def predict(self):
        return self
    
    def _expectation(self) -> None:
        
        return

    def _maximize(self) -> None:
        return

obs = ['a','a','a','a','a','a','a']
hmm = HiddenMarkovModel(np.zeros((102,102)), np.zeros((90,100)))
hmm.fit(obs,10)
print(hmm.n)
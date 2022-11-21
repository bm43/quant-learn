# cliquet-style option pricing class
# author: Hyung Jip Lee
# Nov. 11 2022

from dataclasses import dataclass
import numpy as np
from math import floor

@dataclass
class contractParams:
    K: float = 100.0# principal
    C: float = 0.03 # local cap
    F: float = 0.02 # local floor
    GC: float = 0.08 # global cap
    GF: float = 0.01 # global floor

@dataclass
class CliquetOption:
    
    # grid params:
    N: int = 1024# number of basis points, 2^?
    alpha: int = 10# log asset grid width = 2*alpha

    # contract parameters
    M: int = 100# subintervals in [0,T], there are M+1 points cuz t=0
    r: float = 0.07# interest rate
    q: float = 0.015# dividend yield
    T: float = 2 # years
    
    contract: int = 1# type of contract, between 1 and 5
    """
    1 -> local caps sum
    2 -> local caps sum and floors
    3 -> cliquet local caps sum
    4 -> cliquet local caps sum and floors
    5 -> monthly capped sum
    """
    cp: contractParams

    def __post_init__(self):
        self.cp = contractParams()
        self.dx = 2*self.alpha/(self.N-1)
        self.a = 1/self.dx
        self.dt = self.T/self.M

        # initial xmin
        self.xmin = (1-self.N/2)*self.dx

        self.lc = np.log(1 + self.cp.C)
        self.lf = np.log(1 + self.cp.F)
    
    def compute_contract_price(self):
        return

    def _rnch(self): # risk neutral characteristic function
        return

    def _set_xmin(self) -> float:
        self.klc = floor(self.a(self.lc - self.xmin)) + 1
        xklc = self.xmin + (self.klc - 1)*self.dx
        self.xmin = self.xmin + (self.lc - xklc)

        self.klf = floor(self.a*(self.lf - self.xmin)) + 1

    def _lcfr(self, x: np.array): # locally capped / locally capped and floored return
        if self.contract == 1 or self.contract == 5:
            # for every element in x:
            # if x[i] < lc, then x[i] = np.exp(x[i])-1
            # if x[i] >= lc, then x[i] = C*x[i]
            return np.multiply(np.exp(x)-1, (x<self.lc).astype(int)) + np.multiply(self.C, (x>=self.lc).astype(int))
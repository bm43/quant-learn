# cliquet-style option pricing class
# author: Hyung Jip Lee
# Nov. 11 2022

from dataclasses import dataclass
import numpy as np
from math import floor
from math import pi

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
    M: int = 100 # subintervals in [0,T], there are M+1 points cuz t=0
    r: float = 0.07 # interest rate
    q: float = 0.015 # dividend yield
    T: float = 2 # years
    
    contract: int = 1# type of contract, between 1 and 5
    """
    1 -> local caps sum
    2 -> local caps sum and floors
    3 -> cliquet local caps sum
    4 -> cliquet local caps sum and floors
    5 -> monthly capped sum
    """
    cp: contractParams = contractParams()

    def __post_init__(self):
        
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
        self.klc = floor(self.a*(self.lc - self.xmin)) + 1
        xklc = self.xmin + (self.klc - 1)*self.dx
        self.xmin = self.xmin + (self.lc - xklc)

        self.klf = floor(self.a*(self.lf - self.xmin)) + 1

    def _lcfr(self, x: np.array): # locally capped / locally capped and floored return
        if self.contract == 1 or self.contract == 5:
            # for every element in x:
            # if x[i] < lc, then x[i] = np.exp(x[i])-1
            # if x[i] >= lc, then x[i] = C*x[i]
            return np.multiply(np.exp(x)-1, (x<self.lc).astype(int)) + np.multiply(self.C, (x>=self.lc).astype(int))
        elif self.contract == 2 or self.contract == 3 or self.contract == 4:
            if (self.klc != self.klf):
                self.dx = (self.lc - self.lf) / (self.klc - self.klf)
                self.a = 1/self.dx
                self.xmin = self.lf - (self.klf - 1)*self.dx
            return np.multiply(self.cp.F * x, (x<=self.lf).astype(int)) + np.multiply(np.exp(x) - 1, (x<self.lc).astype(int)) \
                + np.multiply(self.cp.C * x, (x>=self.lc))

    def _set_vals(self):
        self.A = 32 * self.a**4
        self.C_aN = self.A / self.N
        self.dxi = 2 * pi * self.a / self.N
    
    def _PSI_Matrix(self):
        if self.contract == 2 or self.contract == 3 or self.contract == 4:
            leftGridPt = self.lf - self.dx
            NNM = self.klc - self.klf + 3
        elif self.contract == 1 or self.contract == 5:
            leftGridPt = self.xmin
            NNM = self.klc + 1
        else:
            leftGridPt = self.xmin
            NNM = self.M

        PSI = np.zeros(self.N-1, NNM)

        Neta = 5*(NNM) + 15
        Neta5 = NNM + 3
        g2 = np.sqrt(5 - 2*np.sqrt(10/7))/6
        g3 = np.sqrt(5 + 2*np.sqrt(10/7))/6
        v1 = .5*128/225
        v2 = .5*(322 + 13 * np.sqrt(70))/900
        v3 = .5*(322 - 13*np.sqrt(70))/900
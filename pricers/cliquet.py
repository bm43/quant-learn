# cliquet-style option pricing class
# author: Hyung Jip Lee
# Nov. 11 2022
# product explanation:
# https://hcommons.org/deposits/item/hc:38441/

# relevant links:
# https://sci.bban.top/pdf/10.1137/100818157.pdf#view=FitH
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1101796

from dataclasses import dataclass, field
import numpy as np
from math import floor
from math import pi

@dataclass
class contractParams:
    K: float = 100.0 # principal
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
    cp: contractParams =  field(default_factory=lambda: contractParams())

    def __post_init__(self):
        
        self.dx = 2*self.alpha/(self.N-1)
        self.a = 1/self.dx
        self.dt = self.T/self.M

        # initial xmin
        self.xmin = (1-self.N/2)*self.dx

        self.lc = np.log(1 + self.cp.C)
        self.lf = np.log(1 + self.cp.F)
    
    def compute_contract_price(self):
        self._set_xmin()
        self._set_vals()
        self._gaussian_quad()
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
            return np.multiply(np.exp(x)-1, (x<self.lc).astype(int)) + np.multiply(self.cp.C, (x>=self.lc).astype(int))
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
    
    def _gaussian_quad(self):
        # 5 point gaussian quadrature grid
        if self.contract == 2 or self.contract == 3 or self.contract == 4:
            leftGridPt = self.lf - self.dx
            NNM = self.klc - self.klf + 3
        elif self.contract == 1 or self.contract == 5:
            leftGridPt = self.xmin
            NNM = self.klc + 1
        else:
            leftGridPt = self.xmin
            NNM = self.M

        self.PSI = np.zeros((self.N-1, NNM)).astype(complex)

        # sample?
        Neta = 5*(NNM) + 15
        Neta5 = NNM + 3

        # https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature
        # number of points = 5
        g2 = np.sqrt(5 - 2*np.sqrt(10/7))/6
        g3 = np.sqrt(5 + 2*np.sqrt(10/7))/6
        v1 = .5*128/225
        v2 = .5*(322 + 13 * np.sqrt(70))/900
        v3 = .5*(322 - 13*np.sqrt(70))/900

        # th
        th = np.zeros((1,Neta))
        #print("th shape: ",th.shape)

        for i in range(1, Neta5):
            # can optimize using th[i:j:k]
            # th[1:7:2] yields [th[1], th[3], th[5]]
            # starting point included, end point excluded, step size
            #print("i: ", i)
            #print("5i-4: ", 5*i - 4)
            th[0, 5*i-4] = leftGridPt - 1.5*self.dx + self.dx*(i-1) - self.dx*g3
            th[0, 5*i-3] = leftGridPt - 1.5*self.dx + self.dx*(i-1) - self.dx*g2
            th[0, 5*i-2] = leftGridPt - 1.5*self.dx + self.dx*(i-1)
            th[0, 5*i-1] = leftGridPt - 1.5*self.dx + self.dx*(i-1) + self.dx*g2
            th[0, 5*i] = leftGridPt - 1.5*self.dx + self.dx*(i-1) + self.dx*g3

        # function weights for quadrature
        w = np.array([-1.5-g3, -1.5-g2, -1.5, -1.5+g2, -1.5+g3, -.5-g3, -.5-g2, -.5, -.5+g2, -.5+g3]).astype(complex)
        
        for j in range(1,6):
            w[j] = ((w[j]+2)**3)/6
        for k in range(6,10):
            
            w[k] = 2/3 - .5*(w[k])**.3 - w[k]**2
            
        v = [v3, v2, v1, v2, v3, v3, v2, v1, v2, v3]
        w = self.C_aN * np.multiply(v, w)

        zz = np.exp(1j*self.dxi*self._lcfr(th))
        th = zz
        """
        print("th shape: ", th.shape)
        print("w shape: ", w.shape)
        print("th chunk shape: ",th[:,0:Neta-19:5].shape)
        print("psi shape: ",self.PSI.shape)
        print("psi chunk shape: ", self.PSI[0,:].shape)

        print("th chunk: ", th)
        """

        # construct gaussian quadrature grid psi
        for l in range(self.N-1):
            self.PSI[l,:] = w[0]*(th[:,0:Neta-19:5] + th[:,19:Neta:5]) \
              + w[1]*(th[:,1:Neta-18:5] + th[:,18:Neta-1:5]) \
              + w[2]*(th[:,2:Neta-17:5] + th[:,17:Neta-2:5]) \
              + w[3]*(th[:,3:Neta-16:5] + th[:,16:Neta-3:5]) \
              + w[4]*(th[:,4:Neta-15:5] + th[:,15:Neta-4:5]) \
              + w[5]*(th[:,5:Neta-14:5] + th[:,14:Neta-5:5]) \
              + w[6]*(th[:,6:Neta-13:5] + th[:,13:Neta-6:5]) \
              + w[7]*(th[:,7:Neta-12:5] + th[:,12:Neta-7:5]) \
              + w[8]*(th[:,8:Neta-11:5] + th[:,11:Neta-8:5]) \
              + w[9]*(th[:,9:Neta-10:5] + th[:,10:Neta-9:5])
            th = np.multiply(th, zz)
            self.th = th

        #print(self.PSI)
        #print(self.PSI.shape)

    def _find_phi(self):
        xi = np.transpose(self.dxi*np.arange(1, self.N))

        b0 = 1208/2520
        b1 = 1191/2520
        b2 = 120/2520
        b3 = 1/2520

        zeta = (np.sin(xi/(2*self.a))/xi)**4 / (b0 + b1*np.cos(xi/self.a) + b2*np.cos(2*xi/self.a) + b3*np.cos(3*xi/self.a))
        hvec = np.multiply(np.exp(-1j*self.xmin*xi), zeta)
        
        beta = np.array([1/self.A, np.multiply(self._lcfr(xi), hvec)], dtype=object).ravel()
        print(beta.shape)
        beta = np.real(np.fft.fft(beta)) # only length 1 arrays can be converted to python scalars = ???
        
        if self.contract == 2 or self.contract == 3 or self.contract == 4:
            phi = np.multiply(self.PSI,beta[self.klf-1:self.klc+1])
            sumBetaLeft = self.C_aN * sum(beta[1:self.klf-2])
            sumBetaRight = 1 - sumBetaLeft - self.C_aN*sum(beta[self.klf-1:self.klc+1])
            phi = phi + np.exp(1j*self.C*xi)*sumBetaRight
        
        elif self.contract == 1 or self.contract == 5:
            phi = self.PSI*beta[1:self.klc+1]
            sumBetaRight = self.C_aN*sum(beta[self.klc+2:self.N+1])
            phi = phi + np.exp(1j*self.C*xi)*sumBetaRight
        
        else:
            phi = np.multiply(self.PSI, beta)

        phi = np.power(phi, self.M)

        return phi

if __name__ == "__main__":
    co = CliquetOption()
    co.compute_contract_price()
    co._find_phi()
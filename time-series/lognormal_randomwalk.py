from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import math

@dataclass
class lognormalSimulate:
    S_0 : float
    r : float
    mu : float
    sig : float

    def _lognormal_density(self, x, mu, sig):
        return (1/(x*sig*(2*math.pi)**0.5)) * np.exp(-(np.log(x)-mu)**2 / 2*sig**2)

    def plot_lognormal(self) -> None:
        x = np.linspace(0.01,100,1000)
        
        plt.plot(x, self._lognormal_density(x, self.mu, self.sig))
        plt.show()
        return
    
    def _make_random_walk(self, length: int) -> np.array:
        
        T = np.zeros((1,length))
        T[0] = self.S_0
        U = np.random.uniform(0, 1, length)
        Z = np.random.normal(0, 1, length)
        for n in range(1,length):
            if U[n] > self._lognormal_density(T[n-1]+Z[n], self.mu, self.sig) / self._lognormal_density(T[n-1], self.mu, self.sig):
                T[n] = T[n-1]
            else:
                T[n] = T[n-1] + Z[n]

        return T

    def plot_lognormal_randomwalk(self, length: int) -> None:
        T = self._make_random_walk(length)
        plt.plot(T)
        plt.show()
        return

lns = lognormalSimulate(100.0, 0.1, 2.3, 0.9)
#lns.plot_lognormal()
lns.plot_lognormal_randomwalk(100)
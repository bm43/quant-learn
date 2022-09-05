"""
implementing:
1. locally capped contracts pricer
2. cliquet option pricer
in a black-scholes market,

based on the following paper:
https://sci.bban.top/pdf/10.1137/100818157.pdf#view=FitH

Locally capped contract:
these contracts combine a guaranteed payoff with a bonus equal to
some accumulation of the capped periodic returns of a reference portfolio

explanations on this product with some math:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1101796
"""

from attr import dataclass
import numpy as np
from scipy.stats import norm
from numpy.random import normal
from typing import field, Union
from math import pi
from scipy.integrate import quad

"""
notes:
payoff at maturity is:
1 + g + max(0, sum( min( (S[t]-S[t-1])/S[t-1], c) - g)
g: guaranteed return
S[t]: underlying price at t
c: cap rate at each period

time period from 0 to T divided by n
t_1 = T/n, t_2 = 2T/n ... t_n = T

valuation of this product can be done using black scholes
"""

# rough plan:

######################################
# I. globally capped contract pricer #
######################################

"""
Globally capped contracts are replicated by a portfolio
containing two call options
Indeed, price of gcc depends on the price diff of those
2 options."""

def d_j(j: int, S: float, K: float, r: float, sigma: float, T: float):
  return (np.log(S/K) + (r + ( (-1)**(j-1) )*0.5*sigma**2)*T) / (sigma*((T**0.5)))

# black scholes call price:
def C(S: float, K: float, r: float, sigma: float, T: float):
    return S * norm.cdf(d_j(1, S, K, r, sigma, T)) - K * np.exp(-r*T) * norm.cdf(d_j(2, S, K, r, sigma, T))

def GloballyCappedContractPricer(c: float, g: float, S: float,\
    r: float, T: float):
    return (1+g)*np.exp(-r*T) + (1/S) *( C(S*(1+g)) - C(S*(1+c)))

#######################################
# II. locally-capped contract pricer  #
#######################################

"""
There is no closed-form solution
for price of locally-capped contracts.
So we can use monte carlo simulation
or find semi-closed form.

In this case, we derive a semi-closed form of
the monthly sum cap's price.
"""

@dataclass
class MonthlySumCapPrice:
    
    # details of product
    S: np.array # underlying asset prices from t_0 to T
    r: float = 0.03 # risk free rate
    g: float = 0.03 # guaranteed rate at T
    T: float = 3.0 # maturity in years
    delta: float = 0.01 # time interval btwn price ticks
    sig: float = 0.0 # volatility of S
    eta: float = 0.02 # constant yield of dividend of designated reference equity index

    K: float = 300 # initial investment net of fees and commission

    def __post_init__(self) -> None:
        """note: assuming clean data
        i.e. regular time intervals between price ticks
        up to maturity T
        """
        n = self.S.shape[0] # length of price data
        self.delta = self.T/n # tk - tk-1 = delta
        self.sig = self.S.std

    def _risk_neutral_prob(self, x: float, m_xi: float, sig: float, delta: float) -> float:
        # cumulative normal dist with
        # mean = (r - eta - ((sig**2)/2) * delta)
        # std = (sig**2) * delta
        stddev = (sig**2) * delta
        return norm(m_xi, stddev).cdf(x)

    def _expected_Ck(self, c: float, g: float, n: float, r: float, eta: float, sig: float, delta: float) -> float:

        def integrand(y, g, n, delta, sig, m_xi):
            return (y - 1 - g/n) * ((2*pi)**0.5 * delta * sig*y)**(-1) * np.exp((-(np.log(y) - m_xi)**2)/(2*sig*sig*delta))

        m_xi = r - eta - ((sig**2)/2) * delta

        integral = quad(integrand, 0, c+1, args=(g, n, delta, sig, m_xi))
        x = (m_xi - np.log(1+c))/(sig*delta**0.5)

        return (c-g/n) * self._risk_neutral_prob(x, m_xi, sig, delta) + integral    

    def _characteristic_Ck(self, t, c: float, g: float, n: float, r: float, eta: float, sig: float, delta: float) -> float:
        m_xi = r - eta - ((sig**2)/2) * delta

        first_term = np.exp(-1j*t*(1+g/n))

        def integrand(x, t, m_xi, sigma, delta):
            input = (m_xi - np.log(x)) / (sigma * delta)
            return np.exp(1j*t*x) * self._risk_neutral_prob(input, m_xi, sig, delta)

        integral = quad(integrand, 0, 1+c, args=(t, m_xi, sig, delta))[0]

        second_term = 1 + 1j*t * integral
        phi_Ct = first_term * second_term
        return phi_Ct.real

    def _expected_absolute_price_sum_C(self, t, c: float, g: float, n: float, r: float, eta: float, sig: float, delta: float) -> float:
        # Theta_L := abs(sum(Lk))
        def integrand(t, c: float, g: float, n: float, r: float, eta: float, sig: float, delta: float):
            return 1 - self._characteristic_Ck(t, c, g, n, r, eta, sig, delta)/(t*t)
        return 2/pi * quad(integrand, 0, 1000, args=(c, g, n, r, eta, sig, delta))
    
    def _time_zero_price_contract(self, t, c: float, g: float, n: float, r: float, eta: float, sig: float, delta: float):
        return

    def _expected_Zk(self):
        return
    
    def _characteristic_Zk(self):
        return
    
    def _expected_Absolute_price_sum_Z(self):
        return

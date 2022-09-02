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
from typing import field, Union

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
    
    S: np.array # underlying asset prices from t_0 to T

    r: float = field(init=False, default=0.03) # risk free rate
    g: float = field(init=False, default=0.03) # guaranteed rate at T
    T: float = field(init=False, default=3.0) # maturity in years
    delta: float = field(init=False) # time interval btwn price ticks
    sig: float = field(init=False) # volatility of S
    eta: float = field(init=False) # constant yield of dividend of designated reference equity index

    def __post_init__(self) -> None:
        """note: assuming clean data
        i.e. regular time intervals between price ticks
        up to maturity T
        """
        n = self.S.shape[0] # length of price data
        self.delta = self.T/n # tk - tk-1 = delta
        self.sig = self.S.std

    def _risk_neutral_prob_dist(self, r: float, eta: float, sig: float, delta: float) -> norm:
        # cumulative normal dist with
        # mean = (r - eta - ((sig**2)/2) * delta)
        # std = (sig**2) * delta

        # somethings' weird here... you should return a number
        return norm(loc = r - eta - ((sig**2)/2) * delta, scale = (sig**2) * delta)

    def expected_Ck(self, c, g, n, m_xi, sig, delta):


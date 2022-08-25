# https://sci.bban.top/pdf/10.1137/100818157.pdf#view=FitH
# locally capped contracts:
"""
These contracts combine a guaranteed payoff with a bonus equal to
some accumulation of the capped periodic returns of a reference portfolio
"""
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1101796
import numpy as np
from scipy.stats import norm

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

# import relative path to use .hpp file?

def d_j(j: int, S: float, K: float, r: float, sigma: float, T: float):
  return (np.log(S/K) + (r + ( (-1)**(j-1) )*0.5*sigma**2)*T) / (sigma*((T**0.5)))

# black scholes call price:
def C(S: float, K: float, r: float, sigma: float, T: float):
    return S * norm.cdf(d_j(1, S, K, r, sigma, T)) - K * np.exp(-r*T) * norm.cdf(d_j(2, S, K, r, sigma, T))

"""
Globally capped contracts are replicated by a portfolio
containing two call options
Indeed, price of gcc depends on the price diff of those
2 options."""
def GloballyCappedContractPricer(c: float, g: float, S: float,\
    K: float, r: float, sigma: float, T: float):
    return (1+g)*np.exp(-r*T) + (1/S) *( C(S*(1+g)) - C(S*(1+c)))

"""
There is no closed-form solution
for price of locally-capped contracts.
So we can use monte carlo simulation
or 
"""
def LocallyCappedContractPricer():
    maturity = 1
    n = 12
    return
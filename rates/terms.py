# paper by patrick hagan, graeme west
# implements yield curve construction

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds

@dataclass
class ZeroRateCurve:
    """
    most bonds give many cash flows at different points in time,
    which means spot rates fit better than a single discount rate"""

    """Builds zero-rate curve using ZCBs with continuous discounting"""

    maturities : np.ndarray
    # time it takes for a bond to mature

    zero_rates : np.ndarray = None
    # risk free rate values (exponent function)

    zcb_prices : np.ndarray = None

    def __post_init__(self):
        self.disc_factors = self.discount_factor(self.maturities, self.zero_rates)
        self.zcb_prices = self.disc_factors * 100.0

    def discount_factor(self, t: float, rcont: float) -> float:
        """computes the discount factor corresponding to continuous rates
        = maximum amount you give up today to receive
        $1 in t years
        
        t = maturity of zcb
        rcont = the continuous rate
        """
        return np.exp(-rcont * t) # Z(0, t)

    def rates_from_discount_factors(self, t: float, discount_factor: float) -> float:
        """computes the continuous rates from discount factors
        = rate you would receive if you bought the zcb today
        and held it until maturity t in years
        
        t : float = maturity of zcb or rate
        
        discount_factor = zcb discount factor
        
        returns rate corresponding to discount factors
        """
        return (1.0 / t) * np.log(1 / discount_factor)

    def linear_interp(self, t: float) -> float:
        """performs linear interpolation of spot rate
        k: float = maturity of payment date of $1
        returns rate corresponding to maturity t
        """
        r = self.zero_rates
        expiry = self.maturities
        tmin = expiry[0]
        tmax = expiry[-1]

        if t < tmin or t > tmax:
            print("Maturity out of bounds Error")
            return 0.0

        # find index where t1 < t < t2
        told = len(expiry[expiry < t]) - 1
        tnew = told + 1
        terms = (r[tnew] - r[told]) / (expiry[tnew] - expiry[told])
        return terms * (t - expiry[told]) + r[told]

    # get cubic spline 
    def cubic_interp(self, t: float) -> float:
        return 0.0
    
    def build_curve(self, fit_type: Optional[str] = "linear") -> pd.Series:
        """builds a zero rate curve based on type of interpolation
        fit_type = linear_spot, constant_fwd, or cubic_spline
        Returns zero rate curve for all maturities
        """
        tmax = self.maturities[-1]
        knot_points = np.arange(0, tmax, 0.01)
        fit_type += "_interp"
        zero_rates = map(getattr(self, fit_type), knot_points)

        return pd.Series(zero_rates, index=knot_points)


if __name__ == "__main__":
    maturities = np.arange(0,10) # Z

    # https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202211:
    zero_rates = np.array([3.72, 4.00, 4.23, 4.35, 4.58, 5.75, 4.54, 4.48, 4.27, 4.18])

    zrc = ZeroRateCurve(maturities, zero_rates)
    curve = zrc.build_curve()
    
    plt.plot(curve.values),plt.title('Yield Curve')
    plt.show()
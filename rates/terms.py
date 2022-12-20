# paper by patrick hagan, graeme west
# implements yield curve construction

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds
import bisect

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
            raise ValueError("Maturity out of bounds Error")

        # find index where t1 < t < t2
        told = len(expiry[expiry < t]) - 1
        tnew = told + 1
        terms = (r[tnew] - r[told]) / (expiry[tnew] - expiry[told]) # terms = (y2 - y1) / (x2 - x1)
        return terms * (t - expiry[told]) + r[told] # terms * (x - x1) + y1

    # get cubic spline 

    def _get_delta_x(self, x: np.array) -> np.array: # changes in x
        print(x)
        return np.array([x[i+1] - x[i] for i in range(len(x)-1)])

    def _create_tridiagonalmatrix(self, n: int, h: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        creates tridiagonal matrix for cubic spline interpolation.
        n: int, length of maturities
        h: List, delta x on maturities

        return: 3 Lists A B and C representing the tridiagonal matrix
        """
        A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
        B = [2] * n
        C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
        return A, B, C

    def _create_target(self, n: int, h: List[float], y: List[float]):
        # n = length of maturities
        # h = delta x
        # y = zero_rates
        return [0] + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i-1]) for i in range(1, n - 1)] + [0]

    def _solve_tridiagonalsystem(self, A: List[float], B: List[float], C: List[float], D: List[float]) -> List:
        c_p = C + [0]
        d_p = [0] * len(B)
        X = [0] * len(B)

        c_p[0] = C[0] / B[0]
        d_p[0] = D[0] / B[0]

        for i in range(1, len(B)):
            c_p[i] = c_p[i] / (B[i] - c_p[i - 1] * A[i - 1])
            d_p[i] = (D[i] - d_p[i - 1] * A[i - 1]) / (B[i] - c_p[i - 1] * A[i - 1])

        X[-1] = d_p[-1]
        for i in range(len(B) - 2, -1, -1):
            X[i] = d_p[i] - c_p[i] * X[i + 1]

        return X

    def _get_curve_coeffs(self): # notice how type of t is different from linear interp
        # t = knot points, cf. build_curve
        n = len(self.maturities)
        y = self.zero_rates

        if n < 3:
            raise ValueError('length of self.maturities not long enough')
        if n != len(self.zero_rates):
            raise ValueError('Array lengths are different')

        h = self._get_delta_x(self.maturities)
        self.h = h
        if any(v < 0 for v in self.maturities):
            raise ValueError('maturities must be strictly increasing')

        A, B, C = self._create_tridiagonalmatrix(n, h)
        D = self._create_target(n, h, y)

        M = self._solve_tridiagonalsystem(A, B, C, D)
        
        coefficients = [[(M[i+1]-M[i])*h[i]*h[i]/6, M[i]*h[i]*h[i]/2, (y[i+1] - y[i] - (M[i+1]+2*M[i])*h[i]*h[i]/6), y[i]] for i in range(len(self.maturities)-1)]
        
        return coefficients

    def cubic_spline_interp(self, t: float): # spline
        coeffs = self._get_curve_coeffs() # compute spline
        idx = min(bisect.bisect(self.knot_points, t)-1, len(self.maturities)-2)
        z = (t - self.knot_points[idx]) / self.h[idx]
        C = coeffs[idx]
        return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]
    
    def build_curve(self, fit_type: Optional[str] = "linear") -> pd.Series:
        """builds a zero rate curve based on type of interpolation
        fit_type = linear_spot, constant_fwd, or cubic_spline
        Returns zero rate curve for all maturities
        """
        tmax = self.maturities[-1]
        
        fit_type += "_interp"
        
        self.knot_points = np.arange(0, tmax, 0.01)
        zero_rates = map(getattr(self, fit_type), self.knot_points)

        return pd.Series(zero_rates, index=self.knot_points)
        


if __name__ == "__main__":
    maturities = np.arange(0,10) # X, in this case maturities were arbitrarily chosen

    # https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value_month=202211:
    zero_rates = np.array([3.72, 4.00, 4.23, 4.35, 4.58, 5.75, 4.54, 4.48, 4.27, 4.18]) # y

    zrc = ZeroRateCurve(maturities, zero_rates)
    curve = zrc.build_curve("cubic_spline")

    plt.plot(curve.values),plt.title('Yield Curve')
    plt.show()
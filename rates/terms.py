# paper by patrick hagan, graeme west
# methods for constructing yield curve

import pandas as pd
import numpy as np

class ZeroRateCurve:
    """Builds zero-rate curve using ZCBs with continuous discounting"""

    def __init__(
        self,
        maturities: np.ndarray,
        zcb: np.ndarray = None,
        zero_rates: np.ndarray = None,
    ):
        """Default Constructor used to initialize zero rate curve
        :param maturities: maturity corresponding to each
         zero coupon bond in the zcb array
        :type maturities: np.ndarray
         shape = (n_samples,)
        :param zcb: zero coupon bond prices traded for various maturities
        :type rates: np.ndarray
         shape = (n_samples,)
        """
        self.maturities = maturities
        self.zero_rates = zero_rates
        self.disc_factors = self.discount_factor(maturities, zero_rates)
        self.zcb_prices = self.disc_factors * 100.0

    def discount_factor(self, t: float, rcont: float):
        """computes the discount factor corresponding to continuous rates
        This is the maximum amount you give up today to receive
        $1 in t years
        Parameters
        ----------
        t : float
            maturity of zcb
        rcont : float
            the continuous rate
        Returns
        -------
        float
            discount factor corresponding to continuous rate
        """
        return np.exp(-rcont * t)

    def rates_from_discount_factors(self, t: float, discount_factor: float):
        """computes the continuous rates from discount factors
        = rate you would receive if you bought the zcb today
        and held it until maturity t in years
        
        t : float = maturity of zcb or rate
        
        :param discount_factor: zcb discount factor
        :type discount_factor: float
        :return: rate corresponding to discount factors
        :rtype: float
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

    def build_curve(self, fit_type: str = "linear") -> pd.Series:
        """builds a zero rate curve based on type of interpolation
        Parameters
        ----------
        fit_type : str, optional, default="linear_spot"
            type of fit
            supports one of linear_spot, constant_fwd, cubic_spline
        Returns
        -------
        pd.Series
            zero rate curve for all maturities
        """
        tmax = self.maturities[-1]
        knot_points = np.arange(0, tmax, 0.01)
        fit_type += "_interp"
        zero_rates = map(getattr(self, fit_type), knot_points)

        return pd.Series(zero_rates, index=knot_points)
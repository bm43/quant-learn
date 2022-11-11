import statsmodels.api as sm
import matplotlib.pyplot as plt
#from statsmodels.tsa.regime_switching.tests.test_markov_regression import fedfunds
from dataclasses import dataclass

# testing time series to distinguish real US equities price time series
# from time series generated from Gaussian random number generator
# https://www.frontiersin.org/articles/10.3389/fnhum.2015.00319/full
# https://www.stat.colostate.edu/~piotr/normalityJTSA.pdf

#plot | kde, N
#QQ   | standardized residual

@dataclass
class TS_normality:

    data: list = None

    def decompose(self):
        return

    def show_analysis(self):

        # data:
        plt.subplot(221), plt.plot(self.data), plt.title("data")

        # Histogram and estimated density:
        plt.subplot(222)
        plt.plot()
        plt.plot()
        plt.legend(['KDE', 'N', "Hist"])
        
        # QQ plot:
        plt.subplot(223), plt.plot(), plt.title("Q-Q")

        # Standardized residual
        plt.subplot(224), plt.plot(), plt.title("Std. residual")

        plt.show()

    def show_analysis_sm(self):
        """
        p = d = q = range(0, 2)
        pdq = list( itertools.product(p, d, q))
        seasonal_pdq = [ (x[0], x[1], x[2], 12)  for x in pdq ]
        """
        mod = sm.tsa.statespace.SARIMAX(
        self.data,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
        )
        res = mod.fit()
        res.plot_diagnostics(figsize=(12,10))
        plt.show()
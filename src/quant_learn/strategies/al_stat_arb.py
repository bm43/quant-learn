# https://shorturl.at/U7HPe

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class AvellanedaLeeStatArb:
    def __init__(self, lookback=252*2, z_entry=1.25, z_exit=0.5):  # ~1-2yr
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.pca = PCA(n_components=3)  # Top 3 factors
        
    def fit_factors_pca(self, returns: pd.DataFrame) -> tuple:
        # PCA factors
        cov = returns.cov()
        factors = self.pca.fit_transform(returns)
        betas = self.pca.components_.T  # Loadings
        return factors, betas
    
    def fit_factors_etf(self, stock_rets: pd.Series, etf_rets: pd.DataFrame) -> np.ndarray:
        # ETF regression (Method 2)
        X = etf_rets.assign(const=1)
        model = LinearRegression().fit(X, stock_rets)
        return model.coef_[:-1]  # Betas ex-const
    
    def compute_residuals(self, returns: pd.DataFrame, betas: np.ndarray) -> pd.Series:
        # Idiosyncratic residuals
        systematic = returns @ betas
        residuals = np.log(returns / systematic).mean(axis=1)  # Log-price residuals
        return (residuals - residuals.rolling(self.lookback).mean()) / residuals.rolling(self.lookback).std()
    
    def generate_signals(self, residuals: pd.Series) -> pd.DataFrame:
        # Z-score signals: long if z < -entry, short if z > entry
        z = residuals
        signals = pd.DataFrame(index=z.index, columns=['position', 'signal'])
        position = 0
        for t in z.index:
            if position == 0:
                if z[t] > self.z_entry: position = -1  # Short
                elif z[t] < -self.z_entry: position = 1  # Long
            elif abs(z[t]) < self.z_exit:
                position = 0  # Exit
            signals.loc[t, 'position'] = position
            signals.loc[t, 'signal'] = z[t]
        return signals

# Usage
# df = pd.read_csv('sp500_prices.csv', index_col=0, parse_dates=True).pct_change()
# arb = AvellanedaLeeStatArb()
# factors, betas = arb.fit_factors_pca(df)
# residuals = arb.compute_residuals(df, betas)
# signals = arb.generate_signals(residuals)

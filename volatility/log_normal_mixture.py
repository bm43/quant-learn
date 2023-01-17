# https://quant.opengamma.io/Mixed_Log-Normal-Volatility-Model.pdf
# implementation of the Normal Log-Normal Mixture Model

import math
import numpy as np
from numpy.typing import ArrayLike


class NormalLogNormalMixtureModel:

    def __init__(self) -> None:
        self.mu: float = 0.2 # initial value for mu
        self.sigma: float = 0.1 # initial value for sigma
        self.kappa: float = 0.5 # initial value for kappa
        self.num_iterations: int = 1000

    def _log_likelihood(self, data, mu, sigma, kappa) -> float:
        
        log_likelihood = 0
        
        for i in range(len(data)):
            alpha = math.log(mu * mu / math.sqrt(sigma * sigma + mu * mu))
            beta = math.sqrt(math.log(sigma * sigma / (mu * mu) + 1))
            z = math.log(data[i])
            p = (1 / (math.sqrt(2 * math.pi) * beta)) * math.exp(-0.5 * (z - alpha)**2 / beta**2)
            q = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * (z - mu)**2 / sigma**2)
            
            print('kappa: ', kappa)
            print('p: ', p)
            print('q: ', q)
            print('denom: ', kappa * p + (1 - kappa) * q)

            if p < 1e-10:
                w = 0
            else:
                w = kappa * p / (kappa * p + (1 - kappa) * q) # division by zero when p = 0 and kappa = 1 in this case

            print('w: ',w)
            print('input to log: ', w * p + (1 - w) * q)
            log_likelihood += math.log(w * p + (1 - w) * q)
            print('log_likelihood: ',log_likelihood)
        return log_likelihood

    def _EM(self, data, mu, sigma, kappa, num_iterations) -> list:
        log_likelihood_old = 0
        log_likelihood_new = self._log_likelihood(data, mu, sigma, kappa)
        
        for i in range(num_iterations):
            w = []
            mu_new = 0
            sigma_new = 0
            kappa_new = 0
            alpha = math.log(mu * mu / math.sqrt(sigma * sigma + mu * mu))
            beta = math.sqrt(math.log(sigma * sigma / (mu * mu) + 1))
            
            for j in range(len(data)):
                z = math.log(data[j])
                p = (1 / (math.sqrt(2 * math.pi) * beta)) * math.exp(-0.5 * (z - alpha)**2 / beta**2)
                q = (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * (z - mu)**2 / sigma**2)
                weight = kappa * p / (kappa * p + (1 - kappa) * q)
                w.append(weight)
                mu_new += weight * z
                sigma_new += weight * (z - mu)**2
                kappa_new += weight

            mu_new /= kappa_new
            sigma_new = math.sqrt(sigma_new / kappa_new)
            kappa_new /= len(data)

            mu = mu_new
            sigma = sigma_new
            kappa = kappa_new

            log_likelihood_old = log_likelihood_new
            log_likelihood_new = self._log_likelihood(data, mu, sigma, kappa)

        parameters = [mu, sigma, kappa]
        return parameters

    def fit(self, data):
        self._EM(data, self.mu, self.sigma, self.kappa, self.num_iterations)
        return self

if __name__ == "__main__":
    residuals = np.random.normal(23, 1.1, 1000)
    nlnmm = NormalLogNormalMixtureModel()
    parameters = nlnmm.fit(residuals)
    print("mu: ", parameters[0])
    print("sigma: ", parameters[1])
    print("kappa: ", parameters[2])
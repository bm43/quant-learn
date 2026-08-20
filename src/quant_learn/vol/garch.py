import numpy as np
from scipy.optimize import minimize

class Garch:
    """GARCH(1,1):  var_t = omega + alpha * eps_{t-1}^2 + beta * var_{t-1}"""

    def __init__(self):
        self.params = None

    @staticmethod
    def simulate(omega, alpha, beta, n, seed=None):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        eps = np.zeros(n)
        var = np.zeros(n)
        var[0] = omega / (1.0 - alpha - beta)
        eps[0] = z[0] * np.sqrt(var[0])
        for t in range(1, n):
            var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
            eps[t] = z[t] * np.sqrt(var[t])
        return eps, var

    @staticmethod
    def variance_path(eps, omega, alpha, beta):
        n = len(eps)
        var = np.zeros(n)
        var[0] = eps.var()
        for t in range(1, n):
            var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        return var

    def _neg_log_likelihood(self, params, eps):
        omega, alpha, beta = params
        var = np.maximum(self.variance_path(eps, omega, alpha, beta), 1e-12)
        return 0.5 * np.sum(np.log(var) + eps ** 2 / var)

    def fit(self, eps):
        eps = np.asarray(eps, dtype=float)
        eps = eps - eps.mean()
        start = [eps.var() * 0.05, 0.10, 0.85]
        bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]
        constraint = {"type": "ineq", "fun": lambda p: 1.0 - p[1] - p[2]}
        result = minimize(self._neg_log_likelihood, start, args=(eps,),
                          method="SLSQP", bounds=bounds, constraints=constraint)
        self.params = result.x
        return self.params
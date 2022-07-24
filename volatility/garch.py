import numpy as np

# below paper was useful when implementing this
# https://math.berkeley.edu/~btw/thesis4.pdf
class Garch:
    def __init__(self, order: tuple = (1, 1), mean: bool = True):
        self.mean = mean # whether X_t has a non-zero mean
        self.order = order
    
    def simulate_garch_process(self, params: tuple = (.001, .2, .25), size: int = 500, plot: bool = True) -> np.ndarray:
        '''
        input parameters, size, and whether to plot or not
        output is an np.array of variance that is perfectly
        garch process
        '''
        a0 = params[0]
        a1 = params[1]
        beta = params[2]

        Z_t = np.random.standard_normal(size)# white noise
        X_t = np.zeros_like(Z_t)# time series init
        var = np.zeros_like(Z_t)# std dev init


        # maybe modify so it depends on the given order?
        for t in range(1, size):
            var[t] = a0 + a1*(X_t[t-1])**2 + beta*var[t-1]
            X_t[t] = Z_t[t] * np.sqrt(var[t])


        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121), plt.plot(X_t), plt.title('X_t')
            plt.subplot(122), plt.plot(var), plt.title('variance')
            plt.show()

        return var

    def simulate_garch_volatility(self, X_t: np.ndarray, params: np.ndarray):
        omega, alpha, beta = params[0], params[1], params[2]
        vol = np.zeros(X_t.shape[0])
        vol[0] = X_t.var()
        for t in range(1, X_t.shape[0]): # length of input time series
            vol[t] = omega + alpha*X_t[t]**2 + beta*vol[t-1]
        return vol


    def quasi_max_likelihood(self, x: np.ndarray, vol: np.ndarray):
        ret = (1/x.shape[0]) * ((2*np.log(vol) + (x/vol)**2).sum())
        return ret

    def omega_constraint(self, params):
        return 1 - params[1] - params[2]

    def objective_function(self, x: np.ndarray, params: np.ndarray):
        vol = self.simulate_garch_volatility(x,params)
        return self.quasi_max_likelihood(x, vol)

    def fit(self, X_t: np.ndarray):
        # unfinished
        if self.mean:
            X_t -= X_t.mean()
        init_params = np.array([X_t.var(), .09, .9])
        bounds = [(np.finfo(np.float64).eps, 2*X_t.var(ddof=1)), (0,1), (0,1)]
        constraint = {"type": "ineq", "fun": self.omega_constraint}
        from scipy.optimize import minimize
        self.optim_params = minimize(
            self.objective_function,
            init_params,
            method = "SLSQP",
            bounds=bounds,
            args=(X_t),
            constraints=constraint,
        )['x']

        return self.optim_params

'''
g = Garch()
X_t = np.random.normal(size = 1000)
g.fit(X_t)
print(g.optim_params)
'''

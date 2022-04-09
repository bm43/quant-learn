import numpy as np

class Garch:
    def __init__(self, order=(1,1), mean=True):
        self.mean = mean
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
        sigma = np.zeros_like(Z_t)# std dev init


        # maybe modify so it depends on the given order?
        for t in range(1, size):
            sigma[t] = np.sqrt( a0 + a1*(X_t[t-1])**2 + beta*sigma[t-1] )
            X_t[t] = sigma[t] * Z_t[t]
        var = sigma**2
        import matplotlib.pyplot as plt
        plt.subplot(121), plt.plot(X_t), plt.title('X_t')
        plt.subplot(122), plt.plot(var), plt.title('variance')
        plt.show()

        return var

    def train_garch_model(self, train: np.ndarray) -> tuple:
        params = (.1, .1, .1)
        return params
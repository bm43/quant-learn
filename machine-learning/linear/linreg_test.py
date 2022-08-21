import numpy as np
from linreg import LinearRegression
import matplotlib.pyplot as plt

def generate_dumb_data(rows: int):
        
        X = np.random.normal(50, 10, rows).reshape(rows)
        #print("X shape is: ", X.shape)
        TRUEMEAN = 20
        TRUESTD = 10
        # y is column vector, make it follow a linear trend
        y = np.array([3 * X[i] + np.random.normal(TRUEMEAN, TRUESTD) for i in range(rows)]).reshape(rows,1)
        #print("y shape is: ", y.shape)
        return X, y

def test_plot(X, y, beta_hat):
    plt.scatter(X, y)
    plt.plot(X, beta_hat[0] * X + beta_hat[1], 'r')
    plt.show()
    return

X, y = generate_dumb_data(100)

lr = LinearRegression()

lr.fit(X,y)

print(lr.weights)

test_plot(X, y, lr.weights)

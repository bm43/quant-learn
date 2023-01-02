import numpy as np
from linreg import LinearRegression
import matplotlib.pyplot as plt

def generate_dumb_data(rows: int):
        
        X = np.random.normal(50, 10, rows).reshape(rows)#.reshape(-1,1)
        
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
"""
  File "<string>", line 3, in __init__
TypeError: 'numpy.ndarray' object is not callable
????????????????????????
"""

lr.fit(X, y, get_metric=True)

print(lr.theta)

test_plot(X, y, lr.theta)


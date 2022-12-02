# implementation of basic neural network
# author: Hyung Jip Lee

# X: n x p dataset, n is number of samples, p is number of features
# y: n x 1 labels
# beta: p x 1 weights


import numpy as np
from dataclasses import dataclass, field
from math import sqrt

# vanilla neural net for k-class classification task:
@dataclass
class ClassifierNeuralNet():

  #X: np.ndarray # N x p
  #y: np.ndarray # N x K labels, probability of each class for x_i

  # weights and properties:
  #alpha: np.array = field(init=False, default_factory=np.array([])) # feature extraction layer weights, N M x p matrices
  M: int = 10 # how many features are we extracting per data point x_i
  
  #beta: np.array = field(init=False, default_factory=np.array([])) # classification weights receiving extracted features as input, N M x k matrices. output of this layer goes into ridge functions
  # how many classes are there
  lr: float = 0.01

  def _xavier_init(self, p, K):
    a = self.N*self.p + self.p*self.M # no. of nodes in previous layer + no. of nodes in current layer
    self.alpha = np.random.uniform(-sqrt(6)/sqrt(a), sqrt(6)/sqrt(a), size=(self.p, self.M)) # N pxM matrices

    b = self.N*self.M + self.M*self.K
    self.beta = np.random.uniform(-sqrt(6)/sqrt(b), sqrt(6)/sqrt(b), size=(self.M, self.K)) # N Mxk matrices

  def _sigmoid(self, x):
    return 1/(1+np.exp(-x))

  def _output_func(self, T):
    eT = np.exp(T)
    return eT/sum(eT)

  def _cross_entropy_loss(self, y, yhat):
    return -y @ np.log(yhat.T)

  def _update_weights(self):
    # gradient descent
    self.alpha = self.alpha - self.lr * (self.losses[-1] - self.losses[-2])
    return

  def fit(self, X, y):
    """Trains the ppc model
    params:
    ------

    X: N x p matrix, N examples (rows) and p features (columns)
    y: N x K matrix, N examples (rows) and K classes (columns). Probability that an example from X belongs to some class.

    returns:
    --------
    self, the trained model
    """
    self.N, self.p = X.shape[0], X.shape[1]
    
    # features array
    self.K = y.shape[1] # how many classes
    Z = np.zeros((self.N, self.M)) # features
    yhat = np.zeros((self.N, self.K)) # self._ridge(Z) # map(ridge function, Z), different function for each index

    # init weights
    self._xavier_init(p, K)

    # weights on examples in dataset, applies to X[i, :]
    self.w = np.zeros((self.K, 1))
    
    # save loss values
    self.losses = [0.0]

    train_step = 0

    for i in range(self.N):
      Z[i, :] = self._sigmoid(np.matmul(X[i, :], self.alpha)) # 1xp x pxM = 1xM
      yhat[i, :] = self._output_func(np.matmul(Z[i, :], self.beta)) # apply ridge function
      self.losses.append(self._cross_entropy_loss(y[i, :], yhat[i, :]))
      self._update_weights()
    return self

  def predict(self, X) -> np.array:
    return X @ self.alpha @ self.beta # input_N x K

if __name__ == "__main__":
  N = 100
  p = 13
  K = 3
  X = np.random.normal(5,  1, size=(N, p))
  y = np.random.randint(0, 1, size=(N, K))
  
  nn = ClassifierNeuralNet()
  nn.fit(X, y)
  print(nn.alpha, nn.beta)
  
# implementation of projection pursuit regressor
# author: Hyung Jip Lee

# X: n x p dataset, n is number of samples, p is number of features
# y: n x 1 labels
# beta: p x 1 weights


import numpy as np
from dataclasses import dataclass

# vanilla nerual net for k-class classification task:
@dataclass
class ProjectionPursuitClassifier():

  X: np.ndarray # N x p
  y: np.ndarray # N x K labels, probability of each class for x_i

  # parameter and properties:
  alpha: np.ndarray # feature extraction layer weights, M x p matrix
  # M: int # how many features are we extracting per data point x_i
  
  beta: np.ndarray # classification weights receiving extracted features as input, M x k matrix. output of this layer goes into ridge functions
  # K: int # how many classes are there

  def __post_init__(self):
    self.N, self.p = self.X.shape[0], self.X.shape[1]
    
    self.M = 10 # no. of features to be extracted, = no. of columns for Z
    self._xavier_init()
    #self.alpha = np.zeros((M, 1))
    self.beta = np.zeros((M, K))

  def _xavier_init(self):
    self.alpha = np.zeros((self.N, self.M, self.p)) # N Mxp matrices
    

  def _sigmoid(self, x):
    return 1/(1+np.exp(-x))

  def fit(self, X, y):
    # features array
    K = y.shape[0]
    Z = np.zeros((N, M))
    for i in range(self.N):
      Z[i, :] = np.matmul(X[i,:], self.alpha[i, :, :]) # 1xp x pxM = 1xM

    return

  def predict(self, X):
    return

if __name__ == "__main__":
  ppr = ProjectionPursuitClassifier()
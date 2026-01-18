import numpy as np

# dimensions:
# input tokens = 1 x N list of tokens. (some text sequence, in this example)
# including [BOS] and [EOS] tokens

# 1 x N becomes N x d_k which are embeddings.
# where d_k is dimension of the model (how many embeddings per token)
# X = tokens^T x ? = (N x 1) x (1 x d_k) = N x d_k input embeddings

# then what is this 1 x d_k vector?

# then we multiply this X to get Q, K, and V:

# Q = X x W_q = (N x d_k) x (d_k x d_k) = N x d_k
# K = X x W_k = (N x d_k) x (d_k x d_k) = N x d_k
# V = X x W_v = (N x d_k) x (d_k x d_k) = N x d_k

def softmax(x: np.array):
    # numerically stable version?
    return np.exp(x) / np.sum(np.exp(x))

def attention(Q: np.array, K: np.array, V: np.array, d_k: int):
    # attention formula:
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    return np.matmul(softmax(scores), V) 

class Encoder():
    def __init__(self, d_k:int = 384, n_h: int = 12):
        self.d_k = d_k
        self.n_h = n_h
        
        self.W_q = np.random.randn(self.d_k, self.d_k) * 0.01
        self.W_k = np.random.randn(self.d_k, self.d_k) * 0.01
        self.W_v = np.random.randn(self.d_k, self.d_k) * 0.01

    
    def forward(self):
        
        return self
# implements product quantization that reduces vector memory usage
# https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
# author: Hyung Jip Lee
# 2022-09-27

"""
input vector x is split into m distinct subvectors uj , 1 ≤
j ≤ m of dimension D∗ = D/m, where D is a multiple
of m. The subvectors are quantized separately using m
distinct quantizers"""

import numpy as np
from scipy.cluster.vq import kmeans2, vq
from dataclasses import dataclass
from typing import field

@dataclass
class ProductQuantization:

    M: int # number of sub-spaces
    Dstar: int = None # dimension of each subvector in the input vector
    kstar: int = 256 # no. of codewords for each subspace
    verbose: bool = True # print log or not
    codewords: np.ndarray = None

    if kstar <= 256:
        code_dtype: np.dtype = np.uint8
    elif kstar <= 2**16:
        code_dtype: np.dtype = np.uint16
    else:
        code_dtype: np.dtype = np.uint32
    
    #codewords: np.ndarray = field(init=False, default=np.array([]))
    # codewords[i][j] = jth codeword in ith subspace

    if verbose:
            print("M: {}, Ks: {}, code_dtype: {}".format(M, kstar, code_dtype))

    def fit(self, x: np.ndarray, iter: int = 25, seed: int = 343):
        
        # check if x.dtype is np.float32, x.ndim is 2, kstar < N, D/M is int
        # assert
        assert x.dtype == np.float32
        assert x.ndim == 2
        N, D = x.shape
        assert self.Ks < N
        assert D % self.M == 0

        self.Dstar = int(D/self.M)

        np.random.seed(seed)

        self.codewords = np.zeros((self.M, self.kstar, self.Dstar), dtype=np.float32)

        for m in range(self.M):
            if self.verbose:
                print("Subspace {} / {}".format(m, self.M)) # training progress
            subv = x[:, m*self.Dstar : (m+1) * self.Dstar]
            self.codewords[m], _ = kmeans2(subv, self.kstar, iter=iter, minit="points")

        return self

# distance matrix class?
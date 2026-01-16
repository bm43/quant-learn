# implementation of pca in python
# author: Hyung Jip Lee
# 21 Oct. 2022

import numpy as np
from dataclasses import dataclass

@dataclass
class PCA:

    n_components: int = 3

    def fit(self, X: np.ndarray):
        
        # get eigen val and vec
        X_norm = X - np.mean(X, axis=0)
        covar = np.cov(X_norm, rowvar = False)
        self.eig, self.V = np.linalg.eigh(covar) # eig is eigenvalues, V is eigenvectors
        
        # sort the eigenvals and vecs in descending order
        sort_idx = np.argsort(eig)[::-1]
        self.eig = eig[sort_idx][:self.n_components]
        self.V = V[:,sort_idx][:,0:self.n_components]

        self.X_transformed = np.dot(self.eig.transpose(), X_norm.transpose()).transpose()

        return self
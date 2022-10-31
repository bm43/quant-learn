import pandas as pd
import numpy as np
from typing import field
from dataclasses import dataclass

@dataclass
class Node:
    feature = field(init=False, default=None)
    threshold = field(init=False, default=None)
    left = field(init=False, default=None)
    right = field(init=False, default=None)
    value = field(init=False, default=None)

    def is_leaf(self):
        if self.value is None:
            return False
        else:
            return True

@dataclass
class DecisionTreeClassifier:

    # dataset properties
    X: pd.DataFrame
    y: pd.Series
    n_samples: int
    n_features: int
    n_unique_labels: int

    # tree properties
    max_depth: int
    n_leaves: int
    criterion: str = "gini" # or entropy
    max_depth: int = 100
    min_samples_per_split: int = 2
    root = None
    
    def _stop(self, depth) -> bool:
        if (depth >= self.max_depth or self.n_class_labels==1 \
            or self.n_samples < self.min_samples_per_split):
            return True
        return False

    def _gini(self, y) -> float:
        # bincount counts occurences of each value
        # ex: [1,1,1] -> [0,3]
        # [1,2,3,4] -> [0,1,1,1,1]
        probs = np.bincount(y) / len(y)
        gini = -np.sum([1 - p**2 for p in probs if p > 0])
        return gini
    
    def _entropy(self, y) -> float:
        probs = np.bincount(y) / len(y)
        ent = -np.sum([p * np.log2(p) for p in probs if p > 0])
        return ent

    def _create_split(self, X, thr) -> tuple(np.array, np.array):
        # X 중에 thr 보다 높은 값인 data point = right
        left_idx = np.argwhere(X <= thr).flatten()
        right_idx = np.argwhere(X > thr).flatten()
        return left_idx, right_idx

    def _info_gain(self, X, y, thr, method):
        parent_loss = getattr(self, "_"+method)(y)
        left_idx, right_idx = self._create_split(X, thr)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left==0 or n_right==0:
            return 0

        child_loss = (n_left/n) * getattr(self, "_"+method)(y[left_idx]) \
            + (n_right/n) * getattr(self, "_"+method)(y[right_idx])

        return parent_loss - child_loss

    def _best_split(self, X, y, feats, method) -> tuple(float, float):
        split = {'score': -1, 'feat': None, 'thresh': None}
        for f in feats:
            X_feat = X[:,f]
            thresholds = np.unique(X_feat)
            for thr in thresholds:
                score = self._info_gain(X_feat, y, thr, method)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = f
                    split['threshold'] = thr
        return split['feat'], split['threshold']

    def _build_tree(self, X, y, method, depth=0) -> Node:
        self.n_samples, self.n_features = X.shape
        self.n_unique_labels = len(set(y.values))

        if self._stop(depth):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

        r_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thr = self._best_split(X, y, r_feats, method)

        left_idx, right_idx = self._create_split(X[:, best_feat], best_thr)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feat, best_thr, left_child, right_child)

    def fit(self, X: pd.DataFrame, y: pd.Series, method: str):
        self.root = self._build_tree(X, y, method)
        return self

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X: pd.DataFrame) -> np.array:
        preds = [self._traverse_tree(x, self.root) for x in X]
        return np.array(preds)
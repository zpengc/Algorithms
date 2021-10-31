import numpy
import numpy as np


def mean_squared_error(labels, prediction):
    if labels.ndim != 1:
        print("Error: Input labels must be one dimensional")

    return np.mean((labels - prediction) ** 2)


class Decision_Tree:
    def __init__(self, depth=5, min_leaf_node=5):
        self.decision_boundary = 0
        self.left = None
        self.right = None
        self.depth = depth
        self.min_leaf_node = min_leaf_node
        self.prediction = None

    def train(self, x, y):
        if x.ndim != 1:
            print("Error: Input data set must be one dimensional")
            return
        if len(x) != len(y):
            print("Error: X and y have different lengths")
            return
        if y.ndim != 1:
            print("Error: Data set labels must be one dimensional")
            return
        if len(x) < 2 * self.min_leaf_size:
            self.prediction = np.mean(y)
            return
        if self.depth == 1:
            self.prediction = np.mean(y)
            return
        best_split = 0
        min_error = mean_squared_error(x, np.mean(y)) * 2

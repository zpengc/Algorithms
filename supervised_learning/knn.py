import numpy as np
from utils.data_operations import euclidean_distance


class KNN:
    """
    k nearest neighbors classifier
    """
    def __init__(self, k=5):
        # 提前确定类别数
        self.k = k

    def vote(self, neighbor_labels):
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, x_test, x_train, y_train):
        y_pred = np.empty(x_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(x_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_sample, x) for x in x_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self.vote(k_nearest_neighbors)

        return y_pred

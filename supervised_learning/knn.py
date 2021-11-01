import numpy as np
from utils.data_operations import euclidean_distance
import matplotlib.pyplot as plt
from sklearn import datasets
from utils.data_manipulations import train_test_split
from utils.data_manipulations import normalize
from utils.data_operations import accuracy_score


class KNN:
    """
    k nearest neighbors classifier
    """
    def __init__(self, k=5):
        # 提前确定类别数
        self.k = k

    def vote(self, neighbor_labels):
        """
        根据K个最近的邻居的类别，将占比最多的类别赋给新数据
        :param neighbor_labels: K个邻居的类别
        :return: 所属类别
        """
        counts = np.bincount(neighbor_labels.astype('int'))  # 计算每个值出现的次数
        return counts.argmax()

    def predict(self, x_test, x_train, y_train):
        # 指定形状且未初始化的数组
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


if __name__ == '__main__':
    data = datasets.load_iris()
    x = normalize(data["data"])  # numpy array
    y = data["target"]  # numpy array
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)  # numpy array
    classifier = KNN(k=5)
    y_pred = classifier.predict(X_test, X_train, y_train)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

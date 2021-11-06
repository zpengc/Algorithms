import math
import numpy as np


def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


# 适合预测类别，二类别或者多类别，不适合预测自变量和因变量之间的关系
def accuracy_score(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred, axis=0) / len(y_test)  # axis = 0 表示first axis，在不同的维度数组中有不同的情况
    return accuracy


def calculate_covariance_matrix(x, y=None):
    """ Calculate the covariance matrix for data X """
    if y is None:
        y = x
    n_samples, n_features = np.shape(x)
    # 方阵，此处取平均值相当于数学期望，E( [X-E(X)]^2 )
    covariance_matrix = (1 / (n_samples-1)) * (x - x.mean(axis=0)).T.dot(y - y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


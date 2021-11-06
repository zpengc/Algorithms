import numpy as np
from sklearn import datasets
from utils.data_manipulations import train_test_split
from utils.data_operations import accuracy_score
from utils.data_operations import calculate_covariance_matrix


# https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/
class LDA:
    """
    dimension reduction
    """
    def __init__(self):
        self.w = None

    def transform(self, X, y):
        self.fit(X, y)
        # Project data onto vector
        X_transform = X.dot(self.w)
        return X_transform

    # binary classification:0 / 1
    def fit(self, x_train, y_train):
        # Separate data by class
        x_0 = x_train[y_train == 0]
        x_1 = x_train[y_train == 1]

        # Calculate the covariance matrices of the two datasets
        cov1 = calculate_covariance_matrix(x_0)
        cov2 = calculate_covariance_matrix(x_1)
        cov_tot = cov1 + cov2

        # Calculate the mean of the two datasets
        mean1 = x_0.mean(0)
        mean2 = x_1.mean(0)
        mean_diff = np.atleast_1d(mean1 - mean2)

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)  # pseudo inverse of matrix

    def predict(self, x_test):
        y_pred = []
        for sample in x_test:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred


if __name__ == '__main__':
    # 三个类别，0 1 2
    data = datasets.load_iris()
    X = data["data"]
    y = data["target"]

    # 删除类别为2的数据，标签为0/1，binary classification
    X = X[y != 2]
    y = y[y != 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

import numpy as np
from deep_learning.activation_functions import Sigmoid
import math
from utils.data_manipulations import make_diagonal
from sklearn import datasets
from utils.data_manipulations import normalize
from utils.data_manipulations import train_test_split
from utils.data_operations import accuracy_score


# https://machinelearningmastery.com/logistic-regression-for-machine-learning/
class LogisticRegression:
    """
    Logistic regression is named for the function used at the core of the method, the logistic function.
    logistic function or sigmoid: 1 / (1 + e^-value)
    A key difference from linear regression is that the output value is a binary values (0 or 1) rather than a numeric value.
    """
    def __init__(self, learning_rate=0.01, gradient_descent=True, n_iterations=4000):
        # when gradient_descent is false, we use batch optimization by least squares
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()
        self.n_iterations = n_iterations

    def initialize_parameters(self, x_train):
        n_samples, n_features = np.shape(x_train)
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        # 均匀分布，前闭后开[-limit, limit)，此时w0为0，param相当于W
        self.param = np.random.uniform(-limit, limit, n_features)

    def fit(self, x_train, y_train):
        self.initialize_parameters(x_train)

        for i in range(self.n_iterations):
            # Make a new prediction
            linear_output = x_train.dot(self.param)
            y_pred = self.sigmoid(linear_output)
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                self.param -= self.learning_rate * -(y_train - y_pred).dot(x_train)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(x_train.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(x_train.T.dot(diag_gradient).dot(x_train)).dot(x_train.T).dot(diag_gradient.dot(x_train).dot(self.param) + y_train - y_pred)

    def predict(self, x_test):
        # round_取整
        y_pred = np.round_(self.sigmoid(x_test.dot(self.param))).astype(int)
        return y_pred


if __name__ == '__main__':
    # iris数据集有三个类别,0 1 2
    data = datasets.load_iris()
    # 去掉标签为0的样本，实现binary classification
    X = normalize(data["data"][data["target"] != 0])  # 100个样本，150-50
    y = data["target"][data["target"] != 0]  # 100个标签，150-50
    # 修改标签值,binary classification
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression(gradient_descent=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


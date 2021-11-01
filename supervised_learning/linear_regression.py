import numpy as np

# Coefficient of Determination
# https://www.geeksforgeeks.org/python-coefficient-of-determination-r2-score/
def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    e1 = [a - b for a, b in zip(y_true, y_pred)]
    e1 = sum([_ ** 2 for _ in e1])
    e2 = y_true - y_mean
    e2 = sum([_ ** 2 for _ in e2])
    return 1 - float(e1 / e2)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class LinearRegression:
    def __init__(self, learning_rate=0.001, iter_num=1000):
        self.lr = learning_rate
        self.iter_num = iter_num
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        """
        train
        :param x: x_train
        :param y: y_train(correct label)
        :return:
        """
        n_samples, n_features = x.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.iter_num):
            y_pred = np.dot(x, self.weights) + self.bias
            # compute gradients according to objective function
            dw = (2 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x_test):
        y_pred = np.dot(x_test, self.weights) + self.bias
        return y_pred


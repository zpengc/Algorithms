import numpy as np
import matplotlib.pyplot as plt
from utils.data_manipulations import train_test_split
from sklearn import datasets
from utils.data_operations import accuracy_score


# Coefficient of Determination 决定系统，可决系数
# https://www.geeksforgeeks.org/python-coefficient-of-determination-r2-score/
def r2_score(y_true, y_pred):
    """ evaluate the performance of a linear regression model """
    y_mean = np.mean(y_true)
    e1 = [a - b for a, b in zip(y_true, y_pred)]
    e1 = sum([_ ** 2 for _ in e1])
    e2 = y_true - y_mean
    e2 = sum([_ ** 2 for _ in e2])
    return 1 - float(e1 / e2)


# 真实值和估计值之间的差距
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
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
        for _ in range(self.n_iterations):
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


if __name__ == '__main__':
    # linear_regression不适合accuracy_score
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("mean squared error:", mse)

    r2_score = r2_score(y_test, y_pred)
    print("r2_score:", r2_score)

    y_pred_line = model.predict(X)
    # matplotlib色带设置，viridis为默认色带
    cmap = plt.get_cmap("viridis")
    # create a figure object
    fig = plt.figure()
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('linear regression prediction')
    # 图例
    plt.legend()
    plt.show()

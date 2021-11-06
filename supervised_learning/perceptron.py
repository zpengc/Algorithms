import math
import numpy as np
from deep_learning.activation_functions import Sigmoid
from deep_learning.loss_functions import SquareLoss
import progressbar
from utils.misc import bar_widgets
from sklearn import datasets
from utils.data_manipulations import normalize
from utils.data_manipulations import to_categorical
from utils.data_manipulations import train_test_split
from deep_learning.loss_functions import accuracy_score
from deep_learning.loss_functions import CrossEntropy


# https://machinelearningmastery.com/perceptron-algorithm-for-classification-in-python/
class Perceptron:
    """
    感知机
    The Perceptron is a linear machine learning algorithm for binary classification tasks.
    """
    def __init__(self, n_iterations=20000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self.W = None
        self.w0 = None

    def fit(self, x_train, y_train):
        n_samples, n_features = np.shape(x_train)
        _, n_outputs = np.shape(y_train)

        # Initialize weights between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))

        for _ in self.progressbar(range(self.n_iterations)):
            # Calculate outputs
            # https://stackoverflow.com/questions/42517281/difference-between-numpy-dot-and-a-dotb
            linear_output = x_train.dot(self.W) + self.w0
            # add non_linear via activation function
            activation = self.activation_func(linear_output)
            # 链式法则，损失函数先对activation参数求导，activation再对xW+W0(linear_output)参数求导
            error_gradient = self.loss.gradient(y_train, activation) * self.activation_func.gradient(linear_output)
            # Calculate the gradient of the loss with respect to each weight
            grad_wrt_w = x_train.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)
            # Update weights
            self.W -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate * grad_wrt_w0

    def predict(self, x_test):
        y_pred = self.activation_func(x_test.dot(self.W) + self.w0)
        return y_pred


if __name__ == '__main__':
    data = datasets.load_digits()
    x = normalize(data["data"])
    # 10个类别：0-9
    y = data["target"]

    # One-hot encoding
    y = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, seed=1)

    # Perceptron classifier
    clf = Perceptron(n_iterations=5000,
                     learning_rate=0.001,
                     loss=CrossEntropy,
                     activation_function=Sigmoid)
    clf.fit(x_train, y_train)

    # clf_predict(x_test)返回结果为2-d数组，每一行表示一个样本，每一列表示预测结果
    # 2-d数组中，axis=1表示对于每一行进行操作
    y_pred = np.argmax(clf.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

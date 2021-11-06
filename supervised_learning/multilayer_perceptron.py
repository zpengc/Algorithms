from deep_learning.loss_functions import CrossEntropy
from deep_learning.activation_functions import Sigmoid
from deep_learning.activation_functions import Softmax
import math
from sklearn import datasets
import numpy as np
from utils.data_manipulations import normalize
from utils.data_manipulations import to_categorical
from utils.data_manipulations import train_test_split
from utils.data_operations import accuracy_score


# https://en.wikipedia.org/wiki/Multilayer_perceptron
class MultilayerPerceptron:
    """
    MLP, feedforward artificial neural network An MLP consists of at least three layers of nodes: an input layer,
    a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear
    activation function.
    """
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_layer_activation = Sigmoid()
        self.output_layer_activation = Softmax()
        self.loss = CrossEntropy()
        self.W = None
        self.w0 = None
        self.V = None
        self.v0 = None

    def initialize_weights(self, x, y):
        n_samples, n_features = x.shape
        _, n_outputs = y.shape
        # Hidden layer
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))
        # Output layer
        limit = 1 / math.sqrt(self.n_hidden)
        self.V = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))

    def fit(self, x_train, y_train):
        self.initialize_weights(x_train, y_train)

        for i in range(self.n_iterations):
            print("iteration ", i)
            print("forward pass")
            # hidden layer
            hidden_layer_input = x_train.dot(self.W) + self.w0
            hidden_layer_output = self.hidden_layer_activation(hidden_layer_input)
            # output layer
            output_layer_input = hidden_layer_output.dot(self.V) + self.v0
            output_layer_output = self.output_layer_activation(output_layer_input)

            print("backward pass")
            # OUTPUT LAYER
            # Grad. w.r.t input of output layer
            grad_wrt_out_l_input = self.loss.gradient(y_train, output_layer_output) * self.output_layer_activation.gradient(output_layer_input)
            grad_v = hidden_layer_output.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)

            # HIDDEN LAYER
            # Grad. w.r.t input of hidden layer
            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_layer_activation.gradient(hidden_layer_input)
            grad_w = x_train.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            self.V -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    def predict(self, x_test):
        # Forward pass:
        hidden_input = x_test.dot(self.W) + self.w0
        hidden_output = self.hidden_layer_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V) + self.v0
        y_pred = self.output_layer_activation(output_layer_input)
        return y_pred


if __name__ == "__main__":
    data = datasets.load_digits()
    x = normalize(data["data"])
    y = data["target"]

    # one-hot encoding
    y = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, seed=1)

    # MLP classifier
    clf = MultilayerPerceptron(n_hidden=16, n_iterations=1000, learning_rate=0.01)

    clf.fit(x_train, y_train)
    y_pred = np.argmax(clf.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


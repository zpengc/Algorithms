import numpy as np
from utils.data_operations import accuracy_score


class Loss:
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        return 0.5 * (y_true - y_pred) ** 2

    def gradient(self, y_true, y_pred):
        return -(y_true - y_pred)


class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        # Avoid division by zero
        # Clip (limit) the values in an array
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y_true * np.log(p) - (1 - y_true) * np.log(1 - p)

    def acc(self, y_true, y_pred):
        return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    def gradient(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # 对p求导
        return - (y_true / p) + (1 - y_true) / (1 - p)

import numpy
import numpy as np


# https://www.geeksforgeeks.org/decision-tree/


class Decision_Tree:
    def __init__(self, depth=5, min_leaf_size=5):
        self.decision_boundary = 0
        # left subtree
        self.left = None
        # right subtree
        self.right = None
        self.depth = depth
        self.min_leaf_size = min_leaf_size
        self.prediction = None

    def mean_squared_error(self, labels, prediction):
        """
        >>> tester = Decision_Tree()
        >>> test_labels = np.array([1,2,3,4,5,6,7,8,9,10])
        >>> test_prediction = float(6)
        >>> tester.mean_squared_error(test_labels, test_prediction) == (
        ...     Test_Decision_Tree.helper_mean_squared_error_test(test_labels,
        ...         test_prediction))
        True
        >>> test_labels = np.array([1,2,3])
        >>> test_prediction = float(2)
        >>> tester.mean_squared_error(test_labels, test_prediction) == (
        ...     Test_Decision_Tree.helper_mean_squared_error_test(test_labels,
        ...         test_prediction))
        True
        """
        if labels.ndim != 1:
            print("Error: Input labels must be one dimensional")
        # list broadcast
        return np.mean((labels - prediction) ** 2)

    def train(self, x, y):
        if x.ndim != 1:
            print("Error: Input data set must be one dimensional")
            return
        if len(x) != len(y):
            print("Error: X and y have different lengths")
            return
        if y.ndim != 1:
            print("Error: Data set labels must be one dimensional")
            return
        if len(x) < 2 * self.min_leaf_size:
            self.prediction = np.mean(y)
            return
        if self.depth == 1:
            self.prediction = np.mean(y)
            return
        best_split = 0
        min_error = self.mean_squared_error(x, np.mean(y)) * 2

        for i in range(len(x)):  # split the tree given node i as pivot
            # index from 0 to i-1
            if len(x[:i]) < self.min_leaf_size:
                continue
                # index from i to len(x)-1
            elif len(x[i:]) < self.min_leaf_size:
                continue
            else:
                error_left = self.mean_squared_error(x[:i], np.mean(y[:i]))
                error_right = self.mean_squared_error(x[i:], np.mean(y[i:]))
                error = error_left + error_right
                if error < min_error:
                    best_split = i
                    min_error = error
            if best_split != 0:
                left_X = x[:best_split]
                left_y = y[:best_split]
                right_X = x[best_split:]
                right_y = y[best_split:]

                self.decision_boundary = x[best_split]
                self.left = Decision_Tree(
                    depth=self.depth - 1, min_leaf_size=self.min_leaf_size
                )
                self.right = Decision_Tree(
                    depth=self.depth - 1, min_leaf_size=self.min_leaf_size
                )
                self.left.train(left_X, left_y)
                self.right.train(right_X, right_y)
            else:
                self.prediction = np.mean(y)
        return

    def predict(self, x_test):
        if self.prediction is not None:
            return self.prediction
        elif self.left or self.right is not None:
            if x_test >= self.decision_boundary:
                return self.right.predict(x_test)
            else:
                return self.left.predict(x_test)
        else:
            print("Error: Decision tree not yet trained")
            return None


class Test_Decision_Tree:

    @staticmethod
    def helper_mean_squared_error_test(labels, prediction):
        squared_error_sum = float(0)
        for label in labels:
            squared_error_sum += (label - prediction) ** 2

        return float(squared_error_sum / labels.size)


def main():
    # [-1.0, 1.0) with step 0.005
    x_train = np.arange(-1.0, 1.0, 0.05)
    print("x_train size is {}".format(len(x_train)))
    y_train = np.sin(x_train)
    tree = Decision_Tree(depth=10, min_leaf_size=10)
    tree.train(x_train, y_train)

    x_test = (np.random.rand(10) * 2) - 1
    y_test = np.array([tree.predict(_) for _ in x_test])
    avg_error = np.mean((y_test - x_test) ** 2)

    print("Test values: " + str(x_test))
    print("Predictions: " + str(y_test))
    print("Average error: " + str(avg_error))


if __name__ == '__main__':
    main()
    # https://www.geeksforgeeks.org/testing-in-python-using-doctest-module/
    import doctest
    doctest.testmod(name="mean_squared_error", verbose=True)

import numpy as np
import math
from utils.data_manipulations import train_test_split
from utils.data_manipulations import normalize
from sklearn import datasets
from utils.data_operations import accuracy_score


class NaiveBayes:
    """The Gaussian Naive Bayes classifier. """

    def __init__(self):
        self.x = None
        self.y = None
        self.classes = None
        self.parameters = None

    def fit(self, x_train, y_train):
        self.x, self.y = x_train, y_train
        self.classes = np.unique(y_train)
        self.parameters = []

        # Calculate the mean and variance of each feature for each class
        for idx, cls in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            X_where_c = x_train[np.where(y_train == cls)]
            self.parameters.append([])
            # Add the mean and variance for each feature (column)
            for row in X_where_c.T:
                # col表示所有样本的某一个维度/feature
                parameters = {"mean": row.mean(), "var": row.var()}
                self.parameters[idx].append(parameters)

    def calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        # Added in denominator to prevent division by zero
        eps = 1e-4
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def calculate_prior(self, c):
        """ Calculate the prior of class c
        (samples where class == c / total number of samples)"""
        frequency = np.mean(self.y == c)
        return frequency

    def classify(self, sample):
        """ Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
            or Posterior = Likelihood * Prior / evidence
        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        # Go through list of classes
        for i, cls in enumerate(self.classes):
            # Initialize posterior as prior
            posterior = self.calculate_prior(cls)
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Posterior is product of prior and likelihoods (ignoring evidence)
            for feature_value, params in zip(sample, self.parameters[i]):
                # Likelihood of feature value given distribution of feature values given y
                likelihood = self.calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # Return the class with the largest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, x_test):
        y_pred = [self.classify(sample) for sample in x_test]
        return y_pred


if __name__ == '__main__':
    data = datasets.load_digits()
    X = normalize(data["data"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)


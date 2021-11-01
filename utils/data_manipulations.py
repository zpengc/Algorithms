import numpy as np


def train_test_split(x, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = x[:split_i], x[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return x_train, x_test, y_train, y_test


def shuffle_data(x, y, seed=None):
    """ Random shuffle of the samples in x and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(x.shape[0])  # idx from 0 to x.shape[0]-1
    np.random.shuffle(idx)  # 打乱
    return x[idx], y[idx]


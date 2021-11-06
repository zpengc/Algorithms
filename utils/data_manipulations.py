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


def normalize(x, axis=-1, order=2):
    """ Normalize x data in dataset """
    # order=2表示取2-norm范数，axis=-1表示不返回一个值，返回一个1-d数组
    l2 = np.linalg.norm(x, order, axis)
    # modify 0 real number to 1 real number
    l2[l2 == 0] = 1  # l2存储每个样本数据的2-norm值
    return x / np.expand_dims(l2, axis)  # broadcast, 自动插入一个维度


# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
# see source code for keras
# https://github.com/keras-team/keras/blob/2d183db0372e5ac2a686608cb9da0a9bd4319764/keras/utils/np_utils.py#L9
def to_categorical(y, dimensions=None):
    """ One-hot encoding of label values """
    if not dimensions:
        # y标签里面最大值，因为y标签从0开始，所有加上1
        dimensions = np.amax(y) + 1
    one_hot = np.zeros((y.shape[0], dimensions))
    # np.arange(y.shape[0])元素依次表示行，y元素依次表示列
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# np.diag函数实现相同功能
# https://stackoverflow.com/questions/28598572/how-to-convert-a-column-or-row-matrix-to-a-diagonal-matrix-in-python
def make_diagonal(vector):
    """ Converts a vector into an diagonal matrix """
    dimension = len(vector)
    matrix = np.zeros((dimension, dimension))
    for i in range(len(matrix[0])):
        # 对角线元素依次是vector的元素值
        matrix[i, i] = vector[i]
    return matrix

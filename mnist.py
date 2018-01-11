import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist(data_dir):
    # Load the data
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    # Pre-process the data
    X_train = np.reshape(mnist.train.images, newshape=(-1, 28, 28, 1))
    Y_train = np.reshape(mnist.train.labels, newshape=(-1, 10))
    X_test = np.reshape(mnist.test.images, newshape=(-1, 28, 28, 1))
    Y_test = np.reshape(mnist.test.labels, newshape=(-1, 10))
    X_val = np.reshape(mnist.validation.images, newshape=(-1, 28, 28, 1))
    Y_val = np.reshape(mnist.validation.labels, newshape=(-1, 10))

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


def batch_indices(batch_nb, data_length, batch_size):
    """
    Source: Cleverhans Library: https://github.com/tensorflow/cleverhans/blob/master/cleverhans/utils.py
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

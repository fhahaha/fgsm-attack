import tensorflow as tf
import numpy as np


def flatten(input_shape, x, name='flatten'):
    output_dim = 1

    # Collect all dimensions
    for dim in input_shape[1:]:
        output_dim *= dim

    # Reshape the input accordingly
    output_shape = [input_shape[0], output_dim]
    output = tf.reshape(x, shape=(-1, output_dim))

    return output, output_shape


def linear(input_shape, n_hidden, x, name='linear'):
    output_shape = [input_shape[0], n_hidden]

    init = tf.random_normal([input_shape[1], n_hidden], stddev=1.0, dtype=tf.float32, seed=1234)

    init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0, keep_dims=True))

    W = tf.get_variable(name + '_weights', dtype=tf.float32, initializer=init)

    b = tf.get_variable(name + '_biases', shape=[n_hidden], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

    output = tf.nn.xw_plus_b(x, W, b)

    return output, output_shape


def relu(x, name='relu'):
    activation = tf.nn.relu(x)
    return tf.identity(activation, name=name)


def lrelu(x, a=.1, name='lrelu'):
    if a < 1e-16:
        activation = tf.nn.relu(x)
    else:
        activation = tf.maximum(x, a * x)
    return tf.identity(activation, name=name)


def tanh(x, name='tanh'):
    activation = tf.nn.tanh(x)
    return tf.identity(activation, name=name)


def non_linearity(x, name):
    if name == 'relu':
        return relu(x)
    elif name == 'lrelu':
        return lrelu(x)
    elif name == 'tanh':
        return tanh(x)
    else:
        raise NotImplementedError


def softmax(x, name='softmax'):
    activation = tf.nn.softmax(x)
    return tf.identity(activation, name=name)


def cross_entropy_loss(logits, labels, name='cross-entropy-loss'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy = tf.reduce_mean(diff)
    return tf.identity(cross_entropy, name=name)


def sigmoid_cross_entropy_loss(logits, labels, name='cross-entropy-loss'):
    diff = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy = tf.reduce_mean(diff)
    return tf.identity(cross_entropy, name=name)


def accuracy(logits, y, name='accuracy'):
    pred = tf.argmax(tf.nn.softmax(logits), axis=1)
    ground_truth = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(pred, ground_truth)
    acc = tf.reduce_mean(tf.to_float(correct_prediction))
    return tf.identity(acc, name=name)


def conv2d(x, kernel_size, stride, filters_in, filters_out, padding, name='conv'):
    # input_shape = (-1, W1, H1, D1)        (tensor)
    # kernel = (kernel_size, kernel_size)   (matrix)
    # stride = stride                       (scalar)
    # n_filters = filters_out               (scalar)
    # out_height = ceil(float(in_height) / float(strides[1]))
    # out_width = ceil(float(in_width) / float(strides[2]))

    kernel_shape = tuple((kernel_size, kernel_size)) + (filters_in, filters_out)

    assert len(kernel_shape) == 4
    assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

    init = tf.random_normal(kernel_shape, stddev=1.0, dtype=tf.float32, seed=1234)

    init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=(0, 1, 2)))

    W = tf.get_variable(name + '_weights', dtype=tf.float32, initializer=init)

    bias = tf.get_variable(name + '_biases', shape=(filters_out,), dtype=tf.float32,
                           initializer=tf.zeros_initializer())

    x = tf.nn.conv2d(x, W, (1,) + tuple((stride, stride)) + (1,), padding=padding) + bias

    output_shape = x.get_shape().as_list()

    return x, output_shape


def normalize(x, epsilon=1e-12):
    x_shape = tf.shape(x)
    x = tf.contrib.layers.flatten(x)  # flattens the vector by keeping batches
    x /= (epsilon + tf.reduce_max(tf.abs(x), 1, keep_dims=True))
    square_sum = tf.reduce_sum(tf.square(x), 1, keep_dims=True)
    x_inv_norm = tf.rsqrt(np.sqrt(epsilon) + square_sum)
    x_normalized = tf.multiply(x, x_inv_norm)
    x_normalized = tf.reshape(x_normalized, x_shape)

    return x_normalized

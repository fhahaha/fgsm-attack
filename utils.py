import os
import tensorflow as tf


def create_dir(root_dir, sub_dir=''):
    new_dir = root_dir + '/{:s}'.format(sub_dir)

    # Create the dir if necessary
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    return new_dir


def variable_summaries(var, name=None):
    if not name:
        name = var.op.name

    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + '/stddev', stddev)
    tf.summary.scalar(name + '/max', tf.reduce_max(var))
    tf.summary.scalar(name + '/min', tf.reduce_min(var))
    tf.summary.histogram(name, tf.identity(var))

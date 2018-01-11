import tensorflow as tf
import layers as l

from models import SimpleNet1 as model

# from models import ConvNet as model


# MNIST constants
INPUT_SHAPE = (None, 28, 28, 1)
N_CLASSES = 10


def logits(x, create_summaries):
    logits = model(x, input_shape=INPUT_SHAPE, n_classes=N_CLASSES, create_summaries=create_summaries)
    return logits


def forward(x, create_summaries):
    return logits(x, create_summaries)


def generate_random_perturbation(x, ord='inf', epsilon=0.3):
    noise = tf.random_normal(shape=tf.shape(x), dtype=tf.float32)

    if ord == 'inf':
        normalized_noise = tf.sign(noise)
        scaled_noise = epsilon * normalized_noise

    if ord == 'l2':
        normalized_noise = get_normalized_vector(noise)
        scaled_noise = epsilon * normalized_noise

    return scaled_noise


def generate_adversarial_perturbation(x, y=None, ord='inf', epsilon=0.3):
    logits = forward(x, create_summaries=False)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds = tf.nn.softmax(logits)
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
        y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # stable_logits = logits - tf.reduce_max(logits, axis=1, keep_dims=True)
    stable_logits = logits * 1.0
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=stable_logits, labels=y)
    loss = tf.identity(loss, name='fgsm-loss')

    raw_grad, = tf.gradients(loss, x)

    if ord == 'inf':
        normalized_grad = tf.sign(raw_grad)
        normalized_grad = tf.stop_gradient(normalized_grad)
        scaled_grad = epsilon * normalized_grad

    if ord == 'l2':
        normalized_grad = get_normalized_vector(raw_grad)
        scaled_grad = epsilon * normalized_grad

    return scaled_grad, loss, raw_grad


def random_loss(x, y, ord='inf', epsilon=0.3, name='rnd-loss'):
    perturbation = generate_random_perturbation(x, ord, epsilon)
    x_adv = x + perturbation
    x_adv = tf.clip_by_value(x_adv, clip_value_min=0.0, clip_value_max=1.0)

    adv_logit = forward(x_adv, create_summaries=False)
    adv_loss = l.cross_entropy_loss(adv_logit, y)

    return tf.identity(adv_loss, name=name), perturbation, x_adv


def adversarial_loss(x, y, ord='inf', epsilon=0.3, name='advt-loss'):
    perturbation, _, _ = generate_adversarial_perturbation(x, ord=ord, epsilon=epsilon)
    x_adv = x + perturbation
    x_adv = tf.clip_by_value(x_adv, clip_value_min=0.0, clip_value_max=1.0)

    adv_logit = forward(x_adv, create_summaries=False)
    adv_loss = l.cross_entropy_loss(adv_logit, y)

    return tf.identity(adv_loss, name=name), perturbation, x_adv


def get_normalized_vector(d):
    return l.normalize(d)

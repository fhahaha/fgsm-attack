import os
import argparse
import math
from time import localtime, strftime

import numpy as np

import layers
import train_utils
import gpu_utils
import utils

FLAGS = None


def build_training_graph(x, y, learning_rate, method, optimizer, decay_step):
    print('\nBuilding training graph for method {:s}'.format(method))
    print('Using optimizer {:s}'.format(optimizer))

    # Define global step variable
    global_step = tf.get_variable(
        name='global_step',
        shape=[],  # scalar
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False
    )
    # Build the network
    with tf.name_scope('Logits'):
        logits = train_utils.forward(x, create_summaries=True)

    # Build the network
    with tf.name_scope('Predictions'):
        predictions = layers.softmax(logits)
        utils.variable_summaries(predictions, name='softmax-predictions')

    # Create an op for the loss
    with tf.name_scope('Cross-Entropy-Loss'):
        ce_loss = layers.cross_entropy_loss(logits, y)
        tf.add_to_collection('losses', ce_loss)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):

        if method == 'random':
            with tf.name_scope('RND-Adversarial-Training'):
                rnd_loss, perturbation, x_adv = train_utils.random_loss(x, y, ord='l2', epsilon=3.0)
                additional_loss = rnd_loss
                tf.add_to_collection('losses', rnd_loss)
        elif method == 'advt':
            with tf.name_scope('FGSM-Adversarial-Training'):
                advt_loss, perturbation, x_adv = train_utils.adversarial_loss(x, y, ord='l2', epsilon=3.0)
                additional_loss = advt_loss
                tf.add_to_collection('losses', advt_loss)
        else:
            perturbation = None
            x_adv = None

    # Create an op for the total loss
    with tf.name_scope('Loss'):
        if method == 'advt' or method == 'random':
            loss = (ce_loss + additional_loss) / 2
            loss = tf.identity(loss, name='total-loss')
            tf.add_to_collection('losses', loss)
        else:
            loss = ce_loss
            loss = tf.identity(loss, name='total-loss')
            tf.add_to_collection('losses', loss)

    # Create the optimizer
    with tf.name_scope('Optimizer'):

        # Implement additional learning rate decay
        if decay_step:
            print('Using exponential learning rate decay every {:.2f} steps'.format(decay_step))
            starter_learning_rate = learning_rate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate=0.9,
                                                       staircase=True)
        else:
            learning_rate = tf.constant(learning_rate)

        if optimizer == 'vanilla':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.5)
        elif optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=.1)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError

        trainable_vars = tf.trainable_variables()
        grads_and_vars = optimizer.compute_gradients(loss, trainable_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return loss, train_op, global_step, grads_and_vars, optimizer, perturbation, x_adv


def build_eval_graph(x, y, scope):
    losses = {}

    with tf.variable_scope(scope, reuse=True):
        with tf.name_scope('Logits'):
            logits = train_utils.forward(x, create_summaries=False)

        # Create an op for the loss
        with tf.name_scope('Cross-Entropy-Loss'):
            ce_loss = layers.cross_entropy_loss(logits, y)
            losses['LOSS'] = ce_loss

        # Create op for the accuracy
        with tf.name_scope('Accuracy'):
            acc = layers.accuracy(logits, y)
            losses['ACC'] = acc
            losses['ERR'] = 1.0 - acc

        # Create op for the accuracy against FGSM attacker
        with tf.name_scope('Adversarial-Accuracy'):
            fgsm_perturbation, loss, raw_grad = train_utils.generate_adversarial_perturbation(x, ord='inf',
                                                                                              epsilon=FLAGS.eps_fgsm)

            # Compute adversarial examples
            fgsm_x_adv_unclipped = x + fgsm_perturbation
            fgsm_x_adv = tf.clip_by_value(fgsm_x_adv_unclipped, clip_value_min=0.0, clip_value_max=1.0)
            fgsm_x_adv = tf.stop_gradient(fgsm_x_adv)  # do not backprop through attack
            adv_logits = train_utils.forward(fgsm_x_adv, create_summaries=False)

            # Collect gradients
            raw_grad = tf.contrib.layers.flatten(raw_grad)
            raw_grad = tf.identity(raw_grad, name='fgsm-grad')
            raw_grad_norm = tf.norm(raw_grad, ord='euclidean')
            raw_grad_norm = tf.identity(raw_grad_norm, name='fgsm-grad-l2norm')
            losses['FGSM-GRAD-NORM'] = raw_grad_norm

            # Collect perturbation
            perturbation = tf.contrib.layers.flatten(fgsm_perturbation)
            perturbation = tf.identity(perturbation, name='fgsm-perturbation')
            tf.summary.histogram(perturbation.op.name, perturbation)
            perturbation_norm = tf.norm(perturbation, ord='euclidean')
            perturbation_norm = tf.identity(perturbation_norm, name='fgsm-perturbation-l2norm')
            losses['FGSM-PERTURBATION-NORM'] = perturbation_norm

            # Collect accuracy
            adv_acc = layers.accuracy(adv_logits, y)
            losses['ADVT-ACC'] = adv_acc
            losses['ADVT-ERR'] = 1.0 - adv_acc

    return losses, fgsm_perturbation, fgsm_x_adv


def evaluate(sess, x, y, fgsm_perturbation, fgsm_x_adv, X, Y, epoch, batch_size, losses_eval, file_writer):
    # Compute number of batches
    batches = int(math.ceil(float(len(X)) / FLAGS.batch_size))
    indices = list(range(len(X)))

    # Dictionary used to store evaluated metrics
    losses_eval_val = {}

    # TODO: this is not optimal
    # Create image summary of for clean images
    clean_images = tf.summary.image('test-images', x, max_outputs=2)
    # Create image summary op for FGSM adversarial perturbations
    perturbations = tf.summary.image('fgsm-adversarial-test-perturbations', fgsm_perturbation, max_outputs=2)
    # Create image summary op for FGSM adversarial images
    fgsm_images = tf.summary.image('fgsm-adversarial-test-images', fgsm_x_adv, max_outputs=2)

    for key, value in losses_eval.items():
        losses_eval_val[key] = 0.0

    for batch in range(batches):
        # Compute batch start and end indices
        start, end = batch_indices(
            batch, len(X), batch_size)

        losses, clean_images_summary, perturbations_summary, fgsm_images_summary = sess.run(
            [losses_eval, clean_images, perturbations, fgsm_images],
            feed_dict={x: X[indices[start:end]],
                       y: Y[indices[start:end]]})

        # Wirte image summaries
        file_writer.add_summary(clean_images_summary, epoch)
        file_writer.add_summary(perturbations_summary, epoch)
        file_writer.add_summary(fgsm_images_summary, epoch)

        for key, value in losses.items():
            losses_eval_val[key] += value  # accumulate loss values for each batch

    for key, value in losses_eval.items():
        losses_eval_val[key] /= batches  # compute average

        summary = tf.Summary()
        summary.value.add(tag='Evaluate/{:s}'.format(key), simple_value=losses_eval_val[key])
        file_writer.add_summary(summary, epoch + 1)

        if key == 'LOSS':
            print('Epoch: {:d}, Loss on test data: {:.4f}'.format(epoch + 1, losses_eval_val[key]))

        if key == 'ACC':
            print('Epoch: {:d}, Accuracy on test data: {:.4f}'.format(epoch + 1, losses_eval_val[key]))


def main(_):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    np.random.seed(1234)

    with tf.device('/cpu:0'):
        # Get time stamp
        experiment_ts = strftime("%H-%M-%S", localtime())

        # Create log and checkpoint dir for the current experiment
        FLAGS.log_dir += '/{:s}/{:s}'.format(FLAGS.optimizer, experiment_ts)
        utils.create_dir(FLAGS.log_dir)

        # Get training data
        X_train, Y_train, X_test, Y_test, X_val, Y_val = load_mnist(FLAGS.data_dir)

        # Adding validation data to training data (60.000 training data)
        X_train = np.append(X_train, X_val, axis=0)
        Y_train = np.append(Y_train, Y_val, axis=0)

        print('X_train shape: ', X_train.shape)
        print('X_test shape: ', X_test.shape)

        # Repeat training for all specified models
        for method in FLAGS.methods:

            # Create log dir
            log_dir = FLAGS.log_dir + '/{:s}'.format(method)
            utils.create_dir(log_dir)

            with tf.Graph().as_default() as g:

                with tf.device(FLAGS.device):
                    # Define placeholders for inputs
                    with tf.name_scope('Inputs'):
                        # Inputs
                        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='X')
                        y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

                    with tf.variable_scope('model') as scope:
                        with tf.name_scope('Training-Graph'):
                            # Build the model
                            loss, train_op, global_step, grads_and_vars, optimizer, perturbation, x_adv = build_training_graph(
                                x,
                                y,
                                FLAGS.learning_rate,
                                method,
                                FLAGS.optimizer,
                                FLAGS.decay_step)

                            with tf.device('/cpu:0'):
                                # Add summaries for the training losses
                                with tf.name_scope('Loss-Summaries'):
                                    losses = tf.get_collection('losses')
                                    for entry in losses:
                                        tf.summary.scalar(entry.op.name, entry)

                                # Add histograms for all trainable variables and gradients
                                with tf.name_scope('Trainable-Variable-Summaries'):
                                    for var in tf.trainable_variables():
                                        utils.variable_summaries(var)
                                    for grad, var in grads_and_vars:
                                        if grad is not None:
                                            tf.summary.histogram(var.op.name + '/gradients', grad)
                                            grad_norm = tf.norm(grad, ord='euclidean')
                                            tf.summary.scalar(var.op.name + '/gradients-l2norm', grad_norm)

                        with tf.name_scope('Eval-Graph'):
                            # Create ops used for evaluating the model on test data
                            losses_eval, fgsm_perturbation, fgsm_x_adv = build_eval_graph(x, y, scope)

                        with tf.name_scope('Image-Summary'):
                            # Create image summary op for clean images
                            tf.summary.image('training-images', x, max_outputs=2)
                            if perturbation is not None and x_adv is not None:
                                # Create image summary op for FGSM adversarial images
                                tf.summary.image('fgsm-adversarial-training-perturbations', perturbation, max_outputs=2)
                                # Create image summary op for FGSM adversarial images
                                tf.summary.image('fgsm-adversarial-training-images', x_adv, max_outputs=2)

                    # Create init op
                    with tf.name_scope('Initializer'):
                        init_op = tf.global_variables_initializer()

                    # Create file writer for TensorBoard
                    with tf.device('/cpu:0'):
                        writer_train = tf.summary.FileWriter(log_dir + '/train', graph=g)
                        writer_test = tf.summary.FileWriter(log_dir + '/test')

                    # Merge all the summaries
                    merged = tf.summary.merge_all()

            # Create tf session
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            with tf.Session(graph=g, config=config) as sess:

                # Initialize all variables
                sess.run(init_op)

                print('\nStart training for {:d} epochs.'.format(FLAGS.epochs))

                # Compute number of batches
                batches = int(math.ceil(float(len(X_train)) / FLAGS.batch_size))
                print('Performing {:d} updates per epoch.'.format(batches))

                print('\nEvaluate on test data...')
                evaluate(sess, x, y, fgsm_perturbation, fgsm_x_adv, X_test, Y_test, -1, FLAGS.batch_size, losses_eval,
                         writer_test)

                # Training loop
                for epoch in range(FLAGS.epochs):

                    print('\nStart of epoch {:d}...'.format(epoch + 1))

                    # Write summaries for optimizer parameters
                    # TODO: For adaptive learning rate methods this does not log the adapted learning rate
                    if FLAGS.optimizer == 'vanilla':
                        learning_rate_val = optimizer._learning_rate
                    elif FLAGS.optimizer == 'momentum':
                        learning_rate_val = optimizer._learning_rate_tensor
                    elif FLAGS.optimizer == 'adagrad':
                        learning_rate_val = optimizer._learning_rate_tensor
                    elif FLAGS.optimizer == 'adam':
                        learning_rate_val = optimizer._lr_t
                    else:
                        raise NotImplementedError

                    learning_rate_val = sess.run(learning_rate_val)
                    summary = tf.Summary()
                    summary.value.add(tag='Optimizer/{:s}'.format('learning-rate'), simple_value=learning_rate_val)
                    writer_train.add_summary(summary, epoch)

                    # Shuffle training data
                    indices = list(range(len(X_train)))
                    np.random.shuffle(indices)

                    # Iterate through the training data in batches
                    sum_batch_loss = 0.0
                    for batch in range(batches):
                        # Compute batch start and end indices
                        start, end = batch_indices(
                            batch, len(X_train), FLAGS.batch_size)

                        feed_dict = {x: X_train[indices[start:end]],
                                     y: Y_train[indices[start:end]]}

                        # Perform a single step of stochastic gradient descent
                        batch_summaries, _, batch_loss, step = sess.run([merged, train_op, loss, global_step],
                                                                        feed_dict=feed_dict)

                        # Accumulate the loss
                        sum_batch_loss += batch_loss

                        # Write tensorboard summaries
                        writer_train.add_summary(batch_summaries, step)

                    print('Epoch: {:d}, Cross-Entropy-Loss (training data): {:.4f}'.format(epoch + 1,
                                                                                           sum_batch_loss / batches))

                    # Evaluate on test data
                    if epoch % FLAGS.eval_step == 0 or epoch + 1 == FLAGS.epochs:
                        print('\nEvaluate on test data...')
                        evaluate(sess, x, y, fgsm_perturbation, fgsm_x_adv, X_test, Y_test, epoch, FLAGS.batch_size,
                                 losses_eval, writer_test)

                print('\nPerformed {:.2f} training iterations.'.format(sess.run(global_step)))


if __name__ == "__main__":

    # Example how to run a single experiment:
    # python main.py --methods 'baseline' 'advt' --optimizer='adam' --learning_rate=0.001

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data/mnist',
                        help='Directory for input data.')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for log data.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for storing trained models.')
    parser.add_argument('--eps_fgsm', type=float, default=0.3,
                        help='Strength of the FGSM perturbation (either L2 or Linf norm).')
    parser.add_argument('--lambda', type=float, default=1.0,
                        help='Strength of the regularization term.')
    parser.add_argument('--methods', type=str, nargs='+',
                        help='n of {baseline, random, advt}.')
    parser.add_argument('--optimizer', type=str, default='vanilla',
                        help='One of {vanilla, momentum, adagrad, adam}.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for SGD.')
    parser.add_argument('--decay_step', type=float, default=0.0,
                        help='Decay the learning rate every decay_step.')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs.')
    parser.add_argument('--eval_step', type=int, default=1,
                        help='Evaluate all n epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of samples in a single batch.')
    # If you specify anything of the form --gpu=something this counts as True
    parser.add_argument('--use_gpu', type=bool, default=False,
                        help='Whether to run on gpu.')
    parser.add_argument('--device', type=str, default='/cpu:0',
                        help='Device to create variables on.')

    # Parse command line arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Print command line arguments
    print('\nCommand line arguments:')
    print(FLAGS, '\n')

    if FLAGS.use_gpu:
        BEST_GPU = gpu_utils.find_best_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(BEST_GPU)

        import tensorflow as tf

        FLAGS.device = '/gpu:0'
        print('Setting FLAGS.device = {:s}'.format(FLAGS.device))

    else:
        import tensorflow as tf

    # Load the specified data set
    from mnist import load_mnist, batch_indices

    tf.app.run(main=main)

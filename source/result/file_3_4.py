#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/11 15:22
# @Author  : Jackokie Zhao
# @Site    : www.jackokie.com
# @File    : file_3_4.py
# @Software: PyCharm
# @contact: jackokie@gmail.com

# This coding is the modulation of a paper(https://arxiv.org/pdf/1602.04105.pdf).
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib
from sklearn.manifold import TSNE
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow.contrib import layers

num_epoch = 100
batch_size = 1024
learning_rate = 0.01
train_ratio = 0.9
log_dir = './log/'
orig_file_path = '/home/scl/Documents/ModData/RML2016.10a_dict.dat'
[height, width] = [2, 128]
num_channels = 1
num_kernel_1 = 64
num_kernel_2 = 32
hidden_units_1 = 64
hidden_units_2 = 16
dropout = 0.5
num_classes = 7
train_show_step = 100
test_show_step = 1000
seed = 'jackokie'
reg_val_l1 = 0.001
reg_val_l2 = 0.001


def load_data(data_path, input_shape):
    """ Load the original data for training...
    Parameters:
        data_path: The original data path.
        input_shape:
    Returns:
        train_data: Training data structured.
    """
    # load the original data.
    orig_data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')

    # Get the set of snr & modulations
    mode_snr = list(orig_data.keys())
    mods, snrs = [sorted(list(set(x[i] for x in mode_snr))) for i in [0, 1]]
    mods.remove('BPSK')
    mods.remove('8PSK')
    mods.remove('QAM16')
    mods.remove('QAM64')

    # Build the train set.
    samples = []
    labels = []
    samples_snr = []
    mod2cate = dict()
    cate2mod = dict()
    for cate in range(len(mods)):
        cate2mod[cate] = mods[cate]
        mod2cate[mods[cate]] = cate

    for snr in snrs:
        for mod in mods:
            samples.extend(orig_data[(mod, snr)])
            labels.extend(1000 * [mod2cate[mod]])
            samples_snr.extend(1000 * [snr])

    shape = [len(labels), height, width, 1]
    samples = np.array(samples).reshape(shape)
    samples_snr = np.array(samples_snr)
    labels = np.array(labels)
    return samples, labels, mod2cate, cate2mod, snrs, mods, samples_snr


def accuracy_compute(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels.
    Parameters:
        predictions: The prediction logits matrix.
        labels: The real labels of prediction data.
    Returns:
        accuracy: The predictions' accuracy.
    """
    with tf.name_scope('test_accuracy'):
        accu = 100 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
        tf.summary.scalar('test_accuracy', accu)
    return accu


def conv(data, kernel_shape, activation, name, dropout=1, regularizer=None, reg_val=0):
    """ Convolution layer.
    Parameters:
        data: The input data.
        kernel_shape: The kernel_shape of current convolutional layer.
        activation: The activation function.
        name: The name of current layer.
        dropout: Whether do the dropout work.
        regularizer: Whether use the L2 or L1 regularizer.
        reg_val: regularizer value.
    Return:
        conv_out: The output of current layer.
    """
    if regularizer == 'L1':
        regularizer = layers.l1_regularizer(reg_val)
    elif regularizer == 'L2':
        regularizer = layers.l2_regularizer(reg_val)

    with tf.name_scope(name):
        # Convolution layer 1.
        with tf.variable_scope('conv_weights', regularizer=regularizer):
            conv_weights = tf.Variable(
                tf.truncated_normal(kernel_shape, stddev=0.1, dtype=tf.float32))
        with tf.variable_scope('conv_bias'):
            conv_biases = tf.Variable(
                tf.constant(0.0, dtype=tf.float32, shape=[kernel_shape[3]]))
        with tf.name_scope('conv'):
            conv = tf.nn.conv2d(data, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
        with tf.name_scope('activation'):
            conv_out = activation(tf.nn.bias_add(conv, conv_biases))
        with tf.name_scope('dropout'):
            conv_out = tf.nn.dropout(conv_out, dropout)

        return conv_out


def hidden(data, activation, name, hidden_units, dropout=1, regularizer=None, reg_val=None):
    """ Hidden layer.
    Parameters:
        data: The input data.
        activation: The activation function.
        name: The layer's name.
        hidden_units: Number of hidden_out units.
        dropout: Whether do the dropout job.
        regularizer: Whether use the L2 or L1 regularizer.
        reg_val: regularizer value.
    Return:
        hidden_out: Output of current layer.
    """
    if regularizer == 'L1':
        regularizer = layers.l1_regularizer(reg_val)
    elif regularizer == 'L2':
        regularizer = layers.l2_regularizer(reg_val)

    with tf.name_scope(name):
        # Fully connected layer 1. Note that the '+' operation automatically.
        with tf.variable_scope('fc_weights', regularizer=regularizer):
            input_units = int(data.shape[1])
            fc_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([input_units, hidden_units],
                                    stddev=0.1, dtype=tf.float32))
        with tf.name_scope('fc_bias'):
            fc_biases = tf.Variable(
                tf.constant(0.0, dtype=tf.float32, shape=[hidden_units]))
        with tf.name_scope('activation'):
            hidden_out = activation(tf.nn.xw_plus_b(data, fc_weights, fc_biases))
        if dropout is not None:
            hidden_out = tf.nn.dropout(hidden_out, dropout)
        return hidden_out

def cnn_2_model(input_pl, activation=tf.nn.relu, dropout=1):
    """ CNN 2 Model in the paper.
    Parameters:
        input_pl: The input data placeholder.
        activation: The activation function.
        dropout: Whether use the dholderropout.
    Returns:
        logits: The model output value for each category.
     """
    kernel1 = [1, 5, num_channels, num_kernel_1]
    kernel2 = [2, 6, num_kernel_1, num_kernel_2]
    conv1 = conv(input_pl, kernel1, activation, 'conv_1', dropout)
    # pool = tf.nn.avg_pool(conv1, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = conv(conv1, kernel2, activation, 'conv_2', dropout)

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    flatten = tf.reshape(conv2, [batch_size, width * height * num_kernel_2])

    hidden_1 = hidden(flatten, activation, 'hidden_1', hidden_units_1, dropout)
    logits = hidden(hidden_1, activation, 'hidden_2', num_classes)

    return logits, hidden_1

def eval_in_batches(data, sess, eval_prediction, eval_placeholder, keep_prob):
    """Get all predictions for a dataset by running it in small batches.
    Parameters:
        data: The evaluation data set.
        sess: The session with the graph.
        eval_prediction: The evaluation operator, which output the logits.
        eval_placeholder: The placeholder of evaluation data in the graph.
    Returns:
        predictions: The eval result of the input data, which has the format
                    of [size, num_classes]
    """
    size = data.shape[0]
    if size < batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_classes), dtype=np.float32)
    for begin in range(0, size, batch_size):
        end = begin + batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_placeholder: data[begin:end, ...],
                           keep_prob: 1})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_placeholder: data[-batch_size:, ...],
                           keep_prob: 1})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def hidden_representation(data, sess, hid, input_holder, keep_prob):
    """ Get the low-dimension representation.
    Parameters:
        data: The test data set.
        sess: The session with the graph.
        hid: The representation getting operator, which output the hidden.
        input_holder: The placeholder of test data in the graph.
    """
    rept = sess.run(hid, feed_dict={input_holder: data,
                                    keep_prob: 1})
    print('We have gotten the representation features.')
    return rept


def get_rep_data(orig_samples, orig_labels, snrs):
    """ Get the test data.
    Parameters:
        orig_samples: The samples.
        orig_labels: The label arrays.
        snrs： The SNR arrays.
    Returns:
        samples:
        labels:
    """
    samples_temp = orig_samples[snrs== -4]
    labels_temp = orig_labels[snrs == -4]
    indexes = list(range(len(samples_temp)))

    np.random.shuffle(indexes)
    samples = samples_temp[indexes[1024:2048]]
    labels = labels_temp[indexes[1024:2048]]
    return samples, labels, indexes[1024:2048]


def t_sne(samples, labels, cate2mod):
    """Show the data with t-sne algorithms.
    Parameters:
        samples: The samples to be shown.
        labels: The labels of samples.
    """

    low_dim_file = './low_dim.pkl'
    low_dim_fig = './fig_3_6.png'
    if not os.path.exists(low_dim_file):
        print('Dimension decreasing...')
        tsne = TSNE(perplexity=30, n_components=2, init='pca',
             n_iter=5000, method='exact')
        low_dims = tsne.fit_transform(samples)
        pickle.dump(low_dims, open(low_dim_file, 'wb'))
        print('We have fitted the data.')
    else:
        low_dims = pickle.load(open(low_dim_file, 'rb'))

    plt.figure(figsize=(15, 10))  # in inches
    colors = ['red', 'blue', 'springgreen', 'gold', 'cyan', 'darkorange', 'black', 'deepskyblue', 'purple']

    low_dims_cate = list()
    for i in range(len(set(labels))):
        low_dims_cate.append(low_dims[labels==i])

    holder = list()
    for i in range(len(low_dims_cate)):
        temp = low_dims_cate[i]
        x = temp[:, 0]
        y = temp[:, 1]
        holder.append(plt.scatter(x, y, c=[colors[i]]))

    plt.xticks([])
    plt.yticks([])
    mods = ['AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM64', 'QPSK']

    plt.legend(holder, mods)
    plt.savefig(low_dim_fig)


def build_data(samples, labels):
    """ Build the train and test set.
    Parameters:
        samples: The whole samples we have.
        labels: The samples' labels correspondently.
    Returns:
        train_data: The train set data.
        train_labels: The train data's category labels.
        test_data: The test set data.
        test_labels: The test data's category labels.
    """
    num_samples = len(samples)
    indexes = list(range(num_samples))
    np.random.shuffle(indexes)
    num_train = int(train_ratio * num_samples)
    # Get the indexes of train data and test data.
    train_indexes = indexes[0:num_train]
    test_indexes = indexes[num_train:num_samples]

    # Build the train data and test data.
    train_data = samples[train_indexes]
    train_labels = labels[train_indexes]
    test_data = samples[test_indexes]
    test_labels = labels[test_indexes]

    return train_data, test_data, \
           train_labels, test_labels, \
           train_indexes, test_indexes


def accuracy_snr(predictions, labels, indexes, snrs, samples_snr):
    """ Compute the error rate of difference snr.
    Parameters:
        predictions:
        labels:
        indexes:
        snrs:
        samples_snr:
    Returns:
        acc_snr
    """
    labels = labels.reshape([len(labels), ])
    predict_snr = samples_snr[indexes]

    acc_snr = dict()
    for snr in snrs:
        idx = (predict_snr == snr).reshape([len(labels)])
        samples_temp = predictions[idx]
        labels_temp = labels[idx]
        acc_snr[snr] = accuracy_compute(samples_temp, labels_temp)
    return acc_snr


def acc_snr_show(snrs, acc_snr, path):
    """ Show the train procedure.
    Parameters:
        sd
    Returns:
        Hello
    """
    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc_snr[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN Classification Accuracy with Different SNR")
    plt.savefig(path)

def main():
    # Define the input data.
    input_shape = [batch_size, height, width, num_channels]

    # Load the train data and test data.
    samples, labels, mod2cate, cate2mod, snrs, mods, samples_snr = \
        load_data(orig_file_path, input_shape)

    train_data, test_data, \
    train_labels, test_labels, \
    train_indexes, test_indexes = build_data(samples, labels)

    # Define the input placeholder.
    train_data_node = tf.placeholder(tf.float32, shape=[None, height, width, num_channels])
    train_labels_node = tf.placeholder(tf.int64, shape=[None])

    keep_prob = tf.placeholder("float")
    # eval_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, num_channels))

    # Model.
    logits, hid = cnn_2_model(train_data_node, tf.nn.leaky_relu, keep_prob)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits))

    # Use simple adam for the optimization.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    correct_prediction = tf.equal(tf.argmax(train_prediction, 1), train_labels_node)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('value', accuracy)

    saver = tf.train.Saver()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    # Create a local session to run the training.
    with tf.Session(config=config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print('Initialized!')

        # Loop through training steps.
        num_train = len(train_labels)
        max_step_train = int(num_epoch * num_train / batch_size)
        for step in range(max_step_train):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * batch_size) % (num_train - batch_size)
            batch_data = train_data[offset:(offset + batch_size), ...]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels,
                         keep_prob: 0.5}

            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)

            # print some extra information once reach the evaluation frequency
            if step % train_show_step == 0:
                # fetch some extra nodes' data
                loss_step, train_accu = \
                    sess.run([loss, accuracy],
                             feed_dict=feed_dict)

                # eval_acc = accuracy(predictions, batch_labels, 'train_accuracy')
                print('Step: %d(epoch %.2f)  loss: %.3f, train_accuracy: %.3f%%' %
                      (step, float(step) * batch_size / num_train, loss_step, train_accu * 100))

                if step % test_show_step == 0:  # Test the test set.
                    test_predictions = eval_in_batches(test_data, sess, train_prediction, train_data_node, keep_prob)
                    print('Test accuracy: %.3f%% ' % accuracy_compute(test_predictions, test_labels))

        checkpoint_file = os.path.join(log_dir, 'model_final.ckpt')
        saver.save(sess, checkpoint_file)

        # Low Dimension Representation of data.
        show_samples, show_labels, show_indexes = get_rep_data(samples, labels, samples_snr)
        hidden_rept = hidden_representation(show_samples, sess, hid, train_data_node, keep_prob)
        t_sne(hidden_rept, show_labels, cate2mod)

        show_prediction = eval_in_batches(show_samples, sess, train_prediction, train_data_node, keep_prob)
        show_acc = accuracy_compute(show_prediction, show_labels)
        print('The show examples\' prediction accuracy is: %.3f' % show_acc)

if __name__ == '__main__':
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    main()

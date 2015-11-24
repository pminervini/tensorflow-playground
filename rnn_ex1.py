#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import sys


def gen_data(min_length=50, max_length=55, n_batch=5, input_size=8):
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, input_size - 1)),
                        np.zeros((n_batch, max_length, 1))], axis=-1)
    y = np.zeros((n_batch,))

    for n in range(n_batch):
        length = np.random.randint(min_length, max_length)
        X[n, length:, 0] = 0
        X[n, np.random.randint(length/2-1), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1

        y[n] = np.sum(X[n, :, 0] * X[n, :, 1])

    return X, y


def main(argv):
    num_units = 12
    input_size = 4

    batch_size = 128
    seq_len = 100
    num_epochs = 1024

    print('Building the model ..')

    cell = rnn_cell.BasicLSTMCell(num_units)
    inputs = [tf.placeholder(tf.float32, shape=[batch_size, input_size]) for _ in range(seq_len)]
    result = tf.placeholder(tf.float32, shape=[batch_size])

    outputs, states = rnn.rnn(cell, inputs, dtype=tf.float32)

    W_o = tf.Variable(tf.random_normal([num_units, 1], stddev=0.01))
    b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

    outputs3 = tf.matmul(outputs[-1], W_o) + b_o

    cost = tf.reduce_mean(tf.pow(outputs3-result, 2))
    train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

    temp_x, y_val = gen_data(50, seq_len, batch_size, input_size)
    x_val = []
    for i in range(seq_len):
        x_val.append(temp_x[:, i, :])

    print('Executing the model ..')

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            temp_x, y = gen_data(50, seq_len, batch_size, input_size)
            x = []
            for i in range(seq_len):
                x.append(temp_x[:, i, :])

            temp_dict = {inputs[i]: x[i] for i in range(seq_len)}
            temp_dict.update({result: y})

            print('Updating the parameters ..')
            sess.run(train_op, feed_dict=temp_dict)

            val_dict = {inputs[i]: x_val[i] for i in range(seq_len)}
            val_dict.update({result: y_val})
            c_val = sess.run(cost, feed_dict=val_dict)
            print('Validation cost: {}, on Epoch {}'.format(c_val, epoch))


if __name__ == '__main__':
    main(sys.argv[1:])

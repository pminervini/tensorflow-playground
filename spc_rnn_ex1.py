#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import random
import sys


def read_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def main(argv):

    random.seed(1)
    np.random.seed(1)

    batch_size = 1
    num_steps = 10
    hidden_size = 8
    embedding_size = 20
    vocab_size = 16

    num_epochs = 1024

    input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.float32, [batch_size])

    cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)

    initial_state = cell.zero_state(batch_size, tf.float32)

    embedding = tf.get_variable('embedding', [vocab_size, embedding_size])

    inputs = tf.split(
        1,
        num_steps,
        tf.nn.embedding_lookup(embedding, input_data)
    )

    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state)
    last_output = outputs[-1]

    W = tf.Variable(tf.zeros([hidden_size, 1]))
    b = tf.Variable(tf.zeros([1]))

    y = tf.nn.sigmoid(tf.matmul(last_output, W) + b)

    cost = tf.reduce_mean(tf.abs(y - targets))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            _sentence = np.random.randint(vocab_size, size=(1, 10))

            input_feed = {
                input_data: _sentence
            }
            _y = sess.run(y, feed_dict=input_feed)

            _target = np.random.random_sample(1)

            input_feed = {
                input_data: np.random.randint(vocab_size, size=(1, 10)),
                targets: _target
            }

            c_val = sess.run(cost, feed_dict=input_feed)

            print('Validation cost: {}, on Epoch {}'.format(c_val, epoch))

            sess.run(train_op, feed_dict=input_feed)

if __name__ == '__main__':
    main(sys.argv[1:])

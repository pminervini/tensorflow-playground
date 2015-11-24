#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import sys


def read_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def main(argv):

    batch_size = 1
    num_steps = 10
    size = 8
    embedding_size = 20
    vocab_size = 16

    _input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    _targets = tf.placeholder(tf.int32, [batch_size])

    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    cell = lstm_cell

    _initial_state = cell.zero_state(batch_size, tf.float32)

    embedding = tf.get_variable('embedding', [vocab_size, embedding_size])

    inputs = tf.split(
        1,
        num_steps,
        tf.nn.embedding_lookup(embedding, _input_data)
    )

    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, states = rnn.rnn(cell, inputs, initial_state=_initial_state)
    last_output = outputs[-1]
    last_state = states[-1]

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        input_feed = {
            _input_data: np.random.randint(2, size=(1, 10))
        }

        _last_input = sess.run(inputs[-1], feed_dict=input_feed)
        _last_output = sess.run(last_output, feed_dict=input_feed)
        _last_state = sess.run(last_state, feed_dict=input_feed)

        print('Last input: ', _last_input.shape)
        print('Last output: ', _last_output.shape)
        print('Last state: ', _last_state.shape)


if __name__ == '__main__':
    main(sys.argv[1:])

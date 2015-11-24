#!/usr/bin/env python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

from keras.preprocessing import sequence, text

import random
import sys


def read_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def main(argv):

    batch_size = 8
    num_steps = 20
    hidden_size = 64
    embedding_size = 20
    vocab_size = 100

    num_epochs = 2 ** 16

    random.seed(1)
    np.random.seed(1)

    print('Importing Tweets ..')

    positive_tweets = read_lines('data/spc/positive-train')[:1000]
    negative_tweets = read_lines('data/spc/negative-train')[:1000]

    tweets = positive_tweets + negative_tweets
    labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

    print('Shuffling the training examples ..')

    tweet_label_pairs = list(zip(tweets, labels))
    random.shuffle(tweet_label_pairs)
    tweets, labels = [e[0] for e in tweet_label_pairs], [e[1] for e in tweet_label_pairs]

    print('Tokenizing the text ..')

    tokenizer = text.Tokenizer(nb_words=vocab_size)
    tokenizer.fit_on_texts(tweets)
    sequences = [seq for seq in tokenizer.texts_to_sequences_generator(tweets)]

    X = sequence.pad_sequences(sequences, maxlen=num_steps)

    print('Building the model ..')

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

    cost = tf.reduce_mean(tf.pow(y - targets, 2))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.0

            indices = np.random.permutation(X.shape[0])
            _X = X[indices, :]
            _labels = [labels[index] for index in indices]

            for i in range(0, X.shape[0], batch_size):
                _sentence = _X[i:i + batch_size, :]
                _target = [_labels[j] for j in range(i, i + batch_size)]

                input_feed = {
                    input_data: _sentence,
                    targets: _target
                }

                c_val = sess.run(cost, feed_dict=input_feed)
                epoch_cost += c_val

                sess.run(train_op, feed_dict=input_feed)

            print('Validation cost: {}, on Epoch {}'.format(epoch_cost, epoch))

if __name__ == '__main__':
    main(sys.argv[1:])

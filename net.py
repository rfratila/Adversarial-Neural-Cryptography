# -*- coding: utf-8 -*-

import tensorflow as tf


def _conv1d(input_layer, filter_size, strides, kernel_size, activation):
    return tf.layers.conv1d(
        inputs=input_layer,
        filters=filter_size,
        strides=strides,
        kernel_size=kernel_size,
        activation=activation,
        padding="same")


def _conv_layers(input_layer, strides):
    c1 = _conv1d(input_layer, 2, strides[0], [4], tf.nn.sigmoid)
    c2 = _conv1d(c1, 4, strides[1], [2], tf.nn.sigmoid)
    c3 = _conv1d(c2, 4, strides[2], [1], tf.nn.sigmoid)
    c4 = _conv1d(c3, 1, strides[3], [1], tf.tanh)
    return c4


def _network(input_layer, name, message_length, strides):
    with tf.variable_scope(name):
        hidden_layer = tf.layers.dense(
            inputs=input_layer,
            units=2 * message_length,
            activation=tf.nn.sigmoid)
        hidden_layer_2 = tf.layers.dense(
            inputs=hidden_layer,
            units=message_length,
            activation=tf.nn.tanh)
        output_layer = hidden_layer_2 #_conv_layers(hidden_layer, strides)

    return output_layer


def build_input_layers(message_length, key_length):
    msg = tf.placeholder(dtype=tf.float32, shape=(None, message_length, 1),
                         name="message")
    key = tf.placeholder(dtype=tf.float32, shape=(None, key_length, 1),
                         name="key")
    return msg, key


def build_network(msg, key):
    alice_bob_strides = [1, 2, 1, 1]
    eve_strides = [1, 1, 1, 1]
    message_length = int(msg.shape[1])

    alice_input = tf.concat([msg, key], axis=1)
    alice_output = _network(alice_input, "alice", message_length,
                            alice_bob_strides)

    bob_input = tf.concat([alice_output, key], axis=1)
    bob_output = _network(bob_input, "bob", message_length,
                          alice_bob_strides)

    eve_output = _network(alice_output, "eve", message_length,
                          eve_strides)

    return alice_output, bob_output, eve_output

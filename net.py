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

def _eve_dense_network(input_layer, name, message_length, strides):
    with tf.variable_scope(name):
        hidden_layer = tf.layers.dense(
            inputs=input_layer,
            units=2 * message_length,
            activation=tf.nn.sigmoid)
        hidden_layer_2 = tf.layers.dense(
            inputs=hidden_layer,
            units=2 * message_length,
            activation=tf.nn.sigmoid)
        hidden_layer_3 = tf.layers.dense(
            inputs=hidden_layer_2,
            units=2 * message_length,
            activation=tf.nn.sigmoid)
        hidden_layer_4 = tf.layers.dense(
            inputs=hidden_layer_3,
            units=2 * message_length,
            activation=tf.nn.sigmoid)
        hidden_layer_5 = tf.layers.dense(
            inputs=hidden_layer_4,
            units=message_length,
            activation=tf.nn.tanh)
        
        output_layer = hidden_layer_5 #_conv_layers(hidden_layer, strides)

    return output_layer

def _eve_orig_network(input_layer, name, message_length,strides):
    with tf.variable_scope(name):
        expand_in = tf.expand_dims(input_layer,2)
        hidden_layer = tf.layers.dense(
            inputs=expand_in,
            units=2 * message_length,
            activation=tf.nn.sigmoid)
        output_layer = _conv_layers(hidden_layer, strides)
    return tf.reshape(output_layer, [4096,message_length])

def _eve_conv_network(input_layer, name, message_length, strides):

    with tf.variable_scope(name):
        expand_in = tf.expand_dims(input_layer,2)
        output_layer = _conv_layers(expand_in, strides)
    return tf.reshape(output_layer, [4096,message_length])

def _eve_large_network(input_layer, name, message_length, strides):
    with tf.variable_scope(name):
        expand_in = tf.expand_dims(input_layer,2)
        c1 = _conv1d(expand_in, 2, strides[0], [4], tf.nn.sigmoid)
        c2 = _conv1d(c1, 4, strides[1], [2], tf.nn.sigmoid)
        c3 = _conv1d(c2, 4, strides[2], [2], tf.nn.sigmoid)
        c4 = _conv1d(c3, 4, strides[2], [2], tf.nn.sigmoid)
        c5 = _conv1d(c4, 4, strides[2], [1], tf.nn.sigmoid)
        c6 = _conv1d(c5, 4, strides[2], [2], tf.nn.sigmoid)
        c7 = _conv1d(c6, 4, strides[2], [2], tf.nn.sigmoid)
        c8 = _conv1d(c7, 4, strides[2], [2], tf.nn.sigmoid)
        c9 = _conv1d(c8, 4, strides[2], [1], tf.nn.sigmoid)
        c10 = _conv1d(c9, 1, strides[3], [1], tf.tanh)
    return tf.reshape(c10, [4096,message_length])


def build_input_layers(message_length, key_length):
    msg = tf.placeholder(dtype=tf.float32, shape=(None, message_length),
                         name="message")
    key = tf.placeholder(dtype=tf.float32, shape=(None, key_length),
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

    eve_orig_output = _eve_orig_network(alice_output, "eve_orig", message_length,
                                        eve_strides)

    eve_conv_output = _eve_conv_network(alice_output, "eve_conv", message_length,
                                        eve_strides)

    eve_large_output = _eve_large_network(alice_output, "eve_large", message_length,
                                        eve_strides)

    eve_dense_output = _eve_dense_network(alice_output, "eve_dense", message_length,
                                        eve_strides)

    return alice_output, bob_output, eve_output, eve_orig_output, \
            eve_conv_output, eve_large_output, eve_dense_output

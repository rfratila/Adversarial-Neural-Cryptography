# -*- coding: utf-8 -*-

import time
import datetime
import tensorflow as tf
from datagen import get_random_block
from session_manager import save_session
from net import build_input_layers, build_network


def reconstruction_loss(msg, output):
    """Autoencoder error."""
    return tf.reduce_mean(tf.abs(tf.subtract(msg, output))) / 2


def bits_loss(msg, output, message_length):
    """Autoencoder error in number of different bits."""
    return reconstruction_loss(msg, output) * message_length


message_length = 16  # in bits
key_length = message_length  # in bits
batch = 4096  # Number of messages to train on at once
adv_iter = 120  # Adversarial iterations
max_iter = 200  # Individual agent iterations
learning_rate = 0.0008


if __name__ == "__main__":
    msg, key = build_input_layers(message_length, key_length)
    alice_output, bob_output, eve_output = build_network(msg, key)


    eve_loss = reconstruction_loss(msg, eve_output)
    bob_reconst_loss = reconstruction_loss(msg, bob_output)
    bob_loss = bob_reconst_loss + (0.5 - eve_loss) ** 2

    eve_bit_loss = bits_loss(msg, eve_output, message_length)
    bob_bit_loss = bits_loss(msg, bob_output, message_length)

    AB_vars = (
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alice") +
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "bob")
    )

    E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "eve")

    trainAB = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        bob_loss, var_list=AB_vars)

    trainE = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        eve_loss, var_list=E_vars)

    writer = tf.summary.FileWriter("logs/{}".format(datetime.datetime.now()))
    tf.summary.scalar("eve_error", eve_loss)
    tf.summary.scalar("bob_reconst_error", bob_reconst_loss)
    tf.summary.scalar("bob_error", bob_loss)
    tf.summary.scalar("eve_bit_error", eve_bit_loss)
    tf.summary.scalar("bob_bit_error", bob_bit_loss)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)

        for i in range(adv_iter):
            print("\nIteration:", i)
            start_time = time.time()
            feed_dict = {
                msg: get_random_block(message_length, batch),
                key: get_random_block(key_length, batch)
            }

            print("\tTraining Alice and Bob for {} iterations..."
                  .format(max_iter))
            for j in range(max_iter):
                sess.run(trainAB, feed_dict=feed_dict)

            print("\tTraining Eve for {} iterations...".format(2 * max_iter))
            for j in range(2 * max_iter):
                sess.run(trainE, feed_dict=feed_dict)

            results = [eve_loss, bob_loss, merged_summary]
            eve_error, bob_error, summary = sess.run(results,
                                                     feed_dict=feed_dict)
            writer.add_summary(summary, global_step=i)
            writer.flush()

            print("\tEve error: {:.4f} | Bob error: {:.4f} | Time: {:.2f}s"
                  .format(eve_error, bob_error, time.time() - start_time))
        import pudb; pu.db
        save_session(sess, "alice_bob")

import tensorflow as tf
from datagen import get_key
from datagen import get_plain_text
import numpy as np


class Model:
    def __init__(self, bit_count, name, input_net=None, concatenate_key=False,
                 AB=True, training=True):
        '''
        Meant to hold onto the network object
        Arguments:
            bit_count: the length of the plain text
            AB: Whether you would like to create an Alice/Bob or Eve network
            training: if you want the weights to be trainable
        '''
        self.AB = AB
        self.name = name
        self.input_length = bit_count
        self.trainable = training
        self.concatenate_key = concatenate_key

        if self.concatenate_key:
            self.con_key = tf.placeholder(
                dtype=tf.float32, shape=(None, bit_count, 1))

        if input_net is None:
            self.input_layer = tf.placeholder(
                dtype=tf.float32, shape=(None, self.input_length, 1))
        else:
            self.input_layer = input_net

        self.network = self.create_network()

    def create_network(self):
        with tf.variable_scope(self.name, reuse=True):

            self.strides = [1, 2, 1, 1] if self.AB else [1, 1, 1, 1]
            # Create a 2N x 2N deep dense layer for Alice/Bob and N x 2N if Eve
            if self.AB:
                self.dense_depth = self.input_length - 1
            else:
                self.dense_depth = (2 * self.input_length) - 1

            if self.concatenate_key:
                self.con_input_layer = tf.concat(
                    [self.con_key, self.input_layer], axis=1)
            else:
                self.con_input_layer = self.input_layer

            # Dense layers for mixing the plaintext
            self.dense_layers = tf.layers.dense(
                inputs=self.con_input_layer,
                units=self.input_length,
                trainable=self.trainable
            )

            for i in range(0, self.dense_depth):
                self.dense_layers = tf.layers.dense(
                    inputs=self.dense_layers,
                    units=self.input_length,
                    trainable=self.trainable
                )

            # Four conv layers for transforming the ciphertext
            self.conv1 = tf.layers.conv1d(
                inputs=self.dense_layers,
                filters=2,
                strides=self.strides[0],
                kernel_size=[4],
                padding='same',
                activation=tf.sigmoid,
                trainable=self.trainable
            )

            self.conv2 = tf.layers.conv1d(
                inputs=self.conv1,
                filters=4,
                strides=self.strides[1],
                kernel_size=[2],
                padding='same',
                activation=tf.sigmoid,
                trainable=self.trainable
            )

            self.conv3 = tf.layers.conv1d(
                inputs=self.conv2,
                filters=4,
                strides=self.strides[2],
                kernel_size=[1],
                padding='same',
                activation=tf.sigmoid,
                trainable=self.trainable
            )

            self.conv4 = tf.layers.conv1d(
                inputs=self.conv3,
                filters=1,
                strides=self.strides[3],
                kernel_size=[1],
                padding='same',
                activation=tf.tanh,
                trainable=self.trainable
            )

            self.output = tf.sign(
                tf.sign(self.conv4) + tf.ones_like(self.conv4) / 10)

            return self.output


def main():
    num_bits = 16
    batch = 4096

    print('Generating random plaintexts and key...')
    messages = get_plain_text(N=num_bits, to_generate=batch)
    AB_key = get_key(N=num_bits, batch=batch)

    # Inputs are [batch_size, image_width, image_height, channels]
    train_X = np.expand_dims(messages, axis=2)  # prep for passing into network of size (batch,num_bits,1)
    keys = np.expand_dims(AB_key, axis=2)

    orig = tf.placeholder(dtype=tf.float32, shape=(None, num_bits, 1))

    print('Creating Alice net...')
    alice_net = Model(
        bit_count=num_bits, name = 'Alice', concatenate_key=True,
        AB=True, training=True
    )

    print('Creating Bob net...')
    bob_net = Model(
        bit_count=num_bits, name = 'Bob',input_net=alice_net.network,
        concatenate_key=True, AB=True, training=True
    )

    print('Creating Eve net...')
    eve_net = Model(
        bit_count=num_bits, name = 'Eve',input_net=alice_net.network,
        AB=False, training=True
    )

    # Eves reconstruction error
    eve_loss = tf.reduce_sum(tf.abs(tf.subtract(orig, eve_net.network)))

    # Bobs reconstruction error
    bob_reconst = tf.reduce_sum(tf.abs(tf.subtract(orig, bob_net.network)))

    # Bobs ability to still decode while eve can't
    bob_eve = tf.divide(
        tf.square(tf.subtract(bob_net.input_length/2, eve_loss)),
        tf.square(bob_net.input_length/2)
    )

    bob_fancy_loss = bob_reconst - bob_eve  # not explicitly mentioned in the paper
    total_loss = bob_reconst - eve_loss

    AB_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Alice_Bob")
    train_AB = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(
        total_loss, var_list=AB_varsB)

    E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Eve")
    train_eve = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(
        eve_loss, var_list=E_vars)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # tf.initialize_all_variables()

        feed_dict = {
            alice_net.input_layer: train_X,
            alice_net.con_key: keys
        }

        alice_out = sess.run(alice_net.network, feed_dict=feed_dict)

        eve_out = sess.run(eve_net.network, feed_dict=feed_dict)

        feed_dict = {
            alice_net.input_layer: train_X,
            alice_net.con_key: keys,
            bob_net.con_key: keys
        }

        bob_out = sess.run(bob_net.network, feed_dict=feed_dict)

        for i in range(0, 100):
            print('Iteration:', i)

            feed_dict_AB = {
                alice_net.input_layer: train_X,
                alice_net.con_key: keys,
                bob_net.con_key: keys,
                orig: train_X
            }

            sess.run(train_AB, feed_dict=feed_dict_AB)

            feed_dict_E = {
                alice_net.input_layer: train_X,
                alice_net.con_key: keys,
                orig: train_X
            }

            sess.run(train_eve, feed_dict=feed_dict_E)

            eve_error = sess.run(eve_loss, feed_dict=feed_dict_E)
            bob_error = sess.run(bob_reconst, feed_dict=feed_dict_AB)

            print("    Eve reconstruction error: {} | Bob reconstruction error: {}".format(eve_error, bob_error))

            # get more messages
            messages = get_plain_text(N=num_bits, to_generate=batch)
            train_X = np.expand_dims(messages, axis=2)


if __name__ == "__main__":
    main()

import tensorflow as tf
from datagen import get_key
from datagen import get_plain_text
import numpy as np


num_bits = 16
batch = 1000

messages = get_plain_text(N=num_bits,to_generate=batch)
AB_key = get_key(N=num_bits)


m_with_key = [np.concatenate((AB_key[0],messages[i])) for i in range(batch)]

#Inputs are [batch_size, image_width, image_height, channels]
#input_layer = tf.reshape(m_with_key, [-1, 2*num_bits, 1])
input_layer = tf.placeholder(dtype = tf.float32, shape=(None,2*num_bits,1))
import pudb; pu.db
conv1 = tf.layers.conv1d(inputs=input_layer,
                            filters=32, 
                            kernel_size = [5], 
                            padding='same',
                            activation=tf.nn.relu)

pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[2], strides=2)

conv2 = tf.layers.conv1d(inputs=pool1,
                            filters=32, 
                            kernel_size = [5], 
                            padding='same',
                            activation=tf.nn.relu)

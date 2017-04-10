# -*- coding: utf-8 -*-

"""Utility functions to load and save a tf session."""

import os
import tensorflow as tf


def save_session(session, model_name):
    """Saves a session.

    Args:
        session: TensorFlow session
        model_name (str): name with which to save the weights
    """
    if not os.path.isdir("models"):
        os.makedirs("models")
    saver = tf.train.Saver()
    save_path = "models/{}.ckpt".format(model_name)
    save_path = os.path.abspath(os.path.join(os.getcwd(), save_path))
    saver.save(session, save_path)
    print("Saved session to {}".format(save_path))


def load_session(session, model_name):
    """Loads session weights.

    Note: you must first initialize the session
    and create all the variables and layers.

    Args:
        session: TensorFlow session
        model_name (str): name with which the weights were saved
    """
    saver = tf.train.Saver()
    load_path = "models/{}.ckpt".format(model_name)
    load_path = os.path.abspath(os.path.join(os.getcwd(), load_path))
    saver.restore(session, load_path)
    print("Restored session from {}".format(load_path))

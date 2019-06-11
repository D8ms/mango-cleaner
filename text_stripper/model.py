import tensorflow as tf
from keras import optimizers
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Activation, Reshape, Flatten

class Model:
    def __init__(self, width, height):
        self.raw_ph = tf.placeholder(np.float32, shape=[None, height, width, 1], name="raw_ph")
        self.clean_ph = tf.placeholder(np.float32, shape=[None, height, width, 1], name="raw_ph")
        self.diff_ph = tf.placeholder(np.float32, shape=[None, height, width, 1], name="raw_ph")
        with tf.device('/gpu:0'):
            x = Conv2D(filters=32, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu, input_shape=[None, height, width])(self.raw_ph)
            x = Conv2D(filters=32, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu)(x)
            #self.out1 = Conv2D(filters=1, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu)(x)
            #error1 = tf.reduce_mean(tf.square(self.out1 - self.diff_ph))
            #x = Conv2D(filters=32, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu)(self.out1)
            x = Conv2D(filters=32, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu)(x)
            x = Conv2D(filters=32, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.leaky_relu)(x)
            self.out2 = Conv2D(filters=1, kernel_size=(16, 16), strides=(1, 1), padding='same', kernel_initializer='glorot_normal', activation=tf.nn.tanh)(x)
            error2 = tf.reduce_mean(tf.square(self.out2 - self.diff_ph))
            #self.error = error1 + error2 
            self.error = error2


            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.error)

            tf.summary.scalar("error", self.error)
            #tf.summary.image("out1", self.out1, max_outputs=1)
            tf.summary.image("out2", self.out2, max_outputs=1)
            tf.summary.image("target", self.clean_ph, max_outputs=1)
            tf.summary.image("diffs", self.diff_ph, max_outputs=1)
            self.merged_summary = tf.summary.merge_all()

    def train(self, sess, raws, cleans, diffs):
        error, summary, _ = sess.run(
            [self.error, self.merged_summary, self.train_op],
            feed_dict = {
                self.raw_ph: raws,
                self.clean_ph: cleans,
                self.diff_ph: diffs,
                K.learning_phase(): 1
            }
        )
        return error, summary

    def infer(self, sess, raw):
        return sess.run(
            [self.cleaned],
            feed_dict = {
                self.raw_ph: raw,
                K.learning_phase(): 0
            }
        )[0]

# https://github.com/ex4sperans/variational-inference-with-normalizing-flows/blob/master/run_experiment.py
# https://github.com/LukasRinder/normalizing-flows/tree/master/normalizingflows

import tensorflow as tf

import numpy as np

from utils import *


def safe_log( z ):
  return tf.math.log( 1e-8 + z )


def log_normal_pdf(sample, mean, logvar):
  log2pi = safe_log(2. * np.pi)
  return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)



class CouplingNN( tf.keras.Model ):

  def __init__( self, input_dim, activation ):
    super( CouplingNN, self).__init__()

    self.dense0 = Dense(4*input_dim, activation=activation )
    self.dense1 = Dense(4*input_dim, activation=activation )

    self.dense_log_s = tf.keras.layers.Dense(input_dim, activation=tf.math.tanh)
    self.scale_log_s = tf.Variable( np.zeros(shape=(1, input_dim)), name='scale_log_s', dtype=tf.float32, trainable=True)
    self.shift_log_s = tf.Variable( np.zeros(shape=(1, input_dim)), name='shift_log_s', dtype=tf.float32, trainable=True)

    self.dense_t = tf.keras.layers.Dense(input_dim, activation=tf.math.tanh)
    self.scale_t = tf.Variable( np.zeros(shape=(1, input_dim)), name='scale_t', dtype=tf.float32, trainable=True)
    self.shift_t  = tf.Variable( np.zeros(shape=(1, input_dim)), name='shift_t', dtype=tf.float32, trainable=True)

  def call(self, x, is_train):

    x = self.dense0(x, is_train=is_train)
    x = self.dense1(x, is_train=is_train)

    log_s = self.scale_log_s * self.dense_log_s(x) + self.shift_log_s
    t = self.scale_t * self.dense_t(x) + self.shift_t

    return log_s, t



class CouplingLayer( tf.keras.Model ):

  def __init__( self, input_dimensions, activation, random_shuffle=False ):
    super(CouplingLayer, self).__init__()

    self.nn = CouplingNN(input_dimensions//2, activation)

    self.random_shuffle = random_shuffle

    if self.random_shuffle:
      self.indices_forward = np.random.permutation( input_dimensions )
      self.indices_inverse = np.argsort(self.indices_forward)

    pass


  @tf.function
  def forward( self, x, is_train ):
    if self.random_shuffle:
      x = tf.gather( x, self.indices_forward, axis=1 )
    else:
      x = tf.reverse(x, [1] )

    x_a, x_b = tf.split(x, 2, axis=1)

    y_b = x_b

    log_s, t = self.nn( x_b, is_train )

    y_a = tf.math.exp( log_s ) * x_a + t
    y = tf.concat( [y_a, y_b], axis=1)

    log_det = tf.reduce_sum( log_s, axis=1 )

    return y, log_det

  @tf.function
  def inverse( self, y, is_train ):

      y_a, y_b = tf.split(y, 2, axis=1)

      x_b = y_b

      log_s, t = self.nn( y_b, is_train )

      x_a = (y_a - t) / tf.math.exp( log_s )

      x = tf.concat( [x_a, x_b], axis=1)

      log_det = -tf.reduce_sum( log_s, axis=1 ) # minus!

      if self.random_shuffle:
        x = tf.gather( x, self.indices_inverse, axis=1 )
      else:
        x = tf.reverse(x, [1] )

      return x, log_det



class RealNVP( tf.keras.Model ):
  def __init__(self, input_dimensions, num_layers, activation, random_shuffle=False ):
    super(RealNVP, self).__init__()

    self.input_dimensions = input_dimensions

    self.coupling_layers = []
    for i in range( num_layers ):
      self.coupling_layers.append( CouplingLayer( self.input_dimensions, activation, random_shuffle ) )

  def sample( self, n, is_train ):
    return self.forward( tf.random.normal( mean=0.0, stddev=1.0, shape=[n,self.input_dimensions], dtype=np.float32 ), is_train )

  @tf.function
  def forward( self, z, is_train ):
    log_pzk = tf.reduce_sum( log_normal_pdf( z, 0.0, 0.0 ), axis=1 )

    for cl in self.coupling_layers:


      z, log_pz = cl.forward( z, is_train )

      log_pzk -= log_pz

    return z, log_pzk

  @tf.function
  def inverse( self, zk, is_train ):
    log_pzk = 0.0
    z = zk

    for cl in reversed(self.coupling_layers):
      z, log_pz = cl.inverse( z, is_train )

      log_pzk += log_pz

    log_pzk += tf.reduce_sum( log_normal_pdf( z, 0.0, 0.0 ), axis=1 )

    return z, log_pzk

  def log_pdf( self, z, is_train ):
    _, log_pdf = self.inverse( z, is_train )
    return log_pdf

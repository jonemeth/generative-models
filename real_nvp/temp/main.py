
from __future__ import absolute_import, division, print_function, unicode_literals


import os
import time

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from RealNVP import RealNVP

class MySoftPlus( tf.keras.Model ):
  def __init__( self ):
    super( MySoftPlus, self).__init__()

  def call(self, x):
    return tf.nn.softplus(x) - tf.nn.softplus(0.0)

def lrelu( alpha = 0.1 ):
  return tf.keras.layers.LeakyReLU( alpha = alpha )

def elu( ):
  return tf.keras.layers.Activation('elu')

def prelu( ):
  return tf.keras.layers.PReLU( alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None )

def safe_log( z ):
  return tf.math.log( 1e-8 + z )


def plot_density(density):

  X_LIMS = (-7, 7)
  Y_LIMS = (-7, 7)

  x1 = np.linspace(*X_LIMS, 300)
  x2 = np.linspace(*Y_LIMS, 300)
  x1, x2 = np.meshgrid(x1, x2)
  shape = x1.shape
  x1 = x1.ravel()
  x2 = x2.ravel()

  z = np.c_[x1, x2]

  density_values = density(z).numpy().reshape(shape)

  fig = plt.figure(figsize=(7, 7))
  ax = fig.add_subplot(111)
  ax.imshow(density_values, extent=(*X_LIMS, *Y_LIMS), cmap="summer")
  ax.set_title("True density")

  fig.savefig( 'density.png' )
  plt.close()
  
def log_normal_pdf(sample, mean, logvar):
  log2pi = safe_log(2. * np.pi)
  return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)


def p_z(z):
  #return tf.reduce_prod( tf.math.exp(log_normal_pdf( z, 0.0, -2.0 )), axis=1 )
  
  z1, z2 = tf.split(z, 2, axis=1 )
  norm = tf.math.sqrt(z1 ** 2 + z2 ** 2)

  exp1 = tf.math.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
  exp2 = tf.math.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
  u = 0.5 * ((norm - 4) / 0.4) ** 2 - safe_log(exp1 + exp2)

  return tf.math.exp(-u)

tf.random.set_seed(2)
np.random.seed(2)


model = RealNVP( 2, 64, MySoftPlus, random_shuffle=True )

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)

epochs = 25000
mb_size = 64
xlim = ylim = 5

@tf.function
def compute_loss( mb_size ):
  
  zk, log_q_zk = model.sample( mb_size, is_train = False )

  
  log_p_zk = safe_log( p_z(zk) )
  
  loss = tf.reduce_mean( -log_p_zk + log_q_zk )
  
  return loss


@tf.function
def compute_apply_gradients( mb_size ):
  with tf.GradientTape() as tape:
    loss  = compute_loss( mb_size )
    
    vars_to_train = model.trainable_variables
    
    gradients = tape.gradient(loss, vars_to_train)
    
  optimizer.apply_gradients(zip(gradients, vars_to_train))
  return loss


def plot_samples2(model, n=1000, xlim=4, ylim=4):
    x = np.linspace(-xlim, xlim, n)
    xx, yy = np.meshgrid(x, x)
    zz = np.stack((xx.flatten(), yy.flatten()), axis=-1)
    zk, final_log_prob = model.forward(zz.astype(np.float32), is_train=False)

    qk = tf.math.exp(final_log_prob)

    zk0, zk1 = tf.split( zk, 2, axis=1 )
    zk0 = tf.reshape( zk0, [n, n] )
    zk1 = tf.reshape( zk1, [n, n] )
    qk = tf.reshape( qk, [n, n] )

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect(1)
    ax.pcolormesh( zk0, zk1, qk, cmap="Blues" )

    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )

    ax.set_facecolor(plt.cm.Blues(0.0))
    return ax



plot_density( p_z )

for epoch in range(1, epochs + 1):
    loss = compute_apply_gradients( mb_size  )
    print( epoch, loss.numpy(), flush=True )
    
    if epoch % 500 == 0:
        ax = plot_samples2(model, xlim=xlim, ylim=ylim)
        ax.text(
            0,
            ylim - 2,
            "Iteration #{:06d}".format(epoch),
            horizontalalignment="center",
        )
        plt.savefig(
            "iteration_{:06d}.png".format(epoch),
            bbox_inches="tight",
            pad_inches=0.5,
        )
        plt.close()
    
z, _ = model.sample( 10000 )
plt.scatter( z[:,0], z[:,1], s = 1 )
plt.xlim(-6.5, 6.5)
plt.ylim(-6.5, 6.5)
plt.colorbar()
plt.savefig( 'scatter.png' )
plt.close()


######################################################x


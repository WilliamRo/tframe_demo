from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import GAN
from tframe import pedia

from tframe.layers import Linear
from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Conv2D
from tframe.layers import Deconv2D
from tframe.layers import Rescale
from tframe.layers import Reshape


# region : Vanilla

def vanilla(mark, bn=True):
  model = GAN(z_dim=100, sample_shape=[784], mark=mark, #classes=10,
              output_shape=[28, 28, 1])

  model.G.add(Linear(output_dim=128))
  if bn:
    model.G.add(BatchNorm())
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=256))
  if bn:
    model.G.add(BatchNorm())
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=784))
  model.G.add(Activation('tanh'))

  model.G.add(Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))

  # ============================================================================

  model.D.add(Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  model.D.add(Linear(output_dim=256))
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=128))
  if bn:
    model.D.add(BatchNorm())
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=1))
  model.D.add(Activation('sigmoid'))

  # Build model
  model.build(loss=pedia.cross_entropy, smooth_factor=0.9)

  return model

def vanilla_h3_rs_nbn(mark):
  model = GAN(z_dim=100, sample_shape=[784], mark=mark,
              output_shape=[28, 28, 1])

  model.G.add(Linear(output_dim=64))
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=128))
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=128))
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=784))
  model.G.add(Activation('tanh'))

  model.G.add(Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))
  # ============================================================================
  model.D.add(Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  model.D.add(Linear(output_dim=128))
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=128))
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=64))
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=1))
  model.D.add(Activation('sigmoid'))

  model.build(loss=pedia.cross_entropy, smooth_factor=0.9)

  return model


# endregion : Vanilla


# region : DCGAN

def dcgan(mark):
  # Initiate model
  model = GAN(z_dim=100, sample_shape=[28, 28, 1], mark=mark, classes=10)

  # Define generator
  model.G.add(Linear(output_dim=7*7*128))
  model.G.add(Reshape(shape=[7, 7, 128]))
  model.G.add(BatchNorm())
  model.G.add(Activation.ReLU())

  model.G.add(Deconv2D(filters=128, kernel_size=5, strides=2, padding='same'))
  model.G.add(BatchNorm())
  model.G.add(Activation.ReLU())

  model.G.add(Deconv2D(filters=1, kernel_size=5, strides=2, padding='same'))
  model.G.add(Activation('sigmoid'))
  # model.G.add(Activation('tanh'))

  # model.G.add(Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))

  # Define discriminator
  # model.D.add(Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  model.D.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
  model.D.add(Activation.LeakyReLU())

  model.D.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
  model.D.add(BatchNorm())
  model.D.add(Activation.LeakyReLU())

  model.D.add(Reshape(shape=[7*7*128]))
  model.D.add(Linear(output_dim=1))
  model.D.add(Activation('sigmoid'))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
  model.build(loss=pedia.cross_entropy, G_optimizer=optimizer,
              D_optimizer=optimizer)

  return model


def dcgan_h3_rs_nbn(mark='dcgan_h3_nbn'):
  # Initiate model
  model = GAN(z_dim=100, sample_shape=[28, 28, 1], mark=mark)

  # Define generator
  model.G.add(Linear(output_dim=128*7*7))
  model.G.add(Activation.ReLU())
  model.G.add(Reshape(shape=[7, 7, 128]))

  model.G.add(Deconv2D(filters=64, kernel_size=5, strides=2, padding='same'))
  model.G.add(Activation.ReLU())

  model.G.add(Deconv2D(filters=1, kernel_size=5, strides=2, padding='same'))
  model.G.add(Activation('tanh'))
  model.G.add(Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))

  # Define discriminator
  model.D.add(Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  model.D.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
  model.D.add(Activation.LeakyReLU())

  model.D.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
  model.D.add(Activation.LeakyReLU())

  model.D.add(Reshape(shape=[7*7*128]))
  model.D.add(Linear(output_dim=1))
  model.D.add(Activation('sigmoid'))

  # Build model
  optimizer = tf.train.AdamOptimizer()
  model.build(loss=pedia.cross_entropy, G_optimizer=optimizer,
              D_optimizer=optimizer)

  return model

# endregion : DCGAN


def conditional(mark):
  pass



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import GAN
from tframe import pedia
from tframe import regularizers

from tframe.layers import Linear
from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Conv2D
from tframe.layers import Deconv2D
from tframe.layers import Flatten
from tframe.layers import Rescale
from tframe.layers import Reshape


# region : Vanilla

def vanilla(mark):
  model = GAN(z_dim=100, sample_shape=[3072], mark=mark, classes=10,
              output_shape=[32, 32, 3])

  # Define generator
  model.G.add(Linear(output_dim=128))
  model.G.add(BatchNorm())
  model.G.add(Activation('relu'))


  model.G.add(Linear(output_dim=256))
  model.G.add(BatchNorm())
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=512))
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=3072))
  model.G.add(Activation('tanh'))

  model.G.add(Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))

  # ===========================================================================

  # Define discriminator
  model.D.add(Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  model.D.add(Linear(output_dim=256))
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=128))
  model.D.add(BatchNorm())
  model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=1))
  model.D.add(Activation('sigmoid'))

  # Build model
  D_optimizer = None
  D_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
  G_optimizer = None
  G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
  model.build(loss=pedia.cross_entropy, D_optimizer=D_optimizer,
              G_optimizer=G_optimizer, smooth_factor=0.9)

  return model

# endregion : Vanilla


# region : DCGAN

def dcgan(mark):
  model = GAN(z_dim=100, sample_shape=[32, 32, 3], mark=mark, classes=10)

  nch = 256
  h = 5
  reg = regularizers.L2(strength=1e-7)

  # Define generator
  model.G.add(Linear(output_dim=nch * 4 * 4, weight_regularizer=reg))
  model.G.add(BatchNorm())
  model.G.add(Reshape(shape=(4, 4, nch)))

  model.G.add(Deconv2D(filters=int(nch / 2), kernel_size=5,
                       padding='same', kernel_regularizer=reg))
  model.G.add(BatchNorm())
  model.G.add(Activation.LeakyReLU())

  model.G.add(Deconv2D(filters=int(nch / 2), kernel_size=5, strides=2,
                       padding='same', kernel_regularizer=reg))
  model.G.add(BatchNorm())
  model.G.add(Activation.LeakyReLU())

  model.G.add(Deconv2D(filters=int(nch / 4), kernel_size=5, strides=2,
                       padding='same', kernel_regularizer=reg))
  model.G.add(BatchNorm())
  model.G.add(Activation.LeakyReLU())

  model.G.add(Deconv2D(filters=3, kernel_size=5, strides=2,
                       padding='same', kernel_regularizer=reg))
  model.G.add(Activation('sigmoid'))

  # ===========================================================================

  # Define discriminator
  model.D.add(Conv2D(filters=int(nch / 4), kernel_size=h, strides=2,
                     padding='same', kernel_regularizer=reg))
  model.D.add(Activation.LeakyReLU())

  model.D.add(Conv2D(filters=int(nch / 2), kernel_size=h, strides=2,
                     padding='same', kernel_regularizer=reg))
  model.D.add(Activation.LeakyReLU())

  model.D.add(Conv2D(filters=nch, kernel_size=h, strides=2,
                     padding='same', kernel_regularizer=reg))
  model.D.add(Activation.LeakyReLU())

  model.D.add(Flatten())
  model.D.add(Linear(output_dim=1, weight_regularizer=reg))
  model.D.add(Activation('sigmoid'))

  # Build model
  optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
  model.build(loss=pedia.cross_entropy, G_optimizer=optimizer,
              D_optimizer=optimizer)

  return model

# endregion : DCGAN


'''For some reason, do not delete this line.'''

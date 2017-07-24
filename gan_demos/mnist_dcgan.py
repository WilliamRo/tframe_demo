from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import FLAGS
from tframe import console

from tframe import GAN
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Conv2D, Deconv2D
from tframe.layers import Reshape
from tframe.layers import BatchNorm

from tframe import pedia

from tframe.utils import imtool


def main(_):
  console.suppress_logging()
  # Start
  console.start("MNIST DCGAN DEMO")

  # Define model
  # :: If z_dim and sample_shape are specified here, input layers for
  # .. .. G and D will be automatically generated.
  model = GAN(z_dim=100, sample_shape=[28, 28, 1], mark='dcgan00')

  # .. define generator
  model.G.add(Linear(output_dim=7*7*128))
  model.G.add(Reshape(shape=[7, 7, 128]))
  model.G.add(BatchNorm())
  model.G.add(Activation.ReLU())

  model.G.add(Deconv2D(filters=128, kernel_size=5, strides=2, padding='same'))
  model.G.add(BatchNorm())
  model.G.add(Activation.ReLU())

  model.G.add(Deconv2D(filters=1, kernel_size=5, strides=2, padding='same'))
  model.G.add(Activation('sigmoid'))

  # .. define discriminator
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
  model.build(loss=pedia.cross_entropy, optimizer=optimizer)

  # Train or test
  if FLAGS.train:
    mnist = load_mnist(r'..\data\MNIST', flatten=False, validation_size=0)
    model.train(training_set=mnist['train'], epoch=5, batch_size=128,
                print_cycle=20, snapshot_cycle=200, D_times=1, G_times=1)
  else:
    samples = model.generate(sample_num=16)
    console.show_status('{} samples generated'.format(samples.shape[0]))
    imtool.gan_grid_plot(samples, show=True)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()
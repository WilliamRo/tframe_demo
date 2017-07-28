from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import GAN
from tframe import pedia

from tframe.layers import Linear
from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Rescale


def vanilla(mark, sample_num=9):
  model = GAN(z_dim=100, sample_shape=[784], mark=mark,
              output_shape=[28, 28, 1], sample_num=sample_num)

  # model.G.add(Linear(output_dim=64))
  # model.G.add(BatchNorm())
  # model.G.add(Activation('lrelu'))

  model.G.add(Linear(output_dim=1024))
  # model.G.add(BatchNorm())
  model.G.add(Activation('relu'))

  model.G.add(Linear(output_dim=784))
  model.G.add(Activation('sigmoid'))

  # model.G.add(Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))

  # ============================================================================

  # model.D.add(Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  model.D.add(Linear(output_dim=1024))
  model.D.add(Activation('relu'))

  # model.D.add(Linear(output_dim=64))
  # model.G.add(BatchNorm())
  # model.D.add(Activation('lrelu'))

  model.D.add(Linear(output_dim=1))
  model.D.add(Activation('sigmoid'))

  # D_optimizer = tf.train.GradientDescentOptimizer(0.001)
  D_optimizer = None
  model.build(loss=pedia.cross_entropy, D_optimizer=D_optimizer)

  return model


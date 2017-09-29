from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import GAN
from tframe import pedia

from tframe.layers import Linear
from tframe.layers import Activation


# region : Vanilla

def vanilla(mark):
  model = GAN(z_dim=100, sample_shape=[], mark=mark, classes=10,
              output_shape=[])
  # ===========================================================================

  return model


# endregion : Vanilla


# region : DCGAN
# endregion : DCGAN


'''For some reason, do not delete this line.'''

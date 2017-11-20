from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import VAE
from tframe import console
from tframe import pedia

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Conv2D
from tframe.layers import Deconv2D
from tframe.layers import Linear
from tframe.layers import Rescale
from tframe.layers import Reshape

from tframe.nets import Fork


# region : Vanilla

def vanilla(mark, bn=False):
  z_dim = 100
  model = VAE(z_dim=z_dim, mark=mark, classes=0,
              sample_shape=[784], output_shape=[28, 28, 1])

  # Define encoder
  model.Q.add(Linear(output_dim=128))
  model.Q.add(Activation.ReLU())

  fork = Fork(name='mu_sigma')
  fork.add('mu', Linear(output_dim=z_dim))
  fork.add('sigma', Linear(output_dim=z_dim))

  model.Q.add(fork)

  # Define decoder
  model.P.add(Linear(output_dim=128))
  model.P.add(Activation.ReLU())
  model.P.add(Linear(output_dim=784))
  model.P.add(Activation('sigmoid'))

  # Build model
  model.build()

  return model

# endregion : Vanilla


if __name__ == '__main__':
  console.suppress_logging()
  model = vanilla(mark='test')

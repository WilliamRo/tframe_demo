import tensorflow as tf

from tframe import Classifier

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Conv2D
from tframe.layers import Flatten
from tframe.layers import Linear
from tframe.layers import MaxPool2D
from tframe.layers import Input


SAMPLE_SHAPE = [32, 32, 3]
USE_BN = True


def deep_conv(mark, learning_rate):
  # Initialize Classifier
  model = Classifier(mark=mark)
  model.add(Input(sample_shape=SAMPLE_SHAPE))

  def ConvBNReLU(filters, bn=False):
    model.add(Conv2D(filters=filters, kernel_size=5, padding='same'))
    if bn: model.add(BatchNorm())
    model.add(Activation.ReLU())

  # Add Conv layers
  ConvBNReLU(32, USE_BN)
  ConvBNReLU(32, USE_BN)
  model.add(MaxPool2D(2, 2, 'same'))

  ConvBNReLU(64, USE_BN)
  ConvBNReLU(64, USE_BN)
  model.add(MaxPool2D(2, 2, 'same'))

  ConvBNReLU(128, USE_BN)

  # Add FC layers
  model.add(Flatten())
  model.add(Linear(256))
  model.add(Activation.ReLU())

  model.add(Linear(10))

  # Build model
  model.build(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))

  # Return model
  return model




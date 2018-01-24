from  __future__ import absolute_import
from  __future__ import division
from  __future__ import print_function

import tensorflow as tf

from tframe import Classifier
from tframe import pedia

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Conv2D
from tframe.layers import Dropout
from tframe.layers import Flatten
from tframe.layers import Linear
from tframe.layers import MaxPool2D
from tframe.layers import Input

from tframe import regularizers

import config


# region : Deep Convolutional

def vanilla(mark):
  model = Classifier(mark=mark)
  model.add(Input(sample_shape=[784]))

  def fc_bn_relu(bn=True):
    model.add(Linear(100))
    model.add(Activation('relu'))
    if bn:
      model.add(BatchNorm())

  fc_bn_relu()
  fc_bn_relu()

  model.add(Linear(10))

  # Build model
  model.build(loss='cross_entropy',
              optimizer=tf.train.GradientDescentOptimizer(0.01))

  return model


def deep_conv(mark):
  # Initiate predictor
  model = Classifier(mark=mark)
  model.add(Input(sample_shape=[28, 28, 1]))

  def ConvBNReLU(filters, strength=1.0, bn=True):
    model.add(Conv2D(filters=filters, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.L2(strength=strength)))

    if bn:
      model.add(BatchNorm())

    model.add(Activation('relu'))

  # Conv layers
  reg = 1e-5
  ConvBNReLU(32, reg)
  model.add(Dropout(0.5))
  ConvBNReLU(32, reg)

  model.add(MaxPool2D(2, 2, 'same'))

  ConvBNReLU(64, reg)
  model.add(Dropout(0.5))
  ConvBNReLU(64, reg)

  model.add(MaxPool2D(2, 2, 'same'))

  ConvBNReLU(128, reg)

  # FC layers
  model.add(Flatten())
  model.add(Linear(256))
  # model.add(BatchNorm())
  model.add(Activation('relu'))

  model.add(Linear(256))
  # model.add(BatchNorm())
  model.add(Activation('relu'))

  model.add(Linear(config.y_dim))

  # Build model
  model.build(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4))

  return model


def ka_convnet(mark):
  model = Classifier(mark=mark)
  model.add(Input(sample_shape=config.sample_shape))

  strength = 1e-5
  def ConvLayer(filters, bn=False):
    model.add(Conv2D(filters=filters, kernel_size=5, padding='same',
                     kernel_regularizer=regularizers.L2(strength=strength)))
    if bn:
      model.add(BatchNorm())
    model.add(Activation.ReLU())

  # Define structure
  ConvLayer(32)
  model.add(Dropout(0.5))
  ConvLayer(32, False)
  model.add(Dropout(0.5))
  model.add(MaxPool2D(2, 2, 'same'))
  ConvLayer(64, True)
  model.add(Dropout(0.5))
  model.add(MaxPool2D(2, 2, 'same'))

  model.add(Flatten())
  model.add(Linear(128))
  model.add(Activation.ReLU())
  # model.add(Dropout(0.5))
  model.add(Linear(10))

  # Build model
  model.build(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4))

  return model

# endregion : Deep Convolutional
